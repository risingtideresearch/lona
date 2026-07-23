import type { NumNode, VarName } from "lona/internal";
import {
  KIND_LIT,
  LiteralNum,
  compileJvpRoutineFromTape,
  compileTape,
  type SeededJvpRoutine,
  type VarMap,
} from "lona/internal";
import type {
  ColumnarDefinition,
  ColumnarParamBinding,
  ColumnarStage,
  MapStage,
  ReduceStage,
  ScalarBindings,
  SourceStage,
  WholeColumnStage,
} from "./ir";
import type {
  ColumnarAutodiffResult,
  ColumnarGradRoutine,
  ColumnarRoutineOptions,
  CpuBackendName,
  StagePlacement,
} from "./types";
import {
  compileCpuJvpKernel,
  type CompiledCpuJvpKernel,
} from "./cpu/compile-kernel";

type ExecutableStage = ColumnarStage;

interface CompiledExternalRoots {
  readonly roots: readonly NumNode[];
  readonly routine?: SeededJvpRoutine;
  readonly seeds: Float64Array;
}

interface DualValues {
  readonly values: number[];
  /** Scalar-major, direction-minor. */
  readonly tangents: number[];
}

interface CompiledJvpStage {
  readonly stage: Exclude<ExecutableStage, SourceStage>;
  readonly backend: CpuBackendName;
  readonly kernel: CompiledCpuJvpKernel;
}

const CPU_BACKENDS: ReadonlySet<string> = new Set([
  "js-interp",
  "js-codegen",
  "wasm-interp",
  "wasm-codegen",
]);

function gatherScalarRoots(definition: ColumnarDefinition): NumNode[] {
  const roots: NumNode[] = [];
  const seen = new Set<NumNode>();
  const add = (items: readonly NumNode[]): void => {
    for (const root of items) {
      if (root.kind === KIND_LIT || seen.has(root)) continue;
      seen.add(root);
      roots.push(root);
    }
  };
  const addUsing = (using: ScalarBindings): void => add(using.roots);

  for (const stage of definition.stages) {
    if (stage.kind === "source") continue;
    if (stage.kind === "map") {
      addUsing(stage.using);
    } else if (stage.kind === "reduce") {
      add(stage.initial);
      addUsing(stage.using);
    } else {
      addUsing(stage.using);
    }
  }
  return roots;
}

function identitySeeds(
  routine: SeededJvpRoutine,
  diffVars: readonly VarName[],
): Float64Array {
  const seeds = new Float64Array(routine.numVars * diffVars.length);
  for (let input = 0; input < routine.numVars; input++) {
    for (let direction = 0; direction < diffVars.length; direction++) {
      if (routine.varSlots[input] === diffVars[direction]) {
        seeds[input * diffVars.length + direction] = 1;
      }
    }
  }
  return seeds;
}

function compileExternalRoots(
  roots: readonly NumNode[],
  diffVars: readonly VarName[],
  backends: readonly CpuBackendName[],
): CompiledExternalRoots {
  if (roots.length === 0) {
    return { roots: Object.freeze([]), seeds: new Float64Array(0) };
  }
  const tape = compileTape([...roots]);
  if (!tape) throw new Error("failed to compile columnar autodiff roots");
  const failures: string[] = [];
  for (const backend of backends) {
    try {
      const routine = compileJvpRoutineFromTape(tape, diffVars.length, {
        backend,
      });
      return {
        roots: Object.freeze([...roots]),
        routine,
        seeds: identitySeeds(routine, diffVars),
      };
    } catch (error) {
      failures.push(error instanceof Error ? error.message : String(error));
    }
  }
  throw new Error(
    `failed to compile columnar external-root JVP: ${failures.join("; ")}`,
  );
}

function valuesForVars(routine: SeededJvpRoutine, vars: VarMap): Float64Array {
  const values = vars as Map<VarName, number>;
  return Float64Array.from(
    routine.varSlots.slice(0, routine.numVars),
    (name) => values.get(name) ?? 0,
  );
}

function evaluateExternalRoots(
  compiled: CompiledExternalRoots,
  vars: VarMap,
): DualValues {
  if (!compiled.routine) return { values: [], tangents: [] };
  const result = compiled.routine.evalPacked(
    valuesForVars(compiled.routine, vars),
    compiled.seeds,
  );
  return { values: result.vals, tangents: result.tangents.flat() };
}

function placementKey(
  stage: ExecutableStage,
): keyof NonNullable<Exclude<ColumnarRoutineOptions["placement"], string>> {
  return stage.kind;
}

function effectivePlacement(
  stage: ExecutableStage,
  opts: ColumnarRoutineOptions,
): StagePlacement {
  if (stage.requestedPlacement) return stage.requestedPlacement;
  if (typeof opts.placement === "string") return opts.placement;
  return (
    opts.placement?.[placementKey(stage)] ?? opts.placement?.default ?? "auto"
  );
}

function configuredCpuBackends(
  opts: ColumnarRoutineOptions,
): readonly CpuBackendName[] {
  const configured = opts.backends?.cpu;
  const candidates =
    configured === undefined
      ? (["wasm-codegen"] as const)
      : Array.isArray(configured)
        ? configured
        : [configured];
  if (candidates.length === 0) {
    throw new Error(
      "columnar autodiff CPU backend preference must not be empty",
    );
  }
  return candidates;
}

function stageBackends(
  stage: ExecutableStage,
  opts: ColumnarRoutineOptions,
): readonly CpuBackendName[] {
  const placement = effectivePlacement(stage, opts);
  const autoTargets =
    opts.auto?.targets?.[stage.kind] ?? opts.auto?.targets?.default;
  if (
    placement === "gpu" ||
    (placement === "auto" && autoTargets && !autoTargets.includes("cpu"))
  ) {
    throw new Error(
      `columnar autodiff stage ${stage.id} (${stage.kind}) currently supports CPU placement only`,
    );
  }
  if (stage.requestedBackend) {
    if (!CPU_BACKENDS.has(stage.requestedBackend)) {
      throw new Error(
        `columnar autodiff backend '${stage.requestedBackend}' is not a CPU backend`,
      );
    }
    return [stage.requestedBackend as CpuBackendName];
  }
  return configuredCpuBackends(opts);
}

function compileStage(
  stage: Exclude<ExecutableStage, SourceStage>,
  numDirections: number,
  opts: ColumnarRoutineOptions,
): CompiledJvpStage {
  const roots =
    stage.kind === "map" || stage.kind === "reduce"
      ? stage.kernel.roots
      : stage.roots;
  const inputs =
    stage.kind === "map" || stage.kind === "reduce"
      ? stage.kernel.inputs
      : stage.inputs;
  const failures: string[] = [];
  for (const backend of stageBackends(stage, opts)) {
    try {
      return {
        stage,
        backend,
        kernel: compileCpuJvpKernel(roots, inputs, backend, numDirections),
      };
    } catch (error) {
      failures.push(
        `${backend}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  throw new Error(
    `columnar autodiff stage ${stage.id} (${stage.kind}) failed to compile: ${failures.join("; ")}`,
  );
}

function dualForRoots(
  roots: readonly NumNode[],
  scalarValues: ReadonlyMap<NumNode, { value: number; tangents: number[] }>,
  numDirections: number,
): DualValues {
  const values: number[] = [];
  const tangents: number[] = [];
  for (const root of roots) {
    if (root.kind === KIND_LIT) {
      values.push((root as LiteralNum).value);
      for (let direction = 0; direction < numDirections; direction++) {
        tangents.push(0);
      }
      continue;
    }
    const dual = scalarValues.get(root);
    if (!dual)
      throw new Error("columnar autodiff scalar root was not evaluated");
    values.push(dual.value);
    tangents.push(...dual.tangents);
  }
  return { values, tangents };
}

function appendScalar(
  output: { values: number[]; tangents: number[] },
  dual: DualValues,
  scalar: number,
  numDirections: number,
): void {
  const value = dual.values[scalar];
  if (value === undefined)
    throw new Error(`missing columnar JVP input ${scalar}`);
  output.values.push(value);
  for (let direction = 0; direction < numDirections; direction++) {
    output.tangents.push(
      dual.tangents[scalar * numDirections + direction] ?? 0,
    );
  }
}

function kernelInputs(
  bindings: readonly ColumnarParamBinding[],
  source: DualValues,
  using: DualValues,
  numDirections: number,
  rowIndex = 0,
  rowWidth = 0,
  left?: DualValues,
  right?: DualValues,
): DualValues {
  const output: { values: number[]; tangents: number[] } = {
    values: [],
    tangents: [],
  };
  for (const binding of bindings) {
    switch (binding.kind) {
      case "row":
        appendScalar(
          output,
          source,
          rowIndex * rowWidth + binding.component,
          numDirections,
        );
        break;
      case "uniform":
        appendScalar(output, using, binding.component, numDirections);
        break;
      case "index":
        output.values.push(rowIndex);
        for (let direction = 0; direction < numDirections; direction++) {
          output.tangents.push(0);
        }
        break;
      case "reduce-left":
        appendScalar(output, left!, binding.component, numDirections);
        break;
      case "reduce-right":
        appendScalar(output, right!, binding.component, numDirections);
        break;
      case "materialized":
        appendScalar(output, source, binding.component, numDirections);
        break;
    }
  }
  return output;
}

function rowDual(
  source: DualValues,
  width: number,
  row: number,
  numDirections: number,
): DualValues {
  return {
    values: source.values.slice(row * width, (row + 1) * width),
    tangents: source.tangents.slice(
      row * width * numDirections,
      (row + 1) * width * numDirections,
    ),
  };
}

function evalMap(
  compiled: CompiledJvpStage,
  stage: MapStage,
  source: DualValues,
  using: DualValues,
  numDirections: number,
): DualValues {
  const output: DualValues = { values: [], tangents: [] };
  for (let row = 0; row < stage.count; row++) {
    const inputs = kernelInputs(
      stage.kernel.inputs.map(({ binding }) => binding),
      source,
      using,
      numDirections,
      row,
      stage.inputShape.width,
    );
    const result = compiled.kernel.eval(inputs.values, inputs.tangents);
    output.values.push(...result.values);
    output.tangents.push(...result.tangents);
  }
  return output;
}

function combineReduce(
  compiled: CompiledJvpStage,
  stage: ReduceStage,
  left: DualValues,
  right: DualValues,
  using: DualValues,
  numDirections: number,
): DualValues {
  const inputs = kernelInputs(
    stage.kernel.inputs.map(({ binding }) => binding),
    { values: [], tangents: [] },
    using,
    numDirections,
    0,
    0,
    left,
    right,
  );
  return compiled.kernel.eval(inputs.values, inputs.tangents);
}

function evalReduce(
  compiled: CompiledJvpStage,
  stage: ReduceStage,
  source: DualValues,
  using: DualValues,
  initial: DualValues,
  numDirections: number,
): DualValues {
  if (stage.builtIn) {
    if (stage.inputCount === 0) return initial;
    let level = Array.from({ length: stage.inputCount }, (_, row) =>
      rowDual(source, stage.shape.width, row, numDirections),
    );
    while (level.length > 1) {
      const next: DualValues[] = [];
      for (let index = 0; index < level.length; index += 2) {
        const right = level[index + 1];
        next.push(
          right
            ? combineReduce(
                compiled,
                stage,
                level[index]!,
                right,
                using,
                numDirections,
              )
            : level[index]!,
        );
      }
      level = next;
    }
    return level[0]!;
  }

  if (stage.order === "left") {
    let accumulator = initial;
    for (let row = 0; row < stage.inputCount; row++) {
      accumulator = combineReduce(
        compiled,
        stage,
        accumulator,
        rowDual(source, stage.shape.width, row, numDirections),
        using,
        numDirections,
      );
    }
    return accumulator;
  }

  let level = [
    initial,
    ...Array.from({ length: stage.inputCount }, (_, row) =>
      rowDual(source, stage.shape.width, row, numDirections),
    ),
  ];
  while (level.length > 1) {
    const next: DualValues[] = [];
    for (let index = 0; index < level.length; index += 2) {
      const right = level[index + 1];
      next.push(
        right
          ? combineReduce(
              compiled,
              stage,
              level[index]!,
              right,
              using,
              numDirections,
            )
          : level[index]!,
      );
    }
    level = next;
  }
  return level[0] ?? initial;
}

function evalWhole(
  compiled: CompiledJvpStage,
  stage: WholeColumnStage,
  source: DualValues,
  using: DualValues,
  numDirections: number,
): DualValues {
  const inputs = kernelInputs(
    stage.inputs.map(({ binding }) => binding),
    source,
    using,
    numDirections,
  );
  return compiled.kernel.eval(inputs.values, inputs.tangents);
}

/** Compile CPU forward-mode differentiation for a complete columnar graph. */
export function compileColumnarGradRoutine(
  definition: ColumnarDefinition,
  diffVars: readonly VarName[],
  opts: ColumnarRoutineOptions = {},
): ColumnarGradRoutine {
  const source = definition.stages[0];
  if (!source || source.kind !== "source") {
    throw new Error("columnar autodiff requires a source stage");
  }
  if (!definition.resultShape) {
    throw new Error("columnar autodiff routine must end with column.output()");
  }

  const frozenDiffVars = Object.freeze([...diffVars]);
  const sourceRoots = compileExternalRoots(
    source.roots,
    frozenDiffVars,
    stageBackends(source, opts),
  );
  const scalarRoots = compileExternalRoots(
    gatherScalarRoots(definition),
    frozenDiffVars,
    configuredCpuBackends(opts),
  );
  const compiledStages: CompiledJvpStage[] = [];
  try {
    for (const stage of definition.stages.slice(1)) {
      if (stage.kind === "source") {
        throw new Error("columnar autodiff does not support multiple sources");
      }
      compiledStages.push(compileStage(stage, diffVars.length, opts));
    }
  } catch (error) {
    sourceRoots.routine?.dispose?.();
    scalarRoots.routine?.dispose?.();
    for (const compiled of compiledStages) compiled.kernel.dispose();
    throw error;
  }

  const output = definition.stages[definition.outputStage]!;
  const numRoots = definition.resultShape.values.reduce(
    (total, shape) => total + shape.width,
    0,
  );
  const shape = numRoots === 1 ? "grad" : "jacobian";
  const varSlots: VarName[] = [];
  const seen = new Set<VarName>();
  for (const routine of [sourceRoots.routine, scalarRoots.routine]) {
    for (const name of routine?.varSlots.slice(0, routine.numVars) ?? []) {
      if (seen.has(name)) continue;
      seen.add(name);
      varSlots.push(name);
    }
  }
  let disposed = false;

  return {
    shape,
    diffVars: frozenDiffVars,
    varSlots: Object.freeze(varSlots),
    numVars: varSlots.length,
    numRoots,
    async evalAsync(vars: VarMap): Promise<ColumnarAutodiffResult> {
      if (disposed)
        throw new Error("columnar autodiff routine has been disposed");

      const sourceDual = evaluateExternalRoots(sourceRoots, vars);
      const scalarDual = evaluateExternalRoots(scalarRoots, vars);
      const scalarValues = new Map<
        NumNode,
        { value: number; tangents: number[] }
      >();
      for (let root = 0; root < scalarRoots.roots.length; root++) {
        scalarValues.set(scalarRoots.roots[root]!, {
          value: scalarDual.values[root]!,
          tangents: scalarDual.tangents.slice(
            root * diffVars.length,
            (root + 1) * diffVars.length,
          ),
        });
      }

      const stageValues = new Map<number, DualValues>([
        [source.id, sourceDual],
      ]);
      for (const compiled of compiledStages) {
        const stage = compiled.stage;
        const sourceValue = stageValues.get(stage.source);
        if (!sourceValue) {
          throw new Error(
            `columnar autodiff stage ${stage.id} is missing its source`,
          );
        }
        const using = dualForRoots(
          stage.using.roots,
          scalarValues,
          diffVars.length,
        );
        const result =
          stage.kind === "map"
            ? evalMap(compiled, stage, sourceValue, using, diffVars.length)
            : stage.kind === "reduce"
              ? evalReduce(
                  compiled,
                  stage,
                  sourceValue,
                  using,
                  dualForRoots(stage.initial, scalarValues, diffVars.length),
                  diffVars.length,
                )
              : evalWhole(compiled, stage, sourceValue, using, diffVars.length);
        stageValues.set(stage.id, result);
      }

      const result = stageValues.get(output.id);
      if (!result) throw new Error("columnar autodiff produced no output");
      return shape === "grad"
        ? {
            val: result.values[0]!,
            gradient: result.tangents.slice(0, diffVars.length),
          }
        : {
            vals: result.values,
            jacobian: Array.from({ length: numRoots }, (_, root) =>
              result.tangents.slice(
                root * diffVars.length,
                (root + 1) * diffVars.length,
              ),
            ),
          };
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      sourceRoots.routine?.dispose?.();
      scalarRoots.routine?.dispose?.();
      for (const compiled of compiledStages) compiled.kernel.dispose();
    },
  };
}
