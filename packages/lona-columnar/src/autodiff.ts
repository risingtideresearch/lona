import type {
  BackendName,
  DeviceBufferSlice,
  NumNode,
  VarName,
} from "lona/internal";
import {
  KIND_LIT,
  LiteralNum,
  compileJvpRoutineFromTape,
  compileTape,
  isGpuAvailable,
  mapGpuReadbackBuffer,
  requireGpuDevice,
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
  ExecutionTarget,
  GpuBackendName,
  StagePlacement,
} from "./types";
import {
  compileCpuJvpKernel,
  type CompiledCpuJvpKernel,
} from "./cpu/compile-kernel";
import {
  compileGpuInterpJvpMapKernel,
  compileGpuJvpMapKernel,
  type CompiledGpuJvpMapKernel,
} from "./gpu/jvp-map-kernel";
import {
  compileGpuInterpJvpReduceKernel,
  compileGpuJvpReduceKernel,
  type CompiledGpuJvpReduceKernel,
} from "./gpu/jvp-reduce-kernel";
import {
  compileGpuInterpJvpSourceKernel,
  compileGpuJvpSourceKernel,
  type CompiledGpuJvpSourceKernel,
} from "./gpu/jvp-source-kernel";

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
  readonly placement: ExecutionTarget;
  readonly backend: BackendName;
  readonly cpuKernel?: CompiledCpuJvpKernel;
  readonly gpuMapKernel?: CompiledGpuJvpMapKernel;
  readonly gpuReduceKernel?: CompiledGpuJvpReduceKernel;
}

interface DeviceDualValues {
  readonly location: "device";
  readonly values: DeviceBufferSlice;
  readonly tangents: DeviceBufferSlice;
  readonly count: number;
  readonly width: number;
}

interface HostDualValues extends DualValues {
  readonly location: "host";
}

type StageDualValues = HostDualValues | DeviceDualValues;

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

function configuredGpuBackends(
  opts: ColumnarRoutineOptions,
): readonly GpuBackendName[] {
  const configured = opts.backends?.gpu;
  const candidates =
    configured === undefined
      ? (["gpu-codegen"] as const)
      : Array.isArray(configured)
        ? configured
        : [configured];
  if (candidates.length === 0) {
    throw new Error(
      "columnar autodiff GPU backend preference must not be empty",
    );
  }
  return candidates;
}

function stageTargets(
  stage: ExecutableStage,
  opts: ColumnarRoutineOptions,
): readonly ExecutionTarget[] {
  const placement = effectivePlacement(stage, opts);
  if (placement !== "auto") return [placement];
  const configured =
    opts.auto?.targets?.[stage.kind] ?? opts.auto?.targets?.default;
  if (configured) return configured;
  return stage.kind === "source"
    ? ["cpu", "gpu"]
    : stage.kind === "map" || stage.kind === "reduce"
      ? ["gpu", "cpu"]
      : ["cpu"];
}

function requestedTarget(stage: ExecutableStage): ExecutionTarget | null {
  if (!stage.requestedBackend) return null;
  if (CPU_BACKENDS.has(stage.requestedBackend)) return "cpu";
  if (
    stage.requestedBackend === "gpu-codegen" ||
    stage.requestedBackend === "gpu-interp"
  )
    return "gpu";
  throw new Error(
    `columnar autodiff backend '${stage.requestedBackend}' does not support GPU JVP`,
  );
}

function targetBackends(
  stage: ExecutableStage,
  target: ExecutionTarget,
  opts: ColumnarRoutineOptions,
): readonly (CpuBackendName | GpuBackendName)[] {
  if (stage.requestedBackend) {
    return [stage.requestedBackend as CpuBackendName | GpuBackendName];
  }
  return target === "cpu"
    ? configuredCpuBackends(opts)
    : configuredGpuBackends(opts);
}

interface ResolvedSource {
  readonly placement: ExecutionTarget;
  readonly backend: BackendName;
}

function resolveSource(
  stage: SourceStage,
  opts: ColumnarRoutineOptions,
): ResolvedSource {
  const required = requestedTarget(stage);
  const failures: string[] = [];
  for (const target of stageTargets(stage, opts)) {
    if (required && required !== target) continue;
    if (target === "gpu" && !isGpuAvailable()) {
      failures.push("GPU is unavailable");
      continue;
    }
    for (const backend of targetBackends(stage, target, opts)) {
      return { placement: target, backend };
    }
  }
  throw new Error(
    `columnar autodiff source could not resolve placement: ${failures.join("; ")}`,
  );
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
  const required = requestedTarget(stage);
  const failures: string[] = [];
  for (const target of stageTargets(stage, opts)) {
    if (required && required !== target) continue;
    if (target === "gpu" && !isGpuAvailable()) {
      failures.push("GPU is unavailable");
      continue;
    }
    for (const backend of targetBackends(stage, target, opts)) {
      try {
        if (target === "cpu") {
          return {
            stage,
            placement: "cpu",
            backend,
            cpuKernel: compileCpuJvpKernel(
              roots,
              inputs,
              backend as CpuBackendName,
              numDirections,
            ),
          };
        }
        if (stage.kind === "map") {
          return {
            stage,
            placement: "gpu",
            backend,
            gpuMapKernel:
              backend === "gpu-interp"
                ? compileGpuInterpJvpMapKernel(
                    roots,
                    inputs,
                    stage.inputShape.width,
                    stage.outputShape.width,
                    numDirections,
                    stage.count,
                  )
                : compileGpuJvpMapKernel(
                    roots,
                    inputs,
                    stage.inputShape.width,
                    stage.outputShape.width,
                    numDirections,
                  ),
          };
        }
        if (stage.kind === "reduce") {
          if (!stage.associative || stage.order !== "tree") {
            throw new Error(
              "GPU autodiff reductions require associative: true and tree order",
            );
          }
          return {
            stage,
            placement: "gpu",
            backend,
            gpuReduceKernel:
              backend === "gpu-interp"
                ? compileGpuInterpJvpReduceKernel(
                    roots,
                    inputs,
                    stage.shape.width,
                    numDirections,
                    stage.inputCount + (stage.builtIn ? 0 : 1),
                  )
                : compileGpuJvpReduceKernel(
                    roots,
                    inputs,
                    stage.shape.width,
                    numDirections,
                  ),
          };
        }
        throw new Error("whole-column autodiff stages execute on CPU");
      } catch (error) {
        failures.push(
          `${target}/${backend}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
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
    const result = compiled.cpuKernel!.eval(inputs.values, inputs.tangents);
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
  return compiled.cpuKernel!.eval(inputs.values, inputs.tangents);
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
  return compiled.cpuKernel!.eval(inputs.values, inputs.tangents);
}

function disposeCompiledStage(compiled: CompiledJvpStage): void {
  compiled.cpuKernel?.dispose();
  compiled.gpuMapKernel?.dispose();
  compiled.gpuReduceKernel?.dispose();
}

function flattenedResultWidth(
  definition: ColumnarDefinition,
  output: ColumnarStage,
): number {
  const resultShape = definition.resultShape!;
  if (resultShape.collection === "single" && resultShape.values.length !== 1) {
    throw new Error(
      `columnar autodiff single result must contain one value shape, got ${resultShape.values.length}`,
    );
  }
  const flattened = resultShape.values.reduce((total, shape, index) => {
    if (!Number.isInteger(shape.width) || shape.width < 1) {
      throw new Error(
        `columnar autodiff result value ${index} has invalid width ${shape.width}`,
      );
    }
    return total + shape.width;
  }, 0);
  const stageWidth =
    output.kind === "source"
      ? output.count * output.shape.width
      : output.kind === "map"
        ? output.count * output.outputShape.width
        : output.kind === "reduce"
          ? output.shape.width
          : output.roots.length;
  if (flattened !== stageWidth) {
    throw new Error(
      `columnar autodiff result shape flattens to ${flattened} scalars, but output stage ${output.id} produces ${stageWidth}`,
    );
  }
  return flattened;
}

/** Compile one unblocked CPU/GPU forward-mode pass. */
function compileUnblockedColumnarGradRoutine(
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
  let sourcePlan = resolveSource(source, opts);
  let sourceRoots: CompiledExternalRoots | null = null;
  const scalarRoots = compileExternalRoots(
    gatherScalarRoots(definition),
    frozenDiffVars,
    configuredCpuBackends(opts),
  );
  let gpuSource: CompiledGpuJvpSourceKernel | null = null;
  const compiledStages: CompiledJvpStage[] = [];
  try {
    if (sourcePlan.placement === "gpu") {
      try {
        gpuSource =
          sourcePlan.backend === "gpu-interp"
            ? compileGpuInterpJvpSourceKernel(source.roots, diffVars.length)
            : compileGpuJvpSourceKernel(source.roots, diffVars.length);
      } catch (error) {
        if (
          effectivePlacement(source, opts) !== "auto" ||
          source.requestedBackend ||
          !stageTargets(source, opts).includes("cpu")
        ) {
          throw error;
        }
        sourcePlan = {
          placement: "cpu",
          backend: configuredCpuBackends(opts)[0]!,
        };
      }
    }
    sourceRoots =
      sourcePlan.placement === "cpu"
        ? compileExternalRoots(source.roots, frozenDiffVars, [
            sourcePlan.backend as CpuBackendName,
          ])
        : {
            roots: Object.freeze([...source.roots]),
            seeds: new Float64Array(0),
          };
    for (const stage of definition.stages.slice(1)) {
      if (stage.kind === "source") {
        throw new Error("columnar autodiff does not support multiple sources");
      }
      compiledStages.push(compileStage(stage, diffVars.length, opts));
    }
  } catch (error) {
    sourceRoots?.routine?.dispose?.();
    scalarRoots.routine?.dispose?.();
    gpuSource?.dispose();
    for (const compiled of compiledStages) disposeCompiledStage(compiled);
    throw error;
  }
  if (!sourceRoots) {
    throw new Error("columnar autodiff source compilation produced no roots");
  }

  const output = definition.stages[definition.outputStage]!;
  const numRoots = flattenedResultWidth(definition, output);
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
  for (const name of gpuSource?.varSlots ?? []) {
    if (seen.has(name)) continue;
    seen.add(name);
    varSlots.push(name);
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

      const numDirections = diffVars.length;
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
            root * numDirections,
            (root + 1) * numDirections,
          ),
        });
      }

      const ownedBuffers: GPUBuffer[] = [];
      let encoder: GPUCommandEncoder | null = null;
      const createBuffer = (
        byteLength: number,
        usage: GPUBufferUsageFlags,
      ): DeviceBufferSlice => {
        const device = requireGpuDevice();
        if (byteLength > device.limits.maxBufferSize) {
          throw new Error(
            `columnar GPU JVP buffer requires ${byteLength} bytes; device limit is ${device.limits.maxBufferSize}`,
          );
        }
        const buffer = device.createBuffer({
          size: Math.max(byteLength, 4),
          usage,
        });
        ownedBuffers.push(buffer);
        return { buffer, offset: 0, byteLength };
      };
      const upload = (values: ArrayLike<number>): DeviceBufferSlice => {
        const data = Float32Array.from(values);
        const slice = createBuffer(
          data.byteLength,
          GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        );
        if (data.byteLength > 0) {
          requireGpuDevice().queue.writeBuffer(
            slice.buffer,
            0,
            data.buffer,
            data.byteOffset,
            data.byteLength,
          );
        }
        return slice;
      };
      const ensureDevice = (
        dual: StageDualValues,
        count: number,
        width: number,
      ): DeviceDualValues => {
        if (dual.location === "device") {
          if (dual.count !== count || dual.width !== width) {
            throw new Error("columnar GPU JVP column shape mismatch");
          }
          return dual;
        }
        return {
          location: "device",
          values: upload(dual.values),
          tangents: upload(dual.tangents),
          count,
          width,
        };
      };
      const ensureHost = async (
        dual: StageDualValues,
      ): Promise<HostDualValues> => {
        if (dual.location === "host") return dual;
        const readSlice = (
          slice: DeviceBufferSlice,
        ): { staging: GPUBuffer; copy: () => void } | null => {
          if (slice.byteLength === 0) return null;
          const staging = requireGpuDevice().createBuffer({
            size: slice.byteLength,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
          return {
            staging,
            copy: () => {
              const enc = (encoder ??=
                requireGpuDevice().createCommandEncoder());
              enc.copyBufferToBuffer(
                slice.buffer,
                slice.offset,
                staging,
                0,
                slice.byteLength,
              );
            },
          };
        };
        const valueRead = readSlice(dual.values);
        const tangentRead = readSlice(dual.tangents);
        valueRead?.copy();
        tangentRead?.copy();
        if (encoder) {
          requireGpuDevice().queue.submit([encoder.finish()]);
          encoder = null;
        }
        try {
          const [values, tangents] = await Promise.all([
            valueRead
              ? mapGpuReadbackBuffer(valueRead.staging, dual.values.byteLength)
              : Promise.resolve(new Float32Array(0)),
            tangentRead
              ? mapGpuReadbackBuffer(
                  tangentRead.staging,
                  dual.tangents.byteLength,
                )
              : Promise.resolve(new Float32Array(0)),
          ]);
          return {
            location: "host",
            values: Array.from(values),
            tangents: Array.from(tangents),
          };
        } finally {
          valueRead?.staging.destroy();
          tangentRead?.staging.destroy();
        }
      };
      const allocateOutput = (
        count: number,
        width: number,
      ): DeviceDualValues => ({
        location: "device",
        values: createBuffer(
          count * width * 4,
          GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        ),
        tangents: createBuffer(
          count * width * numDirections * 4,
          GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        ),
        count,
        width,
      });
      const uniformCache: Array<{
        roots: readonly NumNode[];
        dual: DeviceDualValues;
      }> = [];
      const uploadUsing = (
        roots: readonly NumNode[],
        using: DualValues,
      ): DeviceDualValues => {
        const cached = uniformCache.find(
          (entry) =>
            entry.roots.length === roots.length &&
            entry.roots.every((root, index) => root === roots[index]),
        );
        if (cached) return cached.dual;
        const dual: DeviceDualValues = {
          location: "device",
          values: upload(using.values),
          tangents: upload(using.tangents),
          count: 1,
          width: using.values.length,
        };
        uniformCache.push({ roots, dual });
        return dual;
      };

      try {
        const sourceHost: HostDualValues = {
          location: "host",
          ...sourceDual,
        };
        let initialSource: StageDualValues = sourceHost;
        if (sourcePlan.placement === "gpu" && gpuSource) {
          const inputValues = Float32Array.from(
            gpuSource.varSlots,
            (name) => (vars as Map<VarName, number>).get(name) ?? 0,
          );
          const inputTangents = new Float32Array(
            gpuSource.numVars * numDirections,
          );
          for (let input = 0; input < gpuSource.numVars; input++) {
            for (let direction = 0; direction < numDirections; direction++) {
              if (gpuSource.varSlots[input] === diffVars[direction]) {
                inputTangents[input * numDirections + direction] = 1;
              }
            }
          }
          const inputs: DeviceDualValues = {
            location: "device",
            values: upload(inputValues),
            tangents: upload(inputTangents),
            count: 1,
            width: gpuSource.numVars,
          };
          const outputDual = allocateOutput(source.count, source.shape.width);
          encoder ??= requireGpuDevice().createCommandEncoder();
          gpuSource.encode(
            encoder,
            inputs.values,
            inputs.tangents,
            outputDual.values,
            outputDual.tangents,
          );
          initialSource = outputDual;
        } else if (sourcePlan.placement === "gpu") {
          initialSource = ensureDevice(
            sourceHost,
            source.count,
            source.shape.width,
          );
        }
        const stageValues = new Map<number, StageDualValues>([
          [source.id, initialSource],
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
            numDirections,
          );

          if (compiled.placement === "gpu" && stage.kind === "map") {
            const input = ensureDevice(
              sourceValue,
              stage.count,
              stage.inputShape.width,
            );
            const outputDual = allocateOutput(
              stage.count,
              stage.outputShape.width,
            );
            const uniformDual =
              compiled.gpuMapKernel!.uniformWidth > 0
                ? uploadUsing(stage.using.roots, using)
                : null;
            encoder ??= requireGpuDevice().createCommandEncoder();
            compiled.gpuMapKernel!.encode(
              encoder,
              input.values,
              input.tangents,
              uniformDual?.values ?? null,
              uniformDual?.tangents ?? null,
              outputDual.values,
              outputDual.tangents,
              stage.count,
            );
            stageValues.set(stage.id, outputDual);
            continue;
          }

          if (compiled.placement === "gpu" && stage.kind === "reduce") {
            const initial = dualForRoots(
              stage.initial,
              scalarValues,
              numDirections,
            );
            if (stage.inputCount === 0) {
              stageValues.set(stage.id, { location: "host", ...initial });
              continue;
            }
            const sourceInput = ensureDevice(
              sourceValue,
              stage.inputCount,
              stage.shape.width,
            );
            let input = sourceInput;
            let reductionCount = stage.inputCount;
            if (!stage.builtIn) {
              const combined = allocateOutput(
                stage.inputCount + 1,
                stage.shape.width,
              );
              const initialValues = Float32Array.from(initial.values);
              const initialTangents = Float32Array.from(initial.tangents);
              if (initialValues.byteLength > 0) {
                requireGpuDevice().queue.writeBuffer(
                  combined.values.buffer,
                  0,
                  initialValues.buffer,
                  initialValues.byteOffset,
                  initialValues.byteLength,
                );
              }
              if (initialTangents.byteLength > 0) {
                requireGpuDevice().queue.writeBuffer(
                  combined.tangents.buffer,
                  0,
                  initialTangents.buffer,
                  initialTangents.byteOffset,
                  initialTangents.byteLength,
                );
              }
              encoder ??= requireGpuDevice().createCommandEncoder();
              encoder.copyBufferToBuffer(
                sourceInput.values.buffer,
                sourceInput.values.offset,
                combined.values.buffer,
                initialValues.byteLength,
                sourceInput.values.byteLength,
              );
              if (sourceInput.tangents.byteLength > 0) {
                encoder.copyBufferToBuffer(
                  sourceInput.tangents.buffer,
                  sourceInput.tangents.offset,
                  combined.tangents.buffer,
                  initialTangents.byteLength,
                  sourceInput.tangents.byteLength,
                );
              }
              input = combined;
              reductionCount++;
            }
            if (reductionCount === 1) {
              stageValues.set(stage.id, { ...input, count: 1 });
              continue;
            }
            const outputDual = allocateOutput(1, stage.shape.width);
            const uniformDual =
              compiled.gpuReduceKernel!.uniformWidth > 0
                ? uploadUsing(stage.using.roots, using)
                : null;
            encoder ??= requireGpuDevice().createCommandEncoder();
            compiled.gpuReduceKernel!.encode(
              encoder,
              input.values,
              input.tangents,
              uniformDual?.values ?? null,
              uniformDual?.tangents ?? null,
              outputDual.values,
              outputDual.tangents,
              reductionCount,
            );
            stageValues.set(stage.id, outputDual);
            continue;
          }

          const hostSource = await ensureHost(sourceValue);
          const result =
            stage.kind === "map"
              ? evalMap(compiled, stage, hostSource, using, numDirections)
              : stage.kind === "reduce"
                ? evalReduce(
                    compiled,
                    stage,
                    hostSource,
                    using,
                    dualForRoots(stage.initial, scalarValues, numDirections),
                    numDirections,
                  )
                : evalWhole(compiled, stage, hostSource, using, numDirections);
          stageValues.set(stage.id, { location: "host", ...result });
        }

        const stagedResult = stageValues.get(output.id);
        if (!stagedResult)
          throw new Error("columnar autodiff produced no output");
        const result = await ensureHost(stagedResult);
        const expectedTangents = numRoots * numDirections;
        if (
          result.values.length !== numRoots ||
          result.tangents.length !== expectedTangents
        ) {
          throw new Error(
            `columnar autodiff output produced ${result.values.length} values and ${result.tangents.length} tangents; expected ${numRoots} and ${expectedTangents}`,
          );
        }
        return shape === "grad"
          ? {
              val: result.values[0]!,
              gradient: result.tangents.slice(0, numDirections),
            }
          : {
              vals: result.values,
              jacobian: Array.from({ length: numRoots }, (_, root) =>
                result.tangents.slice(
                  root * numDirections,
                  (root + 1) * numDirections,
                ),
              ),
            };
      } finally {
        for (const buffer of ownedBuffers) buffer.destroy();
      }
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      sourceRoots.routine?.dispose?.();
      scalarRoots.routine?.dispose?.();
      gpuSource?.dispose();
      for (const compiled of compiledStages) disposeCompiledStage(compiled);
    },
  };
}

/** Compile CPU/GPU forward autodiff, optionally partitioning tangent directions. */
export function compileColumnarGradRoutine(
  definition: ColumnarDefinition,
  diffVars: readonly VarName[],
  opts: ColumnarRoutineOptions = {},
): ColumnarGradRoutine {
  const configuredBlockSize = opts.autodiff?.tangentBlockSize;
  if (
    configuredBlockSize !== undefined &&
    (!Number.isInteger(configuredBlockSize) || configuredBlockSize < 1)
  ) {
    throw new Error(
      `columnar autodiff tangentBlockSize must be a positive integer, got ${configuredBlockSize}`,
    );
  }
  if (
    diffVars.length === 0 ||
    configuredBlockSize === undefined ||
    configuredBlockSize >= diffVars.length
  ) {
    return compileUnblockedColumnarGradRoutine(definition, diffVars, opts);
  }

  const blockSize = configuredBlockSize;
  const blocks: Array<{
    readonly start: number;
    readonly routine: ColumnarGradRoutine;
  }> = [];
  try {
    for (let start = 0; start < diffVars.length; start += blockSize) {
      blocks.push({
        start,
        routine: compileUnblockedColumnarGradRoutine(
          definition,
          diffVars.slice(start, start + blockSize),
          opts,
        ),
      });
    }
  } catch (error) {
    for (const block of blocks) block.routine.dispose();
    throw error;
  }

  const first = blocks[0]!.routine;
  const varSlots: VarName[] = [];
  const seen = new Set<VarName>();
  for (const { routine } of blocks) {
    if (routine.numRoots !== first.numRoots || routine.shape !== first.shape) {
      for (const block of blocks) block.routine.dispose();
      throw new Error(
        "columnar autodiff tangent blocks disagree on result shape",
      );
    }
    for (const name of routine.varSlots) {
      if (seen.has(name)) continue;
      seen.add(name);
      varSlots.push(name);
    }
  }
  let disposed = false;
  return {
    shape: first.shape,
    diffVars: Object.freeze([...diffVars]),
    varSlots: Object.freeze(varSlots),
    numVars: varSlots.length,
    numRoots: first.numRoots,
    async evalAsync(vars: VarMap): Promise<ColumnarAutodiffResult> {
      if (disposed)
        throw new Error("columnar autodiff routine has been disposed");
      let values: number[] | null = null;
      const jacobian = Array.from({ length: first.numRoots }, () =>
        Array<number>(diffVars.length).fill(0),
      );
      for (const block of blocks) {
        const result = await block.routine.evalAsync(vars);
        const blockValues = "gradient" in result ? [result.val] : result.vals;
        const blockJacobian =
          "gradient" in result ? [result.gradient] : result.jacobian;
        if (!values) values = [...blockValues];
        if (
          blockValues.length !== first.numRoots ||
          blockJacobian.length !== first.numRoots
        ) {
          throw new Error(
            "columnar autodiff tangent block returned an invalid result shape",
          );
        }
        for (let root = 0; root < first.numRoots; root++) {
          const row = blockJacobian[root]!;
          const expectedWidth = block.routine.diffVars.length;
          if (row.length !== expectedWidth) {
            throw new Error(
              `columnar autodiff tangent block returned ${row.length} directions; expected ${expectedWidth}`,
            );
          }
          for (let direction = 0; direction < row.length; direction++) {
            jacobian[root]![block.start + direction] = row[direction]!;
          }
        }
      }
      if (!values)
        throw new Error("columnar autodiff produced no tangent blocks");
      return first.shape === "grad"
        ? { val: values[0]!, gradient: jacobian[0]! }
        : { vals: values, jacobian };
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      for (const block of blocks) block.routine.dispose();
    },
  };
}
