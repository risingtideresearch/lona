/**
 * Routine factories — compile a kernel on a backend, wrap it in the full
 * Routine interface.
 *
 * Compilation is always synchronous. GPU backends require `await initGpu()`
 * to have been called beforehand — they throw if the device isn't ready.
 *
 * Returns `null` only when the DAG cannot be compiled to a tape (e.g. ForeignFn).
 * Throws if the chosen backend fails to compile.
 */
import {
  KIND_FOREIGN,
  allVariables,
  childrenOfNumNode,
  type NumNode,
  type VarName,
} from "../../core/tree";
import {
  compileDynamicSelectTraceTape,
  compileSelectTraceTape,
  compileTape,
  specializeSelectsFromTrace,
  TapeAssertionError,
  traceTapeValues,
} from "../tape";
import type { CompiledTape } from "../tape";
import type {
  BackendName,
  GradKernel,
  JacobianKernel,
  JvpKernel,
  JvpResult,
  KernelEnvelope,
  ValueKernel,
} from "./backend";
import { getBackend } from "./backend";
import type {
  GradRoutine,
  GradientResult,
  JacobianResult,
  JacobianRoutine,
  MultiValueRoutine,
  ValueRoutine,
  VarBatch,
  VarMap,
} from "./types";
import { visitFromLeaves } from "../../dag/traversal";
import { wrapGrad, wrapJacobian, wrapValue } from "./wrapper";
import {
  inferNumPoints,
  resolveVarBatchColumns,
  writeVarBatchRow,
} from "./batch-pack";

export type SelectSpecializationMode = false | "trace" | "full-trace";

export type CompileOpts = {
  /** Pin to a specific backend; otherwise the default is used. */
  backend?: BackendName;
  /** Trace selects on first eval, compile a guarded specialized routine, retrace on guard failure. "trace" uses dynamic selected-path tracing; "full-trace" uses the old full-tape trace. */
  selectSpecialization?: SelectSpecializationMode;
  /**
   * Experimental instrumentation hook. Called at phase boundaries while the
   * corresponding inputs/results are still live, so callers can sample memory.
   * The hook must not mutate the roots or compiled tape.
   */
  diagnosticCheckpoint?: (phase: string) => void;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function compileTapeForRoots(roots: NumNode[]): CompiledTape | null {
  if (roots.length === 0) return null;
  return compileTape(roots);
}

function requireBackend(name: BackendName) {
  const b = getBackend(name);
  if (!b) throw new Error(`Backend '${name}' is not registered`);
  return b;
}

function selectSpecializationMode(
  opts?: CompileOpts,
): Exclude<SelectSpecializationMode, false> | false {
  return opts?.selectSpecialization ?? false;
}

function rejectUnsupportedSelectSpecialization(
  backend: BackendName,
  shape: "value" | "grad" | "jacobian",
): void {
  const supportedValue = new Set<BackendName>([
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ]);
  const supportedGrad = new Set<BackendName>([
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ]);
  // These backends expose a native `compileJacobian` path that consumes the
  // specialized tape. Symbolic backends still require the original roots.
  const supportedJacobian = new Set<BackendName>([
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ]);
  const supported =
    shape === "value"
      ? supportedValue
      : shape === "grad"
        ? supportedGrad
        : supportedJacobian;
  if (!supported.has(backend)) {
    throw new Error(
      `selectSpecialization: "trace" is not supported for ${shape} backend '${backend}'`,
    );
  }
}

function varSlotsForRoots(roots: NumNode[]): VarName[] {
  const vars = new Set<VarName>();
  for (const root of roots) {
    for (const variable of allVariables(root)) vars.add(variable);
  }
  return [...vars];
}

function containsForeignNode(roots: NumNode[]): boolean {
  let found = false;
  for (const root of roots) {
    visitFromLeaves(root, childrenOfNumNode, (node) => {
      if (node.kind === KIND_FOREIGN) found = true;
    });
    if (found) return true;
  }
  return false;
}

function compileFullTraceSelectSpecializedTape(
  roots: NumNode[],
  trace: NonNullable<ReturnType<typeof compileSelectTraceTape>>,
  vars: VarMap,
  derivatives?: VarMap,
): CompiledTape {
  const values = traceTapeValues(trace.tape, vars, derivatives);
  const specialized = specializeSelectsFromTrace(roots, trace, values);
  const tape = compileTape(specialized.roots, {
    guardPrelude: specialized.guardPrelude,
  });
  if (!tape) throw new Error("Failed to compile select-specialized tape");
  return tape;
}

function compileTracingValueRoutine(
  roots: NumNode[],
  backend: BackendName,
  mode: Exclude<SelectSpecializationMode, false>,
): ValueRoutine | MultiValueRoutine | null {
  rejectUnsupportedSelectSpecialization(backend, "value");
  const fullTrace =
    mode === "full-trace" ? compileSelectTraceTape(roots) : null;
  if (mode === "full-trace") {
    if (!fullTrace) return null;
  } else if (roots.length === 0 || containsForeignNode(roots)) return null;
  const b = requireBackend(backend);
  if (!b.compileValue) {
    throw new Error(`Backend '${backend}' cannot compile value routines`);
  }

  const varSlots = fullTrace?.tape.varSlots ?? varSlotsForRoots(roots);
  const numVars = fullTrace?.tape.numVars ?? varSlots.length;
  const numRoots = fullTrace?.tape.rootIndices.length ?? roots.length;
  let currentEval: ((vars: VarMap, derivatives?: VarMap) => number[]) | null =
    null;
  let currentDispose: (() => void) | undefined;

  const retrace = (vars: VarMap, derivatives?: VarMap): void => {
    currentDispose?.();
    const dynamicTrace = fullTrace
      ? null
      : compileDynamicSelectTraceTape(roots, vars, derivatives, varSlots);
    const tape = fullTrace
      ? compileFullTraceSelectSpecializedTape(
          roots,
          fullTrace,
          vars,
          derivatives,
        )
      : dynamicTrace?.tape;
    if (!tape) throw new Error("Failed to compile select-specialized tape");
    const env = b.compileValue!(tape);
    if (!env)
      throw new Error(`Backend '${backend}' failed to compile value routine`);
    if (env.kernel.kind !== "sync-value") {
      env.dispose?.();
      throw new Error(
        `selectSpecialization: "trace" requires a sync value backend; got '${backend}'`,
      );
    }
    currentEval = env.kernel.eval;
    currentDispose = env.dispose;
  };

  const evalArray = (vars: VarMap, derivatives?: VarMap): number[] => {
    if (!currentEval) retrace(vars, derivatives);
    try {
      return currentEval!(vars, derivatives);
    } catch (error) {
      if (!(error instanceof TapeAssertionError)) throw error;
      retrace(vars, derivatives);
      return currentEval!(vars, derivatives);
    }
  };

  const dispose = (): void => currentDispose?.();

  if (numRoots === 1) {
    return {
      shape: "value",
      varSlots,
      numVars,
      eval: (vars, derivatives) => evalArray(vars, derivatives)[0]!,
      evalAsync: (vars, derivatives) =>
        Promise.resolve(evalArray(vars, derivatives)[0]!),
      evalBatch: async (vars, numPoints) => {
        const n = numPoints ?? inferNumPoints(vars);
        if (n === 0) return new Float32Array(0);
        const out = new Float32Array(n);
        const columns = resolveVarBatchColumns(varSlots, numVars, vars);
        const row = new Map<VarName, number>();
        for (let p = 0; p < n; p++) {
          writeVarBatchRow(row, varSlots, numVars, columns, p);
          out[p] = evalArray(row)[0]!;
        }
        return out;
      },
      evalBatchPacked: async (varData, numPoints) => {
        const out = new Float32Array(numPoints);
        const row = new Map<VarName, number>();
        for (let p = 0; p < numPoints; p++) {
          for (let s = 0; s < numVars; s++) {
            row.set(varSlots[s]!, varData[p * numVars + s]!);
          }
          out[p] = evalArray(row)[0]!;
        }
        return out;
      },
      dispose,
    };
  }

  return {
    shape: "multi-value",
    varSlots,
    numVars,
    numRoots,
    eval: evalArray,
    evalAsync: (vars, derivatives) =>
      Promise.resolve(evalArray(vars, derivatives)),
    evalBatch: async (vars: VarBatch, numPoints?: number) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return new Float32Array(0);
      const out = new Float32Array(n * numRoots);
      const columns = resolveVarBatchColumns(varSlots, numVars, vars);
      const row = new Map<VarName, number>();
      for (let p = 0; p < n; p++) {
        writeVarBatchRow(row, varSlots, numVars, columns, p);
        const vals = evalArray(row);
        for (let r = 0; r < numRoots; r++) out[p * numRoots + r] = vals[r]!;
      }
      return out;
    },
    evalBatchPacked: async (varData, numPoints) => {
      const out = new Float32Array(numPoints * numRoots);
      const row = new Map<VarName, number>();
      for (let p = 0; p < numPoints; p++) {
        for (let s = 0; s < numVars; s++) {
          row.set(varSlots[s]!, varData[p * numVars + s]!);
        }
        const vals = evalArray(row);
        for (let r = 0; r < numRoots; r++) out[p * numRoots + r] = vals[r]!;
      }
      return out;
    },
    dispose,
  };
}

function compileTracingGradRoutine(
  roots: NumNode[],
  diffVars: VarName[] | undefined,
  backend: BackendName,
  mode: Exclude<SelectSpecializationMode, false>,
): GradRoutine | null {
  rejectUnsupportedSelectSpecialization(backend, "grad");
  const fullTrace =
    mode === "full-trace" ? compileSelectTraceTape(roots) : null;
  if (mode === "full-trace") {
    if (!fullTrace) return null;
  } else if (roots.length === 0 || containsForeignNode(roots)) return null;
  const b = requireBackend(backend);
  if (!b.compileGrad) {
    throw new Error(`Backend '${backend}' cannot compile grad routines`);
  }

  const varSlots = fullTrace?.tape.varSlots ?? varSlotsForRoots(roots);
  const numVars = fullTrace?.tape.numVars ?? varSlots.length;
  const vars = (diffVars ?? varSlots) as VarName[];
  let currentEval: ((vars: VarMap) => GradientResult) | null = null;
  let currentDispose: (() => void) | undefined;

  const retrace = (evalVars: VarMap): void => {
    currentDispose?.();
    const tape = fullTrace
      ? compileFullTraceSelectSpecializedTape(roots, fullTrace, evalVars)
      : compileDynamicSelectTraceTape(roots, evalVars, undefined, varSlots)
          ?.tape;
    if (!tape) throw new Error("Failed to compile select-specialized tape");
    const env = b.compileGrad!(tape, vars);
    if (!env)
      throw new Error(`Backend '${backend}' failed to compile grad routine`);
    if (env.kernel.kind !== "sync-grad") {
      env.dispose?.();
      throw new Error(
        `selectSpecialization: "trace" requires a sync grad backend; got '${backend}'`,
      );
    }
    currentEval = env.kernel.eval;
    currentDispose = env.dispose;
  };

  const evalSync = (evalVars: VarMap): GradientResult => {
    if (!currentEval) retrace(evalVars);
    try {
      return currentEval!(evalVars);
    } catch (error) {
      if (!(error instanceof TapeAssertionError)) throw error;
      retrace(evalVars);
      return currentEval!(evalVars);
    }
  };

  return {
    shape: "grad",
    varSlots,
    numVars,
    diffVars: vars,
    eval: evalSync,
    evalAsync: (vars) => Promise.resolve(evalSync(vars)),
    evalBatch: async (batchVars, numPoints) => {
      const n = numPoints ?? inferNumPoints(batchVars);
      const columns = resolveVarBatchColumns(varSlots, numVars, batchVars);
      const row = new Map<VarName, number>();
      const out: GradientResult[] = [];
      for (let p = 0; p < n; p++) {
        writeVarBatchRow(row, varSlots, numVars, columns, p);
        out.push(evalSync(row));
      }
      return out;
    },
    dispose: () => currentDispose?.(),
  };
}

function compileTracingJacobianRoutine(
  roots: NumNode[],
  diffVars: VarName[] | undefined,
  backend: BackendName,
  mode: Exclude<SelectSpecializationMode, false>,
): JacobianRoutine | null {
  rejectUnsupportedSelectSpecialization(backend, "jacobian");
  const fullTrace =
    mode === "full-trace" ? compileSelectTraceTape(roots) : null;
  if (mode === "full-trace") {
    if (!fullTrace) return null;
  } else if (roots.length === 0 || containsForeignNode(roots)) return null;
  const b = requireBackend(backend);
  if (!b.compileJacobian) {
    throw new Error(`Backend '${backend}' cannot compile jacobian routines`);
  }

  const varSlots = fullTrace?.tape.varSlots ?? varSlotsForRoots(roots);
  const numVars = fullTrace?.tape.numVars ?? varSlots.length;
  const numRoots = fullTrace?.tape.rootIndices.length ?? roots.length;
  const vars = (diffVars ?? varSlots) as VarName[];
  let currentEval: ((vars: VarMap) => JacobianResult) | null = null;
  let currentDispose: (() => void) | undefined;

  const retrace = (evalVars: VarMap): void => {
    currentDispose?.();
    const tape = fullTrace
      ? compileFullTraceSelectSpecializedTape(roots, fullTrace, evalVars)
      : compileDynamicSelectTraceTape(roots, evalVars, undefined, varSlots)
          ?.tape;
    if (!tape) throw new Error("Failed to compile select-specialized tape");
    const env = b.compileJacobian!(tape, vars);
    if (!env)
      throw new Error(
        `Backend '${backend}' failed to compile jacobian routine`,
      );
    if (env.kernel.kind !== "sync-jacobian") {
      env.dispose?.();
      throw new Error(
        `selectSpecialization: "trace" requires a sync jacobian backend; got '${backend}'`,
      );
    }
    currentEval = env.kernel.eval;
    currentDispose = env.dispose;
  };

  const evalSync = (evalVars: VarMap): JacobianResult => {
    if (!currentEval) retrace(evalVars);
    try {
      return currentEval!(evalVars);
    } catch (error) {
      if (!(error instanceof TapeAssertionError)) throw error;
      retrace(evalVars);
      return currentEval!(evalVars);
    }
  };

  return {
    shape: "jacobian",
    varSlots,
    numVars,
    numRoots,
    diffVars: vars,
    eval: evalSync,
    evalAsync: (vars) => Promise.resolve(evalSync(vars)),
    evalBatch: async (batchVars, numPoints) => {
      const n = numPoints ?? inferNumPoints(batchVars);
      const columns = resolveVarBatchColumns(varSlots, numVars, batchVars);
      const row = new Map<VarName, number>();
      const out: JacobianResult[] = [];
      for (let p = 0; p < n; p++) {
        writeVarBatchRow(row, varSlots, numVars, columns, p);
        out.push(evalSync(row));
      }
      return out;
    },
    dispose: () => currentDispose?.(),
  };
}

// ---------------------------------------------------------------------------
// Value — default: wasm-codegen
// ---------------------------------------------------------------------------

/** Options for compiling an already-fixed tape (select tracing needs roots). */
export type CompileValueTapeOpts = Pick<
  CompileOpts,
  "backend" | "diagnosticCheckpoint"
>;

/**
 * Compile an existing immutable tape for value evaluation. This is the staged
 * counterpart to `compileValueRoutine`, useful when a caller also needs tape
 * metadata or wants to compile the same tape for more than one backend.
 */
export function compileValueRoutineFromTape(
  tape: CompiledTape,
  opts?: CompileValueTapeOpts,
): ValueRoutine | MultiValueRoutine {
  const name = opts?.backend ?? "wasm-codegen";
  const b = requireBackend(name);
  opts?.diagnosticCheckpoint?.(`lona:backend:${name}:start`);
  const env = b.compileValue?.(tape);
  if (!env)
    throw new Error(`Backend '${name}' failed to compile value routine`);
  opts?.diagnosticCheckpoint?.(`lona:backend:${name}:done`);
  return wrapValue(env as KernelEnvelope<ValueKernel>);
}

export interface SeededJvpRoutine {
  readonly varSlots: readonly VarName[];
  readonly numVars: number;
  readonly numRoots: number;
  readonly numDirections: number;
  evalPacked(values: Float64Array, seeds: Float64Array): JvpResult;
  dispose?(): void;
}

/** Compile an immutable tape for arbitrary-seed, multi-root forward JVP. */
export function compileJvpRoutineFromTape(
  tape: CompiledTape,
  numDirections: number,
  opts?: CompileValueTapeOpts,
): SeededJvpRoutine {
  const name = opts?.backend ?? "js-interp";
  const backend = requireBackend(name);
  opts?.diagnosticCheckpoint?.(`lona:backend:${name}:jvp:start`);
  const env = backend.compileJvp?.(tape, numDirections);
  if (!env) {
    throw new Error(`Backend '${name}' failed to compile seeded JVP routine`);
  }
  const typed = env as KernelEnvelope<JvpKernel>;
  if (typed.kernel.kind !== "sync-jvp") {
    typed.dispose?.();
    throw new Error(`Backend '${name}' does not provide a synchronous JVP`);
  }
  opts?.diagnosticCheckpoint?.(`lona:backend:${name}:jvp:done`);
  return {
    varSlots: typed.varSlots,
    numVars: typed.numVars,
    numRoots: typed.kernel.numRoots,
    numDirections: typed.kernel.numDirections,
    evalPacked: typed.kernel.evalPacked.bind(typed.kernel),
    dispose: typed.dispose,
  };
}

export function compileValueRoutine(
  roots: NumNode[],
  opts?: CompileOpts,
): ValueRoutine | MultiValueRoutine | null {
  const name = opts?.backend ?? "wasm-codegen";
  const mode = selectSpecializationMode(opts);
  if (mode) {
    return compileTracingValueRoutine(roots, name, mode);
  }

  opts?.diagnosticCheckpoint?.("lona:tape:start");
  const tape = compileTapeForRoots(roots);
  if (!tape) return null;
  opts?.diagnosticCheckpoint?.("lona:tape:done");
  return compileValueRoutineFromTape(tape, opts);
}

// ---------------------------------------------------------------------------
// Grad / Jacobian — default: wasm-codegen (grad), wasm-interp (jacobian)
// ---------------------------------------------------------------------------

export function compileGradRoutine(
  roots: NumNode[],
  diffVars?: VarName[],
  opts?: CompileOpts,
): GradRoutine | JacobianRoutine | null {
  const mode = selectSpecializationMode(opts);
  const multi = roots.length > 1;
  const name = opts?.backend ?? (multi ? "wasm-interp" : "wasm-codegen");
  if (mode) {
    const selectVars = varSlotsForRoots(roots);
    const vars = (diffVars ?? selectVars) as VarName[];
    return multi
      ? compileTracingJacobianRoutine(roots, vars, name, mode)
      : compileTracingGradRoutine(roots, vars, name, mode);
  }

  const tape = compileTapeForRoots(roots);
  if (!tape) return null;
  const normalVars = (diffVars ??
    tape.varSlots.slice(0, tape.numVars)) as VarName[];
  const b = requireBackend(name);
  // Sym backends compile from roots (symbolic differentiation).
  if (multi && b.compileJacobianFromRoots) {
    const result = b.compileJacobianFromRoots(roots, normalVars);
    if (!result)
      throw new Error(`Backend '${name}' failed to compile jacobian routine`);
    return result;
  }
  if (!multi && b.compileGradFromRoots) {
    const result = b.compileGradFromRoots(roots, normalVars);
    if (!result)
      throw new Error(`Backend '${name}' failed to compile grad routine`);
    return result;
  }
  // Base backends compile from tape (forward AD).
  if (multi) {
    const env = b.compileJacobian?.(tape, normalVars);
    if (!env)
      throw new Error(`Backend '${name}' failed to compile jacobian routine`);
    return wrapJacobian(env as KernelEnvelope<JacobianKernel>);
  }
  const env = b.compileGrad?.(tape, normalVars);
  if (!env) throw new Error(`Backend '${name}' failed to compile grad routine`);
  return wrapGrad(env as KernelEnvelope<GradKernel>);
}
