/**
 * Shared helpers for symbolic grad/jacobian backends.
 *
 * Symbolic backends derive the gradient DAG symbolically (via `gradient()`),
 * compile the expanded [root, partials...] as multi-value, then wrap the
 * multi-value output into GradRoutine / JacobianRoutine.
 *
 * CPU backends produce sync eval (preserving f64 precision).
 * GPU backends produce async eval only (f32, batch dispatch).
 */
import type { NumNode, VarName } from "../../../core/tree";
import { gradient } from "../../../api/diff";
import { compileTape } from "../../tape";
import type {
  Backend,
  BackendName,
  KernelEnvelope,
  ValueKernel,
} from "../backend";
import { getBackend, syncEvalNotSupported } from "../backend";
import { wrapValue } from "../wrapper";
import type {
  GradientResult,
  GradRoutine,
  JacobianResult,
  JacobianRoutine,
  MultiValueRoutine,
  VarBatch,
  VarMap,
} from "../types";
import {
  inferNumPoints,
  resolveVarBatchColumns,
  writeVarBatchRow,
} from "../batch-pack";

// ---------------------------------------------------------------------------
// Shared: compile value on a backend (always multi-root capable)
// ---------------------------------------------------------------------------

function compileMultiValueOnBackend(
  baseBackend: Backend,
  roots: NumNode[],
): MultiValueRoutine | null {
  const tape = compileTape(roots);
  if (!tape) return null;
  const env = baseBackend.compileValue?.(tape);
  if (!env || env instanceof Promise) return null;
  const routine = wrapValue(env as KernelEnvelope<ValueKernel>);
  if (routine.shape !== "multi-value") return null;
  return routine as MultiValueRoutine;
}

// ---------------------------------------------------------------------------
// CPU symbolic grad (sync, f64 precision)
// ---------------------------------------------------------------------------

export function compileSyncSymbolicGrad(
  root: NumNode,
  diffVars: VarName[],
  underlyingName: BackendName,
): GradRoutine | null {
  const base = getBackend(underlyingName);
  if (!base) return null;
  const partials = gradient(root, diffVars);
  const mv = compileMultiValueOnBackend(base, [root, ...partials]);
  if (!mv) return null;

  const { varSlots, numVars } = mv;

  function unpack(vals: number[]): GradientResult {
    return { val: vals[0]!, gradient: vals.slice(1) };
  }

  return {
    shape: "grad",
    varSlots,
    numVars,
    diffVars,
    eval: (vars: VarMap) => unpack(mv.eval(vars)),
    evalAsync: async (vars: VarMap) => unpack(await mv.evalAsync(vars)),
    evalBatch: async (vars: VarBatch, numPoints?: number) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return [];
      const columns = resolveVarBatchColumns(varSlots, numVars, vars);
      const row = new Map<VarName, number>();
      const results: GradientResult[] = new Array(n);
      for (let p = 0; p < n; p++) {
        writeVarBatchRow(row, varSlots, numVars, columns, p);
        results[p] = unpack(mv.eval(row));
      }
      return results;
    },
    dispose: mv.dispose,
  };
}

// ---------------------------------------------------------------------------
// CPU symbolic jacobian (sync, f64 precision)
// ---------------------------------------------------------------------------

export function compileSyncSymbolicJacobian(
  roots: NumNode[],
  diffVars: VarName[],
  underlyingName: BackendName,
): JacobianRoutine | null {
  const base = getBackend(underlyingName);
  if (!base) return null;
  const numRoots = roots.length;
  const numDiff = diffVars.length;
  const stride = 1 + numDiff;

  const expanded: NumNode[] = [];
  for (const root of roots) {
    expanded.push(root);
    for (const p of gradient(root, diffVars)) expanded.push(p);
  }
  const mv = compileMultiValueOnBackend(base, expanded);
  if (!mv) return null;

  const { varSlots, numVars } = mv;

  function unpack(vals: number[]): JacobianResult {
    const valsOut = new Array<number>(numRoots);
    const jacobian: number[][] = [];
    for (let r = 0; r < numRoots; r++) {
      const off = r * stride;
      valsOut[r] = vals[off]!;
      jacobian.push(vals.slice(off + 1, off + stride));
    }
    return { vals: valsOut, jacobian };
  }

  return {
    shape: "jacobian",
    varSlots,
    numVars,
    numRoots,
    diffVars,
    eval: (vars: VarMap) => unpack(mv.eval(vars)),
    evalAsync: async (vars: VarMap) => unpack(await mv.evalAsync(vars)),
    evalBatch: async (vars: VarBatch, numPoints?: number) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return [];
      const columns = resolveVarBatchColumns(varSlots, numVars, vars);
      const row = new Map<VarName, number>();
      const results: JacobianResult[] = new Array(n);
      for (let p = 0; p < n; p++) {
        writeVarBatchRow(row, varSlots, numVars, columns, p);
        results[p] = unpack(mv.eval(row));
      }
      return results;
    },
    dispose: mv.dispose,
  };
}

// ---------------------------------------------------------------------------
// GPU symbolic grad (compile is sync, eval is async)
// ---------------------------------------------------------------------------

export function compileSymbolicGrad(
  root: NumNode,
  diffVars: VarName[],
  underlyingName: BackendName,
): GradRoutine | null {
  const base = getBackend(underlyingName);
  if (!base) return null;
  const partials = gradient(root, diffVars);
  const mv = compileMultiValueOnBackend(base, [root, ...partials]);
  if (!mv) return null;

  const { varSlots, numVars } = mv;
  const numDiff = diffVars.length;
  const stride = 1 + numDiff;
  const symName = `${underlyingName}-sym` as BackendName;

  return {
    shape: "grad",
    varSlots,
    numVars,
    diffVars,
    eval: () => syncEvalNotSupported(symName),
    evalAsync: async (vars: VarMap) => {
      const vals = await mv.evalAsync(vars);
      return { val: vals[0]!, gradient: vals.slice(1) };
    },
    evalBatch: async (vars: VarBatch, numPoints?: number) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return [];
      const packed = await mv.evalBatch(vars, n);
      const results: GradientResult[] = new Array(n);
      for (let p = 0; p < n; p++) {
        const off = p * stride;
        results[p] = {
          val: packed[off]!,
          gradient: Array.from(packed.subarray(off + 1, off + stride)),
        };
      }
      return results;
    },
    evalBatchPacked: mv.evalBatchPacked?.bind(mv),
    dispose: mv.dispose,
  };
}

// ---------------------------------------------------------------------------
// GPU symbolic jacobian (compile is sync, eval is async)
// ---------------------------------------------------------------------------

export function compileSymbolicJacobian(
  roots: NumNode[],
  diffVars: VarName[],
  underlyingName: BackendName,
): JacobianRoutine | null {
  const base = getBackend(underlyingName);
  if (!base) return null;
  const numRoots = roots.length;
  const numDiff = diffVars.length;
  const stride = 1 + numDiff;
  const symName = `${underlyingName}-sym` as BackendName;

  const expanded: NumNode[] = [];
  for (const root of roots) {
    expanded.push(root);
    for (const p of gradient(root, diffVars)) expanded.push(p);
  }
  const mv = compileMultiValueOnBackend(base, expanded);
  if (!mv) return null;

  const { varSlots, numVars } = mv;

  function unpackFromPacked(packed: Float32Array, n: number): JacobianResult[] {
    const pointStride = numRoots * stride;
    const results: JacobianResult[] = new Array(n);
    for (let p = 0; p < n; p++) {
      const pOff = p * pointStride;
      const valsOut = new Array<number>(numRoots);
      const jacobian: number[][] = [];
      for (let r = 0; r < numRoots; r++) {
        const rOff = pOff + r * stride;
        valsOut[r] = packed[rOff]!;
        jacobian.push(Array.from(packed.subarray(rOff + 1, rOff + stride)));
      }
      results[p] = { vals: valsOut, jacobian };
    }
    return results;
  }

  return {
    shape: "jacobian",
    varSlots,
    numVars,
    numRoots,
    diffVars,
    eval: () => syncEvalNotSupported(symName),
    evalAsync: async (vars: VarMap) => {
      const vals = await mv.evalAsync(vars);
      const valsOut = new Array<number>(numRoots);
      const jacobian: number[][] = [];
      for (let r = 0; r < numRoots; r++) {
        const off = r * stride;
        valsOut[r] = vals[off]!;
        jacobian.push(vals.slice(off + 1, off + stride));
      }
      return { vals: valsOut, jacobian };
    },
    evalBatch: async (vars: VarBatch, numPoints?: number) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return [];
      const packed = await mv.evalBatch(vars, n);
      return unpackFromPacked(packed, n);
    },
    dispose: mv.dispose,
  };
}
