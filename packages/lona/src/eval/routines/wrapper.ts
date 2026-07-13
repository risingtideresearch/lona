/**
 * Routine wrappers — turn minimal Kernel objects (produced by backends) into
 * the full Routine API declared in `./types.ts`, synthesizing the paths the
 * backend didn't implement natively.
 */
import type { VarName } from "../../core/tree";
import type {
  GradKernel,
  JacobianKernel,
  KernelEnvelope,
  ValueKernel,
} from "./backend";
import { syncEvalNotSupported } from "./backend";
import {
  packVarBatch,
  inferNumPoints,
  resolveVarBatchColumns,
  writeVarBatchRow,
} from "./batch-pack";
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

// ---------------------------------------------------------------------------
// Value (handles both single-root and multi-root)
// ---------------------------------------------------------------------------

export function wrapValue(
  env: KernelEnvelope<ValueKernel>,
): ValueRoutine | MultiValueRoutine {
  const { kernel, varSlots, numVars, backend } = env;
  const numRoots = kernel.numRoots;

  if (kernel.kind === "sync-value") {
    const evalSync = (vars: VarMap, derivatives?: VarMap) =>
      kernel.eval(vars, derivatives);

    if (numRoots === 1) {
      return {
        shape: "value",
        varSlots,
        numVars,
        eval: (vars: VarMap, derivatives?: VarMap) =>
          evalSync(vars, derivatives)[0]!,
        evalAsync: (vars, derivatives) =>
          Promise.resolve(evalSync(vars, derivatives)[0]!),
        evalBatch: async (vars, numPoints) => {
          const n = numPoints ?? inferNumPoints(vars);
          if (n === 0) return new Float32Array(0);
          const out = new Float32Array(n);
          const columns = resolveVarBatchColumns(varSlots, numVars, vars);
          const row = new Map<VarName, number>();
          for (let p = 0; p < n; p++) {
            writeVarBatchRow(row, varSlots, numVars, columns, p);
            out[p] = evalSync(row)[0]!;
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
            out[p] = evalSync(row)[0]!;
          }
          return out;
        },
        dispose: env.dispose,
      };
    }

    // multi-root
    return {
      shape: "multi-value",
      varSlots,
      numVars,
      numRoots,
      eval: evalSync,
      evalAsync: (vars, derivatives) =>
        Promise.resolve(evalSync(vars, derivatives)),
      evalBatch: async (vars, numPoints) => {
        const n = numPoints ?? inferNumPoints(vars);
        if (n === 0) return new Float32Array(0);
        const out = new Float32Array(n * numRoots);
        const columns = resolveVarBatchColumns(varSlots, numVars, vars);
        const row = new Map<VarName, number>();
        for (let p = 0; p < n; p++) {
          writeVarBatchRow(row, varSlots, numVars, columns, p);
          const vals = evalSync(row);
          for (let r = 0; r < numRoots; r++) {
            out[p * numRoots + r] = vals[r]!;
          }
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
          const vals = evalSync(row);
          for (let r = 0; r < numRoots; r++) {
            out[p * numRoots + r] = vals[r]!;
          }
        }
        return out;
      },
      dispose: env.dispose,
    };
  }

  // async-batch backend (GPU)
  const evalBatchPacked = kernel.evalBatchPacked.bind(kernel);

  if (numRoots === 1) {
    return {
      shape: "value",
      varSlots,
      numVars,
      eval: () => syncEvalNotSupported(backend),
      evalAsync: async (vars) => {
        const data = packVarBatchFromVarMap(varSlots, numVars, vars);
        const out = await evalBatchPacked(data, 1);
        return out[0]!;
      },
      evalBatch: async (vars, numPoints) => {
        const n = numPoints ?? inferNumPoints(vars);
        if (n === 0) return new Float32Array(0);
        const data = packVarBatch(varSlots, numVars, vars, n);
        return evalBatchPacked(data, n);
      },
      evalBatchPacked,
      dispose: env.dispose,
    };
  }

  return {
    shape: "multi-value",
    varSlots,
    numVars,
    numRoots,
    eval: () => syncEvalNotSupported(backend),
    evalAsync: async (vars) => {
      const data = packVarBatchFromVarMap(varSlots, numVars, vars);
      const out = await evalBatchPacked(data, 1);
      const result = new Array<number>(numRoots);
      for (let r = 0; r < numRoots; r++) result[r] = out[r]!;
      return result;
    },
    evalBatch: async (vars, numPoints) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return new Float32Array(0);
      const data = packVarBatch(varSlots, numVars, vars, n);
      return evalBatchPacked(data, n);
    },
    evalBatchPacked,
    dispose: env.dispose,
  };
}

// ---------------------------------------------------------------------------
// Grad
// ---------------------------------------------------------------------------

export function wrapGrad(env: KernelEnvelope<GradKernel>): GradRoutine {
  const { kernel, varSlots, numVars, backend } = env;

  if (kernel.kind === "sync-grad") {
    const evalSync = (vars: VarMap) => kernel.eval(vars);
    return {
      shape: "grad",
      varSlots,
      numVars,
      diffVars: kernel.diffVars,
      eval: evalSync,
      evalAsync: (vars) => Promise.resolve(evalSync(vars)),
      evalBatch: async (vars, numPoints) => {
        const n = numPoints ?? inferNumPoints(vars);
        const columns = resolveVarBatchColumns(varSlots, numVars, vars);
        const row = new Map<VarName, number>();
        const out: GradientResult[] = [];
        for (let p = 0; p < n; p++) {
          writeVarBatchRow(row, varSlots, numVars, columns, p);
          out.push(evalSync(row));
        }
        return out;
      },
      dispose: env.dispose,
    };
  }

  // async-batch grad kernel (GPU AD — all-partials-per-thread)
  const { diffVars, evalBatchPacked: kernelEvalBatchPacked } = kernel;
  const numDiff = diffVars.length;
  const stride = 1 + numDiff;

  /** Unpack one point from a packed [val, ∂v₀, …] layout. */
  function unpackPoint(packed: Float32Array, offset: number): GradientResult {
    return {
      val: packed[offset]!,
      gradient: Array.from(packed.subarray(offset + 1, offset + stride)),
    };
  }

  const evalBatchPacked = kernelEvalBatchPacked.bind(kernel);

  return {
    shape: "grad",
    varSlots,
    numVars,
    diffVars,
    eval: () => syncEvalNotSupported(backend),
    evalAsync: async (vars) => {
      const data = packVarBatchFromVarMap(varSlots, numVars, vars);
      const packed = await evalBatchPacked(data, 1);
      return unpackPoint(packed, 0);
    },
    evalBatch: async (vars, numPoints) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return [];
      const data = packVarBatch(varSlots, numVars, vars, n);
      const packed = await evalBatchPacked(data, n);
      const results: GradientResult[] = new Array(n);
      for (let p = 0; p < n; p++) {
        results[p] = unpackPoint(packed, p * stride);
      }
      return results;
    },
    evalBatchPacked,
    dispose: env.dispose,
  };
}

// ---------------------------------------------------------------------------
// Jacobian
// ---------------------------------------------------------------------------

export function wrapJacobian(
  env: KernelEnvelope<JacobianKernel>,
): JacobianRoutine {
  const { kernel, varSlots, numVars } = env;
  const evalSync = (vars: VarMap) => kernel.eval(vars);

  return {
    shape: "jacobian",
    varSlots,
    numVars,
    numRoots: kernel.numRoots,
    diffVars: kernel.diffVars,
    eval: evalSync,
    evalAsync: (vars) => Promise.resolve(evalSync(vars)),
    evalBatch: async (vars, numPoints) => {
      const n = numPoints ?? inferNumPoints(vars);
      const columns = resolveVarBatchColumns(varSlots, numVars, vars);
      const row = new Map<VarName, number>();
      const out: JacobianResult[] = [];
      for (let p = 0; p < n; p++) {
        writeVarBatchRow(row, varSlots, numVars, columns, p);
        out.push(evalSync(row));
      }
      return out;
    },
    dispose: env.dispose,
  };
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function packVarBatchFromVarMap(
  varSlots: readonly VarName[],
  numVars: number,
  vars: VarMap,
): Float32Array {
  const out = new Float32Array(numVars);
  for (let s = 0; s < numVars; s++) {
    out[s] = (vars as Map<VarName, number>).get(varSlots[s]!) ?? 0;
  }
  return out;
}

// Keep unused-import linter quiet — `VarBatch` is used via the return-type
// declarations on the wrapped routine interfaces.
export type { VarBatch };
