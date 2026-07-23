/**
 * WASM interpreter backend — runs a br_table-based tape interpreter inside
 * a shared WebAssembly module.
 */
import type { VarName } from "../../../../core/tree";
import { compileWasmTapeFromTape, compileWasmSeededJvp } from "./tape-eval";
import {
  registerBackend,
  type JvpKernel,
  type KernelEnvelope,
} from "../../backend";
import type { VarMap } from "../../types";
import { adaptSyncJvpToGrad, adaptSyncJvpToJacobian } from "../_jvp-adapters";
import {
  compileSyncSymbolicGrad,
  compileSyncSymbolicJacobian,
} from "../_symbolic-helpers";

function compileJvp(
  tape: Parameters<typeof compileWasmSeededJvp>[0],
  numDirections: number,
): KernelEnvelope<JvpKernel> {
  return {
    varSlots: tape.varSlots,
    numVars: tape.numVars,
    backend: "wasm-interp",
    kernel: {
      kind: "sync-jvp",
      numRoots: tape.rootIndices.length,
      numDirections,
      evalPacked: compileWasmSeededJvp(tape, numDirections),
    },
  };
}

registerBackend({
  name: "wasm-interp",
  supported: new Set(["value", "grad", "jacobian"]),

  compileJvp,

  compileValue(tape) {
    const numRoots = tape.rootIndices.length;
    const fn = compileWasmTapeFromTape(tape);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "wasm-interp",
      kernel: {
        kind: "sync-value",
        numRoots,
        eval: (vars: VarMap, derivatives?: VarMap) =>
          fn(
            vars as Map<VarName, number>,
            derivatives as Map<VarName, number> | undefined,
          ),
      },
    };
  },

  compileGrad(tape, diffVars) {
    return adaptSyncJvpToGrad(
      tape,
      diffVars,
      compileJvp(tape, diffVars.length),
    );
  },

  compileJacobian(tape, diffVars) {
    return adaptSyncJvpToJacobian(
      tape,
      diffVars,
      compileJvp(tape, diffVars.length),
    );
  },
});

registerBackend({
  name: "wasm-interp-sym",
  supported: new Set(["grad", "jacobian"]),

  compileGradFromRoots(roots, diffVars) {
    return compileSyncSymbolicGrad(roots[0]!, diffVars, "wasm-interp");
  },
  compileJacobianFromRoots(roots, diffVars) {
    return compileSyncSymbolicJacobian(roots, diffVars, "wasm-interp");
  },
});
