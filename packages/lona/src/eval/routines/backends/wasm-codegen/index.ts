/**
 * WASM codegen backend — emits a per-DAG WebAssembly module with direct
 * WASM instructions per tape node.
 */
import type { VarName } from "../../../../core/tree";
import { compileWasmFromTape, compileWasmJvpFromTape } from "./codegen";
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
  tape: Parameters<typeof compileWasmJvpFromTape>[0],
  numDirections: number,
): KernelEnvelope<JvpKernel> {
  return {
    varSlots: tape.varSlots,
    numVars: tape.numVars,
    backend: "wasm-codegen",
    kernel: {
      kind: "sync-jvp",
      numRoots: tape.rootIndices.length,
      numDirections,
      evalPacked: compileWasmJvpFromTape(tape, numDirections),
    },
  };
}

registerBackend({
  name: "wasm-codegen",
  supported: new Set(["value", "grad", "jacobian"]),

  compileJvp,

  compileValue(tape) {
    const numRoots = tape.rootIndices.length;
    const fn = compileWasmFromTape(tape);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "wasm-codegen",
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
  name: "wasm-codegen-sym",
  supported: new Set(["grad", "jacobian"]),

  compileGradFromRoots(roots, diffVars) {
    return compileSyncSymbolicGrad(roots[0]!, diffVars, "wasm-codegen");
  },
  compileJacobianFromRoots(roots, diffVars) {
    return compileSyncSymbolicJacobian(roots, diffVars, "wasm-codegen");
  },
});
