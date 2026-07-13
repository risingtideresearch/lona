/**
 * WASM codegen backend — emits a per-DAG WebAssembly module with direct
 * WASM instructions per tape node.
 */
import type { VarName } from "../../../../core/tree";
import { compileWasmFromTape, compileWasmGradFromTape } from "./codegen";
import { registerBackend } from "../../backend";
import type { VarMap } from "../../types";
import {
  compileSyncSymbolicGrad,
  compileSyncSymbolicJacobian,
} from "../_symbolic-helpers";

registerBackend({
  name: "wasm-codegen",
  supported: new Set(["value", "grad"]),

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
    const fn = compileWasmGradFromTape(tape, diffVars);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "wasm-codegen",
      kernel: {
        kind: "sync-grad",
        diffVars,
        eval: (vars: VarMap) => fn(vars as Map<VarName, number>),
      },
    };
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
