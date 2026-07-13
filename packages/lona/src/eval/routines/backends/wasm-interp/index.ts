/**
 * WASM interpreter backend — runs a br_table-based tape interpreter inside
 * a shared WebAssembly module.
 */
import type { VarName } from "../../../../core/tree";
import {
  compileWasmTapeFromTape,
  compileWasmForwardAutodiff,
  compileWasmForwardAutodiffMulti,
} from "./tape-eval";
import { registerBackend } from "../../backend";
import type { VarMap } from "../../types";
import {
  compileSyncSymbolicGrad,
  compileSyncSymbolicJacobian,
} from "../_symbolic-helpers";

registerBackend({
  name: "wasm-interp",
  supported: new Set(["value", "grad", "jacobian"]),

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
    const fn = compileWasmForwardAutodiff(tape, diffVars);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "wasm-interp",
      kernel: {
        kind: "sync-grad",
        diffVars,
        eval: (vars: VarMap) => fn(vars as Map<VarName, number>),
      },
    };
  },

  compileJacobian(tape, diffVars) {
    const fn = compileWasmForwardAutodiffMulti(tape, diffVars);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "wasm-interp",
      kernel: {
        kind: "sync-jacobian",
        numRoots: tape.rootIndices.length,
        diffVars,
        eval: (vars: VarMap) => fn(vars as Map<VarName, number>),
      },
    };
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
