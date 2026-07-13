/**
 * JS interpreter backend — walks a CompiledTape in JavaScript.
 */
import type { VarName } from "../../../../core/tree";
import {
  evalTape,
  compileForwardAutodiff,
  compileForwardAutodiffMulti,
} from "./tape-eval";
import { registerBackend } from "../../backend";
import type { VarMap } from "../../types";

registerBackend({
  name: "js-interp",
  supported: new Set(["value", "grad", "jacobian"]),

  compileValue(tape) {
    const numRoots = tape.rootIndices.length;
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "js-interp",
      kernel: {
        kind: "sync-value",
        numRoots,
        eval: (vars: VarMap, derivatives?: VarMap) =>
          evalTape(
            tape,
            vars as Map<VarName, number>,
            derivatives as Map<VarName, number> | undefined,
          ),
      },
    };
  },

  compileGrad(tape, diffVars) {
    const fn = compileForwardAutodiff(tape, diffVars);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "js-interp",
      kernel: {
        kind: "sync-grad",
        diffVars,
        eval: (vars: VarMap) => fn(vars as Map<VarName, number>),
      },
    };
  },

  compileJacobian(tape, diffVars) {
    const fn = compileForwardAutodiffMulti(tape, diffVars);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "js-interp",
      kernel: {
        kind: "sync-jacobian",
        numRoots: tape.rootIndices.length,
        diffVars,
        eval: (vars: VarMap) => fn(vars as Map<VarName, number>),
      },
    };
  },
});
