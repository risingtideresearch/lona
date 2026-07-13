/**
 * JS codegen backend — `new Function(...)` compilation of a CompiledTape.
 * Value / multi-value shapes are native; grad / jacobian via symbolic route.
 */
import type { VarName } from "../../../../core/tree";
import { compileFunctionFromTape } from "./codegen";
import { registerBackend } from "../../backend";
import type { VarMap } from "../../types";
import {
  compileSyncSymbolicGrad,
  compileSyncSymbolicJacobian,
} from "../_symbolic-helpers";

registerBackend({
  name: "js-codegen",
  supported: new Set(["value"]),

  compileValue(tape) {
    const numRoots = tape.rootIndices.length;
    const fn = compileFunctionFromTape(tape);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "js-codegen",
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
});

registerBackend({
  name: "js-codegen-sym",
  supported: new Set(["grad", "jacobian"]),

  compileGradFromRoots(roots, diffVars) {
    return compileSyncSymbolicGrad(roots[0]!, diffVars, "js-codegen");
  },
  compileJacobianFromRoots(roots, diffVars) {
    return compileSyncSymbolicJacobian(roots, diffVars, "js-codegen");
  },
});
