/**
 * JS codegen backend — `new Function(...)` compilation of a CompiledTape.
 * Value / multi-value shapes are native; grad / jacobian via symbolic route.
 */
import type { VarName } from "../../../../core/tree";
import { compileFunctionFromTape, compileJvpFunctionFromTape } from "./codegen";
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
  tape: Parameters<typeof compileJvpFunctionFromTape>[0],
  numDirections: number,
): KernelEnvelope<JvpKernel> {
  return {
    varSlots: tape.varSlots,
    numVars: tape.numVars,
    backend: "js-codegen",
    kernel: {
      kind: "sync-jvp",
      numRoots: tape.rootIndices.length,
      numDirections,
      evalPacked: compileJvpFunctionFromTape(tape, numDirections),
    },
  };
}

registerBackend({
  name: "js-codegen",
  supported: new Set(["value", "grad", "jacobian"]),

  compileJvp,

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
