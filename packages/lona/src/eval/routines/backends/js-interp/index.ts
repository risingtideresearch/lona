/**
 * JS interpreter backend — walks a CompiledTape in JavaScript.
 */
import type { VarName } from "../../../../core/tree";
import { evalTape, compileSeededJvp } from "./tape-eval";
import {
  registerBackend,
  type KernelEnvelope,
  type JvpKernel,
} from "../../backend";
import type { VarMap } from "../../types";
import { adaptSyncJvpToGrad, adaptSyncJvpToJacobian } from "../_jvp-adapters";

function compileJvp(
  tape: Parameters<typeof compileSeededJvp>[0],
  numDirections: number,
): KernelEnvelope<JvpKernel> {
  return {
    varSlots: tape.varSlots,
    numVars: tape.numVars,
    backend: "js-interp",
    kernel: {
      kind: "sync-jvp",
      numRoots: tape.rootIndices.length,
      numDirections,
      evalPacked: compileSeededJvp(tape, numDirections),
    },
  };
}

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
    return adaptSyncJvpToGrad(
      tape,
      diffVars,
      compileJvp(tape, diffVars.length),
    );
  },

  compileJvp,

  compileJacobian(tape, diffVars) {
    return adaptSyncJvpToJacobian(
      tape,
      diffVars,
      compileJvp(tape, diffVars.length),
    );
  },
});
