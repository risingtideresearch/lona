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

  compileJvp(tape, numDirections) {
    const inputNames = tape.varSlots.slice(0, tape.numVars);
    const jacobian = compileWasmForwardAutodiffMulti(tape, inputNames);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "wasm-interp",
      kernel: {
        kind: "sync-jvp",
        numRoots: tape.rootIndices.length,
        numDirections,
        evalPacked(values: Float64Array, seeds: Float64Array) {
          if (values.length !== tape.numVars) {
            throw new Error(
              `seeded JVP expected ${tape.numVars} values, got ${values.length}`,
            );
          }
          if (seeds.length !== tape.numVars * numDirections) {
            throw new Error(
              `seeded JVP expected ${tape.numVars * numDirections} seeds, got ${seeds.length}`,
            );
          }
          const vars = new Map<VarName, number>();
          for (let input = 0; input < tape.numVars; input++) {
            vars.set(inputNames[input]!, values[input]!);
          }
          const local = jacobian(vars);
          return {
            vals: local.vals,
            tangents: local.jacobian.map((row) =>
              Array.from({ length: numDirections }, (_, direction) => {
                let result = 0;
                for (let input = 0; input < tape.numVars; input++) {
                  result +=
                    row[input]! * seeds[input * numDirections + direction]!;
                }
                return result;
              }),
            ),
          };
        },
      },
    };
  },

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
