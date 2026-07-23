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

  compileJvp(tape, numDirections) {
    const inputNames = tape.varSlots.slice(0, tape.numVars);
    const gradients = tape.rootIndices.map((rootIndex) =>
      compileWasmGradFromTape(
        { ...tape, rootIndices: [rootIndex] },
        inputNames,
      ),
    );
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "wasm-codegen",
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
          const evaluated = gradients.map((gradient) => gradient(vars));
          return {
            vals: evaluated.map(({ val }) => val),
            tangents: evaluated.map(({ gradient }) =>
              Array.from({ length: numDirections }, (_, direction) => {
                let result = 0;
                for (let input = 0; input < tape.numVars; input++) {
                  result +=
                    gradient[input]! *
                    seeds[input * numDirections + direction]!;
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
