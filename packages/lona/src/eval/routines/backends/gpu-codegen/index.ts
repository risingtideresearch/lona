/**
 * GPU codegen backend — emits a per-DAG WGSL compute shader.
 */
import { compileGpuCodegenFromTape } from "./codegen";
import { registerBackend } from "../../backend";
import { compileSymbolicJacobian } from "../_symbolic-helpers";
import { compileFullDerivativeGrad } from "./symbolic-grad";

registerBackend({
  name: "gpu-codegen",
  supported: new Set(["value"]),

  compileValue(tape) {
    const numRoots = tape.rootIndices.length;
    const gpu = compileGpuCodegenFromTape(tape);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "gpu-codegen",
      kernel: {
        kind: "async-batch-value",
        numRoots,
        evalBatchPacked: (varData: Float32Array, numPoints: number) =>
          gpu.evalBatch(varData, numPoints),
      },
      dispose: () => gpu.destroy(),
    };
  },
});

registerBackend({
  name: "gpu-codegen-sym",
  supported: new Set(["grad", "jacobian"]),
  compileGradFromRoots(roots, diffVars) {
    return compileFullDerivativeGrad(roots[0]!, diffVars);
  },
  compileJacobianFromRoots(roots, diffVars) {
    return compileSymbolicJacobian(roots, diffVars, "gpu-codegen");
  },
});
