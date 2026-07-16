/**
 * GPU tape interpreter backend — WGSL compute shader running a br_table
 * interpreter per GPU thread, one point per thread. Value routines support
 * multi-root tapes (numPoints * numRoots interleaved output); grad is
 * single-root only.
 */
import {
  compileGpuTapeFromTape,
  compileGpuTapeGradFromTape,
} from "./tape-eval";
import { registerBackend } from "../../backend";
import type { VarName } from "../../../../core/tree";

registerBackend({
  name: "gpu-interp",
  supported: new Set(["value", "grad"]),

  compileValue(tape) {
    const gpu = compileGpuTapeFromTape(tape);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "gpu-interp",
      kernel: {
        kind: "async-batch-value",
        numRoots: tape.rootIndices.length,
        evalBatchPacked: (varData: Float32Array, numPoints: number) =>
          gpu.evalBatch(varData, numPoints),
      },
      dispose: () => gpu.destroy(),
    };
  },

  compileGrad(tape, diffVars) {
    // Multi-root differentiation is the jacobian shape, which this backend
    // does not support — the grad shader evaluates a single root.
    if (tape.rootIndices.length > 1) return null;
    const diffSlots: number[] = [];
    for (const v of diffVars) {
      const idx = tape.varSlots.indexOf(v);
      if (idx === -1) return null;
      diffSlots.push(idx);
    }

    const gpu = compileGpuTapeGradFromTape(tape, diffSlots);
    return {
      varSlots: tape.varSlots as readonly VarName[],
      numVars: tape.numVars,
      backend: "gpu-interp" as const,
      kernel: {
        kind: "async-batch-grad" as const,
        diffVars,
        evalBatchPacked: (varData: Float32Array, numPoints: number) =>
          gpu.evalBatch(varData, numPoints),
      },
      dispose: () => gpu.destroy(),
    };
  },
});
