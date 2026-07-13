/**
 * GPU tape interpreter backend — WGSL compute shader running a br_table
 * interpreter per GPU thread, one point per thread.
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
    // The tape-interpreter path only emits one output per point (rootIndex),
    // so it can only handle single-root tapes.
    if (tape.rootIndices.length > 1) return null;
    const gpu = compileGpuTapeFromTape(tape);
    return {
      varSlots: tape.varSlots,
      numVars: tape.numVars,
      backend: "gpu-interp",
      kernel: {
        kind: "async-batch-value",
        numRoots: 1,
        evalBatchPacked: (varData: Float32Array, numPoints: number) =>
          gpu.evalBatch(varData, numPoints),
      },
      dispose: () => gpu.destroy(),
    };
  },

  compileGrad(tape, diffVars) {
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
