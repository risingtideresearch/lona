/**
 * GPU codegen symbolic grad — compile fullDerivative(root) as a single
 * codegen shader and evaluate it with rotated derivative seeds.
 *
 * The fullDerivative DAG is roughly the same size as the original root
 * (no blow-up from explicit partial DAGs). Each partial ∂f/∂vᵢ is
 * obtained by seeding Derivative(vᵢ) = 1 and all other seeds = 0.
 *
 * At eval time:
 *   - Value shader: N threads  → f(x) per point
 *   - Derivative shader: N×D threads → ∂f/∂vᵢ per (point, diffVar)
 *   - Both dispatches run in parallel
 */
import type { NumNode, VarName } from "../../../../core/tree";
import { fullDerivative } from "../../../../api/diff";
import { compileTape } from "../../../tape";
import { compileGpuCodegenFromTape } from "./codegen";
import type {
  GradRoutine,
  GradientResult,
  VarBatch,
  VarMap,
} from "../../types";
import { inferNumPoints, packVarBatch } from "../../batch-pack";
import { syncEvalNotSupported } from "../../backend";

export function compileFullDerivativeGrad(
  root: NumNode,
  diffVars: VarName[],
): GradRoutine | null {
  // Compile both tapes (sync CPU work)
  const diffRoot = fullDerivative(root);
  const derivTape = compileTape([diffRoot]);
  const rootTape = compileTape([root]);
  if (!derivTape || !rootTape) return null;

  // Compile both GPU shaders (sync — GPU must be initialized)
  const derivGpu = compileGpuCodegenFromTape(derivTape);
  const valueGpu = compileGpuCodegenFromTape(rootTape);

  // derivTape.varSlots = [regular vars..., derivative seed vars...]
  // derivTape.numVars = number of regular vars
  const allSlots = derivTape.varSlots;
  const numRegularVars = derivTape.numVars;
  const regularVarSlots = allSlots.slice(0, numRegularVars);
  const totalSlots = allSlots.length;
  const numDiff = diffVars.length;

  // Map each diffVar to its derivative-seed slot index in allSlots
  const derivSlotIndices: number[] = [];
  for (const v of diffVars) {
    const idx = allSlots.indexOf(v, numRegularVars);
    if (idx === -1) {
      derivGpu.destroy();
      valueGpu.destroy();
      return null;
    }
    derivSlotIndices.push(idx);
  }

  /**
   * Build packed varData for the derivative shader.
   * Layout: (N×D) × totalSlots. Each D-block seeds one diffVar.
   */
  function packSeededVarData(
    baseVarData: Float32Array, // N × numRegularVars
    numPoints: number,
  ): Float32Array {
    const numThreads = numPoints * numDiff;
    const data = new Float32Array(numThreads * totalSlots);

    for (let d = 0; d < numDiff; d++) {
      const blockBase = d * numPoints;
      for (let p = 0; p < numPoints; p++) {
        const dstOff = (blockBase + p) * totalSlots;
        // Copy regular vars
        for (let v = 0; v < numRegularVars; v++) {
          data[dstOff + v] = baseVarData[p * numRegularVars + v]!;
        }
        // Seed: 1 for this diffVar, 0 for others (0 is default from Float32Array)
        data[dstOff + derivSlotIndices[d]!] = 1;
      }
    }

    return data;
  }

  async function evalBatchGrad(
    varData: Float32Array, // N × numRegularVars
    numPoints: number,
  ): Promise<GradientResult[]> {
    const numThreads = numPoints * numDiff;

    // Two dispatches in parallel
    const [derivResults, valueResults] = await Promise.all([
      derivGpu.evalBatch(packSeededVarData(varData, numPoints), numThreads),
      valueGpu.evalBatch(varData, numPoints),
    ]);

    // Assemble: derivResults[d*N + p] = ∂f/∂diffVars[d] at point p
    const results: GradientResult[] = new Array(numPoints);
    for (let p = 0; p < numPoints; p++) {
      const gradient = new Array<number>(numDiff);
      for (let d = 0; d < numDiff; d++) {
        gradient[d] = derivResults[d * numPoints + p]!;
      }
      results[p] = { val: valueResults[p]!, gradient };
    }
    return results;
  }

  return {
    shape: "grad",
    varSlots: regularVarSlots,
    numVars: numRegularVars,
    diffVars,
    eval: () => syncEvalNotSupported("gpu-codegen-sym"),
    evalAsync: async (vars: VarMap) => {
      const data = new Float32Array(numRegularVars);
      for (let v = 0; v < numRegularVars; v++) {
        data[v] = (vars as Map<VarName, number>).get(regularVarSlots[v]!) ?? 0;
      }
      return (await evalBatchGrad(data, 1))[0]!;
    },
    evalBatch: async (vars: VarBatch, numPoints?: number) => {
      const n = numPoints ?? inferNumPoints(vars);
      if (n === 0) return [];
      const data = packVarBatch(regularVarSlots, numRegularVars, vars, n);
      return evalBatchGrad(data, n);
    },
    dispose: () => {
      derivGpu.destroy();
      valueGpu.destroy();
    },
  };
}
