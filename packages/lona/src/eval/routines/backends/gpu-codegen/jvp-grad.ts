import type { VarName } from "../../../../core/tree";
import type { CompiledTape } from "../../../tape";
import { mapGpuReadbackBuffer, requireGpuDevice } from "../gpu-util";
import { compileGpuCodegenJvpDeviceKernel } from "./jvp-device";

const WORKGROUP_SIZE = 256;

export interface GpuCodegenGradEval {
  evalBatch(varData: Float32Array, numPoints: number): Promise<Float32Array>;
  readonly varSlots: VarName[];
  readonly numDiffVars: number;
  destroy(): void;
}

/** Host-oriented identity-seed adapter over the caller-owned-buffer JVP kernel. */
export function compileGpuCodegenGradFromTape(
  tape: CompiledTape,
  diffVarSlotIndices: number[],
): GpuCodegenGradEval {
  if (tape.rootIndices.length !== 1) {
    throw new Error("GPU codegen gradient evaluation requires one root");
  }
  for (const slot of diffVarSlotIndices) {
    if (!Number.isInteger(slot) || slot < 0 || slot >= tape.numVars) {
      throw new Error(`invalid GPU codegen differentiation slot: ${slot}`);
    }
  }
  const device = requireGpuDevice();
  const numVars = tape.numVars;
  const numDiffVars = diffVarSlotIndices.length;
  const variables = tape.varSlots.map((_, slot) =>
    slot < numVars
      ? {
          value: `inputValues[tid * ${numVars}u + ${slot}u]`,
          tangent: (direction: number) =>
            `inputTangents[(tid * ${numVars}u + ${slot}u) * ${numDiffVars}u + ${direction}u]`,
        }
      : { value: "0.0", tangent: () => "0.0" },
  );
  const jvp = compileGpuCodegenJvpDeviceKernel(tape, {
    inputWidth: numVars,
    outputWidth: 1,
    uniformWidth: 0,
    numDirections: numDiffVars,
    variables,
  });

  const bindingLimit = device.limits.maxStorageBufferBindingSize;
  const dispatchLimit =
    device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
  const maxPointsFor = (scalarsPerPoint: number): number =>
    scalarsPerPoint === 0
      ? dispatchLimit
      : Math.floor(bindingLimit / (scalarsPerPoint * 4));
  const maxBatch = Math.min(
    dispatchLimit,
    maxPointsFor(numVars),
    maxPointsFor(numVars * numDiffVars),
    maxPointsFor(1 + numDiffVars),
  );
  if (maxBatch < 1) {
    jvp.destroy();
    throw new Error("GPU codegen gradient buffers do not fit device limits");
  }

  let allocatedBatch = 0;
  let inputValues: GPUBuffer | null = null;
  let inputTangents: GPUBuffer | null = null;
  let outputValues: GPUBuffer | null = null;
  let outputTangents: GPUBuffer | null = null;
  let valueStaging: GPUBuffer | null = null;
  let tangentStaging: GPUBuffer | null = null;
  let identitySeeds: Float32Array | null = null;

  const destroyBatch = (): void => {
    inputValues?.destroy();
    inputTangents?.destroy();
    outputValues?.destroy();
    outputTangents?.destroy();
    valueStaging?.destroy();
    tangentStaging?.destroy();
    inputValues = null;
    inputTangents = null;
    outputValues = null;
    outputTangents = null;
    valueStaging = null;
    tangentStaging = null;
    identitySeeds = null;
    allocatedBatch = 0;
  };

  const ensureBatch = (numPoints: number): void => {
    if (allocatedBatch === numPoints) return;
    destroyBatch();
    const create = (bytes: number, usage: GPUBufferUsageFlags) =>
      device.createBuffer({ size: Math.max(bytes, 4), usage });
    try {
      inputValues = create(
        numPoints * numVars * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      inputTangents = create(
        numPoints * numVars * numDiffVars * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      outputValues = create(
        numPoints * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      outputTangents = create(
        numPoints * numDiffVars * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      valueStaging = create(
        numPoints * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      );
      tangentStaging = create(
        numPoints * numDiffVars * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      );
      identitySeeds = new Float32Array(numPoints * numVars * numDiffVars);
      for (let point = 0; point < numPoints; point++) {
        for (let direction = 0; direction < numDiffVars; direction++) {
          const slot = diffVarSlotIndices[direction]!;
          identitySeeds[(point * numVars + slot) * numDiffVars + direction] = 1;
        }
      }
      allocatedBatch = numPoints;
    } catch (error) {
      destroyBatch();
      throw error;
    }
  };

  const evalBatch = async (
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array> => {
    if (!Number.isInteger(numPoints) || numPoints < 0) {
      throw new Error(`invalid GPU codegen gradient point count: ${numPoints}`);
    }
    if (varData.length < numPoints * numVars) {
      throw new Error(
        `GPU codegen gradient received ${varData.length} values; expected ${numPoints * numVars}`,
      );
    }
    if (numPoints === 0) return new Float32Array(0);
    if (numPoints > maxBatch) {
      const stride = 1 + numDiffVars;
      const result = new Float32Array(numPoints * stride);
      for (let offset = 0; offset < numPoints; offset += maxBatch) {
        const end = Math.min(offset + maxBatch, numPoints);
        const chunk = await evalBatch(
          varData.subarray(offset * numVars, end * numVars),
          end - offset,
        );
        result.set(chunk, offset * stride);
      }
      return result;
    }

    ensureBatch(numPoints);
    device.queue.writeBuffer(
      inputValues!,
      0,
      varData.buffer as ArrayBuffer,
      varData.byteOffset,
      numPoints * numVars * 4,
    );
    if (identitySeeds!.byteLength > 0) {
      device.queue.writeBuffer(inputTangents!, 0, identitySeeds!);
    }
    const encoder = device.createCommandEncoder();
    jvp.encode(
      encoder,
      { buffer: inputValues!, offset: 0, byteLength: numPoints * numVars * 4 },
      {
        buffer: inputTangents!,
        offset: 0,
        byteLength: identitySeeds!.byteLength,
      },
      null,
      null,
      { buffer: outputValues!, offset: 0, byteLength: numPoints * 4 },
      {
        buffer: outputTangents!,
        offset: 0,
        byteLength: numPoints * numDiffVars * 4,
      },
      numPoints,
    );
    encoder.copyBufferToBuffer(
      outputValues!,
      0,
      valueStaging!,
      0,
      numPoints * 4,
    );
    if (numDiffVars > 0) {
      encoder.copyBufferToBuffer(
        outputTangents!,
        0,
        tangentStaging!,
        0,
        numPoints * numDiffVars * 4,
      );
    }
    device.queue.submit([encoder.finish()]);
    const [values, tangents] = await Promise.all([
      mapGpuReadbackBuffer(valueStaging!, numPoints * 4),
      mapGpuReadbackBuffer(tangentStaging!, numPoints * numDiffVars * 4),
    ]);
    const stride = 1 + numDiffVars;
    const result = new Float32Array(numPoints * stride);
    for (let point = 0; point < numPoints; point++) {
      result[point * stride] = values[point]!;
      result.set(
        tangents.subarray(point * numDiffVars, (point + 1) * numDiffVars),
        point * stride + 1,
      );
    }
    return result;
  };

  return {
    evalBatch,
    varSlots: tape.varSlots,
    numDiffVars,
    destroy() {
      destroyBatch();
      jvp.destroy();
    },
  };
}
