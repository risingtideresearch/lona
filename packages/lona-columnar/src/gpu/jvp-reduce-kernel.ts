import type { DeviceBufferSlice, NumNode } from "lona/internal";
import {
  compileGpuInterpJvpDeviceKernel,
  requireGpuDevice,
} from "lona/internal";
import { compileColumnarTape } from "../compile-tape";
import type { ColumnarParamInput } from "../ir";
import { compileGpuJvpPackKernel } from "./jvp-interp-pack-kernel";
import { compileColumnarGpuJvpTapeKernel } from "./jvp-tape-kernel";

export interface CompiledGpuJvpReduceKernel {
  readonly width: number;
  readonly uniformWidth: number;
  readonly numDirections: number;
  dispatchCount(count: number): number;
  encode(
    encoder: GPUCommandEncoder,
    inputValues: DeviceBufferSlice,
    inputTangents: DeviceBufferSlice,
    uniformValues: DeviceBufferSlice | null,
    uniformTangents: DeviceBufferSlice | null,
    outputValues: DeviceBufferSlice,
    outputTangents: DeviceBufferSlice,
    count: number,
  ): void;
  dispose(): void;
}

export function compileGpuInterpJvpReduceKernel(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
  width: number,
  numDirections: number,
  maxCount: number,
): CompiledGpuJvpReduceKernel {
  const compiled = compileColumnarTape(roots, inputs);
  const uniformWidth = compiled.tapeInputs.reduce(
    (result, { binding }) =>
      binding.kind === "uniform"
        ? Math.max(result, binding.component + 1)
        : result,
    0,
  );
  const variables = compiled.tapeInputs.map(({ binding }) => {
    switch (binding.kind) {
      case "reduce-left": {
        const scalar = `(tid * 2u) * ${width}u + ${binding.component}u`;
        return {
          value: `inputValues[${scalar}]`,
          tangent: (direction: number) =>
            `inputTangents[(${scalar}) * ${numDirections}u + ${direction}u]`,
        };
      }
      case "reduce-right": {
        const scalar = `(tid * 2u + 1u) * ${width}u + ${binding.component}u`;
        return {
          value: `inputValues[${scalar}]`,
          tangent: (direction: number) =>
            `inputTangents[(${scalar}) * ${numDirections}u + ${direction}u]`,
        };
      }
      case "uniform":
        return {
          value: `uniformValues[${binding.component}u]`,
          tangent: (direction: number) =>
            `uniformTangents[${binding.component * numDirections + direction}u]`,
        };
      default:
        throw new Error(
          `GPU interpreter JVP reduction does not support '${binding.kind}' parameter bindings`,
        );
    }
  });
  const maxPairs = Math.max(Math.floor(maxCount / 2), 1);
  const pack = compileGpuJvpPackKernel(variables, numDirections);
  const interp = compileGpuInterpJvpDeviceKernel(
    compiled.tape,
    numDirections,
    maxPairs,
  );
  const device = requireGpuDevice();
  const packedValueBytes = maxPairs * compiled.tape.numVars * 4;
  const packedTangentBytes = packedValueBytes * numDirections;
  const packedValues = device.createBuffer({
    size: Math.max(packedValueBytes, 4),
    usage: GPUBufferUsage.STORAGE,
  });
  const packedTangents = device.createBuffer({
    size: Math.max(packedTangentBytes, 4),
    usage: GPUBufferUsage.STORAGE,
  });
  const scratchRows = Math.max(Math.ceil(maxCount / 2), 1);
  const valueScratch = Array.from({ length: 2 }, () =>
    device.createBuffer({
      size: scratchRows * width * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    }),
  );
  const tangentScratch = Array.from({ length: 2 }, () =>
    device.createBuffer({
      size: Math.max(scratchRows * width * numDirections * 4, 4),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    }),
  );
  return {
    width,
    uniformWidth,
    numDirections,
    dispatchCount(count): number {
      return Math.ceil(Math.log2(Math.max(count, 1))) * 2;
    },
    encode(
      encoder,
      inputValues,
      inputTangents,
      uniformValues,
      uniformTangents,
      outputValues,
      outputTangents,
      count,
    ): void {
      if (!Number.isInteger(count) || count < 2 || count > maxCount) {
        throw new Error(
          `GPU interpreter JVP reduction count ${count} is outside [2, ${maxCount}]`,
        );
      }
      let level = 0;
      let levelCount = count;
      let levelValues = inputValues;
      let levelTangents = inputTangents;
      while (levelCount > 1) {
        const pairCount = Math.floor(levelCount / 2);
        const outputCount = Math.ceil(levelCount / 2);
        const finalLevel = outputCount === 1;
        const nextValues: DeviceBufferSlice = finalLevel
          ? outputValues
          : {
              buffer: valueScratch[level % 2]!,
              offset: 0,
              byteLength: outputCount * width * 4,
            };
        const nextTangents: DeviceBufferSlice = finalLevel
          ? outputTangents
          : {
              buffer: tangentScratch[level % 2]!,
              offset: 0,
              byteLength: outputCount * width * numDirections * 4,
            };
        const packedValueSlice = {
          buffer: packedValues,
          offset: 0,
          byteLength: pairCount * compiled.tape.numVars * 4,
        };
        const packedTangentSlice = {
          buffer: packedTangents,
          offset: 0,
          byteLength: pairCount * compiled.tape.numVars * numDirections * 4,
        };
        pack.encode(
          encoder,
          levelValues,
          levelTangents,
          uniformValues,
          uniformTangents,
          packedValueSlice,
          packedTangentSlice,
          pairCount,
        );
        interp.encode(
          encoder,
          packedValueSlice,
          packedTangentSlice,
          nextValues,
          nextTangents,
          pairCount,
        );
        if (levelCount % 2 === 1) {
          const sourceScalar = (levelCount - 1) * width;
          const targetScalar = pairCount * width;
          encoder.copyBufferToBuffer(
            levelValues.buffer,
            levelValues.offset + sourceScalar * 4,
            nextValues.buffer,
            nextValues.offset + targetScalar * 4,
            width * 4,
          );
          if (numDirections > 0) {
            encoder.copyBufferToBuffer(
              levelTangents.buffer,
              levelTangents.offset + sourceScalar * numDirections * 4,
              nextTangents.buffer,
              nextTangents.offset + targetScalar * numDirections * 4,
              width * numDirections * 4,
            );
          }
        }
        levelValues = nextValues;
        levelTangents = nextTangents;
        levelCount = outputCount;
        level++;
      }
    },
    dispose(): void {
      pack.destroy();
      interp.destroy();
      packedValues.destroy();
      packedTangents.destroy();
      for (const buffer of valueScratch) buffer.destroy();
      for (const buffer of tangentScratch) buffer.destroy();
    },
  };
}

export function compileGpuJvpReduceKernel(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
  width: number,
  numDirections: number,
): CompiledGpuJvpReduceKernel {
  const compiled = compileColumnarTape(roots, inputs);
  const uniformWidth = compiled.tapeInputs.reduce(
    (result, { binding }) =>
      binding.kind === "uniform"
        ? Math.max(result, binding.component + 1)
        : result,
    0,
  );
  const variables = compiled.tapeInputs.map(({ binding }) => {
    switch (binding.kind) {
      case "reduce-left": {
        const scalar = `(tid * 2u) * ${width}u + ${binding.component}u`;
        return {
          value: `inputValues[${scalar}]`,
          tangent: (direction: number) =>
            `inputTangents[(${scalar}) * ${numDirections}u + ${direction}u]`,
        };
      }
      case "reduce-right": {
        const scalar = `(tid * 2u + 1u) * ${width}u + ${binding.component}u`;
        return {
          value: `inputValues[${scalar}]`,
          tangent: (direction: number) =>
            `inputTangents[(${scalar}) * ${numDirections}u + ${direction}u]`,
        };
      }
      case "uniform":
        return {
          value: `uniformValues[${binding.component}u]`,
          tangent: (direction: number) =>
            `uniformTangents[${binding.component * numDirections + direction}u]`,
        };
      default:
        throw new Error(
          `GPU JVP reduction does not support '${binding.kind}' parameter bindings`,
        );
    }
  });
  const pass = compileColumnarGpuJvpTapeKernel(compiled.tape, {
    inputWidth: width,
    outputWidth: width,
    uniformWidth,
    numDirections,
    variables,
    reductionWidth: width,
  });
  const device = requireGpuDevice();
  let valueScratch: GPUBuffer[] = [];
  let tangentScratch: GPUBuffer[] = [];
  let valueScratchBytes = 0;
  let tangentScratchBytes = 0;

  const ensureScratch = (count: number): void => {
    const rows = Math.ceil(count / 2);
    const requiredValues = Math.max(rows * width * 4, 4);
    const requiredTangents = Math.max(rows * width * numDirections * 4, 4);
    if (requiredValues > valueScratchBytes) {
      for (const buffer of valueScratch) buffer.destroy();
      valueScratchBytes = requiredValues;
      valueScratch = Array.from({ length: 2 }, () =>
        device.createBuffer({
          size: requiredValues,
          usage: GPUBufferUsage.STORAGE,
        }),
      );
    }
    if (requiredTangents > tangentScratchBytes) {
      for (const buffer of tangentScratch) buffer.destroy();
      tangentScratchBytes = requiredTangents;
      tangentScratch = Array.from({ length: 2 }, () =>
        device.createBuffer({
          size: requiredTangents,
          usage: GPUBufferUsage.STORAGE,
        }),
      );
    }
  };

  return {
    width,
    uniformWidth,
    numDirections,
    dispatchCount(count): number {
      return Math.ceil(Math.log2(Math.max(count, 1)));
    },
    encode(
      encoder,
      inputValues,
      inputTangents,
      uniformValues,
      uniformTangents,
      outputValues,
      outputTangents,
      count,
    ): void {
      if (!Number.isInteger(count) || count < 2) {
        throw new Error(
          `GPU JVP reduction encode requires count >= 2, got ${count}`,
        );
      }
      ensureScratch(count);
      let level = 0;
      let levelCount = count;
      let levelValues = inputValues;
      let levelTangents = inputTangents;
      while (levelCount > 1) {
        const outputCount = Math.ceil(levelCount / 2);
        const finalLevel = outputCount === 1;
        const nextValues: DeviceBufferSlice = finalLevel
          ? outputValues
          : {
              buffer: valueScratch[level % 2]!,
              offset: 0,
              byteLength: outputCount * width * 4,
            };
        const nextTangents: DeviceBufferSlice = finalLevel
          ? outputTangents
          : {
              buffer: tangentScratch[level % 2]!,
              offset: 0,
              byteLength: outputCount * width * numDirections * 4,
            };
        pass.encode(
          encoder,
          levelValues,
          levelTangents,
          uniformValues,
          uniformTangents,
          nextValues,
          nextTangents,
          levelCount,
        );
        levelValues = nextValues;
        levelTangents = nextTangents;
        levelCount = outputCount;
        level++;
      }
    },
    dispose(): void {
      pass.destroy();
      for (const buffer of valueScratch) buffer.destroy();
      for (const buffer of tangentScratch) buffer.destroy();
      valueScratch = [];
      tangentScratch = [];
    },
  };
}
