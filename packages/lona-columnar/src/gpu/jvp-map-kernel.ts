import type { DeviceBufferSlice, NumNode } from "lona/internal";
import {
  compileGpuInterpJvpDeviceKernel,
  requireGpuDevice,
} from "lona/internal";
import { compileColumnarTape } from "../compile-tape";
import type { ColumnarParamInput } from "../ir";
import { compileGpuJvpPackKernel } from "./jvp-interp-pack-kernel";
import { compileColumnarGpuJvpTapeKernel } from "./jvp-tape-kernel";

export interface CompiledGpuJvpMapKernel {
  readonly inputWidth: number;
  readonly uniformWidth: number;
  readonly outputWidth: number;
  readonly numDirections: number;
  readonly dispatchCount: number;
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

export function compileGpuInterpJvpMapKernel(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
  inputWidth: number,
  outputWidth: number,
  numDirections: number,
  maxCount: number,
): CompiledGpuJvpMapKernel {
  const compiled = compileColumnarTape(roots, inputs);
  const uniformWidth = compiled.tapeInputs.reduce(
    (width, { binding }) =>
      binding.kind === "uniform"
        ? Math.max(width, binding.component + 1)
        : width,
    0,
  );
  const variables = compiled.tapeInputs.map(({ binding }) => {
    switch (binding.kind) {
      case "row": {
        const scalar = `tid * ${inputWidth}u + ${binding.component}u`;
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
      case "index":
        return { value: "f32(tid)", tangent: () => "0.0" };
      default:
        throw new Error(
          `GPU interpreter JVP map does not support '${binding.kind}' parameter bindings`,
        );
    }
  });
  const pack = compileGpuJvpPackKernel(variables, numDirections);
  const interp = compileGpuInterpJvpDeviceKernel(
    compiled.tape,
    numDirections,
    Math.max(maxCount, 1),
  );
  const device = requireGpuDevice();
  const packedValueBytes = maxCount * compiled.tape.numVars * 4;
  const packedTangentBytes = packedValueBytes * numDirections;
  const packedValues = device.createBuffer({
    size: Math.max(packedValueBytes, 4),
    usage: GPUBufferUsage.STORAGE,
  });
  const packedTangents = device.createBuffer({
    size: Math.max(packedTangentBytes, 4),
    usage: GPUBufferUsage.STORAGE,
  });
  return {
    inputWidth,
    uniformWidth,
    outputWidth,
    numDirections,
    dispatchCount: 2,
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
      if (!Number.isInteger(count) || count < 0 || count > maxCount) {
        throw new Error(
          `GPU interpreter JVP map count ${count} is outside [0, ${maxCount}]`,
        );
      }
      const valueSlice = {
        buffer: packedValues,
        offset: 0,
        byteLength: count * compiled.tape.numVars * 4,
      };
      const tangentSlice = {
        buffer: packedTangents,
        offset: 0,
        byteLength: count * compiled.tape.numVars * numDirections * 4,
      };
      pack.encode(
        encoder,
        inputValues,
        inputTangents,
        uniformValues,
        uniformTangents,
        valueSlice,
        tangentSlice,
        count,
      );
      interp.encode(
        encoder,
        valueSlice,
        tangentSlice,
        outputValues,
        outputTangents,
        count,
      );
    },
    dispose(): void {
      pack.destroy();
      interp.destroy();
      packedValues.destroy();
      packedTangents.destroy();
    },
  };
}

export function compileGpuJvpMapKernel(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
  inputWidth: number,
  outputWidth: number,
  numDirections: number,
): CompiledGpuJvpMapKernel {
  const compiled = compileColumnarTape(roots, inputs);
  const uniformWidth = compiled.tapeInputs.reduce(
    (width, { binding }) =>
      binding.kind === "uniform"
        ? Math.max(width, binding.component + 1)
        : width,
    0,
  );
  const variables = compiled.tapeInputs.map(({ binding }) => {
    switch (binding.kind) {
      case "row": {
        const scalar = `tid * ${inputWidth}u + ${binding.component}u`;
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
      case "index":
        return { value: "f32(tid)", tangent: () => "0.0" };
      default:
        throw new Error(
          `GPU JVP map does not support '${binding.kind}' parameter bindings`,
        );
    }
  });
  const gpu = compileColumnarGpuJvpTapeKernel(compiled.tape, {
    inputWidth,
    outputWidth,
    uniformWidth,
    numDirections,
    variables,
  });
  return {
    inputWidth,
    uniformWidth,
    outputWidth,
    numDirections,
    dispatchCount: gpu.dispatchesPerPass,
    encode: gpu.encode,
    dispose: gpu.destroy,
  };
}
