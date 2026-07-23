import type { NumNode } from "lona/internal";
import type { DeviceBufferSlice } from "lona/internal";
import { compileColumnarTape } from "../compile-tape";
import type { ColumnarParamBinding, ColumnarParamInput } from "../ir";
import { compileColumnarGpuTapeKernel } from "./tape-kernel";

export interface CompiledGpuMapKernel {
  readonly inputWidth: number;
  readonly uniformWidth: number;
  readonly outputWidth: number;
  readonly dispatchCount: number;
  encode(
    encoder: GPUCommandEncoder,
    input: DeviceBufferSlice,
    uniforms: DeviceBufferSlice | null,
    output: DeviceBufferSlice,
    count: number,
  ): void;
  dispose(): void;
}

function mapInputValue(
  binding: ColumnarParamBinding,
  source: readonly number[],
  uniforms: readonly number[],
  inputWidth: number,
  row: number,
): number {
  switch (binding.kind) {
    case "row": {
      const value = source[row * inputWidth + binding.component];
      if (value === undefined) {
        throw new Error(
          `GPU map row component ${binding.component} is missing`,
        );
      }
      return value;
    }
    case "uniform": {
      const value = uniforms[binding.component];
      if (value === undefined) {
        throw new Error(
          `GPU map uniform component ${binding.component} is missing`,
        );
      }
      return value;
    }
    case "index":
      return row;
    case "reduce-left":
    case "reduce-right":
    case "materialized":
      throw new Error(
        `GPU map kernel cannot bind columnar input '${binding.kind}'`,
      );
  }
}

/** Pack map inputs in the actual OP_VAR slot order of the compiled tape. */
export function packGpuMapInputs(
  tapeInputs: readonly ColumnarParamInput[],
  source: readonly number[],
  uniforms: readonly number[],
  inputWidth: number,
  count: number,
): Float32Array {
  if (source.length !== count * inputWidth) {
    throw new Error(
      `GPU map expected ${count * inputWidth} source values, got ${source.length}`,
    );
  }
  const packed = new Float32Array(count * tapeInputs.length);
  for (let row = 0; row < count; row++) {
    for (let slot = 0; slot < tapeInputs.length; slot++) {
      packed[row * tapeInputs.length + slot] = mapInputValue(
        tapeInputs[slot]!.binding,
        source,
        uniforms,
        inputWidth,
        row,
      );
    }
  }
  return packed;
}

export function compileGpuMapKernel(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
  inputWidth: number,
  outputWidth: number,
): CompiledGpuMapKernel {
  const compiled = compileColumnarTape(roots, inputs);
  for (const input of compiled.tapeInputs) {
    const kind = input.binding.kind;
    if (kind !== "row" && kind !== "uniform" && kind !== "index") {
      throw new Error(`GPU map does not support '${kind}' parameter bindings`);
    }
  }
  const uniformWidth = compiled.tapeInputs.reduce(
    (width, { binding }) =>
      binding.kind === "uniform"
        ? Math.max(width, binding.component + 1)
        : width,
    0,
  );
  const variables = compiled.tapeInputs.map(({ binding }) => {
    switch (binding.kind) {
      case "row":
        return `inputData[tid * ${inputWidth}u + ${binding.component}u]`;
      case "uniform":
        return `uniformData[${binding.component}u]`;
      case "index":
        return "f32(tid)";
      default:
        throw new Error(
          `GPU map does not support '${binding.kind}' parameter bindings`,
        );
    }
  });
  const gpu = compileColumnarGpuTapeKernel(compiled.tape, {
    inputWidth,
    outputWidth,
    uniformWidth,
    variables,
  });

  return {
    inputWidth,
    uniformWidth,
    outputWidth,
    dispatchCount: gpu.dispatchesPerPass,
    encode: gpu.encode,
    dispose(): void {
      gpu.destroy();
    },
  };
}
