import type { DeviceBufferSlice, NumNode, VarName } from "lona/internal";
import { compileGpuInterpJvpDeviceKernel, compileTape } from "lona/internal";
import { compileColumnarGpuJvpTapeKernel } from "./jvp-tape-kernel";

export interface CompiledGpuJvpSourceKernel {
  readonly varSlots: readonly VarName[];
  readonly numVars: number;
  readonly numRoots: number;
  readonly numDirections: number;
  readonly dispatchCount: number;
  encode(
    encoder: GPUCommandEncoder,
    inputValues: DeviceBufferSlice,
    inputTangents: DeviceBufferSlice,
    outputValues: DeviceBufferSlice,
    outputTangents: DeviceBufferSlice,
  ): void;
  dispose(): void;
}

export function compileGpuInterpJvpSourceKernel(
  roots: readonly NumNode[],
  numDirections: number,
): CompiledGpuJvpSourceKernel | null {
  if (roots.length === 0) return null;
  const tape = compileTape([...roots]);
  if (!tape) throw new Error("failed to compile GPU autodiff source tape");
  const gpu = compileGpuInterpJvpDeviceKernel(tape, numDirections, 1);
  return {
    varSlots: Object.freeze(tape.varSlots.slice(0, tape.numVars)),
    numVars: tape.numVars,
    numRoots: roots.length,
    numDirections,
    dispatchCount: 1,
    encode(
      encoder,
      inputValues,
      inputTangents,
      outputValues,
      outputTangents,
    ): void {
      gpu.encode(
        encoder,
        inputValues,
        inputTangents,
        outputValues,
        outputTangents,
        1,
      );
    },
    dispose: gpu.destroy,
  };
}

export function compileGpuJvpSourceKernel(
  roots: readonly NumNode[],
  numDirections: number,
): CompiledGpuJvpSourceKernel | null {
  if (roots.length === 0) return null;
  const tape = compileTape([...roots]);
  if (!tape) throw new Error("failed to compile GPU autodiff source tape");
  const variables = tape.varSlots.map((_, slot) =>
    slot < tape.numVars
      ? {
          value: `inputValues[${slot}u]`,
          tangent: (direction: number) =>
            `inputTangents[${slot * numDirections + direction}u]`,
        }
      : { value: "0.0", tangent: () => "0.0" },
  );
  const gpu = compileColumnarGpuJvpTapeKernel(tape, {
    inputWidth: tape.numVars,
    outputWidth: roots.length,
    uniformWidth: 0,
    numDirections,
    variables,
  });
  return {
    varSlots: Object.freeze(tape.varSlots.slice(0, tape.numVars)),
    numVars: tape.numVars,
    numRoots: roots.length,
    numDirections,
    dispatchCount: gpu.dispatchesPerPass,
    encode(
      encoder,
      inputValues,
      inputTangents,
      outputValues,
      outputTangents,
    ): void {
      gpu.encode(
        encoder,
        inputValues,
        inputTangents,
        null,
        null,
        outputValues,
        outputTangents,
        1,
      );
    },
    dispose: gpu.destroy,
  };
}
