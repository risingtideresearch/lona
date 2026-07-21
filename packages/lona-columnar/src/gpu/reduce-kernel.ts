import type { NumNode } from "lona/internal";
import type { DeviceBufferSlice } from "lona/internal";
import { requireGpuDevice } from "lona/internal";
import { compileStructuredTape } from "../compile-tape";
import type { BuiltInReduction, StructuredParamInput } from "../ir";
import { compileStructuredGpuTapeKernel } from "./tape-kernel";

const WORKGROUP_SIZE = 256;

export interface CompiledGpuReduceKernel {
  readonly operation?: BuiltInReduction;
  readonly width: number;
  readonly uniformWidth: number;
  dispatchCount(count: number): number;
  encode(
    encoder: GPUCommandEncoder,
    input: DeviceBufferSlice,
    uniforms: DeviceBufferSlice | null,
    output: DeviceBufferSlice,
    count: number,
  ): void;
  dispose(): void;
}

function combineExpression(operation: BuiltInReduction): string {
  switch (operation) {
    case "sum":
      return "left + right";
    case "product":
      return "left * right";
    case "min":
      return "lonaMin(left, right)";
    case "max":
      return "lonaMax(left, right)";
  }
}

function shader(operation: BuiltInReduction): string {
  // WGSL forbids a NaN float *literal* (validators reject a bitcast of a
  // literal u32 that const-folds to NaN), so the canonical-NaN bit pattern is
  // threaded through the Params uniform from JS instead of baked into the
  // shader source — see `ensureResources`, which writes it alongside
  // `count`/`width`.
  return `struct Params {
  count: u32,
  width: u32,
  nanBits: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputData: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputData: array<f32>;

fn canonicalNaN() -> f32 {
  return bitcast<f32>(params.nanBits);
}

// \`a != a\` rather than a WGSL \`isNan\` builtin: current WGSL has none
// (removed from the spec), and this comparison is well-defined for IEEE-754
// NaN under the default (non-fast-math) numeric mode these compute shaders use.
fn lonaMin(a: f32, b: f32) -> f32 {
  if (a != a || b != b) { return canonicalNaN(); }
  if (a == b) {
    if (a == 0.0) {
      return bitcast<f32>(bitcast<u32>(a) | bitcast<u32>(b));
    }
    return a;
  }
  return select(b, a, a < b);
}

fn lonaMax(a: f32, b: f32) -> f32 {
  if (a != a || b != b) { return canonicalNaN(); }
  if (a == b) {
    if (a == 0.0) {
      return bitcast<f32>(bitcast<u32>(a) & bitcast<u32>(b));
    }
    return a;
  }
  return select(b, a, a > b);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let outputCount = (params.count + 1u) / 2u;
  let scalar = gid.x;
  if (scalar >= outputCount * params.width) { return; }
  let row = scalar / params.width;
  let component = scalar % params.width;
  let leftIndex = (row * 2u) * params.width + component;
  let left = inputData[leftIndex];
  let rightRow = row * 2u + 1u;
  if (rightRow >= params.count) {
    outputData[scalar] = left;
    return;
  }
  let right = inputData[rightRow * params.width + component];
  outputData[scalar] = ${combineExpression(operation)};
}
`;
}

export function compileGpuReduceKernel(
  operation: BuiltInReduction,
  width: number,
): CompiledGpuReduceKernel {
  const device = requireGpuDevice();
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const module = device.createShaderModule({ code: shader(operation) });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module, entryPoint: "main" },
  });

  let parameterBuffers: GPUBuffer[] = [];
  let scratchBuffers: GPUBuffer[] = [];
  let scratchBytes = 0;

  const ensureResources = (count: number): void => {
    const levels = Math.ceil(Math.log2(count));
    if (parameterBuffers.length !== levels) {
      for (const buffer of parameterBuffers) buffer.destroy();
      parameterBuffers = Array.from({ length: levels }, () =>
        device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      );
    }
    const requiredScratch = Math.max(Math.ceil(count / 2) * width * 4, 4);
    if (levels > 1 && requiredScratch > scratchBytes) {
      for (const buffer of scratchBuffers) buffer.destroy();
      scratchBytes = requiredScratch;
      scratchBuffers = Array.from({ length: 2 }, () =>
        device.createBuffer({
          size: requiredScratch,
          usage: GPUBufferUsage.STORAGE,
        }),
      );
    }
  };

  return {
    operation,
    width,
    uniformWidth: 0,
    dispatchCount(count): number {
      return Math.ceil(Math.log2(Math.max(count, 1)));
    },
    encode(encoder, input, _uniforms, output, count): void {
      if (!Number.isInteger(count) || count < 2) {
        throw new Error(
          `GPU reduction encode requires count >= 2, got ${count}`,
        );
      }
      if (input.byteLength < count * width * 4) {
        throw new Error("GPU reduction input buffer is too small");
      }
      if (output.byteLength < width * 4) {
        throw new Error("GPU reduction output buffer is too small");
      }
      if (count * width * 4 > device.limits.maxStorageBufferBindingSize) {
        throw new Error(
          `GPU reduction input exceeds the ${device.limits.maxStorageBufferBindingSize} byte storage binding limit`,
        );
      }
      const firstPassScalars = Math.ceil(count / 2) * width;
      if (
        Math.ceil(firstPassScalars / WORKGROUP_SIZE) >
        device.limits.maxComputeWorkgroupsPerDimension
      ) {
        throw new Error(
          "GPU reduction dispatch exceeds the device workgroup limit",
        );
      }
      ensureResources(count);

      let level = 0;
      let levelCount = count;
      let levelInput = input;
      while (levelCount > 1) {
        const outputCount = Math.ceil(levelCount / 2);
        const finalLevel = outputCount === 1;
        const levelOutput: DeviceBufferSlice = finalLevel
          ? output
          : {
              buffer: scratchBuffers[level % 2]!,
              offset: 0,
              byteLength: outputCount * width * 4,
            };
        device.queue.writeBuffer(
          parameterBuffers[level]!,
          0,
          new Uint32Array([levelCount, width, 0x7fc00000, 0]),
        );
        const bindGroup = device.createBindGroup({
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: parameterBuffers[level]! } },
            {
              binding: 1,
              resource: {
                buffer: levelInput.buffer,
                offset: levelInput.offset,
                size: levelInput.byteLength,
              },
            },
            {
              binding: 2,
              resource: {
                buffer: levelOutput.buffer,
                offset: levelOutput.offset,
                size: levelOutput.byteLength,
              },
            },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          Math.ceil((outputCount * width) / WORKGROUP_SIZE),
        );
        pass.end();
        levelInput = levelOutput;
        levelCount = outputCount;
        level++;
      }
    },
    dispose(): void {
      for (const buffer of parameterBuffers) buffer.destroy();
      for (const buffer of scratchBuffers) buffer.destroy();
      parameterBuffers = [];
      scratchBuffers = [];
    },
  };
}

export function compileGpuTracedReduceKernel(
  roots: readonly NumNode[],
  inputs: readonly StructuredParamInput[],
  width: number,
): CompiledGpuReduceKernel {
  const compiled = compileStructuredTape(roots, inputs);
  const uniformWidth = compiled.tapeInputs.reduce(
    (result, { binding }) =>
      binding.kind === "uniform"
        ? Math.max(result, binding.component + 1)
        : result,
    0,
  );
  const variables = compiled.tapeInputs.map(({ binding }) => {
    switch (binding.kind) {
      case "reduce-left":
        return `inputData[(tid * 2u) * ${width}u + ${binding.component}u]`;
      case "reduce-right":
        return `inputData[(tid * 2u + 1u) * ${width}u + ${binding.component}u]`;
      case "uniform":
        return `uniformData[${binding.component}u]`;
      default:
        throw new Error(
          `GPU reduction does not support '${binding.kind}' parameter bindings`,
        );
    }
  });
  const passKernel = compileStructuredGpuTapeKernel(compiled.tape, {
    inputWidth: width,
    outputWidth: width,
    uniformWidth,
    variables,
    reductionWidth: width,
  });
  const device = requireGpuDevice();
  let scratchBuffers: GPUBuffer[] = [];
  let scratchBytes = 0;

  const ensureScratch = (count: number): void => {
    const required = Math.max(Math.ceil(count / 2) * width * 4, 4);
    if (required > scratchBytes) {
      for (const buffer of scratchBuffers) buffer.destroy();
      scratchBytes = required;
      scratchBuffers = Array.from({ length: 2 }, () =>
        device.createBuffer({
          size: required,
          usage: GPUBufferUsage.STORAGE,
        }),
      );
    }
  };

  return {
    width,
    uniformWidth,
    dispatchCount(count): number {
      return (
        Math.ceil(Math.log2(Math.max(count, 1))) * passKernel.dispatchesPerPass
      );
    },
    encode(encoder, input, uniforms, output, count): void {
      if (!Number.isInteger(count) || count < 2) {
        throw new Error(
          `GPU traced reduction encode requires count >= 2, got ${count}`,
        );
      }
      ensureScratch(count);
      let level = 0;
      let levelCount = count;
      let levelInput = input;
      while (levelCount > 1) {
        const outputCount = Math.ceil(levelCount / 2);
        const finalLevel = outputCount === 1;
        const levelOutput: DeviceBufferSlice = finalLevel
          ? output
          : {
              buffer: scratchBuffers[level % 2]!,
              offset: 0,
              byteLength: outputCount * width * 4,
            };
        passKernel.encode(
          encoder,
          levelInput,
          uniforms,
          levelOutput,
          levelCount,
        );
        levelInput = levelOutput;
        levelCount = outputCount;
        level++;
      }
    },
    dispose(): void {
      passKernel.destroy();
      for (const buffer of scratchBuffers) buffer.destroy();
      scratchBuffers = [];
    },
  };
}
