import type { DeviceBufferSlice } from "lona/internal";
import { requireGpuDevice } from "lona/internal";

const WORKGROUP_SIZE = 256;

export interface JvpPackVariable {
  readonly value: string;
  tangent(direction: number): string;
}

export interface CompiledGpuJvpPackKernel {
  encode(
    encoder: GPUCommandEncoder,
    inputValues: DeviceBufferSlice,
    inputTangents: DeviceBufferSlice,
    uniformValues: DeviceBufferSlice | null,
    uniformTangents: DeviceBufferSlice | null,
    packedValues: DeviceBufferSlice,
    packedTangents: DeviceBufferSlice,
    count: number,
  ): void;
  destroy(): void;
}

export function compileGpuJvpPackKernel(
  variables: readonly JvpPackVariable[],
  numDirections: number,
): CompiledGpuJvpPackKernel {
  const device = requireGpuDevice();
  const width = variables.length;
  const writes = variables
    .map((variable, slot) => {
      const scalar = `tid * ${width}u + ${slot}u`;
      return `  packedValues[${scalar}] = ${variable.value};\n${Array.from(
        { length: numDirections },
        (_, direction) =>
          `  packedTangents[(${scalar}) * ${numDirections}u + ${direction}u] = ${variable.tangent(direction)};`,
      ).join("\n")}`;
    })
    .join("\n");
  const shader = `struct Params { count: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
@group(0) @binding(2) var<storage, read> inputTangents: array<f32>;
@group(0) @binding(3) var<storage, read> uniformValues: array<f32>;
@group(0) @binding(4) var<storage, read> uniformTangents: array<f32>;
@group(0) @binding(5) var<storage, read_write> packedValues: array<f32>;
@group(0) @binding(6) var<storage, read_write> packedTangents: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= params.count) { return; }
${writes}
}`;
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      ...Array.from({ length: 6 }, (_, index) => ({
        binding: index + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type:
            index >= 4 ? ("storage" as const) : ("read-only-storage" as const),
        },
      })),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: {
      module: device.createShaderModule({ code: shader }),
      entryPoint: "main",
    },
  });
  const parameterBuffers = new Map<number, GPUBuffer>();
  const parameterBuffer = (count: number): GPUBuffer => {
    let buffer = parameterBuffers.get(count);
    if (!buffer) {
      buffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      parameterBuffers.set(count, buffer);
    }
    return buffer;
  };
  const emptyRead = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const emptyWriteValues = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const emptyWriteTangents = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const nonemptyRead = (slice: DeviceBufferSlice | null): DeviceBufferSlice =>
    slice && slice.byteLength
      ? slice
      : { buffer: emptyRead, offset: 0, byteLength: 4 };
  const nonemptyWrite = (
    slice: DeviceBufferSlice,
    empty: GPUBuffer,
  ): DeviceBufferSlice =>
    slice.byteLength ? slice : { buffer: empty, offset: 0, byteLength: 4 };
  return {
    encode(
      encoder,
      inputValues,
      inputTangents,
      uniformValues,
      uniformTangents,
      packedValues,
      packedTangents,
      count,
    ) {
      if (count === 0) return;
      const params = parameterBuffer(count);
      device.queue.writeBuffer(params, 0, new Uint32Array([count, 0, 0, 0]));
      const slices = [
        nonemptyRead(inputValues),
        nonemptyRead(inputTangents),
        nonemptyRead(uniformValues),
        nonemptyRead(uniformTangents),
        nonemptyWrite(packedValues, emptyWriteValues),
        nonemptyWrite(packedTangents, emptyWriteTangents),
      ];
      const group = device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          ...slices.map((slice, index) => ({
            binding: index + 1,
            resource: {
              buffer: slice.buffer,
              offset: slice.offset,
              size: Math.max(slice.byteLength, 4),
            },
          })),
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP_SIZE));
      pass.end();
    },
    destroy() {
      for (const buffer of parameterBuffers.values()) buffer.destroy();
      parameterBuffers.clear();
      emptyRead.destroy();
      emptyWriteValues.destroy();
      emptyWriteTangents.destroy();
    },
  };
}
