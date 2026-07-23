import type {
  CompiledTape,
  DeviceBufferSlice,
  TapeJvpVariableWgsl,
} from "lona/internal";
import { emitTapeJvpWgsl, requireGpuDevice } from "lona/internal";

const WORKGROUP_SIZE = 256;
const MAX_SHADER_WORK = 4_000;

export interface CompiledColumnarGpuJvpTapeKernel {
  readonly dispatchesPerPass: 1;
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
  destroy(): void;
}

export interface ColumnarGpuJvpTapeLayout {
  readonly inputWidth: number;
  readonly outputWidth: number;
  readonly uniformWidth: number;
  readonly numDirections: number;
  readonly variables: readonly TapeJvpVariableWgsl[];
  /** When present, each invocation combines two rows into one. */
  readonly reductionWidth?: number;
}

function shader(tape: CompiledTape, layout: ColumnarGpuJvpTapeLayout): string {
  const estimatedWork = tape.opcodes.length * (1 + layout.numDirections);
  if (estimatedWork > MAX_SHADER_WORK) {
    throw new Error(
      `columnar GPU JVP kernel requires ${estimatedWork} node-directions; single-shader limit is ${MAX_SHADER_WORK}`,
    );
  }
  const emitted = emitTapeJvpWgsl(tape, layout.numDirections, (slot) => {
    const binding = layout.variables[slot];
    if (!binding) {
      throw new Error(`columnar GPU JVP has no binding for variable ${slot}`);
    }
    return binding;
  });
  const oddRowGuard = layout.reductionWidth
    ? `  if (tid * 2u + 1u >= params.inputCount) {
${Array.from({ length: layout.reductionWidth }, (_, component) => {
  const scalar = `tid * ${layout.reductionWidth}u + ${component}u`;
  const inputScalar = `(tid * 2u) * ${layout.reductionWidth}u + ${component}u`;
  return `    outputValues[${scalar}] = inputValues[${inputScalar}];
${Array.from(
  { length: layout.numDirections },
  (_, direction) =>
    `    outputTangents[(${scalar}) * ${layout.numDirections}u + ${direction}u] = inputTangents[(${inputScalar}) * ${layout.numDirections}u + ${direction}u];`,
).join("\n")}`;
}).join("\n")}
    return;
  }`
    : "";
  const outputs = emitted.roots
    .map((root, component) => {
      const scalar = `tid * ${layout.outputWidth}u + ${component}u`;
      return `  outputValues[${scalar}] = ${root};
${emitted.tangentRoots[component]!.map(
  (tangent, direction) =>
    `  outputTangents[(${scalar}) * ${layout.numDirections}u + ${direction}u] = ${tangent};`,
).join("\n")}`;
    })
    .join("\n");

  return `struct Params {
  outputCount: u32,
  inputCount: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputValues: array<f32>;
@group(0) @binding(2) var<storage, read> inputTangents: array<f32>;
@group(0) @binding(3) var<storage, read> uniformValues: array<f32>;
@group(0) @binding(4) var<storage, read> uniformTangents: array<f32>;
@group(0) @binding(5) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(6) var<storage, read_write> outputTangents: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= params.outputCount) { return; }
${oddRowGuard}
${emitted.body}
${outputs}
}
`;
}

export function compileColumnarGpuJvpTapeKernel(
  tape: CompiledTape,
  layout: ColumnarGpuJvpTapeLayout,
): CompiledColumnarGpuJvpTapeKernel {
  if (tape.rootIndices.length !== layout.outputWidth) {
    throw new Error(
      `columnar GPU JVP has ${tape.rootIndices.length} outputs; expected ${layout.outputWidth}`,
    );
  }
  if (layout.variables.length !== tape.varSlots.length) {
    throw new Error(
      `columnar GPU JVP expected ${tape.varSlots.length} variable bindings, got ${layout.variables.length}`,
    );
  }
  const device = requireGpuDevice();
  if (device.limits.maxStorageBuffersPerShaderStage < 6) {
    throw new Error(
      `columnar GPU JVP requires 6 storage bindings; device supports ${device.limits.maxStorageBuffersPerShaderStage}`,
    );
  }
  const bindGroupLayout = device.createBindGroupLayout({
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
  const module = device.createShaderModule({ code: shader(tape, layout) });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module, entryPoint: "main" },
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
  const emptyReadBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const emptyWriteBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const nonempty = (
    slice: DeviceBufferSlice | null,
    writable = false,
  ): DeviceBufferSlice =>
    slice && slice.byteLength > 0
      ? slice
      : {
          buffer: writable ? emptyWriteBuffer : emptyReadBuffer,
          offset: 0,
          byteLength: 4,
        };

  return {
    dispatchesPerPass: 1,
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
      if (!Number.isInteger(count) || count < 0) {
        throw new Error(`invalid columnar GPU JVP count: ${count}`);
      }
      if (count === 0) return;
      const outputCount = layout.reductionWidth ? Math.ceil(count / 2) : count;
      const inputScalars = count * layout.inputWidth;
      const outputScalars = outputCount * layout.outputWidth;
      const expected = {
        inputValues: inputScalars * 4,
        inputTangents: inputScalars * layout.numDirections * 4,
        uniformValues: layout.uniformWidth * 4,
        uniformTangents: layout.uniformWidth * layout.numDirections * 4,
        outputValues: outputScalars * 4,
        outputTangents: outputScalars * layout.numDirections * 4,
      };
      const supplied = {
        inputValues: inputValues.byteLength,
        inputTangents: inputTangents.byteLength,
        uniformValues: uniformValues?.byteLength ?? 0,
        uniformTangents: uniformTangents?.byteLength ?? 0,
        outputValues: outputValues.byteLength,
        outputTangents: outputTangents.byteLength,
      };
      for (const key of Object.keys(expected) as Array<keyof typeof expected>) {
        if (supplied[key] < expected[key]) {
          throw new Error(
            `columnar GPU JVP ${key} has ${supplied[key]} bytes; expected ${expected[key]}`,
          );
        }
        if (expected[key] > device.limits.maxStorageBufferBindingSize) {
          throw new Error(
            `columnar GPU JVP ${key} exceeds the ${device.limits.maxStorageBufferBindingSize} byte storage binding limit`,
          );
        }
      }
      const dispatchLimit =
        device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
      if (outputCount > dispatchLimit) {
        throw new Error(
          `columnar GPU JVP output count ${outputCount} exceeds dispatch limit ${dispatchLimit}`,
        );
      }
      const params = parameterBuffer(count);
      device.queue.writeBuffer(
        params,
        0,
        new Uint32Array([outputCount, count, 0, 0]),
      );
      const slices = [
        nonempty(inputValues),
        nonempty(inputTangents),
        nonempty(uniformValues),
        nonempty(uniformTangents),
        nonempty(outputValues, true),
        nonempty(outputTangents, true),
      ];
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
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
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(outputCount / WORKGROUP_SIZE));
      pass.end();
    },
    destroy(): void {
      for (const buffer of parameterBuffers.values()) buffer.destroy();
      parameterBuffers.clear();
      emptyReadBuffer.destroy();
      emptyWriteBuffer.destroy();
    },
  };
}
