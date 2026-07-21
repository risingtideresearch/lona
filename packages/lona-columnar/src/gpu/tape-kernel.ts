import type { CompiledTape } from "lona/internal";
import { emitTapeWgsl } from "lona/internal";
import type { DeviceBufferSlice } from "lona/internal";
import { requireGpuDevice } from "lona/internal";

const WORKGROUP_SIZE = 256;
const MAX_SHADER_NODES = 4_000;

export interface CompiledColumnarGpuTapeKernel {
  readonly dispatchesPerPass: 1;
  encode(
    encoder: GPUCommandEncoder,
    input: DeviceBufferSlice,
    uniforms: DeviceBufferSlice | null,
    output: DeviceBufferSlice,
    count: number,
  ): void;
  destroy(): void;
}

interface ColumnarGpuTapeLayout {
  readonly inputWidth: number;
  readonly outputWidth: number;
  readonly uniformWidth: number;
  readonly variables: readonly string[];
  /** When present, each invocation combines two rows into one. */
  readonly reductionWidth?: number;
}

function shader(tape: CompiledTape, layout: ColumnarGpuTapeLayout): string {
  if (tape.opcodes.length > MAX_SHADER_NODES) {
    throw new Error(
      `columnar GPU kernel has ${tape.opcodes.length} nodes; the single-shader limit is ${MAX_SHADER_NODES}`,
    );
  }
  const emitted = emitTapeWgsl(tape, (slot) => {
    const expression = layout.variables[slot];
    if (expression === undefined) {
      throw new Error(
        `columnar GPU kernel has no binding for variable ${slot}`,
      );
    }
    return expression;
  });
  const oddRowGuard = layout.reductionWidth
    ? `  if (tid * 2u + 1u >= params.inputCount) {
${Array.from(
  { length: layout.reductionWidth },
  (_, component) =>
    `    outputData[tid * ${layout.reductionWidth}u + ${component}u] = inputData[(tid * 2u) * ${layout.reductionWidth}u + ${component}u];`,
).join("\n")}
    return;
  }`
    : "";
  const outputs = emitted.roots
    .map(
      (root, component) =>
        `  outputData[tid * ${layout.outputWidth}u + ${component}u] = ${root};`,
    )
    .join("\n");

  return `struct Params {
  outputCount: u32,
  inputCount: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputData: array<f32>;
@group(0) @binding(2) var<storage, read> uniformData: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputData: array<f32>;

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

/** Compile one columnar map or pairwise-reduction pass. */
export function compileColumnarGpuTapeKernel(
  tape: CompiledTape,
  layout: ColumnarGpuTapeLayout,
): CompiledColumnarGpuTapeKernel {
  if (tape.rootIndices.length !== layout.outputWidth) {
    throw new Error(
      `columnar GPU kernel has ${tape.rootIndices.length} outputs; expected ${layout.outputWidth}`,
    );
  }
  if (layout.variables.length !== tape.varSlots.length) {
    throw new Error(
      `columnar GPU kernel expected ${tape.varSlots.length} variable bindings, got ${layout.variables.length}`,
    );
  }
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
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
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
  const emptyUniformBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });

  return {
    dispatchesPerPass: 1,
    encode(encoder, input, uniforms, output, count): void {
      if (!Number.isInteger(count) || count < 0) {
        throw new Error(`invalid columnar GPU kernel count: ${count}`);
      }
      if (count === 0) return;
      const outputCount = layout.reductionWidth ? Math.ceil(count / 2) : count;
      const expectedInputBytes = count * layout.inputWidth * 4;
      const expectedOutputBytes = outputCount * layout.outputWidth * 4;
      if (input.byteLength < expectedInputBytes) {
        throw new Error(
          `columnar GPU kernel input has ${input.byteLength} bytes; expected ${expectedInputBytes}`,
        );
      }
      if (output.byteLength < expectedOutputBytes) {
        throw new Error(
          `columnar GPU kernel output has ${output.byteLength} bytes; expected ${expectedOutputBytes}`,
        );
      }
      if (
        layout.uniformWidth > 0 &&
        (!uniforms || uniforms.byteLength < layout.uniformWidth * 4)
      ) {
        throw new Error(
          `columnar GPU kernel requires ${layout.uniformWidth * 4} uniform bytes`,
        );
      }
      if (
        expectedInputBytes > device.limits.maxStorageBufferBindingSize ||
        expectedOutputBytes > device.limits.maxStorageBufferBindingSize
      ) {
        throw new Error(
          `columnar GPU kernel exceeds the ${device.limits.maxStorageBufferBindingSize} byte storage binding limit`,
        );
      }
      const dispatchLimit =
        device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
      if (outputCount > dispatchLimit) {
        throw new Error(
          `columnar GPU kernel output count ${outputCount} exceeds the dispatch limit ${dispatchLimit}`,
        );
      }

      const params = parameterBuffer(count);
      device.queue.writeBuffer(
        params,
        0,
        new Uint32Array([outputCount, count, 0, 0]),
      );
      const uniformSlice = uniforms ?? {
        buffer: emptyUniformBuffer,
        offset: 0,
        byteLength: 4,
      };
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          {
            binding: 1,
            resource: {
              buffer: input.buffer,
              offset: input.offset,
              size: Math.max(expectedInputBytes, 4),
            },
          },
          {
            binding: 2,
            resource: {
              buffer: uniformSlice.buffer,
              offset: uniformSlice.offset,
              size: Math.max(uniformSlice.byteLength, 4),
            },
          },
          {
            binding: 3,
            resource: {
              buffer: output.buffer,
              offset: output.offset,
              size: Math.max(expectedOutputBytes, 4),
            },
          },
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
      emptyUniformBuffer.destroy();
    },
  };
}
