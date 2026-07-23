import type { CompiledTape } from "../../../tape";
import type { DeviceBufferSlice } from "../gpu-util";
import { requireGpuDevice } from "../gpu-util";
import { GPU_INTERP_DUAL_OP_CASES } from "./dual-op-cases";

const WORKGROUP_SIZE = 256;

export interface GpuInterpJvpDeviceKernel {
  readonly numVars: number;
  readonly numRoots: number;
  readonly numDirections: number;
  encode(
    encoder: GPUCommandEncoder,
    inputValues: DeviceBufferSlice,
    inputTangents: DeviceBufferSlice,
    outputValues: DeviceBufferSlice,
    outputTangents: DeviceBufferSlice,
    numPoints: number,
  ): void;
  destroy(): void;
}

const SHADER = /* wgsl */ `
struct Params {
  numNodes: u32,
  numRoots: u32,
  numPoints: u32,
  numVars: u32,
  numDirections: u32,
  argAOffset: u32,
  argBOffset: u32,
  literalOffset: u32,
  rootOffset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tapeData: array<u32>;
@group(0) @binding(2) var<storage, read> inputValues: array<f32>;
@group(0) @binding(3) var<storage, read> inputTangents: array<f32>;
@group(0) @binding(4) var<storage, read_write> nodeValues: array<f32>;
@group(0) @binding(5) var<storage, read_write> nodeTangents: array<f32>;
@group(0) @binding(6) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(7) var<storage, read_write> outputTangents: array<f32>;

const DIV_ZERO: f32 = 1e38;
fn cbrt_f32(x: f32) -> f32 { return sign(x) * pow(abs(x), 1.0 / 3.0); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= params.numPoints) { return; }
  let np = params.numPoints;
  let nd = params.numDirections;

  for (var i = 0u; i < params.numNodes; i = i + 1u) {
    let op = tapeData[i];
    let a = tapeData[params.argAOffset + i];
    let b = tapeData[params.argBOffset + i];
    let scalar = i * np + tid;
    let aScalar = a * np + tid;
    let bScalar = b * np + tid;
    let va = nodeValues[aScalar];
    let vb = nodeValues[bScalar];
    var r = 0.0;
    var fa = 0.0;
    var fb = 0.0;

    switch op {
      case 0u { r = bitcast<f32>(tapeData[params.literalOffset + a]); }
      case 1u { r = inputValues[tid * params.numVars + a]; }
${GPU_INTERP_DUAL_OP_CASES}
      default {}
    }
    nodeValues[scalar] = r;
    for (var d = 0u; d < nd; d = d + 1u) {
      let tangentIndex = scalar * nd + d;
      if (op == 0u) { nodeTangents[tangentIndex] = 0.0; }
      else if (op == 1u) { nodeTangents[tangentIndex] = inputTangents[(tid * params.numVars + a) * nd + d]; }
      else { nodeTangents[tangentIndex] = fa * nodeTangents[aScalar * nd + d] + fb * nodeTangents[bScalar * nd + d]; }
    }
  }

  for (var root = 0u; root < params.numRoots; root = root + 1u) {
    let node = tapeData[params.rootOffset + root];
    let source = node * np + tid;
    let outputIndex = tid * params.numRoots + root;
    outputValues[outputIndex] = nodeValues[source];
    for (var d = 0u; d < params.numDirections; d = d + 1u) {
      outputTangents[outputIndex * params.numDirections + d] = nodeTangents[source * params.numDirections + d];
    }
  }
}
`;

interface SharedPipeline {
  readonly layout: GPUBindGroupLayout;
  readonly pipeline: GPUComputePipeline;
}

const pipelines = new WeakMap<GPUDevice, SharedPipeline>();

function sharedPipeline(device: GPUDevice): SharedPipeline {
  const existing = pipelines.get(device);
  if (existing) return existing;
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      ...Array.from({ length: 7 }, (_, i) => ({
        binding: i + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: i >= 3 ? ("storage" as const) : ("read-only-storage" as const),
        },
      })),
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: {
      module: device.createShaderModule({ code: SHADER }),
      entryPoint: "main",
    },
  });
  const result = { layout, pipeline };
  pipelines.set(device, result);
  return result;
}

function uploadMetadata(device: GPUDevice, tape: CompiledTape): GPUBuffer {
  const n = tape.opcodes.length;
  const literalOffset = n * 3;
  const rootOffset = literalOffset + tape.literals.length;
  const data = new Uint32Array(rootOffset + tape.rootIndices.length);
  for (let i = 0; i < n; i++) {
    data[i] = tape.opcodes[i]!;
    data[n + i] = tape.argA[i]!;
    data[n * 2 + i] = tape.argB[i]!;
  }
  const literalBits = new Uint32Array(new Float32Array(tape.literals).buffer);
  data.set(literalBits, literalOffset);
  data.set(tape.rootIndices, rootOffset);
  const buffer = device.createBuffer({
    size: Math.max(data.byteLength, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export function compileGpuInterpJvpDeviceKernel(
  tape: CompiledTape,
  numDirections: number,
  maxPoints: number,
): GpuInterpJvpDeviceKernel {
  if (!Number.isInteger(numDirections) || numDirections < 0)
    throw new Error(
      `invalid GPU interpreter JVP direction count: ${numDirections}`,
    );
  if (!Number.isInteger(maxPoints) || maxPoints < 1)
    throw new Error(`invalid GPU interpreter JVP point limit: ${maxPoints}`);
  const device = requireGpuDevice();
  if (device.limits.maxStorageBuffersPerShaderStage < 7)
    throw new Error("GPU interpreter JVP requires 7 storage buffer bindings");
  const n = tape.opcodes.length;
  const nodeValueBytes = n * maxPoints * 4;
  const nodeTangentBytes = n * maxPoints * numDirections * 4;
  for (const bytes of [nodeValueBytes, nodeTangentBytes]) {
    if (bytes > device.limits.maxStorageBufferBindingSize)
      throw new Error(
        `GPU interpreter JVP scratch requires ${bytes} bytes; storage binding limit is ${device.limits.maxStorageBufferBindingSize}`,
      );
  }
  const maxDispatchPoints =
    device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
  if (maxPoints > maxDispatchPoints)
    throw new Error(
      `GPU interpreter JVP point limit ${maxPoints} exceeds dispatch limit ${maxDispatchPoints}`,
    );
  const metadataBytes =
    (n * 3 + tape.literals.length + tape.rootIndices.length) * 4;
  if (metadataBytes > device.limits.maxStorageBufferBindingSize)
    throw new Error(
      `GPU interpreter JVP metadata requires ${metadataBytes} bytes; storage binding limit is ${device.limits.maxStorageBufferBindingSize}`,
    );
  const metadata = uploadMetadata(device, tape);
  const parameterBuffers = new Map<number, GPUBuffer>();
  const parameterBuffer = (numPoints: number): GPUBuffer => {
    let buffer = parameterBuffers.get(numPoints);
    if (!buffer) {
      buffer = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      parameterBuffers.set(numPoints, buffer);
    }
    return buffer;
  };
  const nodeValues = device.createBuffer({
    size: Math.max(nodeValueBytes, 4),
    usage: GPUBufferUsage.STORAGE,
  });
  const nodeTangents = device.createBuffer({
    size: Math.max(nodeTangentBytes, 4),
    usage: GPUBufferUsage.STORAGE,
  });
  const emptyRead = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const emptyWrite = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const { layout, pipeline } = sharedPipeline(device);
  const slice = (value: DeviceBufferSlice, writable = false) =>
    value.byteLength > 0
      ? value
      : { buffer: writable ? emptyWrite : emptyRead, offset: 0, byteLength: 4 };
  return {
    numVars: tape.numVars,
    numRoots: tape.rootIndices.length,
    numDirections,
    encode(
      encoder,
      inputValues,
      inputTangents,
      outputValues,
      outputTangents,
      numPoints,
    ) {
      if (
        !Number.isInteger(numPoints) ||
        numPoints < 0 ||
        numPoints > maxPoints
      )
        throw new Error(
          `GPU interpreter JVP point count ${numPoints} exceeds compiled limit ${maxPoints}`,
        );
      if (numPoints === 0) return;
      const expected = [
        numPoints * tape.numVars * 4,
        numPoints * tape.numVars * numDirections * 4,
        numPoints * tape.rootIndices.length * 4,
        numPoints * tape.rootIndices.length * numDirections * 4,
      ];
      const supplied = [
        inputValues.byteLength,
        inputTangents.byteLength,
        outputValues.byteLength,
        outputTangents.byteLength,
      ];
      for (let i = 0; i < expected.length; i++) {
        if (expected[i]! > device.limits.maxStorageBufferBindingSize)
          throw new Error(
            `GPU interpreter JVP buffer ${i} requires ${expected[i]} bytes; storage binding limit is ${device.limits.maxStorageBufferBindingSize}`,
          );
        if (supplied[i]! < expected[i]!)
          throw new Error(
            `GPU interpreter JVP buffer ${i} has ${supplied[i]} bytes; expected ${expected[i]}`,
          );
      }
      const rootOffset = n * 3 + tape.literals.length;
      const params = parameterBuffer(numPoints);
      device.queue.writeBuffer(
        params,
        0,
        new Uint32Array([
          n,
          tape.rootIndices.length,
          numPoints,
          tape.numVars,
          numDirections,
          n,
          n * 2,
          n * 3,
          rootOffset,
          0,
          0,
          0,
        ]),
      );
      const slices = [
        slice(inputValues),
        slice(inputTangents),
        {
          buffer: nodeValues,
          offset: 0,
          byteLength: Math.max(nodeValueBytes, 4),
        },
        {
          buffer: nodeTangents,
          offset: 0,
          byteLength: Math.max(nodeTangentBytes, 4),
        },
        slice(outputValues, true),
        slice(outputTangents, true),
      ];
      const group = device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: metadata } },
          ...slices.map((s, i) => ({
            binding: i + 2,
            resource: {
              buffer: s.buffer,
              offset: s.offset,
              size: Math.max(s.byteLength, 4),
            },
          })),
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(numPoints / WORKGROUP_SIZE));
      pass.end();
    },
    destroy() {
      metadata.destroy();
      for (const buffer of parameterBuffers.values()) buffer.destroy();
      parameterBuffers.clear();
      nodeValues.destroy();
      nodeTangents.destroy();
      emptyRead.destroy();
      emptyWrite.destroy();
    },
  };
}
