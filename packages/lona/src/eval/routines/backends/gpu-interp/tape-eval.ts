/**
 * WebGPU compute shader tape evaluator for NumNode DAGs.
 *
 * Unlike the CPU evaluators (which process one point at a time), the GPU
 * version evaluates an entire batch of points in parallel. Each GPU thread
 * runs the full tape interpreter for one (x, y, z) point.
 *
 * Requires the `webgpu` npm package (Node.js WebGPU via Dawn).
 *
 * Notes:
 * - Uses f32 (not f64) — results have reduced precision vs CPU evaluators
 * - Batch-oriented API: callers provide packed batch inputs directly
 * - The values scratch buffer is the memory bottleneck:
 *   batchSize * numNodes * 4 bytes (e.g. 1024 * 17800 * 4 = 73MB)
 */
import type { CompiledTape } from "../../../tape";
import type { VarName } from "../../../../core/tree";
import {
  mapGpuReadbackBuffer,
  requireGpuDevice,
  readGpuBuffer,
} from "../gpu-util";
import {
  compileGpuInterpJvpDeviceKernel,
  type GpuInterpJvpDeviceKernel,
} from "./jvp-device";
export { destroyGpu } from "../gpu-util";

// ---------------------------------------------------------------------------
// WGSL compute shader
// ---------------------------------------------------------------------------

const SHADER_SOURCE = /* wgsl */ `
struct Params {
  numNodes: u32,
  numRoots: u32,
  numPoints: u32,
  numVars: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> opcodes: array<u32>;
@group(0) @binding(2) var<storage, read> argA: array<u32>;
@group(0) @binding(3) var<storage, read> argB: array<u32>;
@group(0) @binding(4) var<storage, read> literals: array<f32>;
@group(0) @binding(5) var<storage, read> varData: array<f32>;
@group(0) @binding(6) var<storage, read_write> vals: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
@group(0) @binding(8) var<storage, read> rootIndices: array<u32>;

const DIV_ZERO: f32 = 1e38;

fn cbrt_f32(x: f32) -> f32 {
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= params.numPoints) { return; }

  let np = params.numPoints;
  let nv = params.numVars;

  for (var i = 0u; i < params.numNodes; i = i + 1u) {
    let op = opcodes[i];
    let a  = argA[i];
    let b  = argB[i];
    let va = vals[a * np + tid];
    let vb = vals[b * np + tid];

    var r: f32 = 0.0;

    switch op {
      case  0u { r = literals[a]; }
      case  1u { r = varData[tid * nv + a]; }
      case 10u { r = sqrt(va); }
      case 11u { r = cbrt_f32(va); }
      case 12u { r = cos(va); }
      case 13u { r = acos(va); }
      case 14u { r = asin(va); }
      case 15u { r = tan(va); }
      case 16u { r = atan(va); }
      case 17u { r = log(va); }
      case 18u { r = exp(va); }
      case 19u { r = abs(va); }
      case 20u { r = -va; }
      case 21u { r = sin(va); }
      case 22u { r = sign(va); }
      case 23u { r = select(0.0, 1.0, va == 0.0); }
      case 24u { r = tanh(va); }
      case 25u { r = log(1.0 + va); }
      case 26u { r = va; }
      case 40u { r = va + vb; }
      case 41u { r = va - vb; }
      case 42u { r = va * vb; }
      case 43u { r = select(DIV_ZERO, va / vb, vb != 0.0); }
      case 44u { r = va % vb; }
      case 45u { r = atan2(va, vb); }
      case 46u { r = min(va, vb); }
      case 47u { r = max(va, vb); }
      case 48u { r = sign(va - vb); }
      case 49u { r = select(vb, va, va == 0.0); }
      case 50u { r = select(va, vb, va == 0.0); }
      case 51u { r = va; }
      case 52u { r = va; }
      default  { }
    }

    vals[i * np + tid] = r;
  }

  for (var r = 0u; r < params.numRoots; r = r + 1u) {
    output[tid * params.numRoots + r] = vals[rootIndices[r] * np + tid];
  }
}
`;

const WORKGROUP_SIZE = 256;

// ---------------------------------------------------------------------------
// Cached pipelines — initialized once by initGpuTapeEval()
// ---------------------------------------------------------------------------

let evalPipeline: GPUComputePipeline | null = null;
let evalBindGroupLayout: GPUBindGroupLayout | null = null;

function ensureEvalPipeline(device: GPUDevice) {
  if (evalPipeline) return;
  const shaderModule = device.createShaderModule({ code: SHADER_SOURCE });
  evalBindGroupLayout = device.createBindGroupLayout({
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
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 7,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 8,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
  evalPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [evalBindGroupLayout],
    }),
    compute: { module: shaderModule, entryPoint: "main" },
  });
}

/**
 * Eagerly initialize the GPU device and compile the shared value pipeline.
 * Device JVP pipelines are initialized lazily by their lower-level compiler.
 * Returns false if the GPU is unavailable.
 */
export function initGpuTapeEval(): boolean {
  try {
    const device = requireGpuDevice();
    ensureEvalPipeline(device);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// GpuTapeEval
// ---------------------------------------------------------------------------

export interface GpuTapeEval {
  /**
   * Evaluate the tape for a batch of points.
   * @param varData Flat Float32Array with numPoints * varSlots.length entries,
   *   in varSlots order: [v0_p0, v1_p0, ..., v0_p1, v1_p1, ...]
   * @param numPoints Number of points in the batch
   * @returns Float32Array with numPoints * numRoots results, interleaved
   *   per point: [r0_p0, r1_p0, ..., r0_p1, r1_p1, ...]
   */
  evalBatch(varData: Float32Array, numPoints: number): Promise<Float32Array>;

  /** All variable names in slot order */
  readonly varSlots: VarName[];

  destroy(): void;
}

interface TapeGpuBuffers {
  opcodes: GPUBuffer;
  argA: GPUBuffer;
  argB: GPUBuffer;
  literals: GPUBuffer;
  rootIndices: GPUBuffer;
  numNodes: number;
  numRoots: number;
  /** First root index — used by the single-root grad path. */
  rootIndex: number;
  varSlots: VarName[];
}

function uploadTapeBuffers(
  device: GPUDevice,
  tape: CompiledTape,
): TapeGpuBuffers {
  const N = tape.opcodes.length;

  // Upload opcodes as u32
  const opcU32 = new Uint32Array(N);
  for (let i = 0; i < N; i++) opcU32[i] = tape.opcodes[i]!;
  const opcBuf = createStorageBuffer(device, opcU32.buffer as ArrayBuffer);

  // Upload args as exact-length u32 arrays. Compiled tapes may be views over
  // growable backing storage, so copying `.buffer` directly could include
  // unused capacity beyond the tape's logical length.
  const argAU32 = new Uint32Array(N);
  argAU32.set(tape.argA);
  const argABuf = createStorageBuffer(device, argAU32.buffer as ArrayBuffer);

  const argBU32 = new Uint32Array(N);
  argBU32.set(tape.argB);
  const argBBuf = createStorageBuffer(device, argBU32.buffer as ArrayBuffer);

  // Upload literals as f32 (downcast from f64)
  const litF32 = new Float32Array(tape.literals.length);
  for (let i = 0; i < tape.literals.length; i++) litF32[i] = tape.literals[i]!;
  const litBuf = createStorageBuffer(
    device,
    litF32.length > 0 ? (litF32.buffer as ArrayBuffer) : new ArrayBuffer(4), // WGSL requires non-zero buffer
  );

  // Upload root indices as u32
  const rootU32 = new Uint32Array(tape.rootIndices);
  const rootBuf = createStorageBuffer(device, rootU32.buffer as ArrayBuffer);

  return {
    opcodes: opcBuf,
    argA: argABuf,
    argB: argBBuf,
    literals: litBuf,
    rootIndices: rootBuf,
    numNodes: N,
    numRoots: tape.rootIndices.length,
    rootIndex: tape.rootIndices[0],
    varSlots: tape.varSlots,
  };
}

function createStorageBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  const buf = device.createBuffer({
    size: Math.max(data.byteLength, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
  buf.unmap();
  return buf;
}

function createGpuTapeEval(
  device: GPUDevice,
  tapeBuffers: TapeGpuBuffers,
): GpuTapeEval {
  const { numNodes, numRoots, varSlots } = tapeBuffers;

  // Compute max batch size based on device limits
  const maxStorageBuf = device.limits.maxStorageBufferBindingSize;
  const valuesPerPoint = numNodes * 4; // f32 per node
  const maxByBuffer = Math.floor(maxStorageBuf / valuesPerPoint);
  const maxByDispatch =
    device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
  const maxBatch = Math.min(maxByBuffer, maxByDispatch);

  ensureEvalPipeline(device);
  const pipeline = evalPipeline!;
  const bindGroupLayout = evalBindGroupLayout!;

  const numVars = varSlots.length;

  // Pre-allocated buffers for the current batch size
  let allocatedBatch = 0;
  let paramsBuf: GPUBuffer | null = null;
  let varDataBuf: GPUBuffer | null = null;
  let valuesBuf: GPUBuffer | null = null;
  let outputBuf: GPUBuffer | null = null;
  let stagingBuf: GPUBuffer | null = null;
  let bindGroup: GPUBindGroup | null = null;

  function ensureBatchBuffers(batchSize: number) {
    if (batchSize === allocatedBatch) return;
    paramsBuf?.destroy();
    varDataBuf?.destroy();
    valuesBuf?.destroy();
    outputBuf?.destroy();
    stagingBuf?.destroy();

    allocatedBatch = batchSize;

    paramsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    varDataBuf = device.createBuffer({
      size: Math.max(batchSize * numVars * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    valuesBuf = device.createBuffer({
      size: numNodes * batchSize * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    outputBuf = device.createBuffer({
      size: batchSize * numRoots * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    stagingBuf = device.createBuffer({
      size: batchSize * numRoots * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: tapeBuffers.opcodes } },
        { binding: 2, resource: { buffer: tapeBuffers.argA } },
        { binding: 3, resource: { buffer: tapeBuffers.argB } },
        { binding: 4, resource: { buffer: tapeBuffers.literals } },
        { binding: 5, resource: { buffer: varDataBuf } },
        { binding: 6, resource: { buffer: valuesBuf } },
        { binding: 7, resource: { buffer: outputBuf } },
        { binding: 8, resource: { buffer: tapeBuffers.rootIndices } },
      ],
    });
  }

  async function evalBatch(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array> {
    if (numPoints === 0) return new Float32Array(0);
    if (numPoints > maxBatch) {
      const results = new Float32Array(numPoints * numRoots);
      for (let offset = 0; offset < numPoints; offset += maxBatch) {
        const end = Math.min(offset + maxBatch, numPoints);
        const batchData = varData.subarray(offset * numVars, end * numVars);
        const batchResults = await evalBatch(batchData, end - offset);
        results.set(batchResults, offset * numRoots);
      }
      return results;
    }

    ensureBatchBuffers(numPoints);

    // Upload params
    const paramsData = new Uint32Array([
      numNodes,
      numRoots,
      numPoints,
      numVars,
    ]);
    device.queue.writeBuffer(paramsBuf!, 0, paramsData);

    // Upload varData
    device.queue.writeBuffer(
      varDataBuf!,
      0,
      varData.buffer as ArrayBuffer,
      varData.byteOffset,
      varData.byteLength,
    );

    // Dispatch compute
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup!);
    pass.dispatchWorkgroups(Math.ceil(numPoints / WORKGROUP_SIZE));
    pass.end();

    device.queue.submit([encoder.finish()]);

    return readGpuBuffer(
      device,
      outputBuf!,
      stagingBuf!,
      numPoints * numRoots * 4,
    );
  }

  function destroy() {
    paramsBuf?.destroy();
    varDataBuf?.destroy();
    valuesBuf?.destroy();
    outputBuf?.destroy();
    stagingBuf?.destroy();
    tapeBuffers.opcodes.destroy();
    tapeBuffers.argA.destroy();
    tapeBuffers.argB.destroy();
    tapeBuffers.literals.destroy();
    tapeBuffers.rootIndices.destroy();
  }

  return { evalBatch, varSlots, destroy };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function compileGpuTapeFromTape(tape: CompiledTape): GpuTapeEval {
  const device = requireGpuDevice();
  const tapeBuffers = uploadTapeBuffers(device, tape);
  return createGpuTapeEval(device, tapeBuffers);
}

// ---------------------------------------------------------------------------
// GPU forward-mode autodiff — host adapter over the device JVP interpreter
// ---------------------------------------------------------------------------

export interface GpuTapeGradEval {
  /** Output is [value, derivative0, ...] for each point. */
  evalBatch(varData: Float32Array, numPoints: number): Promise<Float32Array>;
  readonly varSlots: VarName[];
  readonly numDiffVars: number;
  destroy(): void;
}

/**
 * Wrap the caller-owned-buffer JVP primitive for the ordinary batch-grad API.
 * Identity seeds are packed on the host; primal and tangent outputs are copied
 * in one submission and interleaved after readback.
 */
export function compileGpuTapeGradFromTape(
  tape: CompiledTape,
  diffVarSlotIndices: number[],
): GpuTapeGradEval {
  if (tape.rootIndices.length !== 1) {
    throw new Error("GPU interpreter gradient evaluation requires one root");
  }
  for (const slot of diffVarSlotIndices) {
    if (!Number.isInteger(slot) || slot < 0 || slot >= tape.numVars) {
      throw new Error(`invalid GPU interpreter differentiation slot: ${slot}`);
    }
  }

  const device = requireGpuDevice();
  const numVars = tape.numVars;
  const numDiffVars = diffVarSlotIndices.length;
  const bindingLimit = device.limits.maxStorageBufferBindingSize;
  const dispatchLimit =
    device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
  const maxPointsFor = (scalarsPerPoint: number): number =>
    scalarsPerPoint === 0
      ? dispatchLimit
      : Math.floor(bindingLimit / (scalarsPerPoint * 4));
  const maxBatch = Math.min(
    dispatchLimit,
    maxPointsFor(tape.opcodes.length),
    maxPointsFor(tape.opcodes.length * numDiffVars),
    maxPointsFor(numVars),
    maxPointsFor(numVars * numDiffVars),
    maxPointsFor(1 + numDiffVars),
  );
  if (maxBatch < 1) {
    throw new Error("GPU interpreter gradient tape does not fit device limits");
  }

  let allocatedBatch = 0;
  let jvp: GpuInterpJvpDeviceKernel | null = null;
  let inputValues: GPUBuffer | null = null;
  let inputTangents: GPUBuffer | null = null;
  let outputValues: GPUBuffer | null = null;
  let outputTangents: GPUBuffer | null = null;
  let valueStaging: GPUBuffer | null = null;
  let tangentStaging: GPUBuffer | null = null;
  let identitySeeds: Float32Array | null = null;

  const destroyBatch = (): void => {
    jvp?.destroy();
    inputValues?.destroy();
    inputTangents?.destroy();
    outputValues?.destroy();
    outputTangents?.destroy();
    valueStaging?.destroy();
    tangentStaging?.destroy();
    jvp = null;
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
    jvp = compileGpuInterpJvpDeviceKernel(tape, numDiffVars, numPoints);
    const createStorage = (bytes: number, usage = GPUBufferUsage.STORAGE) =>
      device.createBuffer({ size: Math.max(bytes, 4), usage });
    try {
      inputValues = createStorage(
        numPoints * numVars * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      inputTangents = createStorage(
        numPoints * numVars * numDiffVars * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      outputValues = createStorage(
        numPoints * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      outputTangents = createStorage(
        numPoints * numDiffVars * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      valueStaging = createStorage(
        numPoints * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      );
      tangentStaging = createStorage(
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
      throw new Error(
        `invalid GPU interpreter gradient point count: ${numPoints}`,
      );
    }
    if (varData.length < numPoints * numVars) {
      throw new Error(
        `GPU interpreter gradient received ${varData.length} values; expected ${numPoints * numVars}`,
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
    if (identitySeeds!.byteLength > 0)
      device.queue.writeBuffer(inputTangents!, 0, identitySeeds!);

    const encoder = device.createCommandEncoder();
    jvp!.encode(
      encoder,
      { buffer: inputValues!, offset: 0, byteLength: numPoints * numVars * 4 },
      {
        buffer: inputTangents!,
        offset: 0,
        byteLength: identitySeeds!.byteLength,
      },
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
    destroy: destroyBatch,
  };
}
