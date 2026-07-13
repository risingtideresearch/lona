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
import { requireGpuDevice, readGpuBuffer } from "../gpu-util";
export { destroyGpu } from "../gpu-util";

// ---------------------------------------------------------------------------
// WGSL compute shader
// ---------------------------------------------------------------------------

const SHADER_SOURCE = /* wgsl */ `
struct Params {
  numNodes: u32,
  rootIndex: u32,
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

  output[tid] = vals[params.rootIndex * np + tid];
}
`;

const WORKGROUP_SIZE = 256;

// ---------------------------------------------------------------------------
// Cached pipelines — initialized once by initGpuTapeEval()
// ---------------------------------------------------------------------------

let evalPipeline: GPUComputePipeline | null = null;
let evalBindGroupLayout: GPUBindGroupLayout | null = null;

let gradPipeline: GPUComputePipeline | null = null;
let gradBindGroupLayout: GPUBindGroupLayout | null = null;

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
    ],
  });
  evalPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [evalBindGroupLayout],
    }),
    compute: { module: shaderModule, entryPoint: "main" },
  });
}

function ensureGradPipeline(device: GPUDevice) {
  if (gradPipeline) return;
  const shaderModule = device.createShaderModule({ code: GRAD_SHADER_SOURCE });
  gradBindGroupLayout = device.createBindGroupLayout({
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
  gradPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [gradBindGroupLayout],
    }),
    compute: { module: shaderModule, entryPoint: "main" },
  });
}

/**
 * Eagerly initialize the GPU device, compile the value-only and grad
 * shaders, and build their pipelines. Subsequent calls to
 * compileGpuTapeFromSerialized / compileGpuTapeGrad skip these costs.
 * Returns false if the GPU is unavailable.
 */
export function initGpuTapeEval(): boolean {
  try {
    const device = requireGpuDevice();
    ensureEvalPipeline(device);
    ensureGradPipeline(device);
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
   * @returns Float32Array with one result per point
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
  numNodes: number;
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

  // Upload argA as u32 (all values are non-negative)
  const argAU32 = new Uint32Array(tape.argA.buffer.slice(0));
  const argABuf = createStorageBuffer(device, argAU32.buffer as ArrayBuffer);

  // Upload argB as u32
  const argBU32 = new Uint32Array(tape.argB.buffer.slice(0));
  const argBBuf = createStorageBuffer(device, argBU32.buffer as ArrayBuffer);

  // Upload literals as f32 (downcast from f64)
  const litF32 = new Float32Array(tape.literals.length);
  for (let i = 0; i < tape.literals.length; i++) litF32[i] = tape.literals[i]!;
  const litBuf = createStorageBuffer(
    device,
    litF32.length > 0 ? (litF32.buffer as ArrayBuffer) : new ArrayBuffer(4), // WGSL requires non-zero buffer
  );

  return {
    opcodes: opcBuf,
    argA: argABuf,
    argB: argBBuf,
    literals: litBuf,
    numNodes: N,
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
  const { numNodes, rootIndex, varSlots } = tapeBuffers;

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
      size: batchSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    stagingBuf = device.createBuffer({
      size: batchSize * 4,
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
      ],
    });
  }

  async function evalBatch(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array> {
    if (numPoints === 0) return new Float32Array(0);
    if (numPoints > maxBatch) {
      const results = new Float32Array(numPoints);
      for (let offset = 0; offset < numPoints; offset += maxBatch) {
        const end = Math.min(offset + maxBatch, numPoints);
        const batchData = varData.subarray(offset * numVars, end * numVars);
        const batchResults = await evalBatch(batchData, end - offset);
        results.set(batchResults, offset);
      }
      return results;
    }

    ensureBatchBuffers(numPoints);

    // Upload params
    const paramsData = new Uint32Array([
      numNodes,
      rootIndex,
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

    return readGpuBuffer(device, outputBuf!, stagingBuf!, numPoints * 4);
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
// GPU forward-mode autodiff tape evaluator — all-partials-per-thread
// ---------------------------------------------------------------------------
//
// Each GPU thread computes value + ALL D partial derivatives in one pass.
// The shader propagates [value, ∂/∂v₀, ∂/∂v₁, …, ∂/∂vD-1] per node per
// thread (stride = 1 + D). Only N threads are needed (one per point),
// not N×D. The derivative factors (fa, fb) are computed once per node and
// applied to all D derivative slots in a loop.

const GRAD_SHADER_SOURCE = /* wgsl */ `
struct Params {
  numNodes: u32,
  rootIndex: u32,
  numPoints: u32,
  numVars: u32,
  numDiffVars: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> opcodes: array<u32>;
@group(0) @binding(2) var<storage, read> argA: array<u32>;
@group(0) @binding(3) var<storage, read> argB: array<u32>;
@group(0) @binding(4) var<storage, read> literals: array<f32>;
@group(0) @binding(5) var<storage, read> varData: array<f32>;
@group(0) @binding(6) var<storage, read_write> vals: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
@group(0) @binding(8) var<storage, read> diffVarSlots: array<u32>;

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
  let nd = params.numDiffVars;
  let stride = 1u + nd;

  for (var i = 0u; i < params.numNodes; i = i + 1u) {
    let op = opcodes[i];
    let a  = argA[i];
    let b  = argB[i];
    let base  = (i * np + tid) * stride;
    let aBase = (a * np + tid) * stride;
    let bBase = (b * np + tid) * stride;

    let va = vals[aBase];
    let vb = vals[bBase];

    var r: f32 = 0.0;
    var fa: f32 = 0.0;
    var fb: f32 = 0.0;

    switch op {
      case  0u { r = literals[a]; }
      case  1u { r = varData[tid * nv + a]; }
      case 10u { r = sqrt(va); fa = select(0.5 / r, 0.0, r == 0.0); }
      case 11u { r = cbrt_f32(va); fa = select(1.0 / (3.0 * r * r), 0.0, r == 0.0); }
      case 12u { r = cos(va); fa = -sin(va); }
      case 13u { r = acos(va); fa = -1.0 / sqrt(1.0 - va * va); }
      case 14u { r = asin(va); fa = 1.0 / sqrt(1.0 - va * va); }
      case 15u { r = tan(va); let c = cos(va); fa = 1.0 / (c * c); }
      case 16u { r = atan(va); fa = 1.0 / (1.0 + va * va); }
      case 17u { r = log(va); fa = 1.0 / va; }
      case 18u { r = exp(va); fa = r; }
      case 19u { r = abs(va); fa = select(select(-1.0, 0.0, va == 0.0), 1.0, va > 0.0); }
      case 20u { r = -va; fa = -1.0; }
      case 21u { r = sin(va); fa = cos(va); }
      case 22u { r = sign(va); }
      case 23u { r = select(0.0, 1.0, va == 0.0); }
      case 24u { r = tanh(va); fa = 1.0 - r * r; }
      case 25u { r = log(1.0 + va); fa = 1.0 / (1.0 + va); }
      case 26u { r = va; fa = 1.0; }
      case 40u { r = va + vb; fa = 1.0; fb = 1.0; }
      case 41u { r = va - vb; fa = 1.0; fb = -1.0; }
      case 42u { r = va * vb; fa = vb; fb = va; }
      case 43u {
        if (vb == 0.0) { r = DIV_ZERO; }
        else { r = va / vb; fa = 1.0 / vb; fb = -va / (vb * vb); }
      }
      case 44u { r = va % vb; fa = 1.0; }
      case 45u {
        r = atan2(va, vb);
        let denom = va * va + vb * vb;
        if (denom != 0.0) { fa = vb / denom; fb = -va / denom; }
      }
      case 46u { let useA = va <= vb; r = select(vb, va, useA); fa = select(0.0, 1.0, useA); fb = select(1.0, 0.0, useA); }
      case 47u { let useA = va >= vb; r = select(vb, va, useA); fa = select(0.0, 1.0, useA); fb = select(1.0, 0.0, useA); }
      case 48u { r = sign(va - vb); }
      case 49u { let z = va == 0.0; r = select(va, vb, z); fa = select(1.0, 0.0, z); fb = select(0.0, 1.0, z); }
      case 50u { let nz = va != 0.0; r = select(va, vb, nz); fa = select(1.0, 0.0, nz); fb = select(0.0, 1.0, nz); }
      case 51u { r = va; fa = 1.0; }
      case 52u { r = va; fa = 1.0; }
      default  { }
    }

    vals[base] = r;

    if (op == 0u) {
      // LIT: all derivatives are 0
      for (var d = 0u; d < nd; d = d + 1u) {
        vals[base + 1u + d] = 0.0;
      }
    } else if (op == 1u) {
      // VAR: seed derivatives
      for (var d = 0u; d < nd; d = d + 1u) {
        vals[base + 1u + d] = select(0.0, 1.0, a == diffVarSlots[d]);
      }
    } else {
      // All other ops: dr = fa * da + fb * db
      for (var d = 0u; d < nd; d = d + 1u) {
        vals[base + 1u + d] = fa * vals[aBase + 1u + d] + fb * vals[bBase + 1u + d];
      }
    }
  }

  // Output: [val, dv0, dv1, ...] per point
  let rootBase = (params.rootIndex * np + tid) * stride;
  for (var k = 0u; k < stride; k = k + 1u) {
    output[tid * stride + k] = vals[rootBase + k];
  }
}
`;

// ---------------------------------------------------------------------------
// GpuTapeGradEval — all-partials-per-thread
// ---------------------------------------------------------------------------

export interface GpuTapeGradEval {
  /**
   * Evaluate value + all partial derivatives for a batch of points.
   * @param varData  Flat f32: numPoints × numVars entries in varSlots order
   * @param numPoints  Number of points in the batch
   * @returns Float32Array of numPoints × (1 + numDiffVars):
   *   [val₀, ∂v₀₀, ∂v₀₁, …, val₁, ∂v₁₀, …]
   */
  evalBatch(varData: Float32Array, numPoints: number): Promise<Float32Array>;

  readonly varSlots: VarName[];
  readonly numDiffVars: number;
  destroy(): void;
}

function createGpuTapeGradEval(
  device: GPUDevice,
  tapeBuffers: TapeGpuBuffers,
  diffVarSlotIndices: number[],
): GpuTapeGradEval {
  const { numNodes, rootIndex, varSlots } = tapeBuffers;
  const numVars = varSlots.length;
  const numDiffVars = diffVarSlotIndices.length;
  const stride = 1 + numDiffVars;

  ensureGradPipeline(device);
  const pipeline = gradPipeline!;
  const bindGroupLayout = gradBindGroupLayout!;

  // Upload diffVarSlots once (small buffer, immutable)
  const diffVarSlotsU32 = new Uint32Array(diffVarSlotIndices);
  const diffVarSlotsBuf = createStorageBuffer(
    device,
    diffVarSlotsU32.buffer as ArrayBuffer,
  );

  const maxStorageBuf = device.limits.maxStorageBufferBindingSize;
  const valsPerPoint = numNodes * stride * 4;
  const maxByBuffer = Math.floor(maxStorageBuf / valsPerPoint);
  const maxByDispatch =
    device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
  const maxBatch = Math.min(maxByBuffer, maxByDispatch);

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
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    varDataBuf = device.createBuffer({
      size: Math.max(batchSize * numVars * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    valuesBuf = device.createBuffer({
      size: numNodes * batchSize * stride * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    const outputSize = batchSize * stride * 4;
    outputBuf = device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    stagingBuf = device.createBuffer({
      size: outputSize,
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
        { binding: 8, resource: { buffer: diffVarSlotsBuf } },
      ],
    });
  }

  async function evalBatch(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array> {
    if (numPoints === 0) return new Float32Array(0);
    if (numPoints > maxBatch) {
      const results = new Float32Array(numPoints * stride);
      for (let offset = 0; offset < numPoints; offset += maxBatch) {
        const end = Math.min(offset + maxBatch, numPoints);
        const batchData = varData.subarray(offset * numVars, end * numVars);
        const batchResults = await evalBatch(batchData, end - offset);
        results.set(batchResults, offset * stride);
      }
      return results;
    }

    ensureBatchBuffers(numPoints);

    const paramsData = new Uint32Array([
      numNodes,
      rootIndex,
      numPoints,
      numVars,
      numDiffVars,
    ]);
    device.queue.writeBuffer(paramsBuf!, 0, paramsData);
    device.queue.writeBuffer(
      varDataBuf!,
      0,
      varData.buffer as ArrayBuffer,
      varData.byteOffset,
      varData.byteLength,
    );

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
      numPoints * stride * 4,
    );
  }

  function destroy() {
    paramsBuf?.destroy();
    varDataBuf?.destroy();
    valuesBuf?.destroy();
    outputBuf?.destroy();
    stagingBuf?.destroy();
    diffVarSlotsBuf.destroy();
    tapeBuffers.opcodes.destroy();
    tapeBuffers.argA.destroy();
    tapeBuffers.argB.destroy();
    tapeBuffers.literals.destroy();
  }

  return { evalBatch, varSlots, numDiffVars, destroy };
}

// ---------------------------------------------------------------------------
// Public API — GPU forward-mode autodiff
// ---------------------------------------------------------------------------

/**
 * Compile a GPU tape grad evaluator. All D partial derivatives are
 * computed per thread — only N threads needed, no CPU-side expansion.
 * Output is N × (1 + D) f32s: [val, ∂v₀, ∂v₁, …] per point.
 */
export function compileGpuTapeGradFromTape(
  tape: CompiledTape,
  diffVarSlotIndices: number[],
): GpuTapeGradEval {
  const device = requireGpuDevice();
  const tapeBuffers = uploadTapeBuffers(device, tape);
  return createGpuTapeGradEval(device, tapeBuffers, diffVarSlotIndices);
}

// destroyGpu re-exported from gpu-util.ts
