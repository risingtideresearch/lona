/**
 * WebGPU codegen evaluator.
 *
 * Large tapes are split into separately compiled shader pipelines. Values
 * consumed by a later chunk are written to a storage buffer; values whose
 * lifetime is contained in one chunk remain shader-local `let` bindings.
 * All chunk dispatches are encoded in order and submitted together.
 *
 * Uses f32 — results have reduced precision vs CPU f64 evaluators.
 */
import type { VarName } from "../../../../core/tree";
import { type CompiledTape, OP_ADD, OP_LIT, OP_VAR } from "../../../tape";
import { requireGpuDevice, readGpuBuffer } from "../gpu-util";
import {
  binaryTapeOpWgsl,
  formatWgslF32,
  unaryTapeOpWgsl,
  wgslValueRef,
} from "./emit-tape-wgsl";

const WORKGROUP_SIZE = 256;
/** Conservative enough to keep Tint/Dawn away from monolithic shader stalls. */
const DEFAULT_MAX_CHUNK_NODES = 4_000;

interface ChunkPlan {
  start: number;
  end: number;
}

interface GpuPlan {
  chunks: ChunkPlan[];
  escapeSlot: Int32Array;
  numEscapeSlots: number;
}

/** Plan contiguous chunks and identify values read by a later chunk. */
export function planGpuCodegenChunks(
  tape: CompiledTape,
  maxChunkNodes = DEFAULT_MAX_CHUNK_NODES,
): GpuPlan {
  if (!Number.isInteger(maxChunkNodes) || maxChunkNodes < 1) {
    throw new Error(`Invalid GPU codegen chunk node limit: ${maxChunkNodes}`);
  }

  const N = tape.opcodes.length;
  const chunks: ChunkPlan[] = [];
  if (N === 0) chunks.push({ start: 0, end: 0 });
  for (let start = 0; start < N; start += maxChunkNodes) {
    chunks.push({ start, end: Math.min(start + maxChunkNodes, N) });
  }

  const lastReader = new Int32Array(N).fill(-1);
  for (let i = 0; i < N; i++) {
    const op = tape.opcodes[i]!;
    if (op === OP_LIT || op === OP_VAR) continue;
    lastReader[tape.argA[i]!] = i;
    if (op >= OP_ADD) lastReader[tape.argB[i]!] = i;
  }

  const escapeSlot = new Int32Array(N).fill(-1);
  let numEscapeSlots = 0;
  for (const chunk of chunks) {
    for (let i = chunk.start; i < chunk.end; i++) {
      if (lastReader[i]! >= chunk.end) escapeSlot[i] = numEscapeSlots++;
    }
  }
  return { chunks, escapeSlot, numEscapeSlots };
}

function generateChunkShader(
  tape: CompiledTape,
  chunk: ChunkPlan,
  escapeSlot: Int32Array,
): string {
  const lines: string[] = [];
  const rootsAtNode = new Map<number, number[]>();
  for (let r = 0; r < tape.rootIndices.length; r++) {
    const node = tape.rootIndices[r]!;
    const positions = rootsAtNode.get(node) ?? [];
    positions.push(r);
    rootsAtNode.set(node, positions);
  }

  const ref = (node: number): string => {
    if (node >= chunk.start && node < chunk.end) return wgslValueRef(node);
    const slot = escapeSlot[node]!;
    if (slot < 0) {
      throw new Error(
        `GPU codegen internal error: node ${node} is not available in chunk ` +
          `[${chunk.start}, ${chunk.end})`,
      );
    }
    return `scratch[${slot}u * params.numPoints + tid]`;
  };

  for (let i = chunk.start; i < chunk.end; i++) {
    const op = tape.opcodes[i]!;
    const a = tape.argA[i]!;
    const b = tape.argB[i]!;
    let expression: string | null;
    if (op === OP_LIT) expression = formatWgslF32(tape.literals[a]!);
    else if (op === OP_VAR) {
      expression = `varData[tid * params.numVars + ${a}u]`;
    } else {
      expression =
        unaryTapeOpWgsl(op, ref(a)) ?? binaryTapeOpWgsl(op, ref(a), ref(b));
    }

    if (expression === null) {
      throw new Error(
        `GPU codegen does not support tape opcode ${op} at node ${i}`,
      );
    }
    lines.push(`  let ${wgslValueRef(i)} = ${expression};`);

    const slot = escapeSlot[i]!;
    if (slot >= 0) {
      lines.push(
        `  scratch[${slot}u * params.numPoints + tid] = ${wgslValueRef(i)};`,
      );
    }
    for (const rootPosition of rootsAtNode.get(i) ?? []) {
      lines.push(
        `  output[tid * ${tape.rootIndices.length}u + ${rootPosition}u] = ${wgslValueRef(i)};`,
      );
    }
  }

  return `struct Params {
  numPoints: u32,
  numVars: u32,
  numInputPoints: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> varData: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= params.numPoints) { return; }
${lines.join("\n")}
}
`;
}

export interface GpuCodegenEval {
  evalBatch(varData: Float32Array, numPoints: number): Promise<Float32Array>;
  /** @internal Encode directly into caller-owned device storage without readback. */
  encodeBatchToBuffer(
    encoder: GPUCommandEncoder,
    varData: Float32Array,
    numPoints: number,
    output: {
      readonly buffer: GPUBuffer;
      readonly offset: number;
      readonly byteLength: number;
    },
  ): number;
  /** Total synchronous shader-module and pipeline creation time (ms). */
  shaderCompileMs: number;
  readonly varSlots: VarName[];
  readonly numRoots: number;
  readonly numChunks: number;
  readonly numEscapeSlots: number;
  destroy(): void;
}

export interface CompileGpuCodegenOptions {
  /** Maximum tape nodes emitted into one shader module. */
  maxChunkNodes?: number;
}

function createGpuCodegenEval(
  device: GPUDevice,
  tape: CompiledTape,
  plan: GpuPlan,
): GpuCodegenEval {
  const numVars = tape.varSlots.length;
  const numRoots = tape.rootIndices.length;
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
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const t0 = performance.now();
  const pipelines = plan.chunks.map((chunk, chunkIndex) => {
    try {
      const code = generateChunkShader(tape, chunk, plan.escapeSlot);
      const module = device.createShaderModule({ code });
      return device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint: "main" },
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(
        `GPU codegen failed to compile chunk ${chunkIndex + 1}/${plan.chunks.length} ` +
          `(nodes ${chunk.start}..${chunk.end - 1} of ${tape.opcodes.length}): ${detail}`,
      );
    }
  });
  const shaderCompileMs = performance.now() - t0;

  let allocatedBatch = 0;
  let paramsBuf: GPUBuffer | null = null;
  let varDataBuf: GPUBuffer | null = null;
  let scratchBuf: GPUBuffer | null = null;
  let outputBuf: GPUBuffer | null = null;
  let stagingBuf: GPUBuffer | null = null;
  let bindGroup: GPUBindGroup | null = null;

  function destroyBuffers(): void {
    paramsBuf?.destroy();
    varDataBuf?.destroy();
    scratchBuf?.destroy();
    outputBuf?.destroy();
    stagingBuf?.destroy();
  }

  function ensureBatchBuffers(batchSize: number): void {
    if (batchSize === allocatedBatch) return;
    destroyBuffers();
    allocatedBatch = batchSize;

    paramsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    varDataBuf = device.createBuffer({
      size: Math.max(batchSize * numVars * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    scratchBuf = device.createBuffer({
      size: Math.max(batchSize * plan.numEscapeSlots * 4, 4),
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
        { binding: 1, resource: { buffer: varDataBuf } },
        { binding: 2, resource: { buffer: scratchBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });
  }

  function encodeBatchToBuffer(
    encoder: GPUCommandEncoder,
    varData: Float32Array,
    numPoints: number,
    output: {
      readonly buffer: GPUBuffer;
      readonly offset: number;
      readonly byteLength: number;
    },
  ): number {
    if (!Number.isInteger(numPoints) || numPoints <= 0) {
      throw new Error(`Invalid GPU codegen batch size: ${numPoints}`);
    }
    if (varData.length !== numPoints * numVars) {
      throw new Error(
        `GPU codegen expected ${numPoints * numVars} input values, got ${varData.length}`,
      );
    }
    const requiredBytes = numPoints * numRoots * 4;
    if (output.byteLength < requiredBytes) {
      throw new Error(
        `GPU codegen output has ${output.byteLength} bytes; expected ${requiredBytes}`,
      );
    }
    ensureBatchBuffers(numPoints);
    device.queue.writeBuffer(
      paramsBuf!,
      0,
      new Uint32Array([numPoints, numVars, 0, 0]),
    );
    if (varData.byteLength > 0) {
      device.queue.writeBuffer(
        varDataBuf!,
        0,
        varData.buffer as ArrayBuffer,
        varData.byteOffset,
        varData.byteLength,
      );
    }
    const externalBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf! } },
        { binding: 1, resource: { buffer: varDataBuf! } },
        { binding: 2, resource: { buffer: scratchBuf! } },
        {
          binding: 3,
          resource: {
            buffer: output.buffer,
            offset: output.offset,
            size: requiredBytes,
          },
        },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, externalBindGroup);
    for (const pipeline of pipelines) {
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(Math.ceil(numPoints / WORKGROUP_SIZE));
    }
    pass.end();
    return pipelines.length;
  }

  async function evalBatch(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array> {
    if (!Number.isInteger(numPoints) || numPoints < 0) {
      throw new Error(`Invalid GPU codegen batch size: ${numPoints}`);
    }
    if (varData.length !== numPoints * numVars) {
      throw new Error(
        `GPU codegen expected ${numPoints * numVars} input values, got ${varData.length}`,
      );
    }
    if (numPoints === 0) return new Float32Array(0);

    const limit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      device.limits.maxBufferSize,
    );
    const pointsFor = (valuesPerPoint: number): number =>
      valuesPerPoint === 0
        ? Number.MAX_SAFE_INTEGER
        : Math.floor(limit / 4 / valuesPerPoint);
    const maxPoints = Math.min(
      pointsFor(numVars),
      pointsFor(plan.numEscapeSlots),
      pointsFor(numRoots),
      device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE,
    );
    if (maxPoints < 1) {
      throw new Error(
        `GPU codegen scratch requirements exceed device limits: ` +
          `${plan.numEscapeSlots} cross-chunk values, ${limit} byte binding limit`,
      );
    }
    if (numPoints > maxPoints) {
      const results = new Float32Array(numPoints * numRoots);
      for (let offset = 0; offset < numPoints; offset += maxPoints) {
        const end = Math.min(offset + maxPoints, numPoints);
        const part = await evalBatch(
          varData.subarray(offset * numVars, end * numVars),
          end - offset,
        );
        results.set(part, offset * numRoots);
      }
      return results;
    }

    ensureBatchBuffers(numPoints);
    device.queue.writeBuffer(
      paramsBuf!,
      0,
      new Uint32Array([numPoints, numVars, 0, 0]),
    );
    if (varData.byteLength > 0) {
      device.queue.writeBuffer(
        varDataBuf!,
        0,
        varData.buffer as ArrayBuffer,
        varData.byteOffset,
        varData.byteLength,
      );
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bindGroup!);
    for (const pipeline of pipelines) {
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(Math.ceil(numPoints / WORKGROUP_SIZE));
    }
    pass.end();
    device.queue.submit([encoder.finish()]);

    return readGpuBuffer(
      device,
      outputBuf!,
      stagingBuf!,
      numPoints * numRoots * 4,
    );
  }

  return {
    evalBatch,
    encodeBatchToBuffer,
    shaderCompileMs,
    varSlots: tape.varSlots,
    numRoots,
    numChunks: plan.chunks.length,
    numEscapeSlots: plan.numEscapeSlots,
    destroy: destroyBuffers,
  };
}

export function compileGpuCodegenFromTape(
  tape: CompiledTape,
  options: CompileGpuCodegenOptions = {},
): GpuCodegenEval {
  if (tape.rootIndices.length === 0) {
    throw new Error("GPU codegen requires at least one tape root");
  }
  const plan = planGpuCodegenChunks(
    tape,
    options.maxChunkNodes ?? DEFAULT_MAX_CHUNK_NODES,
  );
  return createGpuCodegenEval(requireGpuDevice(), tape, plan);
}
