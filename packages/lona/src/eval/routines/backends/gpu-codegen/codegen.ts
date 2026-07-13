/**
 * WebGPU codegen evaluator — generates a WGSL compute shader per DAG.
 *
 * Instead of interpreting a tape on the GPU (gpu-tape-eval.ts), this
 * generates a WGSL shader where each DAG node becomes a direct WGSL
 * statement. The GPU compiler can then optimize across operations.
 *
 * Advantages over the tape interpreter:
 * - No values scratch buffer (huge memory savings for large DAGs)
 * - No switch dispatch overhead per node
 * - Only 3 bind group entries (vs 8 for tape)
 * - GPU compiler can fuse/reorder operations
 *
 * Trade-off: shader compilation is per-DAG (one-time cost).
 *
 * Uses f32 — results have reduced precision vs CPU f64 evaluators.
 */
import type { VarName } from "../../../../core/tree";
import {
  type CompiledTape,
  OP_LIT,
  OP_VAR,
  OP_SQRT,
  OP_CBRT,
  OP_COS,
  OP_ACOS,
  OP_ASIN,
  OP_TAN,
  OP_ATAN,
  OP_LOG,
  OP_EXP,
  OP_ABS,
  OP_NEG,
  OP_SIN,
  OP_SIGN,
  OP_NOT,
  OP_TANH,
  OP_LOG1P,
  OP_DEBUG,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_MOD,
  OP_ATAN2,
  OP_MIN,
  OP_MAX,
  OP_COMPARE,
  OP_AND,
  OP_OR,
  OP_ASSERT_ZERO,
  OP_ASSERT_NONZERO,
} from "../../../tape";
import { requireGpuDevice, readGpuBuffer } from "../gpu-util";

const WORKGROUP_SIZE = 256;

// ---------------------------------------------------------------------------
// WGSL code generation
// ---------------------------------------------------------------------------

function varRef(i: number): string {
  return `_${i}`;
}

// unaryWgsl and binaryWgsl operate on tape opcodes (defined below generateShaderFromTape)

/** Opcode → WGSL unary expression */
function unaryWgsl(op: number, operand: string): string | null {
  switch (op) {
    case OP_SQRT:
      return `sqrt(${operand})`;
    case OP_CBRT:
      return `sign(${operand}) * pow(abs(${operand}), 0.33333333)`;
    case OP_COS:
      return `cos(${operand})`;
    case OP_ACOS:
      return `acos(${operand})`;
    case OP_ASIN:
      return `asin(${operand})`;
    case OP_TAN:
      return `tan(${operand})`;
    case OP_ATAN:
      return `atan(${operand})`;
    case OP_LOG:
      return `log(${operand})`;
    case OP_EXP:
      return `exp(${operand})`;
    case OP_ABS:
      return `abs(${operand})`;
    case OP_NEG:
      return `(-${operand})`;
    case OP_SIN:
      return `sin(${operand})`;
    case OP_SIGN:
      return `sign(${operand})`;
    case OP_NOT:
      return `select(0.0, 1.0, ${operand} == 0.0)`;
    case OP_TANH:
      return `tanh(${operand})`;
    case OP_LOG1P:
      return `log(1.0 + ${operand})`;
    case OP_DEBUG:
    case OP_ASSERT_ZERO:
    case OP_ASSERT_NONZERO:
      return operand;
  }
  return null;
}

/** Opcode → WGSL binary expression */
function binaryWgsl(op: number, left: string, right: string): string | null {
  switch (op) {
    case OP_ADD:
      return `(${left} + ${right})`;
    case OP_SUB:
      return `(${left} - ${right})`;
    case OP_MUL:
      return `(${left} * ${right})`;
    case OP_DIV:
      return `select(1e38, ${left} / ${right}, ${right} != 0.0)`;
    case OP_MOD:
      return `(${left} % ${right})`;
    case OP_ATAN2:
      return `atan2(${left}, ${right})`;
    case OP_MIN:
      return `min(${left}, ${right})`;
    case OP_MAX:
      return `max(${left}, ${right})`;
    case OP_COMPARE:
      return `sign(${left} - ${right})`;
    case OP_AND:
      return `select(${right}, ${left}, ${left} == 0.0)`;
    case OP_OR:
      return `select(${left}, ${right}, ${left} == 0.0)`;
  }
  return null;
}

function generateShaderFromTape(tape: CompiledTape): string {
  const lines: string[] = [];
  const numNodes = tape.opcodes.length;

  for (let i = 0; i < numNodes; i++) {
    const op = tape.opcodes[i]!;
    const a = tape.argA[i]!;
    const b = tape.argB[i]!;
    const v = varRef(i);

    if (op === OP_LIT) {
      lines.push(`  let ${v} = ${formatF32(tape.literals[a]!)};`);
    } else if (op === OP_VAR) {
      lines.push(`  let ${v} = varData[tid * numVarsu + ${a}u];`);
    } else if (op === OP_DEBUG) {
      lines.push(`  let ${v} = ${varRef(a)};`);
    } else {
      // Try unary first, then binary
      const uExpr = unaryWgsl(op, varRef(a));
      if (uExpr) {
        lines.push(`  let ${v} = ${uExpr};`);
      } else {
        const bExpr = binaryWgsl(op, varRef(a), varRef(b));
        if (bExpr) {
          lines.push(`  let ${v} = ${bExpr};`);
        }
      }
    }
  }

  const body = lines.join("\n");

  return `struct Params {
  numPoints: u32,
  numVars: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> varData: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= params.numPoints) { return; }

  let numVarsu = params.numVars;

${body}

  output[tid] = ${varRef(tape.rootIndices[0])};
}
`;
}

/** Format a number as a WGSL f32 literal */
function formatF32(v: number): string {
  if (!Number.isFinite(v)) {
    if (v === Infinity) return "1e38";
    if (v === -Infinity) return "-1e38";
    return "0.0"; // NaN → 0
  }
  const s = v.toString();
  // Ensure it has a decimal point so WGSL treats it as float
  if (s.includes(".") || s.includes("e") || s.includes("E")) return s;
  return s + ".0";
}

// ---------------------------------------------------------------------------
// GpuCodegenEval
// ---------------------------------------------------------------------------

export interface GpuCodegenEval {
  evalBatch(varData: Float32Array, numPoints: number): Promise<Float32Array>;
  /** Time spent compiling the WGSL shader (ms) */
  shaderCompileMs: number;
  /** All variable names in slot order */
  readonly varSlots: VarName[];
  /** Number of output values per point (1 for single-root, N for multi-root) */
  readonly numRoots: number;
  destroy(): void;
}

function createGpuCodegenEval(
  device: GPUDevice,
  shaderSource: string,
  varSlots: VarName[],
  numRoots = 1,
): GpuCodegenEval {
  const t0 = performance.now();
  const shaderModule = device.createShaderModule({ code: shaderSource });
  const shaderCompileMs = performance.now() - t0;

  const numVars = varSlots.length;

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

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module: shaderModule, entryPoint: "main" },
  });

  let allocatedBatch = 0;
  let paramsBuf: GPUBuffer | null = null;
  let varDataBuf: GPUBuffer | null = null;
  let outputBuf: GPUBuffer | null = null;
  let stagingBuf: GPUBuffer | null = null;
  let bindGroup: GPUBindGroup | null = null;

  function ensureBatchBuffers(batchSize: number) {
    if (batchSize === allocatedBatch) return;
    paramsBuf?.destroy();
    varDataBuf?.destroy();
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
        { binding: 2, resource: { buffer: outputBuf } },
      ],
    });
  }

  async function evalBatch(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array> {
    if (numPoints === 0) return new Float32Array(0);

    const maxByBuffer = Math.floor(
      device.limits.maxStorageBufferBindingSize / (Math.max(numVars, 1) * 4),
    );
    const maxByDispatch =
      device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE;
    const maxPoints = Math.min(maxByBuffer, maxByDispatch);
    if (numPoints > maxPoints) {
      const results = new Float32Array(numPoints * numRoots);
      for (let offset = 0; offset < numPoints; offset += maxPoints) {
        const end = Math.min(offset + maxPoints, numPoints);
        const batchData = varData.subarray(offset * numVars, end * numVars);
        const batchResults = await evalBatch(batchData, end - offset);
        results.set(batchResults, offset * numRoots);
      }
      return results;
    }

    ensureBatchBuffers(numPoints);

    const paramsData = new Uint32Array([numPoints, numVars, 0, 0]);
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
      numPoints * numRoots * 4,
    );
  }

  function destroy() {
    paramsBuf?.destroy();
    varDataBuf?.destroy();
    outputBuf?.destroy();
    stagingBuf?.destroy();
  }

  return { evalBatch, shaderCompileMs, varSlots, numRoots, destroy };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function compileGpuCodegenFromTape(tape: CompiledTape): GpuCodegenEval {
  const device = requireGpuDevice();
  const numRoots = tape.rootIndices.length;
  if (numRoots === 1) {
    const source = generateShaderFromTape(tape);
    return createGpuCodegenEval(device, source, tape.varSlots);
  }
  const source = generateMultiShaderFromTape(tape);
  return createGpuCodegenEval(device, source, tape.varSlots, numRoots);
}

// ---------------------------------------------------------------------------
// Multi-root GPU codegen
// ---------------------------------------------------------------------------

function generateMultiShaderFromTape(tape: CompiledTape): string {
  const rootIndices = tape.rootIndices;
  const numRoots = rootIndices.length;
  const lines: string[] = [];
  const numNodes = tape.opcodes.length;

  for (let i = 0; i < numNodes; i++) {
    const op = tape.opcodes[i]!;
    const a = tape.argA[i]!;
    const b = tape.argB[i]!;
    const v = varRef(i);

    if (op === OP_LIT) {
      lines.push(`  let ${v} = ${formatF32(tape.literals[a]!)};`);
    } else if (op === OP_VAR) {
      lines.push(`  let ${v} = varData[tid * numVarsu + ${a}u];`);
    } else if (op === OP_DEBUG) {
      lines.push(`  let ${v} = ${varRef(a)};`);
    } else {
      const uExpr = unaryWgsl(op, varRef(a));
      if (uExpr) {
        lines.push(`  let ${v} = ${uExpr};`);
      } else {
        const bExpr = binaryWgsl(op, varRef(a), varRef(b));
        if (bExpr) {
          lines.push(`  let ${v} = ${bExpr};`);
        }
      }
    }
  }

  // Write all roots to output, interleaved per point.
  const outputLines: string[] = [];
  for (let i = 0; i < numRoots; i++) {
    outputLines.push(
      `  output[tid * ${numRoots}u + ${i}u] = ${varRef(rootIndices[i]!)};`,
    );
  }

  const body = lines.join("\n");
  const outputBody = outputLines.join("\n");

  return `struct Params {
  numPoints: u32,
  numVars: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> varData: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= params.numPoints) { return; }

  let numVarsu = params.numVars;

${body}

${outputBody}
}
`;
}
