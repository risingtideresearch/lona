/**
 * Per-DAG WASM codegen evaluator.
 *
 * Compiles a CompiledTape into a WASM module where each tape entry becomes
 * direct WASM instructions. The module is AOT-compiled by the WASM engine,
 * avoiding V8 JIT size limits that affect the JS codegen approach.
 *
 * Large tapes are split across multiple WASM functions to stay under V8's
 * per-function body size limit (~7.3 MB). Values that are only read within
 * the same chunk stay in f64 locals on the fast path; values referenced by
 * a later chunk are additionally written to an internal linear memory so
 * the next chunk can load them. A tiny entry function calls each chunk in
 * order and returns the root slot.
 *
 * Uses the shared tape compilation from tape-opcodes.ts.
 */
import type { VarName } from "../../../../core/tree";
import type { CompiledFn, CompiledMultiFn } from "../compiled-fn";
import type { GradientResult } from "../../../tape";
import { bindVarMap } from "../../../tape";
import {
  type CompiledTape,
  OP_LIT,
  OP_VAR,
  OP_ADD,
  OP_ASSERT_ZERO,
  OP_ASSERT_NONZERO,
} from "../../../tape";
import {
  WasmBuilder,
  W_LOCAL_GET,
  W_LOCAL_SET,
  W_F64_CONST,
  W_F64_LOAD,
  W_F64_STORE,
  W_I32_CONST,
  W_CALL,
  W_END,
  VT_F64,
  ALL_IMPORTS,
  IMPORT_INDEX,
  UNARY_IMPORT,
  BINARY_IMPORT,
  emitNumOp,
} from "../wasm-builder";

// ---------------------------------------------------------------------------
// Chunk planning
// ---------------------------------------------------------------------------

/**
 * Target max body-bytes per chunk. V8 rejects functions larger than ~7.3 MB
 * (size 7654321), and escape loads/stores add some slack on top of the raw
 * estimate — keep a comfortable margin.
 */
const DEFAULT_MAX_CHUNK_BYTES = 3_000_000;

/**
 * Conservative cap for parameters plus declared locals in one function.
 * V8 currently rejects functions at roughly 50,000 locals. Staying below
 * that implementation limit also leaves room for emitter temporaries.
 */
const DEFAULT_MAX_FUNCTION_LOCALS = 40_000;

/** Conservative upper-bound byte cost per tape op inside a function body. */
function estimateOpBytes(op: number): number {
  switch (op) {
    case OP_LIT:
      return 16; // f64.const(1+8) + local.set(1+<=5)
    case OP_VAR:
      return 10; // local.get + local.set
    // Expensive control-flow ops
    case 43 /* OP_DIV */:
    case 48 /* OP_COMPARE */:
    case 22 /* OP_SIGN */:
      return 48;
    case 23 /* OP_NOT */:
    case 49 /* OP_AND */:
    case 50 /* OP_OR */:
      return 32;
    // Imported unary/binary: call + index
    case 21 /* OP_SIN */:
    case 12 /* OP_COS */:
    case 15 /* OP_TAN */:
    case 14 /* OP_ASIN */:
    case 13 /* OP_ACOS */:
    case 16 /* OP_ATAN */:
    case 18 /* OP_EXP */:
    case 17 /* OP_LOG */:
    case 25 /* OP_LOG1P */:
    case 11 /* OP_CBRT */:
    case 24 /* OP_TANH */:
    case 44 /* OP_MOD */:
    case 45 /* OP_ATAN2 */:
      return 20;
    default:
      // Native unary/binary: 2-3 local.get + op byte + local.set
      return 18;
  }
}

/**
 * Additional bytes added when an op reads/writes a cross-chunk escape slot.
 * Reads: `i32.const 0 + f64.load memarg` (~7-10 B per escaping arg).
 * Writes: `i32.const 0 + local.get + f64.store memarg` (~10-14 B).
 * Used as a safety fudge in the chunk size estimator.
 */
const ESCAPE_READ_FUDGE = 12;
const ESCAPE_WRITE_FUDGE = 14;

interface ChunkPlan {
  start: number; // first tape index (inclusive)
  end: number; // last tape index (exclusive)
}

/**
 * Split the tape into contiguous chunks bounded by both estimated emitted
 * bytes and function-local count. `localsPerNode` is greater than one for
 * forward-mode autodiff, where every node stores a primal and derivatives.
 */
function planChunks(
  tape: CompiledTape,
  maxBytes: number,
  maxFunctionLocals: number,
  localsPerNode = 1,
): ChunkPlan[] {
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
    throw new Error(`Invalid WASM chunk byte limit: ${maxBytes}`);
  }
  if (!Number.isInteger(maxFunctionLocals) || maxFunctionLocals <= 0) {
    throw new Error(`Invalid WASM function-local limit: ${maxFunctionLocals}`);
  }
  if (!Number.isInteger(localsPerNode) || localsPerNode <= 0) {
    throw new Error(`Invalid WASM locals-per-node value: ${localsPerNode}`);
  }

  // Count parameters conservatively as locals. Engines differ in how their
  // diagnostics describe this limit, but both consume local indices.
  const availableLocals = maxFunctionLocals - tape.varSlots.length;
  const maxNodes = Math.floor(availableLocals / localsPerNode);
  if (maxNodes < 1) {
    throw new Error(
      `WASM codegen cannot fit one tape node in a function: ` +
        `${tape.varSlots.length} parameters, ${localsPerNode} locals/node, ` +
        `${maxFunctionLocals} total-local limit`,
    );
  }

  const chunks: ChunkPlan[] = [];
  const N = tape.opcodes.length;
  if (N === 0) return [{ start: 0, end: 0 }];

  let chunkStart = 0;
  // Base per-chunk overhead: locals decl + end byte + function header.
  let chunkBytes = 32;

  for (let i = 0; i < N; i++) {
    // Include worst-case escape overhead so the estimator stays safe even
    // before we know the final escape set.
    const est =
      estimateOpBytes(tape.opcodes[i]!) +
      ESCAPE_READ_FUDGE * 2 +
      ESCAPE_WRITE_FUDGE;
    const exceedsBytes = chunkBytes + est > maxBytes;
    const exceedsLocals = i - chunkStart >= maxNodes;
    if ((exceedsBytes || exceedsLocals) && i > chunkStart) {
      chunks.push({ start: chunkStart, end: i });
      chunkStart = i;
      chunkBytes = 32;
    }
    chunkBytes += est;
  }
  chunks.push({ start: chunkStart, end: N });
  return chunks;
}

// ---------------------------------------------------------------------------
// Liveness / escape analysis
// ---------------------------------------------------------------------------

/**
 * For each tape slot, compute a dense escape slot index (or -1 if the
 * value is only read within the chunk that produces it).
 *
 * Escape slot i gets byte offset `i * 8` in the module's linear memory.
 * Returns the slot assignment array plus the total number of escape slots.
 */
function computeEscapes(
  tape: CompiledTape,
  chunks: ChunkPlan[],
): {
  escapeSlot: Int32Array; // tape index → escape slot (or -1)
  numEscapeSlots: number;
} {
  const N = tape.opcodes.length;
  const lastReader = new Int32Array(N);
  for (let i = 0; i < N; i++) lastReader[i] = -1;

  // Single forward scan: each op pulls in its tape-slot arguments.
  for (let i = 0; i < N; i++) {
    const op = tape.opcodes[i]!;
    if (op === OP_LIT || op === OP_VAR) continue;
    // argA is a tape slot for every unary/binary op.
    lastReader[tape.argA[i]!] = i;
    // Binary ops (op >= OP_ADD) also read argB.
    if (op >= OP_ADD) {
      lastReader[tape.argB[i]!] = i;
    }
  }
  // Root is "read" by the entry function after every chunk has run.
  lastReader[tape.rootIndices[0]] = N;

  // Build chunkOf[i]: the chunk containing tape slot i.
  const chunkOf = new Int32Array(N);
  for (let c = 0; c < chunks.length; c++) {
    const { start, end } = chunks[c]!;
    for (let i = start; i < end; i++) chunkOf[i] = c;
  }

  // A slot escapes iff its last reader lives strictly beyond its chunk's end.
  const escapeSlot = new Int32Array(N);
  let numEscapeSlots = 0;
  for (let i = 0; i < N; i++) {
    const producerChunk = chunkOf[i]!;
    const chunkEnd = chunks[producerChunk]!.end;
    const reader = lastReader[i]!;
    if (reader === -1) {
      // Dead code (or unread intermediate that isn't the root). Shouldn't
      // happen for a well-formed DAG but stay defensive.
      escapeSlot[i] = -1;
      continue;
    }
    if (reader >= chunkEnd) {
      escapeSlot[i] = numEscapeSlots++;
    } else {
      escapeSlot[i] = -1;
    }
  }

  return { escapeSlot, numEscapeSlots };
}

// ---------------------------------------------------------------------------
// Emit a single chunk function body
// ---------------------------------------------------------------------------

/** Emit `f64.load offset=slot*8` starting from `i32.const 0` as address. */
function emitEscapeLoad(body: WasmBuilder, slot: number): void {
  body.byte(W_I32_CONST);
  body.i32(0);
  body.byte(W_F64_LOAD);
  body.u32(3); // alignment exponent (8-byte)
  body.u32(slot * 8);
}

/**
 * Emit `f64.store offset=slot*8` of the f64 currently in `srcLocalIdx`,
 * starting from `i32.const 0` as address.
 */
function emitEscapeStore(
  body: WasmBuilder,
  slot: number,
  srcLocalIdx: number,
): void {
  body.byte(W_I32_CONST);
  body.i32(0);
  body.byte(W_LOCAL_GET);
  body.u32(srcLocalIdx);
  body.byte(W_F64_STORE);
  body.u32(3);
  body.u32(slot * 8);
}

function emitChunkBody(
  tape: CompiledTape,
  chunk: ChunkPlan,
  numVars: number,
  escapeSlot: Int32Array,
  getCallIdx: (name: string) => number,
): Uint8Array {
  const chunkSize = chunk.end - chunk.start;
  const body = new WasmBuilder(Math.max(chunkSize * 24, 512));

  // Locals: one f64 per tape node produced in this chunk.
  if (chunkSize > 0) {
    body.u32(1);
    body.u32(chunkSize);
    body.byte(VT_F64);
  } else {
    body.u32(0);
  }

  // Local index layout:
  //   [0 .. numVars-1]            : variable params
  //   [numVars .. numVars+chunkSize-1] : node values (local[numVars + (i - chunk.start)])
  const localBase = numVars;

  const localOfNode = (tapeIdx: number): number =>
    localBase + (tapeIdx - chunk.start);

  /**
   * Push the f64 value of tape slot `tapeIdx` onto the stack. If the
   * value was produced in this chunk use its local; otherwise load it
   * from the cross-chunk escape memory.
   */
  const pushSlot = (w: WasmBuilder, tapeIdx: number): void => {
    if (tapeIdx >= chunk.start && tapeIdx < chunk.end) {
      w.byte(W_LOCAL_GET);
      w.u32(localOfNode(tapeIdx));
      return;
    }
    const slot = escapeSlot[tapeIdx]!;
    // A producer outside this chunk must have been marked as escaping
    // (its last reader is this chunk or later).
    emitEscapeLoad(w, slot);
  };

  for (let i = chunk.start; i < chunk.end; i++) {
    const op = tape.opcodes[i]!;
    const a = tape.argA[i]!;
    const b = tape.argB[i]!;
    const dstLocal = localOfNode(i);

    if (op === OP_LIT) {
      body.byte(W_F64_CONST);
      body.f64(tape.literals[a]!);
      body.byte(W_LOCAL_SET);
      body.u32(dstLocal);
    } else if (op === OP_VAR) {
      // `a` is the var slot index == param index in this function.
      body.byte(W_LOCAL_GET);
      body.u32(a);
      body.byte(W_LOCAL_SET);
      body.u32(dstLocal);
    } else {
      const pushA = (w: WasmBuilder) => pushSlot(w, a);
      const pushB = (w: WasmBuilder) => pushSlot(w, b);
      emitNumOp(body, op, pushA, pushB, getCallIdx);
      body.byte(W_LOCAL_SET);
      body.u32(dstLocal);
    }

    // If this node escapes its chunk, mirror the value to memory.
    const slot = escapeSlot[i]!;
    if (slot >= 0) {
      emitEscapeStore(body, slot, dstLocal);
    }
  }

  body.byte(W_END);
  return body.toBytes();
}

// ---------------------------------------------------------------------------
// Emit the entry function body (calls each chunk, returns root from memory)
// ---------------------------------------------------------------------------

function emitEntryBody(
  numChunks: number,
  numVars: number,
  firstChunkFuncIdx: number,
  rootEscapeSlot: number,
): Uint8Array {
  const body = new WasmBuilder(64 + numChunks * (2 * numVars + 4));

  // No additional locals — the var params are reused as call arguments.
  body.u32(0);

  for (let c = 0; c < numChunks; c++) {
    for (let p = 0; p < numVars; p++) {
      body.byte(W_LOCAL_GET);
      body.u32(p);
    }
    body.byte(W_CALL);
    body.u32(firstChunkFuncIdx + c);
  }

  // Return the root value from its escape slot in memory.
  emitEscapeLoad(body, rootEscapeSlot);

  body.byte(W_END);
  return body.toBytes();
}

// ---------------------------------------------------------------------------
// Scan which imports are needed
// ---------------------------------------------------------------------------

function collectUsedImports(tape: CompiledTape): Set<string> {
  const used = new Set<string>();
  for (let i = 0; i < tape.opcodes.length; i++) {
    const op = tape.opcodes[i]!;
    const imp = UNARY_IMPORT[op] ?? BINARY_IMPORT[op];
    if (imp) used.add(imp);
  }
  return used;
}

/** Adds imports needed by forward-mode derivative rules (e.g. sin→cos). */
function collectGradExtraImports(tape: CompiledTape, used: Set<string>): void {
  const DERIV_EXTRA: Record<number, string[]> = {
    21 /* SIN */: ["cos"],
    12 /* COS */: ["sin"],
    15 /* TAN */: ["cos"],
    14 /* ASIN */: [],
    13 /* ACOS */: [],
    16 /* ATAN */: [],
    18 /* EXP */: ["exp"],
    17 /* LOG */: [],
    25 /* LOG1P */: [],
    11 /* CBRT */: ["cbrt"],
    24 /* TANH */: ["tanh"],
  };
  for (let i = 0; i < tape.opcodes.length; i++) {
    const extras = DERIV_EXTRA[tape.opcodes[i]!];
    if (extras) for (const e of extras) used.add(e);
  }
}

// ---------------------------------------------------------------------------
// Assemble the full WASM module
// ---------------------------------------------------------------------------

function emitWasmModule(
  tape: CompiledTape,
  maxChunkBytes: number,
  maxFunctionLocals: number,
): {
  bytes: Uint8Array;
  importNames: string[];
  numChunks: number;
} {
  const numVars = tape.varSlots.length;
  const N = tape.opcodes.length;

  // --- Chunk + escape analysis ---
  const chunks = planChunks(tape, maxChunkBytes, maxFunctionLocals);
  const { escapeSlot, numEscapeSlots } = computeEscapes(tape, chunks);
  const rootEscapeSlot = N === 0 ? -1 : escapeSlot[tape.rootIndices[0]]!;

  // Memory size: one f64 per escape slot, rounded up to 64 KiB pages.
  // Guarantee at least 1 page so the memory section is always valid.
  const bytesNeeded = Math.max(1, numEscapeSlots * 8);
  const numPages = Math.max(1, Math.ceil(bytesNeeded / 65536));

  // --- Imports ---
  const usedImportNames = collectUsedImports(tape);
  const importNames: string[] = [];
  const importFuncIndex: Record<string, number> = {};
  for (const def of ALL_IMPORTS) {
    if (usedImportNames.has(def.name)) {
      importFuncIndex[def.name] = importNames.length;
      importNames.push(def.name);
    }
  }
  const numImports = importNames.length;
  const getCallIdx = (name: string) => importFuncIndex[name]!;

  // Function index layout:
  //   0 .. numImports-1       : imported functions
  //   numImports .. +numChunks-1 : chunk functions
  //   numImports + numChunks  : entry function ("eval")
  const firstChunkFuncIdx = numImports;
  const entryFuncIdx = numImports + chunks.length;

  // --- Type section ---
  // type 0: entry (f64^numVars) -> f64
  // type 1: chunk (f64^numVars) -> ()
  // type 2: unary import (f64) -> f64  [if any unary import used]
  // type 3: binary import (f64, f64) -> f64  [if any binary import used]
  const typeSec = new WasmBuilder();
  const hasUnary = importNames.some(
    (n) => ALL_IMPORTS[IMPORT_INDEX[n]!]!.paramCount === 1,
  );
  const hasBinary = importNames.some(
    (n) => ALL_IMPORTS[IMPORT_INDEX[n]!]!.paramCount === 2,
  );

  const entryTypeIdx = 0;
  const chunkTypeIdx = 1;
  let typeCount = 2;
  const unaryTypeIdx = hasUnary ? typeCount++ : -1;
  const binaryTypeIdx = hasBinary ? typeCount++ : -1;

  typeSec.u32(typeCount);

  // type 0: entry
  typeSec.byte(0x60);
  typeSec.u32(numVars);
  for (let i = 0; i < numVars; i++) typeSec.byte(VT_F64);
  typeSec.u32(1);
  typeSec.byte(VT_F64);

  // type 1: chunk
  typeSec.byte(0x60);
  typeSec.u32(numVars);
  for (let i = 0; i < numVars; i++) typeSec.byte(VT_F64);
  typeSec.u32(0);

  if (hasUnary) {
    typeSec.byte(0x60);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
  }
  if (hasBinary) {
    typeSec.byte(0x60);
    typeSec.u32(2);
    typeSec.byte(VT_F64);
    typeSec.byte(VT_F64);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
  }

  // --- Import section ---
  const importSec = new WasmBuilder();
  importSec.u32(numImports);
  for (const name of importNames) {
    const def = ALL_IMPORTS[IMPORT_INDEX[name]!]!;
    importSec.str("env");
    importSec.str(name);
    importSec.byte(0x00); // kind: func
    importSec.u32(def.paramCount === 1 ? unaryTypeIdx : binaryTypeIdx);
  }

  // --- Function section ---
  // N chunk functions (type 1) + 1 entry function (type 0).
  const funcSec = new WasmBuilder();
  funcSec.u32(chunks.length + 1);
  for (let c = 0; c < chunks.length; c++) funcSec.u32(chunkTypeIdx);
  funcSec.u32(entryTypeIdx);

  // --- Memory section ---
  // Internal memory sized to hold all cross-chunk escape slots.
  const memSec = new WasmBuilder();
  memSec.u32(1); // 1 memory
  memSec.byte(0x00); // flags: min-only
  memSec.u32(numPages);

  // --- Export section ---
  const exportSec = new WasmBuilder();
  exportSec.u32(1);
  exportSec.str("eval");
  exportSec.byte(0x00); // kind: func
  exportSec.u32(entryFuncIdx);

  // --- Code section ---
  // Each chunk's body + entry body, each preceded by its length.
  const chunkBodies: Uint8Array[] = [];
  for (const chunk of chunks) {
    chunkBodies.push(
      emitChunkBody(tape, chunk, numVars, escapeSlot, getCallIdx),
    );
  }
  // For N === 0 (empty tape / unreachable) force a 0-slot store+load pair
  // so the entry's load-root still resolves. Easiest: pretend the root
  // escapes to slot 0 and zero the memory implicitly. The CompiledTape
  // builder never returns a null root, so this path is theoretical.
  const entryBody = emitEntryBody(
    chunks.length,
    numVars,
    firstChunkFuncIdx,
    rootEscapeSlot < 0 ? 0 : rootEscapeSlot,
  );

  const codeSec = new WasmBuilder(
    chunkBodies.reduce((s, b) => s + b.length + 8, 0) + entryBody.length + 16,
  );
  codeSec.u32(chunks.length + 1);
  for (const cb of chunkBodies) {
    codeSec.u32(cb.length);
    codeSec.bytes(cb);
  }
  codeSec.u32(entryBody.length);
  codeSec.bytes(entryBody);

  // --- Assemble module ---
  // Total estimate: code + other sections ~= code size + a few KB.
  const codeBytes = codeSec.length;
  const mod = new WasmBuilder(codeBytes + 4096);
  mod.bytes([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  mod.section(1, typeSec);
  if (numImports > 0) mod.section(2, importSec);
  mod.section(3, funcSec);
  mod.section(5, memSec);
  mod.section(7, exportSec);
  mod.section(10, codeSec);

  return {
    bytes: mod.toBytes(),
    importNames,
    numChunks: chunks.length,
  };
}

// ---------------------------------------------------------------------------
// Instantiate and wrap
// ---------------------------------------------------------------------------

function compileModuleBytes(
  bytes: Uint8Array,
  tape: CompiledTape,
): WebAssembly.Module {
  try {
    return new WebAssembly.Module(bytes as unknown as BufferSource);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      `WASM codegen failed for a ${tape.opcodes.length}-node tape with ` +
        `${tape.varSlots.length} parameters: ${detail}`,
    );
  }
}

function emitAndWrap(
  tape: CompiledTape,
  maxChunkBytes: number,
  maxFunctionLocals: number,
): CompiledFn {
  const { bytes, importNames } = emitWasmModule(
    tape,
    maxChunkBytes,
    maxFunctionLocals,
  );

  const env: Record<string, (...args: number[]) => number> = {};
  for (const name of importNames) {
    env[name] = ALL_IMPORTS[IMPORT_INDEX[name]!]!.jsFn;
  }

  const module = compileModuleBytes(bytes, tape);
  const instance = new WebAssembly.Instance(module, { env });
  const wasmEval = instance.exports.eval as (...args: number[]) => number;

  const slots = tape.varSlots;
  const args = new Array<number>(slots.length);

  return (vars, derivatives) => {
    bindVarMap(tape, vars, derivatives, args);
    return wasmEval(...args);
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface CompileWasmOptions {
  /**
   * Maximum estimated body byte size per emitted WASM function. Defaults
   * to {@link DEFAULT_MAX_CHUNK_BYTES}, well under the V8 ~7.3 MB cap.
   * Lower values exercise the multi-chunk path (useful in tests).
   */
  maxChunkBytes?: number;
  /**
   * Maximum parameters plus declared locals in an emitted function. Defaults
   * to a conservative value below V8's current implementation limit.
   * Lower values are useful for exercising local-count chunking in tests.
   */
  maxFunctionLocals?: number;
}

export function compileWasmFromTape(
  tape: CompiledTape,
  options: CompileWasmOptions = {},
): CompiledMultiFn {
  const maxChunkBytes = options.maxChunkBytes ?? DEFAULT_MAX_CHUNK_BYTES;
  const maxFunctionLocals =
    options.maxFunctionLocals ?? DEFAULT_MAX_FUNCTION_LOCALS;
  const numRoots = tape.rootIndices.length;
  if (numRoots === 1) {
    // Single-root path: the WASM entry returns f64 directly.
    const fn = emitAndWrap(tape, maxChunkBytes, maxFunctionLocals);
    return (vars, derivatives) => [fn(vars, derivatives)];
  }
  // Multi-root path: results written to WASM memory, read back as number[].
  return emitMultiAndWrap(tape, maxChunkBytes, maxFunctionLocals);
}

// ---------------------------------------------------------------------------
// Multi-root WASM codegen (internal)
// ---------------------------------------------------------------------------

/**
 * Emit a multi-root entry function body. After calling all chunks, it writes
 * each root value from its escape slot to the output region at the start of
 * memory (offset 0 .. numRoots*8).
 */
function emitMultiEntryBody(
  numChunks: number,
  numVars: number,
  firstChunkFuncIdx: number,
  rootEscapeSlots: number[],
  outputBaseOffset: number,
): Uint8Array {
  const numRoots = rootEscapeSlots.length;
  const body = new WasmBuilder(
    64 + numChunks * (2 * numVars + 4) + numRoots * 24,
  );

  // No additional locals.
  body.u32(0);

  // Call each chunk, passing all var params.
  for (let c = 0; c < numChunks; c++) {
    for (let p = 0; p < numVars; p++) {
      body.byte(W_LOCAL_GET);
      body.u32(p);
    }
    body.byte(W_CALL);
    body.u32(firstChunkFuncIdx + c);
  }

  // Copy each root value from its escape slot to the output region.
  for (let i = 0; i < numRoots; i++) {
    // Store address
    body.byte(W_I32_CONST);
    body.i32(0);
    // Load the root value from its escape slot
    emitEscapeLoad(body, rootEscapeSlots[i]!);
    // Store to output region
    body.byte(W_F64_STORE);
    body.u32(3); // alignment
    body.u32(outputBaseOffset + i * 8);
  }

  body.byte(W_END);
  return body.toBytes();
}

/**
 * Assemble a multi-root WASM module. The entry function returns void; results
 * are written to the output region in exported memory.
 */
function emitMultiWasmModule(
  tape: CompiledTape,
  maxChunkBytes: number,
  maxFunctionLocals: number,
): {
  bytes: Uint8Array;
  importNames: string[];
  numChunks: number;
  /** Byte offset of the output region in memory */
  outputBaseOffset: number;
} {
  const rootIndices = tape.rootIndices;
  const numRoots = rootIndices.length;
  const numVars = tape.varSlots.length;

  // --- Chunk + escape analysis ---
  const chunks = planChunks(tape, maxChunkBytes, maxFunctionLocals);
  const { escapeSlot, numEscapeSlots } = computeEscapes(tape, chunks);

  // Force all roots to have escape slots (they may already from computeEscapes
  // since the last root is always marked, but others might not be).
  let nextEscape = numEscapeSlots;
  for (const ri of rootIndices) {
    if (escapeSlot[ri]! < 0) {
      escapeSlot[ri] = nextEscape++;
    }
  }

  // Output region starts after all escape slots.
  const outputBaseOffset = nextEscape * 8;

  // Memory: escape slots + output region.
  const bytesNeeded = Math.max(1, outputBaseOffset + numRoots * 8);
  const numPages = Math.max(1, Math.ceil(bytesNeeded / 65536));

  const rootEscapeSlots = rootIndices.map((ri) => escapeSlot[ri]!);

  // --- Imports ---
  const usedImportNames = collectUsedImports(tape);
  const importNames: string[] = [];
  const importFuncIndex: Record<string, number> = {};
  for (const def of ALL_IMPORTS) {
    if (usedImportNames.has(def.name)) {
      importFuncIndex[def.name] = importNames.length;
      importNames.push(def.name);
    }
  }
  const numImports = importNames.length;
  const getCallIdx = (name: string) => importFuncIndex[name]!;

  const firstChunkFuncIdx = numImports;
  const entryFuncIdx = numImports + chunks.length;

  // --- Type section ---
  // type 0: entry (f64^numVars) -> () [void for multi-root]
  // type 1: chunk (f64^numVars) -> ()
  // type 2+: import types
  const typeSec = new WasmBuilder();
  const hasUnary = importNames.some(
    (n) => ALL_IMPORTS[IMPORT_INDEX[n]!]!.paramCount === 1,
  );
  const hasBinary = importNames.some(
    (n) => ALL_IMPORTS[IMPORT_INDEX[n]!]!.paramCount === 2,
  );

  // For multi-root, entry and chunk have the same type: (f64^numVars) -> ()
  const entryTypeIdx = 0;
  const chunkTypeIdx = 0; // Same type!
  let typeCount = 1;
  const unaryTypeIdx = hasUnary ? typeCount++ : -1;
  const binaryTypeIdx = hasBinary ? typeCount++ : -1;

  typeSec.u32(typeCount);

  // type 0: (f64^numVars) -> ()
  typeSec.byte(0x60);
  typeSec.u32(numVars);
  for (let i = 0; i < numVars; i++) typeSec.byte(VT_F64);
  typeSec.u32(0); // void return

  if (hasUnary) {
    typeSec.byte(0x60);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
  }
  if (hasBinary) {
    typeSec.byte(0x60);
    typeSec.u32(2);
    typeSec.byte(VT_F64);
    typeSec.byte(VT_F64);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
  }

  // --- Import section ---
  const importSec = new WasmBuilder();
  importSec.u32(numImports);
  for (const name of importNames) {
    const def = ALL_IMPORTS[IMPORT_INDEX[name]!]!;
    importSec.str("env");
    importSec.str(name);
    importSec.byte(0x00);
    importSec.u32(def.paramCount === 1 ? unaryTypeIdx : binaryTypeIdx);
  }

  // --- Function section ---
  const funcSec = new WasmBuilder();
  funcSec.u32(chunks.length + 1);
  for (let c = 0; c < chunks.length; c++) funcSec.u32(chunkTypeIdx);
  funcSec.u32(entryTypeIdx);

  // --- Memory section ---
  const memSec = new WasmBuilder();
  memSec.u32(1);
  memSec.byte(0x00);
  memSec.u32(numPages);

  // --- Export section ---
  // Export both the entry function and memory.
  const exportSec = new WasmBuilder();
  exportSec.u32(2);
  exportSec.str("eval");
  exportSec.byte(0x00); // kind: func
  exportSec.u32(entryFuncIdx);
  exportSec.str("mem");
  exportSec.byte(0x02); // kind: memory
  exportSec.u32(0); // memory index 0

  // --- Code section ---
  const chunkBodies: Uint8Array[] = [];
  for (const chunk of chunks) {
    chunkBodies.push(
      emitChunkBody(tape, chunk, numVars, escapeSlot, getCallIdx),
    );
  }
  const entryBody = emitMultiEntryBody(
    chunks.length,
    numVars,
    firstChunkFuncIdx,
    rootEscapeSlots,
    outputBaseOffset,
  );

  const codeSec = new WasmBuilder(
    chunkBodies.reduce((s, b) => s + b.length + 8, 0) + entryBody.length + 16,
  );
  codeSec.u32(chunks.length + 1);
  for (const cb of chunkBodies) {
    codeSec.u32(cb.length);
    codeSec.bytes(cb);
  }
  codeSec.u32(entryBody.length);
  codeSec.bytes(entryBody);

  // --- Assemble module ---
  const codeBytes = codeSec.length;
  const mod = new WasmBuilder(codeBytes + 4096);
  mod.bytes([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  mod.section(1, typeSec);
  if (numImports > 0) mod.section(2, importSec);
  mod.section(3, funcSec);
  mod.section(5, memSec);
  mod.section(7, exportSec);
  mod.section(10, codeSec);

  return {
    bytes: mod.toBytes(),
    importNames,
    numChunks: chunks.length,
    outputBaseOffset,
  };
}

function emitMultiAndWrap(
  tape: CompiledTape,
  maxChunkBytes: number,
  maxFunctionLocals: number,
): CompiledMultiFn {
  const rootIndices = tape.rootIndices;
  const numRoots = rootIndices.length;
  const { bytes, importNames, outputBaseOffset } = emitMultiWasmModule(
    tape,
    maxChunkBytes,
    maxFunctionLocals,
  );

  const env: Record<string, (...args: number[]) => number> = {};
  for (const name of importNames) {
    env[name] = ALL_IMPORTS[IMPORT_INDEX[name]!]!.jsFn;
  }

  const module = compileModuleBytes(bytes, tape);
  const instance = new WebAssembly.Instance(module, { env });
  const wasmEval = instance.exports.eval as (...args: number[]) => void;
  const wasmMem = instance.exports.mem as WebAssembly.Memory;

  const slots = tape.varSlots;
  const args = new Array<number>(slots.length);

  return (vars, derivatives) => {
    bindVarMap(tape, vars, derivatives, args);
    wasmEval(...args);
    // Read results from the output region in WASM memory.
    const view = new Float64Array(wasmMem.buffer, outputBaseOffset, numRoots);
    // Copy out as number[] — the buffer may be detached on next call.
    return Array.from(view);
  };
}

// ---------------------------------------------------------------------------
// Forward-mode autodiff — per-DAG WASM codegen
// ---------------------------------------------------------------------------
//
// Like the value-only codegen above, but each tape node gets `stride`
// f64 locals (value + one partial per diffVar). Derivative propagation is
// unrolled at compile time since numDiff is known.

export type WasmGradFn = (vars: Map<VarName, number>) => GradientResult;

/**
 * Emit the forward-mode chunk body. Each node i gets locals
 * `localBase + (i - chunk.start) * stride` through `+ stride - 1`.
 * Local 0 is the primal, locals 1..numDiff are the partials.
 */
function emitGradChunkBody(
  tape: CompiledTape,
  chunk: ChunkPlan,
  numVars: number,
  numDiff: number,
  stride: number,
  seedBaseOffset: number,
  escapeSlot: Int32Array,
  getCallIdx: (name: string) => number,
): Uint8Array {
  const chunkSize = chunk.end - chunk.start;
  const body = new WasmBuilder(Math.max(chunkSize * stride * 32, 512));

  // Locals: stride f64 per tape node in this chunk.
  if (chunkSize > 0) {
    body.u32(1);
    body.u32(chunkSize * stride);
    body.byte(VT_F64);
  } else {
    body.u32(0);
  }

  const localBase = numVars;
  const localOfDual = (tapeIdx: number, d: number): number =>
    localBase + (tapeIdx - chunk.start) * stride + d;

  /** Push primal (d=0) or derivative (d>0) of a tape slot. */
  const pushDual = (w: WasmBuilder, tapeIdx: number, d: number): void => {
    if (tapeIdx >= chunk.start && tapeIdx < chunk.end) {
      w.byte(W_LOCAL_GET);
      w.u32(localOfDual(tapeIdx, d));
      return;
    }
    const slot = escapeSlot[tapeIdx]!;
    emitEscapeLoad(w, slot * stride + d);
  };

  for (let i = chunk.start; i < chunk.end; i++) {
    const op = tape.opcodes[i]!;
    const a = tape.argA[i]!;
    const b = tape.argB[i]!;
    const dst = (d: number) => localOfDual(i, d);

    if (op === OP_LIT) {
      body.byte(W_F64_CONST);
      body.f64(tape.literals[a]!);
      body.byte(W_LOCAL_SET);
      body.u32(dst(0));
      for (let d = 0; d < numDiff; d++) {
        body.byte(W_F64_CONST);
        body.f64(0);
        body.byte(W_LOCAL_SET);
        body.u32(dst(1 + d));
      }
    } else if (op === OP_VAR) {
      // Primal: var param
      body.byte(W_LOCAL_GET);
      body.u32(a);
      body.byte(W_LOCAL_SET);
      body.u32(dst(0));
      // Derivatives: arbitrary input-major seeds loaded from memory.
      for (let d = 0; d < numDiff; d++) {
        body.byte(W_I32_CONST);
        body.i32(seedBaseOffset + (a * numDiff + d) * 8);
        body.byte(W_F64_LOAD);
        body.u32(3);
        body.u32(0);
        body.byte(W_LOCAL_SET);
        body.u32(dst(1 + d));
      }
    } else {
      // Compute primal using emitNumOp
      const pushA = (w: WasmBuilder) => pushDual(w, a, 0);
      const pushB = (w: WasmBuilder) => pushDual(w, b, 0);
      emitNumOp(body, op, pushA, pushB, getCallIdx);
      body.byte(W_LOCAL_SET);
      body.u32(dst(0));

      // Compute derivatives (unrolled)
      emitGradDerivatives(
        body,
        op,
        numDiff,
        (w, d) => pushDual(w, a, 1 + d),
        (w, d) => pushDual(w, b, 1 + d),
        pushA,
        pushB,
        (d, w) => {
          w.byte(W_LOCAL_SET);
          w.u32(dst(1 + d));
        },
        getCallIdx,
      );
    }

    // Escape: store stride f64s to memory
    const slot = escapeSlot[i]!;
    if (slot >= 0) {
      for (let d = 0; d < stride; d++) {
        emitEscapeStore(body, slot * stride + d, dst(d));
      }
    }
  }

  body.byte(W_END);
  return body.toBytes();
}

/**
 * Emit unrolled derivative computations for one op.
 * For each d in [0, numDiff), pushes the derivative value and calls storeDeriv.
 */
function emitGradDerivatives(
  body: WasmBuilder,
  op: number,
  numDiff: number,
  /** Push d-th derivative of operand A */
  pushAd: (w: WasmBuilder, d: number) => void,
  /** Push d-th derivative of operand B */
  pushBd: (w: WasmBuilder, d: number) => void,
  /** Push primal of operand A */
  pushA: (w: WasmBuilder) => void,
  /** Push primal of operand B */
  pushB: (w: WasmBuilder) => void,
  /** Store result: calls local.set for derivative d */
  storeDeriv: (d: number, w: WasmBuilder) => void,
  getCallIdx: (name: string) => number,
): void {
  // Classify op to determine derivative rule
  if (op === OP_ASSERT_ZERO || op === OP_ASSERT_NONZERO) {
    emitGradUnaryDerivs(
      body,
      op,
      numDiff,
      pushAd,
      pushA,
      storeDeriv,
      getCallIdx,
    );
  } else if (op >= OP_ADD) {
    // Binary op
    emitGradBinaryDerivs(
      body,
      op,
      numDiff,
      pushAd,
      pushBd,
      pushA,
      pushB,
      storeDeriv,
    );
  } else {
    // Unary op
    emitGradUnaryDerivs(
      body,
      op,
      numDiff,
      pushAd,
      pushA,
      storeDeriv,
      getCallIdx,
    );
  }
}

function emitGradUnaryDerivs(
  body: WasmBuilder,
  op: number,
  numDiff: number,
  pushAd: (w: WasmBuilder, d: number) => void,
  pushA: (w: WasmBuilder) => void,
  storeDeriv: (d: number, w: WasmBuilder) => void,
  getCallIdx: (name: string) => number,
): void {
  // For ops with zero derivative
  const zeroDerivs = op === 22 /* SIGN */ || op === 23; /* NOT */
  if (zeroDerivs) {
    for (let d = 0; d < numDiff; d++) {
      body.byte(W_F64_CONST);
      body.f64(0);
      storeDeriv(d, body);
    }
    return;
  }

  // For ops where deriv = coeff * a_d, emit coeff once per d
  // (or use the formula specific to each op)
  for (let d = 0; d < numDiff; d++) {
    switch (op) {
      case 20: // NEG: -a_d
        pushAd(body, d);
        body.byte(0x9a); // f64.neg
        break;
      case 26: // DEBUG: a_d
      case OP_ASSERT_ZERO:
      case OP_ASSERT_NONZERO:
        pushAd(body, d);
        break;
      case 19: // ABS: sign(a) * a_d
        pushA(body);
        body.byte(W_F64_CONST);
        body.f64(0);
        body.byte(0x64); // f64.gt
        pushA(body);
        body.byte(W_F64_CONST);
        body.f64(0);
        body.byte(0x63); // f64.lt
        body.byte(0x6b); // i32.sub
        body.byte(0xb7); // f64.convert_i32_s
        pushAd(body, d);
        body.byte(0xa2); // f64.mul
        break;
      case 10: // SQRT: 0.5/sqrt(a) * a_d
        body.byte(W_F64_CONST);
        body.f64(0.5);
        pushA(body);
        body.byte(0x9f); // f64.sqrt
        body.byte(0xa3); // f64.div
        pushAd(body, d);
        body.byte(0xa2); // f64.mul
        break;
      case 11: // CBRT: 1/(3*cbrt(a)^2) * a_d
        body.byte(W_F64_CONST);
        body.f64(1);
        body.byte(W_F64_CONST);
        body.f64(3);
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("cbrt"));
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("cbrt"));
        body.byte(0xa2);
        body.byte(0xa2);
        body.byte(0xa3); // mul, mul, div
        pushAd(body, d);
        body.byte(0xa2); // f64.mul
        break;
      case 21: // SIN: cos(a) * a_d
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("cos"));
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 12: // COS: -sin(a) * a_d
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("sin"));
        body.byte(0x9a); // f64.neg
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 15: // TAN: 1/cos(a)^2 * a_d
        body.byte(W_F64_CONST);
        body.f64(1);
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("cos"));
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("cos"));
        body.byte(0xa2);
        body.byte(0xa3); // mul, div
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 14: // ASIN: 1/sqrt(1-a^2) * a_d
        body.byte(W_F64_CONST);
        body.f64(1);
        body.byte(W_F64_CONST);
        body.f64(1);
        pushA(body);
        pushA(body);
        body.byte(0xa2); // a*a
        body.byte(0xa1);
        body.byte(0x9f);
        body.byte(0xa3); // sub, sqrt, div
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 13: // ACOS: -1/sqrt(1-a^2) * a_d
        body.byte(W_F64_CONST);
        body.f64(-1);
        body.byte(W_F64_CONST);
        body.f64(1);
        pushA(body);
        pushA(body);
        body.byte(0xa2);
        body.byte(0xa1);
        body.byte(0x9f);
        body.byte(0xa3);
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 16: // ATAN: 1/(1+a^2) * a_d
        body.byte(W_F64_CONST);
        body.f64(1);
        body.byte(W_F64_CONST);
        body.f64(1);
        pushA(body);
        pushA(body);
        body.byte(0xa2); // a*a
        body.byte(0xa0);
        body.byte(0xa3); // add, div
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 18: // EXP: exp(a) * a_d
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("exp"));
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 17: // LOG: 1/a * a_d
        body.byte(W_F64_CONST);
        body.f64(1);
        pushA(body);
        body.byte(0xa3); // div
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 25: // LOG1P: 1/(1+a) * a_d
        body.byte(W_F64_CONST);
        body.f64(1);
        body.byte(W_F64_CONST);
        body.f64(1);
        pushA(body);
        body.byte(0xa0);
        body.byte(0xa3); // add, div
        pushAd(body, d);
        body.byte(0xa2);
        break;
      case 24: // TANH: (1-tanh(a)^2) * a_d
        body.byte(W_F64_CONST);
        body.f64(1);
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("tanh"));
        pushA(body);
        body.byte(W_CALL);
        body.u32(getCallIdx("tanh"));
        body.byte(0xa2);
        body.byte(0xa1); // mul, sub
        pushAd(body, d);
        body.byte(0xa2);
        break;
      default:
        body.byte(W_F64_CONST);
        body.f64(0);
        break;
    }
    storeDeriv(d, body);
  }
}

function emitGradBinaryDerivs(
  body: WasmBuilder,
  op: number,
  numDiff: number,
  pushAd: (w: WasmBuilder, d: number) => void,
  pushBd: (w: WasmBuilder, d: number) => void,
  pushA: (w: WasmBuilder) => void,
  pushB: (w: WasmBuilder) => void,
  storeDeriv: (d: number, w: WasmBuilder) => void,
): void {
  // Zero-derivative binary ops
  if (op === 48 /* COMPARE */) {
    for (let d = 0; d < numDiff; d++) {
      body.byte(W_F64_CONST);
      body.f64(0);
      storeDeriv(d, body);
    }
    return;
  }

  for (let d = 0; d < numDiff; d++) {
    switch (op) {
      case 40: // ADD: a_d + b_d
        pushAd(body, d);
        pushBd(body, d);
        body.byte(0xa0); // f64.add
        break;
      case 41: // SUB: a_d - b_d
        pushAd(body, d);
        pushBd(body, d);
        body.byte(0xa1); // f64.sub
        break;
      case 42: // MUL: a_d*b + a*b_d
        pushAd(body, d);
        pushB(body);
        body.byte(0xa2);
        pushA(body);
        pushBd(body, d);
        body.byte(0xa2);
        body.byte(0xa0);
        break;
      case 43: // DIV: (a_d*b - a*b_d) / b^2
        pushAd(body, d);
        pushB(body);
        body.byte(0xa2);
        pushA(body);
        pushBd(body, d);
        body.byte(0xa2);
        body.byte(0xa1);
        pushB(body);
        pushB(body);
        body.byte(0xa2);
        body.byte(0xa3);
        break;
      case 44: // MOD: a_d
        pushAd(body, d);
        break;
      case 45: // ATAN2: (b*a_d - a*b_d) / (a^2+b^2)
        pushB(body);
        pushAd(body, d);
        body.byte(0xa2);
        pushA(body);
        pushBd(body, d);
        body.byte(0xa2);
        body.byte(0xa1);
        pushA(body);
        pushA(body);
        body.byte(0xa2);
        pushB(body);
        pushB(body);
        body.byte(0xa2);
        body.byte(0xa0);
        body.byte(0xa3);
        break;
      case 46: // MIN: a<=b ? a_d : b_d
        pushAd(body, d);
        pushBd(body, d);
        pushA(body);
        pushB(body);
        body.byte(0x65); // f64.le
        body.byte(0x1b); // select
        break;
      case 47: // MAX: a>=b ? a_d : b_d
        pushAd(body, d);
        pushBd(body, d);
        pushA(body);
        pushB(body);
        body.byte(0x66); // f64.ge
        body.byte(0x1b); // select
        break;
      case 49: // AND: a==0 ? a_d : b_d
        pushAd(body, d);
        pushBd(body, d);
        pushA(body);
        body.byte(W_F64_CONST);
        body.f64(0);
        body.byte(0x61); // f64.eq
        body.byte(0x1b);
        break;
      case 50: // OR: a==0 ? b_d : a_d
        pushBd(body, d);
        pushAd(body, d);
        pushA(body);
        body.byte(W_F64_CONST);
        body.f64(0);
        body.byte(0x61);
        body.byte(0x1b);
        break;
      default:
        body.byte(W_F64_CONST);
        body.f64(0);
        break;
    }
    storeDeriv(d, body);
  }
}

export type WasmJvpFn = (
  values: Float64Array,
  seeds: Float64Array,
) => { vals: number[]; tangents: number[][] };

function emitJvpAndWrap(
  tape: CompiledTape,
  numDirections: number,
  maxChunkBytes: number,
  maxFunctionLocals: number,
): WasmJvpFn {
  if (!Number.isInteger(numDirections) || numDirections < 0) {
    throw new Error(
      `seeded JVP direction count must be a non-negative integer, got ${numDirections}`,
    );
  }
  const stride = 1 + numDirections;
  const numVars = tape.varSlots.length;

  // Derivative emission is approximately proportional to stride for both
  // body bytes and locals.
  const chunks = planChunks(
    tape,
    Math.floor(maxChunkBytes / stride),
    maxFunctionLocals,
    stride,
  );
  const { escapeSlot } = computeEscapes(tape, chunks);

  // Every root must escape so the wrapper can read multi-root results.
  let nextEscape = 0;
  for (let i = 0; i < tape.opcodes.length; i++) {
    if (escapeSlot[i]! >= 0) {
      nextEscape = Math.max(nextEscape, escapeSlot[i]! + 1);
    }
  }
  for (const root of tape.rootIndices) {
    if (escapeSlot[root]! < 0) escapeSlot[root] = nextEscape++;
  }

  // Escaped duals occupy the first region; arbitrary seeds follow them.
  const seedBaseOffset = nextEscape * stride * 8;
  const bytesNeeded = Math.max(1, seedBaseOffset + numVars * numDirections * 8);
  const numPages = Math.max(1, Math.ceil(bytesNeeded / 65536));

  const usedImportNames = collectUsedImports(tape);
  collectGradExtraImports(tape, usedImportNames);
  const importNames: string[] = [];
  const importFuncIndex: Record<string, number> = {};
  for (const def of ALL_IMPORTS) {
    if (usedImportNames.has(def.name)) {
      importFuncIndex[def.name] = importNames.length;
      importNames.push(def.name);
    }
  }
  const numImports = importNames.length;
  const getCallIdx = (name: string) => importFuncIndex[name]!;
  const firstChunkFuncIdx = numImports;
  const entryFuncIdx = numImports + chunks.length;

  const typeSec = new WasmBuilder();
  const hasUnary = importNames.some(
    (name) => ALL_IMPORTS[IMPORT_INDEX[name]!]!.paramCount === 1,
  );
  const hasBinary = importNames.some(
    (name) => ALL_IMPORTS[IMPORT_INDEX[name]!]!.paramCount === 2,
  );
  const funcTypeIdx = 0;
  let typeCount = 1;
  const unaryTypeIdx = hasUnary ? typeCount++ : -1;
  const binaryTypeIdx = hasBinary ? typeCount++ : -1;

  typeSec.u32(typeCount);
  typeSec.byte(0x60);
  typeSec.u32(numVars);
  for (let i = 0; i < numVars; i++) typeSec.byte(VT_F64);
  typeSec.u32(0);
  if (hasUnary) {
    typeSec.byte(0x60);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
  }
  if (hasBinary) {
    typeSec.byte(0x60);
    typeSec.u32(2);
    typeSec.byte(VT_F64);
    typeSec.byte(VT_F64);
    typeSec.u32(1);
    typeSec.byte(VT_F64);
  }

  const importSec = new WasmBuilder();
  importSec.u32(numImports);
  for (const name of importNames) {
    const def = ALL_IMPORTS[IMPORT_INDEX[name]!]!;
    importSec.str("env");
    importSec.str(name);
    importSec.byte(0x00);
    importSec.u32(def.paramCount === 1 ? unaryTypeIdx : binaryTypeIdx);
  }

  const funcSec = new WasmBuilder();
  funcSec.u32(chunks.length + 1);
  for (let c = 0; c < chunks.length; c++) funcSec.u32(funcTypeIdx);
  funcSec.u32(funcTypeIdx);

  const memSec = new WasmBuilder();
  memSec.u32(1);
  memSec.byte(0x00);
  memSec.u32(numPages);

  const exportSec = new WasmBuilder();
  exportSec.u32(2);
  exportSec.str("eval");
  exportSec.byte(0x00);
  exportSec.u32(entryFuncIdx);
  exportSec.str("mem");
  exportSec.byte(0x02);
  exportSec.u32(0);

  const chunkBodies = chunks.map((chunk) =>
    emitGradChunkBody(
      tape,
      chunk,
      numVars,
      numDirections,
      stride,
      seedBaseOffset,
      escapeSlot,
      getCallIdx,
    ),
  );
  const entryBody = (() => {
    const body = new WasmBuilder(64);
    body.u32(0);
    for (let chunk = 0; chunk < chunks.length; chunk++) {
      for (let param = 0; param < numVars; param++) {
        body.byte(W_LOCAL_GET);
        body.u32(param);
      }
      body.byte(W_CALL);
      body.u32(firstChunkFuncIdx + chunk);
    }
    body.byte(W_END);
    return body.toBytes();
  })();

  const codeSec = new WasmBuilder();
  codeSec.u32(chunks.length + 1);
  for (const body of chunkBodies) {
    codeSec.u32(body.length);
    codeSec.bytes(body);
  }
  codeSec.u32(entryBody.length);
  codeSec.bytes(entryBody);

  const mod = new WasmBuilder(codeSec.length + 4096);
  mod.bytes([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  mod.section(1, typeSec);
  if (numImports > 0) mod.section(2, importSec);
  mod.section(3, funcSec);
  mod.section(5, memSec);
  mod.section(7, exportSec);
  mod.section(10, codeSec);

  const env: Record<string, (...args: number[]) => number> = {};
  for (const name of importNames) {
    env[name] = ALL_IMPORTS[IMPORT_INDEX[name]!]!.jsFn;
  }
  const module = compileModuleBytes(mod.toBytes(), tape);
  const instance = new WebAssembly.Instance(module, { env });
  const wasmEval = instance.exports.eval as (...args: number[]) => void;
  const wasmMem = instance.exports.mem as WebAssembly.Memory;
  const args = new Array<number>(numVars).fill(0);

  return (values, seeds) => {
    if (values.length !== tape.numVars) {
      throw new Error(
        `seeded JVP expected ${tape.numVars} values, got ${values.length}`,
      );
    }
    if (seeds.length !== tape.numVars * numDirections) {
      throw new Error(
        `seeded JVP expected ${tape.numVars * numDirections} seeds, got ${seeds.length}`,
      );
    }
    args.fill(0);
    for (let input = 0; input < tape.numVars; input++)
      args[input] = values[input]!;
    const seedView = new Float64Array(
      wasmMem.buffer,
      seedBaseOffset,
      numVars * numDirections,
    );
    seedView.fill(0);
    seedView.set(seeds);
    wasmEval(...args);

    const vals = new Array<number>(tape.rootIndices.length);
    const tangents: number[][] = [];
    for (let root = 0; root < tape.rootIndices.length; root++) {
      const offset = escapeSlot[tape.rootIndices[root]!]! * stride * 8;
      const result = new Float64Array(wasmMem.buffer, offset, stride);
      vals[root] = result[0]!;
      tangents.push(Array.from(result.subarray(1)));
    }
    return { vals, tangents };
  };
}

export function compileWasmJvpFromTape(
  tape: CompiledTape,
  numDirections: number,
  options: CompileWasmOptions = {},
): WasmJvpFn {
  return emitJvpAndWrap(
    tape,
    numDirections,
    options.maxChunkBytes ?? DEFAULT_MAX_CHUNK_BYTES,
    options.maxFunctionLocals ?? DEFAULT_MAX_FUNCTION_LOCALS,
  );
}

export function compileWasmGradFromTape(
  tape: CompiledTape,
  diffVars: VarName[],
  options: CompileWasmOptions = {},
): WasmGradFn {
  const jvp = compileWasmJvpFromTape(tape, diffVars.length, options);
  const seeds = new Float64Array(tape.numVars * diffVars.length);
  for (let input = 0; input < tape.numVars; input++) {
    for (let direction = 0; direction < diffVars.length; direction++) {
      if (tape.varSlots[input] === diffVars[direction]) {
        seeds[input * diffVars.length + direction] = 1;
      }
    }
  }
  return (vars) => {
    const values = Float64Array.from(
      tape.varSlots.slice(0, tape.numVars),
      (name) => vars.get(name) ?? 0,
    );
    const result = jvp(values, seeds);
    return { val: result.vals[0]!, gradient: result.tangents[0]! };
  };
}
