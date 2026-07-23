/**
 * WASM-based tape interpreters for NumNode DAG evaluation.
 *
 * Two interpreters share a single WebAssembly.Memory and bump allocator:
 *
 *   1. **evalTape** — value-only evaluation. Each tape slot stores one f64.
 *   2. **evalGrad** — forward-mode autodiff. Each tape slot stores a dual
 *      number [value, ∂/∂x₁, …, ∂/∂xₙ].
 *
 * Both are generic WASM modules with a br_table-based interpreter loop.
 * Tape data is copied into WASM linear memory once; the modules are
 * compiled once at import time and reused across all tapes.
 */
import { bindVarSlots } from "../../../tape";
import {
  type CompiledTape,
  MAX_OPCODE,
  NUM_CASES,
  VALID_OPS,
  OP_LIT,
  OP_VAR,
} from "../../../tape";
import type { VarName } from "../../../../core/tree";
import type { GradientResult, JacobianResult } from "../../../tape";
import {
  WasmBuilder as WB,
  type GradLocals,
  W_BLOCK,
  W_LOOP,
  W_END,
  W_BR,
  W_BR_IF,
  W_BR_TABLE,
  W_LOCAL_GET,
  W_LOCAL_SET,
  W_I32_LOAD,
  W_F64_LOAD,
  W_I32_LOAD8_U,
  W_F64_STORE,
  W_I32_CONST,
  W_I32_ADD,
  W_I32_MUL,
  W_I32_SHL,
  W_I32_GE_U,
  VT_I32,
  VT_F64,
  BT_VOID,
  ALL_IMPORTS,
  emitNumOp,
  emitNumOpGrad,
} from "../wasm-builder";

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

const NUM_FUNC_IMPORTS = ALL_IMPORTS.length;

/** Map import name → function index */
const impIdx: Record<string, number> = {};
for (let i = 0; i < ALL_IMPORTS.length; i++) {
  impIdx[ALL_IMPORTS[i]!.name] = i;
}
const getCallIdx = (name: string) => impIdx[name]!;

// ---------------------------------------------------------------------------
// Shared WASM module builder
// ---------------------------------------------------------------------------
//
// Both the value-only and grad interpreters share the same structure:
// type section (unary/binary import types + function type), import section,
// func section, export section, and a br_table interpreter loop.

interface TapeModuleConfig {
  /** Name of the exported function */
  exportName: string;
  /** Number of i32 parameters */
  numParams: number;
  /** Number of i32 locals (after params) */
  numI32Locals: number;
  /** Number of f64 locals */
  numF64Locals: number;
  /** Return type: true for f64, false for void */
  returnsF64: boolean;
  /** Emit code before the main loop (e.g. stride caching). */
  emitPrologue?: (body: WB) => void;
  /**
   * Emit code before the br_table switch: load op/argA/argB into locals.
   * Receives local indices for i (loop counter), op, argA, argB.
   */
  emitIterSetup: (
    body: WB,
    locals: { L_I: number; L_OP: number; L_A: number; L_B: number },
    params: { P_LEN: number; P_OPC: number; P_AA: number; P_AB: number },
  ) => void;
  /** Emit the case body for one opcode inside the br_table. */
  emitCaseBody: (body: WB, op: number) => void;
  /** Emit code after the br_table (store result, etc.). */
  emitPostSwitch: (body: WB) => void;
  /** Emit code after the loop ends (return value, etc.). */
  emitEpilogue?: (body: WB) => void;
  /** Local indices for i, op (used for loop counter + br_table dispatch). */
  L_I: number;
  L_OP: number;
  /** Param indices for len, opc, argA, argB */
  P_LEN: number;
  P_OPC: number;
  P_AA: number;
  P_AB: number;
}

function buildTapeModule(cfg: TapeModuleConfig): Uint8Array {
  // --- Type section ---
  const typeSec = new WB();
  typeSec.u32(3);
  // type 0: f64 -> f64
  typeSec.byte(0x60);
  typeSec.u32(1);
  typeSec.byte(VT_F64);
  typeSec.u32(1);
  typeSec.byte(VT_F64);
  // type 1: (f64, f64) -> f64
  typeSec.byte(0x60);
  typeSec.u32(2);
  typeSec.byte(VT_F64);
  typeSec.byte(VT_F64);
  typeSec.u32(1);
  typeSec.byte(VT_F64);
  // type 2: function signature
  typeSec.byte(0x60);
  typeSec.u32(cfg.numParams);
  for (let i = 0; i < cfg.numParams; i++) typeSec.byte(VT_I32);
  if (cfg.returnsF64) {
    typeSec.u32(1);
    typeSec.byte(VT_F64);
  } else {
    typeSec.u32(0);
  }

  // --- Import section ---
  const importSec = new WB();
  importSec.u32(NUM_FUNC_IMPORTS + 1);
  importSec.str("env");
  importSec.str("memory");
  importSec.byte(0x02);
  importSec.byte(0x00);
  importSec.u32(1);
  for (const def of ALL_IMPORTS) {
    importSec.str("env");
    importSec.str(def.name);
    importSec.byte(0x00);
    importSec.u32(def.paramCount === 1 ? 0 : 1);
  }

  // --- Function + Export sections ---
  const funcSec = new WB();
  funcSec.u32(1);
  funcSec.u32(2);

  const exportSec = new WB();
  exportSec.u32(1);
  exportSec.str(cfg.exportName);
  exportSec.byte(0x00);
  exportSec.u32(NUM_FUNC_IMPORTS);

  // --- Code section (function body) ---
  const body = new WB();
  body.u32(2);
  body.u32(cfg.numI32Locals);
  body.byte(VT_I32);
  body.u32(cfg.numF64Locals);
  body.byte(VT_F64);

  cfg.emitPrologue?.(body);

  // Main loop: block $break { loop $loop { ... } }
  body.byte(W_BLOCK);
  body.byte(BT_VOID);
  body.byte(W_LOOP);
  body.byte(BT_VOID);

  // br_if $break if i >= len
  body.byte(W_LOCAL_GET);
  body.u32(cfg.L_I);
  body.byte(W_LOCAL_GET);
  body.u32(cfg.P_LEN);
  body.byte(W_I32_GE_U);
  body.byte(W_BR_IF);
  body.u32(1);

  // Load op, argA, argB for this iteration
  cfg.emitIterSetup(
    body,
    { L_I: cfg.L_I, L_OP: cfg.L_OP, L_A: 0, L_B: 0 },
    { P_LEN: cfg.P_LEN, P_OPC: cfg.P_OPC, P_AA: cfg.P_AA, P_AB: cfg.P_AB },
  );

  // br_table switch
  body.byte(W_BLOCK);
  body.byte(BT_VOID);
  for (let c = 0; c < NUM_CASES; c++) {
    body.byte(W_BLOCK);
    body.byte(BT_VOID);
  }
  body.byte(W_LOCAL_GET);
  body.u32(cfg.L_OP);
  body.byte(W_BR_TABLE);
  body.u32(NUM_CASES);
  for (let v = 0; v <= MAX_OPCODE; v++) {
    body.u32(VALID_OPS.has(v) ? v : NUM_CASES);
  }
  body.u32(NUM_CASES);

  for (let c = 0; c < NUM_CASES; c++) {
    body.byte(W_END);
    cfg.emitCaseBody(body, c);
    if (c < NUM_CASES - 1) {
      body.byte(W_BR);
      body.u32(NUM_CASES - c - 1);
    }
  }
  body.byte(W_END); // $after

  cfg.emitPostSwitch(body);

  // i++
  body.byte(W_LOCAL_GET);
  body.u32(cfg.L_I);
  body.byte(W_I32_CONST);
  body.i32(1);
  body.byte(W_I32_ADD);
  body.byte(W_LOCAL_SET);
  body.u32(cfg.L_I);

  body.byte(W_BR);
  body.u32(0);
  body.byte(W_END); // loop
  body.byte(W_END); // block

  cfg.emitEpilogue?.(body);

  body.byte(W_END); // function

  const codeSec = new WB();
  codeSec.u32(1);
  const bodyBytes = body.toBytes();
  codeSec.u32(bodyBytes.length);
  codeSec.bytes(bodyBytes);

  const mod = new WB();
  mod.bytes([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  mod.section(1, typeSec);
  mod.section(2, importSec);
  mod.section(3, funcSec);
  mod.section(7, exportSec);
  mod.section(10, codeSec);
  return mod.toBytes();
}

// ---------------------------------------------------------------------------
// Value-only interpreter — evalTape
// ---------------------------------------------------------------------------

// Params: all i32 offsets/lengths
const P_LEN = 0,
  P_ROOT = 1;
const P_OPC = 2,
  P_AA = 3,
  P_AB = 4,
  P_LIT = 5,
  P_VARS = 6,
  P_VAL = 7;
const EVAL_NUM_PARAMS = 8;
const L_I = 8,
  L_OP = 9,
  L_A = 10,
  L_B = 11,
  L_RES = 12;

/** Emit: f64.load(baseParam + idxLocal * 8) */
function emitLoadF64(b: WB, baseParam: number, idxLocal: number) {
  b.byte(W_LOCAL_GET);
  b.u32(baseParam);
  b.byte(W_LOCAL_GET);
  b.u32(idxLocal);
  b.byte(W_I32_CONST);
  b.i32(3);
  b.byte(W_I32_SHL);
  b.byte(W_I32_ADD);
  b.byte(W_F64_LOAD);
  b.u32(3);
  b.u32(0);
}

function emitLoadVal(b: WB, idxLocal: number) {
  emitLoadF64(b, P_VAL, idxLocal);
}

function emitEvalCaseBody(b: WB, op: number) {
  if (op === OP_LIT) {
    emitLoadF64(b, P_LIT, L_A);
    b.byte(W_LOCAL_SET);
    b.u32(L_RES);
  } else if (op === OP_VAR) {
    emitLoadF64(b, P_VARS, L_A);
    b.byte(W_LOCAL_SET);
    b.u32(L_RES);
  } else if (VALID_OPS.has(op)) {
    const pushA = (w: WB) => emitLoadVal(w, L_A);
    const pushB = (w: WB) => emitLoadVal(w, L_B);
    emitNumOp(b, op, pushA, pushB, getCallIdx);
    b.byte(W_LOCAL_SET);
    b.u32(L_RES);
  }
}

/** Emit: load op, argA, argB from tape arrays into locals */
function emitStandardIterSetup(
  body: WB,
  iLocal: number,
  opLocal: number,
  aLocal: number,
  bLocal: number,
  opcParam: number,
  aaParam: number,
  abParam: number,
) {
  // op = i32.load8_u(opcOff + i)
  body.byte(W_LOCAL_GET);
  body.u32(opcParam);
  body.byte(W_LOCAL_GET);
  body.u32(iLocal);
  body.byte(W_I32_ADD);
  body.byte(W_I32_LOAD8_U);
  body.u32(0);
  body.u32(0);
  body.byte(W_LOCAL_SET);
  body.u32(opLocal);

  // aIdx = i32.load(argAOff + i * 4)
  body.byte(W_LOCAL_GET);
  body.u32(aaParam);
  body.byte(W_LOCAL_GET);
  body.u32(iLocal);
  body.byte(W_I32_CONST);
  body.i32(2);
  body.byte(W_I32_SHL);
  body.byte(W_I32_ADD);
  body.byte(W_I32_LOAD);
  body.u32(2);
  body.u32(0);
  body.byte(W_LOCAL_SET);
  body.u32(aLocal);

  // bIdx = i32.load(argBOff + i * 4)
  body.byte(W_LOCAL_GET);
  body.u32(abParam);
  body.byte(W_LOCAL_GET);
  body.u32(iLocal);
  body.byte(W_I32_CONST);
  body.i32(2);
  body.byte(W_I32_SHL);
  body.byte(W_I32_ADD);
  body.byte(W_I32_LOAD);
  body.u32(2);
  body.u32(0);
  body.byte(W_LOCAL_SET);
  body.u32(bLocal);
}

function emitEvalPostSwitch(body: WB): void {
  // values[i] = result
  body.byte(W_LOCAL_GET);
  body.u32(P_VAL);
  body.byte(W_LOCAL_GET);
  body.u32(L_I);
  body.byte(W_I32_CONST);
  body.i32(3);
  body.byte(W_I32_SHL);
  body.byte(W_I32_ADD);
  body.byte(W_LOCAL_GET);
  body.u32(L_RES);
  body.byte(W_F64_STORE);
  body.u32(3);
  body.u32(0);
}

function buildEvalModule(): Uint8Array {
  return buildTapeModule({
    exportName: "evalTape",
    numParams: EVAL_NUM_PARAMS,
    numI32Locals: 4,
    numF64Locals: 1,
    returnsF64: true,
    P_LEN,
    P_OPC,
    P_AA,
    P_AB,
    L_I,
    L_OP,
    emitIterSetup: (body) => {
      emitStandardIterSetup(body, L_I, L_OP, L_A, L_B, P_OPC, P_AA, P_AB);
    },
    emitCaseBody: emitEvalCaseBody,
    emitPostSwitch: emitEvalPostSwitch,
    emitEpilogue: (body) => {
      // Return values[rootIdx]
      body.byte(W_LOCAL_GET);
      body.u32(P_VAL);
      body.byte(W_LOCAL_GET);
      body.u32(P_ROOT);
      body.byte(W_I32_CONST);
      body.i32(3);
      body.byte(W_I32_SHL);
      body.byte(W_I32_ADD);
      body.byte(W_F64_LOAD);
      body.u32(3);
      body.u32(0);
    },
  });
}

// ---------------------------------------------------------------------------
// Forward-mode autodiff interpreter — evalGrad
// ---------------------------------------------------------------------------

const GRAD_NUM_PARAMS = 11;

const GRAD_LOCALS: GradLocals = {
  P_VAL: 7,
  P_LIT: 5,
  P_VARS: 6,
  P_NUMDIFF: 9,
  P_SEEDS: 10,
  L_BASE: GRAD_NUM_PARAMS + 4,
  L_ABASE: GRAD_NUM_PARAMS + 5,
  L_BBASE: GRAD_NUM_PARAMS + 6,
  L_AIDX: GRAD_NUM_PARAMS + 2,
  L_BIDX: GRAD_NUM_PARAMS + 3,
  L_STRIDEBYTES: GRAD_NUM_PARAMS + 8,
  L_D: GRAD_NUM_PARAMS + 7,
  L_VA: GRAD_NUM_PARAMS + 9,
  L_VB: GRAD_NUM_PARAMS + 10,
  L_COEFF: GRAD_NUM_PARAMS + 11,
};

const GRAD_L_I = GRAD_NUM_PARAMS;
const GRAD_L_OP = GRAD_NUM_PARAMS + 1;
const GRAD_P_STRIDE = 8;

function buildGradModule(): Uint8Array {
  return buildTapeModule({
    exportName: "evalGrad",
    numParams: GRAD_NUM_PARAMS,
    numI32Locals: 9,
    numF64Locals: 4,
    returnsF64: false,
    P_LEN: 0,
    P_OPC: 2,
    P_AA: 3,
    P_AB: 4,
    L_I: GRAD_L_I,
    L_OP: GRAD_L_OP,
    emitPrologue: (body) => {
      // L_STRIDEBYTES = P_STRIDE * 8
      body.byte(W_LOCAL_GET);
      body.u32(GRAD_P_STRIDE);
      body.byte(W_I32_CONST);
      body.i32(3);
      body.byte(W_I32_SHL);
      body.byte(W_LOCAL_SET);
      body.u32(GRAD_LOCALS.L_STRIDEBYTES);
    },
    emitIterSetup: (body) => {
      // L_BASE = L_I * L_STRIDEBYTES
      body.byte(W_LOCAL_GET);
      body.u32(GRAD_L_I);
      body.byte(W_LOCAL_GET);
      body.u32(GRAD_LOCALS.L_STRIDEBYTES);
      body.byte(W_I32_MUL);
      body.byte(W_LOCAL_SET);
      body.u32(GRAD_LOCALS.L_BASE);

      emitStandardIterSetup(
        body,
        GRAD_L_I,
        GRAD_L_OP,
        GRAD_LOCALS.L_AIDX,
        GRAD_LOCALS.L_BIDX,
        2,
        3,
        4,
      );
    },
    emitCaseBody: (body, op) => {
      emitNumOpGrad(body, op, GRAD_LOCALS, getCallIdx);
    },
    emitPostSwitch: () => {
      // grad interpreter stores results inline in emitNumOpGrad
    },
  });
}

// ---------------------------------------------------------------------------
// Shared WASM memory and singleton instances
// ---------------------------------------------------------------------------

let sharedMemory: WebAssembly.Memory | null = null;
let wasmEvalTape: ((...args: number[]) => number) | null = null;
let wasmEvalGrad: ((...args: number[]) => number) | null = null;
let memoryOffset = 0;

function align8(n: number) {
  return (n + 7) & ~7;
}
function align4(n: number) {
  return (n + 3) & ~3;
}

function instantiateWasm(bytes: Uint8Array): WebAssembly.Instance {
  const module = new WebAssembly.Module(bytes as unknown as BufferSource);
  const env: WebAssembly.ModuleImports = { memory: sharedMemory! };
  for (const def of ALL_IMPORTS) env[def.name] = def.jsFn;
  return new WebAssembly.Instance(module, { env });
}

function ensureEvalTape() {
  if (wasmEvalTape) return;

  sharedMemory = new WebAssembly.Memory({ initial: 64 });
  memoryOffset = 0;

  const instance = instantiateWasm(buildEvalModule());
  wasmEvalTape = instance.exports.evalTape as (...args: number[]) => number;

  // Warm up: run a tiny tape to force Turbofan compilation at load time.
  const buf = sharedMemory.buffer;
  new Uint8Array(buf)[0] = 0; // OP_LIT
  new Int32Array(buf)[2] = 0;
  new Int32Array(buf)[4] = 0;
  new Float64Array(buf)[3] = 0;
  wasmEvalTape(1, 0, 0, 8, 16, 24, 0, 32);
  new Float64Array(buf)[4] = 0;
}

function ensureEvalGrad() {
  if (wasmEvalGrad) return;
  ensureEvalTape();
  const instance = instantiateWasm(buildGradModule());
  wasmEvalGrad = instance.exports.evalGrad as (...args: number[]) => number;
}

// Eager initialization — precompile the value-only interpreter.
ensureEvalTape();

function ensureMemory(needed: number) {
  const mem = sharedMemory!;
  const available = mem.buffer.byteLength - memoryOffset;
  if (needed > available) {
    const pagesNeeded = Math.ceil((needed - available) / 65536);
    mem.grow(pagesNeeded);
  }
}

// ---------------------------------------------------------------------------
// Copy tape into WASM memory
// ---------------------------------------------------------------------------

interface TapeLayout {
  len: number;
  rootIdx: number;
  opcodesOff: number;
  argAOff: number;
  argBOff: number;
  literalsOff: number;
  varsOff: number;
  valuesOff: number;
  varSlots: VarName[];
  numVars: number;
}

function loadTapeIntoMemory(tape: CompiledTape): TapeLayout {
  ensureEvalTape();

  const N = tape.opcodes.length;
  const L = tape.literals.length;
  const V = tape.varSlots.length;

  const opcodesSize = align4(N);
  const argSize = N * 4;
  const totalSize = align8(opcodesSize + argSize * 2) + (L + V + N) * 8;

  ensureMemory(totalSize);

  const base = memoryOffset;
  memoryOffset += totalSize;

  const opcodesOff = base;
  const argAOff = opcodesOff + opcodesSize;
  const argBOff = argAOff + argSize;
  const literalsOff = align8(argBOff + argSize);
  const varsOff = literalsOff + L * 8;
  const valuesOff = varsOff + V * 8;

  const buf = sharedMemory!.buffer;
  new Uint8Array(buf, opcodesOff, N).set(tape.opcodes);
  new Int32Array(buf, argAOff, N).set(tape.argA);
  new Int32Array(buf, argBOff, N).set(tape.argB);
  new Float64Array(buf, literalsOff, L).set(tape.literals);

  return {
    len: N,
    rootIdx: tape.rootIndices[0],
    opcodesOff,
    argAOff,
    argBOff,
    literalsOff,
    varsOff,
    valuesOff,
    varSlots: tape.varSlots,
    numVars: tape.numVars,
  };
}

/** Write variable values from a Map into the vars region of WASM memory. */
function writeVarsToMemory(
  varsOff: number,
  varSlots: VarName[],
  numVars: number,
  vars: Map<VarName, number>,
  derivatives?: Map<VarName, number>,
) {
  const f64View = new Float64Array(
    sharedMemory!.buffer,
    varsOff,
    varSlots.length,
  );
  bindVarSlots(varSlots, numVars, vars, derivatives, f64View);
}

// ---------------------------------------------------------------------------
// interpretTapeValues — array-shaped public entry point.
//
// Same signature as `js-interp/tape-eval.ts:interpretTapeValues`, so
// the two backends are interchangeable from the caller's perspective.
// Used by LiveTape's wasm sweep backend (live-tape-wasm-interp-sweep.ts).
//
// State note: per-opcodes-array layouts are cached in a WeakMap so
// that repeated calls with the same growable buffers reuse one wasm
// memory region. The bump allocator never frees, so each fresh
// `opcodes` Uint8Array (e.g., after the live tape doubles its
// capacity) leaks the previous layout's region. Acceptable for the
// expected workload (tapes that grow then stabilize); a smarter
// allocator is future work.
// ---------------------------------------------------------------------------

interface InterpLayout {
  opcodesOff: number;
  argAOff: number;
  argBOff: number;
  literalsOff: number;
  varsOff: number;
  valuesOff: number;
  opcodesCapacity: number;
  literalsCapacity: number;
  varsCapacity: number;
}

const interpLayoutCache = new WeakMap<Uint8Array, InterpLayout>();

function allocateInterpLayout(N: number, L: number, V: number): InterpLayout {
  const opcodesSize = align4(N);
  const argSize = N * 4;
  const totalSize = align8(opcodesSize + argSize * 2) + (L + V + N) * 8;
  ensureMemory(totalSize);
  const base = memoryOffset;
  memoryOffset += totalSize;
  const opcodesOff = base;
  const argAOff = opcodesOff + opcodesSize;
  const argBOff = argAOff + argSize;
  const literalsOff = align8(argBOff + argSize);
  const varsOff = literalsOff + L * 8;
  const valuesOff = varsOff + V * 8;
  return {
    opcodesOff,
    argAOff,
    argBOff,
    literalsOff,
    varsOff,
    valuesOff,
    opcodesCapacity: N,
    literalsCapacity: L,
    varsCapacity: V,
  };
}

export function interpretTapeValues(
  opcodes: Uint8Array,
  argA: Int32Array,
  argB: Int32Array,
  literals: Float64Array,
  len: number,
  varValues: Float64Array,
  values: Float64Array,
): void {
  ensureEvalTape();
  if (len === 0) return;

  let layout = interpLayoutCache.get(opcodes);
  if (
    !layout ||
    layout.opcodesCapacity < opcodes.length ||
    layout.literalsCapacity < literals.length ||
    layout.varsCapacity < varValues.length
  ) {
    layout = allocateInterpLayout(
      opcodes.length,
      literals.length,
      varValues.length,
    );
    interpLayoutCache.set(opcodes, layout);
  }

  // Copy structural state every call: the live tape may have appended
  // entries since the cache was populated. The valid range is [0, len);
  // beyond that the source arrays may hold garbage.
  const buf = sharedMemory!.buffer;
  new Uint8Array(buf, layout.opcodesOff, len).set(opcodes.subarray(0, len));
  new Int32Array(buf, layout.argAOff, len).set(argA.subarray(0, len));
  new Int32Array(buf, layout.argBOff, len).set(argB.subarray(0, len));
  new Float64Array(buf, layout.literalsOff, literals.length).set(literals);
  new Float64Array(buf, layout.varsOff, varValues.length).set(varValues);

  wasmEvalTape!(
    len,
    len - 1,
    layout.opcodesOff,
    layout.argAOff,
    layout.argBOff,
    layout.literalsOff,
    layout.varsOff,
    layout.valuesOff,
  );

  const valuesView = new Float64Array(buf, layout.valuesOff, len);
  values.set(valuesView.subarray(0, len));
}

// ---------------------------------------------------------------------------
// Value-only — public API
// ---------------------------------------------------------------------------

export type CompiledMultiFn = (
  vars: Map<VarName, number>,
  derivatives?: Map<VarName, number>,
) => number[];

export function compileWasmTapeFromTape(tape: CompiledTape): CompiledMultiFn {
  const rootIndices = tape.rootIndices;
  const numRoots = rootIndices.length;
  const layout = loadTapeIntoMemory(tape);
  const fn = wasmEvalTape!;
  const {
    len,
    opcodesOff,
    argAOff,
    argBOff,
    literalsOff,
    varsOff,
    valuesOff,
    varSlots,
    numVars: layoutNumVars,
  } = layout;

  if (numRoots === 1) {
    // Single-root: use the WASM return value directly.
    return (vars, derivatives) => {
      writeVarsToMemory(varsOff, varSlots, layoutNumVars, vars, derivatives);
      return [
        fn(
          len,
          layout.rootIdx,
          opcodesOff,
          argAOff,
          argBOff,
          literalsOff,
          varsOff,
          valuesOff,
        ) as number,
      ];
    };
  }

  // Multi-root: run the interpreter up to the last root, then read all roots
  // from the values buffer.
  const lastRootIdx = rootIndices[rootIndices.length - 1]!;

  return (vars, derivatives) => {
    writeVarsToMemory(varsOff, varSlots, layoutNumVars, vars, derivatives);
    fn(
      len,
      lastRootIdx,
      opcodesOff,
      argAOff,
      argBOff,
      literalsOff,
      varsOff,
      valuesOff,
    );

    const values = new Float64Array(sharedMemory!.buffer, valuesOff, len);
    const result = new Array<number>(numRoots);
    for (let i = 0; i < numRoots; i++) {
      result[i] = values[rootIndices[i]!]!;
    }
    return result;
  };
}

// ---------------------------------------------------------------------------
// Forward-mode autodiff — public API
// ---------------------------------------------------------------------------

export type WasmForwardAutodiffFn = (
  vars: Map<VarName, number>,
) => GradientResult;

export type WasmSeededJvpFn = (
  values: Float64Array,
  seeds: Float64Array,
) => { vals: number[]; tangents: number[][] };

/** Compile a direct arbitrary-seed, multi-root JVP in the WASM interpreter. */
export function compileWasmSeededJvp(
  tape: CompiledTape,
  numDirections: number,
): WasmSeededJvpFn {
  if (!Number.isInteger(numDirections) || numDirections < 0) {
    throw new Error(
      `seeded JVP direction count must be a non-negative integer, got ${numDirections}`,
    );
  }
  ensureEvalGrad();

  const N = tape.opcodes.length;
  const L = tape.literals.length;
  const V = tape.varSlots.length;
  const stride = 1 + numDirections;
  const opcodesSize = align4(N);
  const argSize = N * 4;
  const seedsSize = V * numDirections * 8;
  const fixedSize =
    align8(opcodesSize + argSize * 2) + L * 8 + V * 8 + seedsSize;
  const totalSize = align8(fixedSize) + N * stride * 8;

  ensureMemory(totalSize);
  const base = memoryOffset;
  memoryOffset += totalSize;
  const opcodesOff = base;
  const argAOff = opcodesOff + opcodesSize;
  const argBOff = argAOff + argSize;
  const literalsOff = align8(argBOff + argSize);
  const varsOff = literalsOff + L * 8;
  const seedsOff = varsOff + V * 8;
  const valuesOff = align8(seedsOff + seedsSize);

  const buf = sharedMemory!.buffer;
  new Uint8Array(buf, opcodesOff, N).set(tape.opcodes);
  new Int32Array(buf, argAOff, N).set(tape.argA);
  new Int32Array(buf, argBOff, N).set(tape.argB);
  new Float64Array(buf, literalsOff, L).set(tape.literals);

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
    const current = sharedMemory!.buffer;
    const varsView = new Float64Array(current, varsOff, V);
    varsView.fill(0);
    varsView.set(values);
    const seedsView = new Float64Array(current, seedsOff, V * numDirections);
    seedsView.fill(0);
    seedsView.set(seeds);

    wasmEvalGrad!(
      N,
      tape.rootIndices[0],
      opcodesOff,
      argAOff,
      argBOff,
      literalsOff,
      varsOff,
      valuesOff,
      stride,
      numDirections,
      seedsOff,
    );

    const vals = new Array<number>(tape.rootIndices.length);
    const tangents: number[][] = [];
    for (let root = 0; root < tape.rootIndices.length; root++) {
      const result = new Float64Array(
        current,
        valuesOff + tape.rootIndices[root]! * stride * 8,
        stride,
      );
      vals[root] = result[0]!;
      tangents.push(Array.from(result.subarray(1)));
    }
    return { vals, tangents };
  };
}

export function compileWasmForwardAutodiff(
  tape: CompiledTape,
  diffVars: VarName[],
): WasmForwardAutodiffFn {
  const jvp = compileWasmSeededJvp(tape, diffVars.length);
  const inputs = compileIdentityInputs(tape, diffVars);
  return (vars) => {
    const result = jvp(inputs.values(vars), inputs.seeds);
    return { val: result.vals[0]!, gradient: result.tangents[0]! };
  };
}

export type WasmForwardAutodiffMultiFn = (
  vars: Map<VarName, number>,
) => JacobianResult;

export function compileWasmForwardAutodiffMulti(
  tape: CompiledTape,
  diffVars: VarName[],
): WasmForwardAutodiffMultiFn {
  const jvp = compileWasmSeededJvp(tape, diffVars.length);
  const inputs = compileIdentityInputs(tape, diffVars);
  return (vars) => {
    const result = jvp(inputs.values(vars), inputs.seeds);
    return { vals: result.vals, jacobian: result.tangents };
  };
}

function compileIdentityInputs(
  tape: CompiledTape,
  diffVars: readonly VarName[],
): {
  values(vars: ReadonlyMap<VarName, number>): Float64Array;
  seeds: Float64Array;
} {
  const seeds = new Float64Array(tape.numVars * diffVars.length);
  for (let input = 0; input < tape.numVars; input++) {
    for (let direction = 0; direction < diffVars.length; direction++) {
      if (tape.varSlots[input] === diffVars[direction]) {
        seeds[input * diffVars.length + direction] = 1;
      }
    }
  }
  return {
    values: (vars) =>
      Float64Array.from(
        tape.varSlots.slice(0, tape.numVars),
        (name) => vars.get(name) ?? 0,
      ),
    seeds,
  };
}
