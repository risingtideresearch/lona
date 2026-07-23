/**
 * Shared WASM binary builder, instruction opcodes, import definitions,
 * and operation emission helpers.
 *
 * Used by wasm-eval.ts (per-DAG WASM codegen) and wasm-tape-eval.ts
 * (generic WASM tape interpreter).
 */
import {
  assertTapeValue,
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
  DIV_BY_ZERO_FALLBACK,
  VALID_OPS,
} from "../../tape";

// ---------------------------------------------------------------------------
// WASM binary byte builder
// ---------------------------------------------------------------------------

export class WasmBuilder {
  private buf: Uint8Array;
  private view: DataView;
  private pos: number = 0;

  constructor(initialCapacity: number = 4096) {
    this.buf = new Uint8Array(initialCapacity);
    this.view = new DataView(this.buf.buffer);
  }

  get length() {
    return this.pos;
  }

  private grow(needed: number) {
    let newSize = this.buf.length * 2;
    while (this.pos + needed > newSize) newSize *= 2;
    const newBuf = new Uint8Array(newSize);
    newBuf.set(this.buf);
    this.buf = newBuf;
    this.view = new DataView(newBuf.buffer);
  }

  byte(b: number) {
    if (this.pos + 1 > this.buf.length) this.grow(1);
    this.buf[this.pos++] = b & 0xff;
  }

  bytes(arr: ArrayLike<number>) {
    const len = arr.length;
    if (this.pos + len > this.buf.length) this.grow(len);
    if (arr instanceof Uint8Array) {
      this.buf.set(arr, this.pos);
      this.pos += len;
    } else {
      for (let i = 0; i < len; i++) this.buf[this.pos++] = arr[i]!;
    }
  }

  /** Unsigned LEB128 (max 5 bytes for u32) */
  u32(n: number) {
    if (this.pos + 5 > this.buf.length) this.grow(5);
    do {
      let b = n & 0x7f;
      n >>>= 7;
      if (n !== 0) b |= 0x80;
      this.buf[this.pos++] = b;
    } while (n !== 0);
  }

  /** Signed LEB128 (max 5 bytes for i32) */
  i32(n: number) {
    if (this.pos + 5 > this.buf.length) this.grow(5);
    n |= 0;
    while (true) {
      const b = n & 0x7f;
      n >>= 7;
      if ((n === 0 && (b & 0x40) === 0) || (n === -1 && (b & 0x40) !== 0)) {
        this.buf[this.pos++] = b;
        return;
      }
      this.buf[this.pos++] = b | 0x80;
    }
  }

  /** IEEE 754 f64, little-endian */
  f64(v: number) {
    if (this.pos + 8 > this.buf.length) this.grow(8);
    this.view.setFloat64(this.pos, v, true);
    this.pos += 8;
  }

  /** Length-prefixed string (UTF-8) */
  str(s: string) {
    const enc = new TextEncoder().encode(s);
    this.u32(enc.length);
    this.bytes(enc);
  }

  /** Emit a section: id byte + length-prefixed content */
  section(id: number, content: WasmBuilder) {
    this.byte(id);
    const data = content.toBytes();
    this.u32(data.length);
    this.bytes(data);
  }

  /**
   * Returns a Uint8Array view of the written bytes (no copy).
   * The returned view references the internal buffer — do not hold on
   * to it after further writes to the builder.
   */
  toBytes(): Uint8Array {
    return this.buf.subarray(0, this.pos);
  }
}

// ---------------------------------------------------------------------------
// WASM instruction opcodes
// ---------------------------------------------------------------------------

// Control flow
export const W_BLOCK = 0x02;
export const W_LOOP = 0x03;
export const W_IF = 0x04;
export const W_ELSE = 0x05;
export const W_END = 0x0b;
export const W_BR = 0x0c;
export const W_BR_IF = 0x0d;
export const W_BR_TABLE = 0x0e;

// Calls
export const W_CALL = 0x10;

// Variable access
export const W_LOCAL_GET = 0x20;
export const W_LOCAL_SET = 0x21;
export const W_LOCAL_TEE = 0x22;

// Memory
export const W_I32_LOAD = 0x28;
export const W_F64_LOAD = 0x2b;
export const W_I32_LOAD8_U = 0x2d;
export const W_F64_STORE = 0x39;

// Constants
export const W_I32_CONST = 0x41;
export const W_F64_CONST = 0x44;

// i32 comparison / arithmetic
export const W_I32_LT_U = 0x49;
export const W_I32_GE_U = 0x4f;
export const W_I32_SUB = 0x6b;
export const W_I32_ADD = 0x6a;
export const W_I32_MUL = 0x6c;
export const W_I32_SHL = 0x74;

// Select
export const W_SELECT = 0x1b;

// f64 comparison
export const W_F64_EQ = 0x61;
export const W_F64_NE = 0x62;
export const W_F64_LT = 0x63;
export const W_F64_GT = 0x64;
export const W_F64_LE = 0x65;
export const W_F64_GE = 0x66;

// f64 arithmetic
export const W_F64_ABS = 0x99;
export const W_F64_NEG = 0x9a;
export const W_F64_SQRT = 0x9f;
export const W_F64_ADD = 0xa0;
export const W_F64_SUB = 0xa1;
export const W_F64_MUL = 0xa2;
export const W_F64_DIV = 0xa3;
export const W_F64_MIN = 0xa4;
export const W_F64_MAX = 0xa5;

// Conversions
export const W_F64_CONVERT_I32_S = 0xb7;

// Value types / block types
export const VT_I32 = 0x7f;
export const VT_F64 = 0x7c;
export const BT_VOID = 0x40;
export const BT_F64 = 0x7c;

// ---------------------------------------------------------------------------
// WASM import definitions (JS math functions callable from WASM)
// ---------------------------------------------------------------------------

export interface ImportDef {
  name: string;
  paramCount: 1 | 2;
  jsFn: (...args: number[]) => number;
}

export const ALL_IMPORTS: ImportDef[] = [
  { name: "sin", paramCount: 1, jsFn: Math.sin },
  { name: "cos", paramCount: 1, jsFn: Math.cos },
  { name: "tan", paramCount: 1, jsFn: Math.tan },
  { name: "asin", paramCount: 1, jsFn: Math.asin },
  { name: "acos", paramCount: 1, jsFn: Math.acos },
  { name: "atan", paramCount: 1, jsFn: Math.atan },
  { name: "exp", paramCount: 1, jsFn: Math.exp },
  { name: "log", paramCount: 1, jsFn: Math.log },
  { name: "log1p", paramCount: 1, jsFn: Math.log1p },
  { name: "cbrt", paramCount: 1, jsFn: Math.cbrt },
  { name: "tanh", paramCount: 1, jsFn: Math.tanh },
  { name: "atan2", paramCount: 2, jsFn: Math.atan2 },
  { name: "fmod", paramCount: 2, jsFn: (a: number, b: number) => a % b },
  {
    name: "assert_zero",
    paramCount: 1,
    jsFn: (a: number) => assertTapeValue("zero", a),
  },
  {
    name: "assert_nonzero",
    paramCount: 1,
    jsFn: (a: number) => assertTapeValue("nonzero", a),
  },
];

export const IMPORT_INDEX: Record<string, number> = {};
for (let i = 0; i < ALL_IMPORTS.length; i++) {
  IMPORT_INDEX[ALL_IMPORTS[i]!.name] = i;
}

/** Map tape opcode → import function name (null = native WASM or inline) */
export const UNARY_IMPORT: Record<number, string | null> = {
  [OP_SQRT]: null,
  [OP_ABS]: null,
  [OP_NEG]: null,
  [OP_NOT]: null,
  [OP_DEBUG]: null,
  [OP_ASSERT_ZERO]: "assert_zero",
  [OP_ASSERT_NONZERO]: "assert_nonzero",
  [OP_SIGN]: null,
  [OP_SIN]: "sin",
  [OP_COS]: "cos",
  [OP_TAN]: "tan",
  [OP_ASIN]: "asin",
  [OP_ACOS]: "acos",
  [OP_ATAN]: "atan",
  [OP_EXP]: "exp",
  [OP_LOG]: "log",
  [OP_LOG1P]: "log1p",
  [OP_CBRT]: "cbrt",
  [OP_TANH]: "tanh",
};

export const BINARY_IMPORT: Record<number, string | null> = {
  [OP_ADD]: null,
  [OP_SUB]: null,
  [OP_MUL]: null,
  [OP_DIV]: null,
  [OP_AND]: null,
  [OP_OR]: null,
  [OP_COMPARE]: null,
  [OP_MOD]: "fmod",
  [OP_ATAN2]: "atan2",
  [OP_MIN]: null,
  [OP_MAX]: null,
};

// ---------------------------------------------------------------------------
// emitNumOp — shared operation emission for unary/binary tape opcodes
// ---------------------------------------------------------------------------

type EmitPush = (b: WasmBuilder) => void;

/**
 * Emit WASM instructions for a unary or binary tape opcode.
 * Leaves one f64 on the WASM stack (the result).
 *
 * `pushA`/`pushB` are called to push operand values onto the stack.
 * They may be called multiple times (for ops like NOT, DIV, AND, OR).
 *
 * `getCallIdx` resolves an import function name to its WASM function index.
 */
export function emitNumOp(
  b: WasmBuilder,
  op: number,
  pushA: EmitPush,
  pushB: EmitPush,
  getCallIdx: (name: string) => number,
): void {
  switch (op) {
    // --- Native unary ---
    case OP_DEBUG:
      pushA(b);
      break;
    case OP_NEG:
      pushA(b);
      b.byte(W_F64_NEG);
      break;
    case OP_ABS:
      pushA(b);
      b.byte(W_F64_ABS);
      break;
    case OP_SQRT:
      pushA(b);
      b.byte(W_F64_SQRT);
      break;
    case OP_NOT:
      pushA(b);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_EQ);
      b.byte(W_IF);
      b.byte(BT_F64);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_ELSE);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_END);
      break;

    case OP_SIGN:
      // sign(a) = f64((a > 0) - (a < 0))
      pushA(b);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_GT);
      pushA(b);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_LT);
      b.byte(W_I32_SUB);
      b.byte(W_F64_CONVERT_I32_S);
      break;

    // --- Imported unary ---
    case OP_ASSERT_ZERO:
    case OP_ASSERT_NONZERO:
    case OP_SIN:
    case OP_COS:
    case OP_TAN:
    case OP_ASIN:
    case OP_ACOS:
    case OP_ATAN:
    case OP_EXP:
    case OP_LOG:
    case OP_LOG1P:
    case OP_CBRT:
    case OP_TANH:
      pushA(b);
      b.byte(W_CALL);
      b.u32(getCallIdx(UNARY_IMPORT[op]!));
      break;

    // --- Native binary ---
    case OP_ADD:
      pushA(b);
      pushB(b);
      b.byte(W_F64_ADD);
      break;
    case OP_SUB:
      pushA(b);
      pushB(b);
      b.byte(W_F64_SUB);
      break;
    case OP_MUL:
      pushA(b);
      pushB(b);
      b.byte(W_F64_MUL);
      break;
    case OP_MIN:
      pushA(b);
      pushB(b);
      b.byte(W_F64_MIN);
      break;
    case OP_MAX:
      pushA(b);
      pushB(b);
      b.byte(W_F64_MAX);
      break;
    case OP_DIV:
      pushB(b);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_NE);
      b.byte(W_IF);
      b.byte(BT_F64);
      pushA(b);
      pushB(b);
      b.byte(W_F64_DIV);
      b.byte(W_ELSE);
      b.byte(W_F64_CONST);
      b.f64(DIV_BY_ZERO_FALLBACK);
      b.byte(W_END);
      break;
    case OP_AND:
      // (a == 0) ? a : b — branchless via select
      pushA(b); // value if cond true (a)
      pushB(b); // value if cond false (b)
      pushA(b);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_EQ); // i32: 1 if a==0
      b.byte(W_SELECT);
      break;
    case OP_OR:
      // (a == 0) ? b : a — branchless via select
      pushB(b); // value if cond true (b)
      pushA(b); // value if cond false (a)
      pushA(b);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_EQ); // i32: 1 if a==0
      b.byte(W_SELECT);
      break;
    case OP_COMPARE:
      // sign(a - b) = f64((a-b > 0) - (a-b < 0)) — no import needed
      pushA(b);
      pushB(b);
      b.byte(W_F64_SUB);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_GT); // i32: (a-b) > 0
      pushA(b);
      pushB(b);
      b.byte(W_F64_SUB);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_LT); // i32: (a-b) < 0
      b.byte(W_I32_SUB); // i32: sign
      b.byte(W_F64_CONVERT_I32_S);
      break;

    // --- Imported binary ---
    case OP_MOD:
    case OP_ATAN2:
      pushA(b);
      pushB(b);
      b.byte(W_CALL);
      b.u32(getCallIdx(BINARY_IMPORT[op]!));
      break;
  }
}

// ---------------------------------------------------------------------------
// emitNumOpGrad — forward-mode autodiff op emission
// ---------------------------------------------------------------------------
//
// Emits WASM instructions for one opcode's forward-mode derivative rule.
// Each tape slot stores a dual number [value, ∂/∂x₁, …, ∂/∂xₙ].
// The emitted code computes both the primal value and propagates all
// partial derivatives in a loop over numDiff.

/**
 * WASM local/param indices needed by the forward-mode grad emitter.
 * Allows the module builder to define its own layout and pass it in.
 */
export interface GradLocals {
  /** Param: base offset of values array in WASM memory */
  P_VAL: number;
  /** Param: base offset of literals array */
  P_LIT: number;
  /** Param: base offset of variable values array */
  P_VARS: number;
  /** Param: number of differentiation variables */
  P_NUMDIFF: number;
  /** Param: base offset of input-major, direction-minor f64 seed matrix */
  P_SEEDS: number;
  /** Local: byte offset of current node's dual in values array */
  L_BASE: number;
  /** Local: byte offset of operand A's dual */
  L_ABASE: number;
  /** Local: byte offset of operand B's dual */
  L_BBASE: number;
  /** Local: argA index value */
  L_AIDX: number;
  /** Local: argB index value */
  L_BIDX: number;
  /** Local: stride in bytes (cached) */
  L_STRIDEBYTES: number;
  /** Local: derivative loop counter (i32) */
  L_D: number;
  /** Local: f64 value of operand A */
  L_VA: number;
  /** Local: f64 value of operand B */
  L_VB: number;
  /** Local: f64 derivative coefficient */
  L_COEFF: number;
}

// ---- internal helpers (all parameterised on GradLocals) ----

/** Emit: f64.load from address already on the WASM stack */
function gradLoadF64(b: WasmBuilder) {
  b.byte(W_F64_LOAD);
  b.u32(3);
  b.u32(0);
}

/** Emit: push P_VAL + localIdx onto the WASM stack */
function gradValAddr(b: WasmBuilder, g: GradLocals, localIdx: number) {
  b.byte(W_LOCAL_GET);
  b.u32(g.P_VAL);
  b.byte(W_LOCAL_GET);
  b.u32(localIdx);
  b.byte(W_I32_ADD);
}

/** Emit: f64.load(P_VAL + baseLocal) — loads primal value of a dual */
function gradLoadPrimal(b: WasmBuilder, g: GradLocals, baseLocal: number) {
  gradValAddr(b, g, baseLocal);
  gradLoadF64(b);
}

/** Emit: compute L_ABASE = L_AIDX * L_STRIDEBYTES, L_VA = primal of A */
function gradLoadOperandA(b: WasmBuilder, g: GradLocals) {
  b.byte(W_LOCAL_GET);
  b.u32(g.L_AIDX);
  b.byte(W_LOCAL_GET);
  b.u32(g.L_STRIDEBYTES);
  b.byte(W_I32_MUL);
  b.byte(W_LOCAL_SET);
  b.u32(g.L_ABASE);
  gradLoadPrimal(b, g, g.L_ABASE);
  b.byte(W_LOCAL_SET);
  b.u32(g.L_VA);
}

/** Emit: compute L_BBASE = L_BIDX * L_STRIDEBYTES, L_VB = primal of B */
function gradLoadOperandB(b: WasmBuilder, g: GradLocals) {
  b.byte(W_LOCAL_GET);
  b.u32(g.L_BIDX);
  b.byte(W_LOCAL_GET);
  b.u32(g.L_STRIDEBYTES);
  b.byte(W_I32_MUL);
  b.byte(W_LOCAL_SET);
  b.u32(g.L_BBASE);
  gradLoadPrimal(b, g, g.L_BBASE);
  b.byte(W_LOCAL_SET);
  b.u32(g.L_VB);
}

/** Emit: store f64 constant into primal slot at L_BASE */
function gradStorePrimalConst(b: WasmBuilder, g: GradLocals, val: number) {
  gradValAddr(b, g, g.L_BASE);
  b.byte(W_F64_CONST);
  b.f64(val);
  b.byte(W_F64_STORE);
  b.u32(3);
  b.u32(0);
}

/** Emit: push address P_VAL + baseLocal + (1+L_D)*8 onto WASM stack */
function gradDerivAddr(b: WasmBuilder, g: GradLocals, baseLocal: number) {
  b.byte(W_LOCAL_GET);
  b.u32(g.P_VAL);
  b.byte(W_LOCAL_GET);
  b.u32(baseLocal);
  b.byte(W_I32_ADD);
  b.byte(W_LOCAL_GET);
  b.u32(g.L_D);
  b.byte(W_I32_CONST);
  b.i32(1);
  b.byte(W_I32_ADD);
  b.byte(W_I32_CONST);
  b.i32(3);
  b.byte(W_I32_SHL);
  b.byte(W_I32_ADD);
}

/** Emit loop preamble: L_D=0, block $break, loop $loop, br_if L_D>=P_NUMDIFF */
function gradLoopHead(b: WasmBuilder, g: GradLocals) {
  b.byte(W_I32_CONST);
  b.i32(0);
  b.byte(W_LOCAL_SET);
  b.u32(g.L_D);
  b.byte(W_BLOCK);
  b.byte(BT_VOID);
  b.byte(W_LOOP);
  b.byte(BT_VOID);
  b.byte(W_LOCAL_GET);
  b.u32(g.L_D);
  b.byte(W_LOCAL_GET);
  b.u32(g.P_NUMDIFF);
  b.byte(W_I32_GE_U);
  b.byte(W_BR_IF);
  b.u32(1);
}

/** Emit loop tail: L_D++, br $loop, end loop, end block */
function gradLoopTail(b: WasmBuilder, g: GradLocals) {
  b.byte(W_LOCAL_GET);
  b.u32(g.L_D);
  b.byte(W_I32_CONST);
  b.i32(1);
  b.byte(W_I32_ADD);
  b.byte(W_LOCAL_SET);
  b.u32(g.L_D);
  b.byte(W_BR);
  b.u32(0);
  b.byte(W_END);
  b.byte(W_END);
}

/**
 * Emit loop: result_d = L_COEFF * a_d  (unary chain rule).
 * L_COEFF must be set before calling.
 */
function gradUnaryDerivLoop(b: WasmBuilder, g: GradLocals) {
  gradLoopHead(b, g);
  // dst addr
  gradDerivAddr(b, g, g.L_BASE);
  // value = L_COEFF * values[L_ABASE + (1+d)*8]
  b.byte(W_LOCAL_GET);
  b.u32(g.L_COEFF);
  gradDerivAddr(b, g, g.L_ABASE);
  gradLoadF64(b);
  b.byte(W_F64_MUL);
  b.byte(W_F64_STORE);
  b.u32(3);
  b.u32(0);
  gradLoopTail(b, g);
}

/** Emit loop: result_d = 0 */
function gradZeroDerivLoop(b: WasmBuilder, g: GradLocals) {
  gradLoopHead(b, g);
  gradDerivAddr(b, g, g.L_BASE);
  b.byte(W_F64_CONST);
  b.f64(0);
  b.byte(W_F64_STORE);
  b.u32(3);
  b.u32(0);
  gradLoopTail(b, g);
}

/** Emit loop: result_d = src_d  (copy from srcBaseLocal) */
function gradCopyDerivLoop(
  b: WasmBuilder,
  g: GradLocals,
  srcBaseLocal: number,
) {
  gradLoopHead(b, g);
  gradDerivAddr(b, g, g.L_BASE);
  gradDerivAddr(b, g, srcBaseLocal);
  gradLoadF64(b);
  b.byte(W_F64_STORE);
  b.u32(3);
  b.u32(0);
  gradLoopTail(b, g);
}

/** Emit loop: result_d = L_VA * a_d + L_VB * b_d  (binary chain rule) */
function gradBinaryDerivLoop(b: WasmBuilder, g: GradLocals) {
  gradLoopHead(b, g);
  gradDerivAddr(b, g, g.L_BASE);
  // L_VA * a_d
  b.byte(W_LOCAL_GET);
  b.u32(g.L_VA);
  gradDerivAddr(b, g, g.L_ABASE);
  gradLoadF64(b);
  b.byte(W_F64_MUL);
  // + L_VB * b_d
  b.byte(W_LOCAL_GET);
  b.u32(g.L_VB);
  gradDerivAddr(b, g, g.L_BBASE);
  gradLoadF64(b);
  b.byte(W_F64_MUL);
  b.byte(W_F64_ADD);
  b.byte(W_F64_STORE);
  b.u32(3);
  b.u32(0);
  gradLoopTail(b, g);
}

// ---- per-opcode emission ----

function emitGradUnary(
  b: WasmBuilder,
  op: number,
  g: GradLocals,
  getCallIdx: (name: string) => number,
) {
  gradLoadOperandA(b, g);

  switch (op) {
    case OP_NEG:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_NEG);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(-1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_ABS:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_ABS);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_GT);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_LT);
      b.byte(W_I32_SUB);
      b.byte(W_F64_CONVERT_I32_S);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_SQRT:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_SQRT);
      b.byte(W_LOCAL_TEE);
      b.u32(g.L_VB);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_NE);
      b.byte(W_IF);
      b.byte(BT_F64);
      b.byte(W_F64_CONST);
      b.f64(0.5);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_DIV);
      b.byte(W_ELSE);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_END);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_SIN:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("sin"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("cos"));
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_COS:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("cos"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("sin"));
      b.byte(W_F64_NEG);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_TAN:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("tan"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("cos"));
      b.byte(W_LOCAL_TEE);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_MUL);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VB);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_ASIN:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("asin"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_MUL);
      b.byte(W_F64_SUB);
      b.byte(W_F64_SQRT);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_ACOS:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("acos"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(-1);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_MUL);
      b.byte(W_F64_SUB);
      b.byte(W_F64_SQRT);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_ATAN:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("atan"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_MUL);
      b.byte(W_F64_ADD);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_EXP:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("exp"));
      b.byte(W_LOCAL_TEE);
      b.u32(g.L_COEFF);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_LOG:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("log"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_LOG1P:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("log1p"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_ADD);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_CBRT:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("cbrt"));
      b.byte(W_LOCAL_TEE);
      b.u32(g.L_VB);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_NE);
      b.byte(W_IF);
      b.byte(BT_F64);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_F64_CONST);
      b.f64(3);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_MUL);
      b.byte(W_F64_MUL);
      b.byte(W_F64_DIV);
      b.byte(W_ELSE);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_END);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_TANH:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(getCallIdx("tanh"));
      b.byte(W_LOCAL_TEE);
      b.u32(g.L_VB);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_MUL);
      b.byte(W_F64_SUB);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_SIGN:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_GT);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_LT);
      b.byte(W_I32_SUB);
      b.byte(W_F64_CONVERT_I32_S);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      gradZeroDerivLoop(b, g);
      break;
    case OP_NOT:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_EQ);
      b.byte(W_IF);
      b.byte(BT_F64);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_ELSE);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_END);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      gradZeroDerivLoop(b, g);
      break;
    case OP_DEBUG:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_ASSERT_ZERO:
    case OP_ASSERT_NONZERO:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_CALL);
      b.u32(
        getCallIdx(op === OP_ASSERT_ZERO ? "assert_zero" : "assert_nonzero"),
      );
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
  }
}

function emitGradBinary(
  b: WasmBuilder,
  op: number,
  g: GradLocals,
  getCallIdx: (name: string) => number,
) {
  gradLoadOperandA(b, g);
  gradLoadOperandB(b, g);

  switch (op) {
    case OP_ADD:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_ADD);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VB);
      gradBinaryDerivLoop(b, g);
      break;
    case OP_SUB:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_SUB);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(-1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VB);
      gradBinaryDerivLoop(b, g);
      break;
    case OP_MUL:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_MUL);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      // coeffA = b, coeffB = a → swap VA/VB
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_COEFF);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VB);
      gradBinaryDerivLoop(b, g);
      break;
    case OP_DIV:
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_NE);
      b.byte(W_IF);
      b.byte(BT_VOID);
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_DIV);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      // invB2 = 1/(b*b)
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_MUL);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      // coeffA = b*invB2 = 1/b, coeffB = -a*invB2
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_COEFF);
      b.byte(W_F64_MUL);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_NEG);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_COEFF);
      b.byte(W_F64_MUL);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VA);
      gradBinaryDerivLoop(b, g);
      b.byte(W_ELSE);
      gradStorePrimalConst(b, g, DIV_BY_ZERO_FALLBACK);
      gradZeroDerivLoop(b, g);
      b.byte(W_END);
      break;
    case OP_MOD:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_CALL);
      b.u32(getCallIdx("fmod"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_F64_CONST);
      b.f64(1);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      gradUnaryDerivLoop(b, g);
      break;
    case OP_ATAN2:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_CALL);
      b.u32(getCallIdx("atan2"));
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_MUL);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_MUL);
      b.byte(W_F64_ADD);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_COEFF);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_COEFF);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_NE);
      b.byte(W_IF);
      b.byte(BT_VOID);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_COEFF);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_NEG);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_COEFF);
      b.byte(W_F64_DIV);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VB);
      b.byte(W_LOCAL_SET);
      b.u32(g.L_VA);
      gradBinaryDerivLoop(b, g);
      b.byte(W_ELSE);
      gradZeroDerivLoop(b, g);
      b.byte(W_END);
      break;
    case OP_MIN:
    case OP_MAX:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(op === OP_MIN ? W_F64_MIN : W_F64_MAX);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(op === OP_MIN ? W_F64_LE : W_F64_GE);
      b.byte(W_IF);
      b.byte(BT_VOID);
      gradCopyDerivLoop(b, g, g.L_ABASE);
      b.byte(W_ELSE);
      gradCopyDerivLoop(b, g, g.L_BBASE);
      b.byte(W_END);
      break;
    case OP_COMPARE:
      gradValAddr(b, g, g.L_BASE);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_SUB);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_GT);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VB);
      b.byte(W_F64_SUB);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_LT);
      b.byte(W_I32_SUB);
      b.byte(W_F64_CONVERT_I32_S);
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      gradZeroDerivLoop(b, g);
      break;
    case OP_AND:
    case OP_OR:
      gradValAddr(b, g, g.L_BASE);
      if (op === OP_AND) {
        b.byte(W_LOCAL_GET);
        b.u32(g.L_VA);
        b.byte(W_LOCAL_GET);
        b.u32(g.L_VB);
        b.byte(W_LOCAL_GET);
        b.u32(g.L_VA);
        b.byte(W_F64_CONST);
        b.f64(0);
        b.byte(W_F64_EQ);
        b.byte(W_SELECT);
      } else {
        b.byte(W_LOCAL_GET);
        b.u32(g.L_VB);
        b.byte(W_LOCAL_GET);
        b.u32(g.L_VA);
        b.byte(W_LOCAL_GET);
        b.u32(g.L_VA);
        b.byte(W_F64_CONST);
        b.f64(0);
        b.byte(W_F64_EQ);
        b.byte(W_SELECT);
      }
      b.byte(W_F64_STORE);
      b.u32(3);
      b.u32(0);
      b.byte(W_LOCAL_GET);
      b.u32(g.L_VA);
      b.byte(W_F64_CONST);
      b.f64(0);
      b.byte(W_F64_EQ);
      b.byte(W_IF);
      b.byte(BT_VOID);
      gradCopyDerivLoop(b, g, op === OP_AND ? g.L_ABASE : g.L_BBASE);
      b.byte(W_ELSE);
      gradCopyDerivLoop(b, g, op === OP_AND ? g.L_BBASE : g.L_ABASE);
      b.byte(W_END);
      break;
  }
}

/**
 * Emit the forward-mode autodiff case body for one opcode.
 *
 * For OP_LIT/OP_VAR: handles primal + derivative seeding.
 * For unary/binary ops: computes primal + propagates derivatives.
 *
 * This is the grad counterpart of `emitNumOp`.
 */
export function emitNumOpGrad(
  b: WasmBuilder,
  op: number,
  g: GradLocals,
  getCallIdx: (name: string) => number,
): void {
  if (op === OP_LIT) {
    gradValAddr(b, g, g.L_BASE);
    b.byte(W_LOCAL_GET);
    b.u32(g.P_LIT);
    b.byte(W_LOCAL_GET);
    b.u32(g.L_AIDX);
    b.byte(W_I32_CONST);
    b.i32(3);
    b.byte(W_I32_SHL);
    b.byte(W_I32_ADD);
    gradLoadF64(b);
    b.byte(W_F64_STORE);
    b.u32(3);
    b.u32(0);
    gradZeroDerivLoop(b, g);
  } else if (op === OP_VAR) {
    // primal
    gradValAddr(b, g, g.L_BASE);
    b.byte(W_LOCAL_GET);
    b.u32(g.P_VARS);
    b.byte(W_LOCAL_GET);
    b.u32(g.L_AIDX);
    b.byte(W_I32_CONST);
    b.i32(3);
    b.byte(W_I32_SHL);
    b.byte(W_I32_ADD);
    gradLoadF64(b);
    b.byte(W_F64_STORE);
    b.u32(3);
    b.u32(0);
    // Copy this variable's arbitrary tangent seeds into its dual lanes.
    gradLoopHead(b, g);
    gradDerivAddr(b, g, g.L_BASE);
    b.byte(W_LOCAL_GET);
    b.u32(g.P_SEEDS);
    b.byte(W_LOCAL_GET);
    b.u32(g.L_AIDX);
    b.byte(W_LOCAL_GET);
    b.u32(g.P_NUMDIFF);
    b.byte(W_I32_MUL);
    b.byte(W_LOCAL_GET);
    b.u32(g.L_D);
    b.byte(W_I32_ADD);
    b.byte(W_I32_CONST);
    b.i32(3);
    b.byte(W_I32_SHL);
    b.byte(W_I32_ADD);
    gradLoadF64(b);
    b.byte(W_F64_STORE);
    b.u32(3);
    b.u32(0);
    gradLoopTail(b, g);
  } else if (
    (VALID_OPS.has(op) && op >= 10 && op <= 29) ||
    op === OP_ASSERT_ZERO ||
    op === OP_ASSERT_NONZERO
  ) {
    emitGradUnary(b, op, g, getCallIdx);
  } else if (VALID_OPS.has(op) && op >= 40) {
    emitGradBinary(b, op, g, getCallIdx);
  }
}
