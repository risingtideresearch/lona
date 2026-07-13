/**
 * Shared tape opcode constants, string→opcode maps, and the set of valid
 * opcodes used by the WASM br_table dispatch.
 *
 * The numeric values of `OP_*` for LIT / VAR / unary / binary nodes are
 * intentionally identical to the corresponding `KIND_*` values on NumNode,
 * so tape compilers can push `node.kind` directly without an extra lookup.
 */

export const OP_LIT = 0;
export const OP_VAR = 1;

// Unary (10–29)
export const OP_SQRT = 10;
export const OP_CBRT = 11;
export const OP_COS = 12;
export const OP_ACOS = 13;
export const OP_ASIN = 14;
export const OP_TAN = 15;
export const OP_ATAN = 16;
export const OP_LOG = 17;
export const OP_EXP = 18;
export const OP_ABS = 19;
export const OP_NEG = 20;
export const OP_SIN = 21;
export const OP_SIGN = 22;
export const OP_NOT = 23;
export const OP_TANH = 24;
export const OP_LOG1P = 25;
export const OP_DEBUG = 26;

// Binary (40–50)
export const OP_ADD = 40;
export const OP_SUB = 41;
export const OP_MUL = 42;
export const OP_DIV = 43;
export const OP_MOD = 44;
export const OP_ATAN2 = 45;
export const OP_MIN = 46;
export const OP_MAX = 47;
export const OP_COMPARE = 48;
export const OP_AND = 49;
export const OP_OR = 50;

// Assertions (value-producing pass-through guards for specialized tapes)
export const OP_ASSERT_ZERO = 51;
export const OP_ASSERT_NONZERO = 52;

export const MAX_OPCODE = 52;
export const NUM_CASES = MAX_OPCODE + 1;

export const DIV_BY_ZERO_FALLBACK = 1e50;

export const UNARY_OP_MAP: Record<string, number> = {
  SQRT: OP_SQRT,
  CBRT: OP_CBRT,
  COS: OP_COS,
  ACOS: OP_ACOS,
  ASIN: OP_ASIN,
  TAN: OP_TAN,
  ATAN: OP_ATAN,
  LOG: OP_LOG,
  EXP: OP_EXP,
  ABS: OP_ABS,
  NEG: OP_NEG,
  SIN: OP_SIN,
  SIGN: OP_SIGN,
  NOT: OP_NOT,
  TANH: OP_TANH,
  LOG1P: OP_LOG1P,
  DEBUG: OP_DEBUG,
};

export const BINARY_OP_MAP: Record<string, number> = {
  ADD: OP_ADD,
  SUB: OP_SUB,
  MUL: OP_MUL,
  DIV: OP_DIV,
  MOD: OP_MOD,
  ATAN2: OP_ATAN2,
  MIN: OP_MIN,
  MAX: OP_MAX,
  COMPARE: OP_COMPARE,
  AND: OP_AND,
  OR: OP_OR,
};

/** Set of all valid opcodes (for WASM br_table) */
export const VALID_OPS = new Set([
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
]);
