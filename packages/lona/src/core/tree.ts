import { visitFromLeaves } from "../dag/traversal";
import { UnaryOperation, BinaryOperation } from "../types";
export type VarName = string | symbol;

// ---------------------------------------------------------------------------
// Kind tags — numeric dispatch keys
// ---------------------------------------------------------------------------
//
// Every NumNode carries a numeric `kind` field so evaluators can dispatch
// via a dense integer switch instead of a chain of `instanceof` checks.
// V8 compiles integer switches to jump tables (or a short linear compare
// chain); walking a chain of six `instanceof` checks through a polymorphic
// IC is an order of magnitude slower on hot paths.
//
// The numeric values are kept **identical** to the `OP_*` opcodes in
// `eval/tape-opcodes.ts`, so `compileTape` can push `node.kind`
// directly as a tape opcode byte for literals / variables / unary /
// binary ops. Drift between the two is guarded by a test in
// `tree.test.ts`.
//
// `Derivative`, `ForeignFn`, and `SelectOp` have kinds ≥ 60 because they
// never appear directly in the current linear compiled tape, so they're
// outside the opcode space.

export const KIND_LIT = 0;
export const KIND_VAR = 1;

// Unary ops (10-29). Order must match the `OP_*` unary constants in
// tape-opcodes.ts.
export const KIND_SQRT = 10;
export const KIND_CBRT = 11;
export const KIND_COS = 12;
export const KIND_ACOS = 13;
export const KIND_ASIN = 14;
export const KIND_TAN = 15;
export const KIND_ATAN = 16;
export const KIND_LOG = 17;
export const KIND_EXP = 18;
export const KIND_ABS = 19;
export const KIND_NEG = 20;
export const KIND_SIN = 21;
export const KIND_SIGN = 22;
export const KIND_NOT = 23;
export const KIND_TANH = 24;
export const KIND_LOG1P = 25;
export const KIND_DEBUG = 26;

// Binary ops (40-59). Order must match the `OP_*` binary constants in
// tape-opcodes.ts.
export const KIND_ADD = 40;
export const KIND_SUB = 41;
export const KIND_MUL = 42;
export const KIND_DIV = 43;
export const KIND_MOD = 44;
export const KIND_ATAN2 = 45;
export const KIND_MIN = 46;
export const KIND_MAX = 47;
export const KIND_COMPARE = 48;
export const KIND_AND = 49;
export const KIND_OR = 50;

// Non-tape node kinds (outside the opcode space).
export const KIND_DERIVATIVE = 60;
export const KIND_FOREIGN = 61;
export const KIND_SELECT = 62;

/** Range check: is `kind` any unary operator (10-29, including DEBUG)? */
export function isUnaryKind(kind: number): boolean {
  return kind >= 10 && kind <= 29;
}

/** Range check: is `kind` any binary operator (40-59)? */
export function isBinaryKind(kind: number): boolean {
  return kind >= 40 && kind <= 59;
}

// Operation string → kind lookup. Used at construction time in the
// UnaryOp / BinaryOp constructors to turn the existing `operation`
// string into a fast dispatch tag.
const UNARY_KIND: Record<UnaryOperation, number> = {
  SQRT: KIND_SQRT,
  CBRT: KIND_CBRT,
  COS: KIND_COS,
  ACOS: KIND_ACOS,
  ASIN: KIND_ASIN,
  TAN: KIND_TAN,
  ATAN: KIND_ATAN,
  LOG: KIND_LOG,
  EXP: KIND_EXP,
  ABS: KIND_ABS,
  NEG: KIND_NEG,
  SIN: KIND_SIN,
  SIGN: KIND_SIGN,
  NOT: KIND_NOT,
  TANH: KIND_TANH,
  LOG1P: KIND_LOG1P,
  DEBUG: KIND_DEBUG,
};

const BINARY_KIND: Record<BinaryOperation, number> = {
  ADD: KIND_ADD,
  SUB: KIND_SUB,
  MUL: KIND_MUL,
  DIV: KIND_DIV,
  MOD: KIND_MOD,
  ATAN2: KIND_ATAN2,
  MIN: KIND_MIN,
  MAX: KIND_MAX,
  COMPARE: KIND_COMPARE,
  AND: KIND_AND,
  OR: KIND_OR,
};

// ---------------------------------------------------------------------------
// NumNode class hierarchy
// ---------------------------------------------------------------------------

export class NumNode {
  readonly operation: string = "NONE";
  /**
   * Numeric dispatch tag. Set in each subclass constructor. See the
   * `KIND_*` constants above. Intended for `switch (node.kind)` dispatch
   * in hot-path evaluators; `instanceof` is still correct but slower.
   */
  readonly kind: number = -1;
}

export class Derivative extends NumNode {
  readonly kind = KIND_DERIVATIVE;
  constructor(readonly variable: Variable) {
    super();
  }
}

export class UnaryOp extends NumNode {
  readonly kind: number;
  constructor(
    readonly operation: UnaryOperation,
    readonly original: NumNode,
  ) {
    super();
    this.kind = UNARY_KIND[operation]!;
  }
}

export class DebugNode extends UnaryOp {
  // Inherits `kind = KIND_DEBUG` from the UnaryOp constructor, which
  // looks up `UNARY_KIND["DEBUG"]` at construction time.
  constructor(
    readonly original: NumNode,
    readonly debug: string,
  ) {
    super("DEBUG", original);
    this.debug = debug;
  }
}

export class BinaryOp extends NumNode {
  readonly kind: number;
  constructor(
    readonly operation: BinaryOperation,
    readonly left: NumNode,
    readonly right: NumNode,
  ) {
    super();
    this.kind = BINARY_KIND[operation]!;
  }
}

export class LiteralNum extends NumNode {
  readonly operation = "LITERAL";
  readonly kind = KIND_LIT;
  // `value` is `readonly` because LiteralNum participates in hash-consing
  // (see tree-cons.ts). Mutating it would silently corrupt every cached
  // DAG that references this node.
  constructor(readonly value: number) {
    super();
  }
}

export function asLiteralValue(node: NumNode): number | undefined {
  return node.kind === KIND_LIT ? (node as LiteralNum).value : undefined;
}

export class Variable extends NumNode {
  readonly operation = "VAR";
  readonly kind = KIND_VAR;
  // `name` is `readonly` for the same hash-cons reasons as LiteralNum.value.
  constructor(readonly name: VarName) {
    super();
  }
}

export class ForeignFn extends NumNode {
  readonly operation = "FOREIGN";
  readonly kind = KIND_FOREIGN;
  constructor(
    /** The input NumNodes whose values are passed to evalFn */
    readonly inputs: readonly NumNode[],
    /** Evaluates the foreign function given concrete input values */
    readonly evalFn: (inputValues: readonly number[]) => number,
    /** Produces the derivative w.r.t. the i-th input, as a NumNode */
    readonly diffFn: (inputIndex: number) => NumNode,
  ) {
    super();
  }
}

export class SelectOp extends NumNode {
  readonly operation = "SELECT";
  readonly kind = KIND_SELECT;
  constructor(
    readonly condition: NumNode,
    readonly ifNonZero: NumNode,
    readonly ifZero: NumNode,
  ) {
    super();
  }
}

// Sentinel literal singletons. These must be constructed with `new` rather
// than `litNode`: when this module loads the cons module may not yet be
// initialized because of the circular import (tree ↔ tree-cons).
// `litNode` seeds its own table from these sentinels on first use, so
// `litNode(0) === ZERO_NODE` holds at runtime.
export const ZERO_NODE = new LiteralNum(0);
export const ONE_NODE = new LiteralNum(1);
export const TWO_NODE = new LiteralNum(2);
export const NEG_ONE_NODE = new LiteralNum(-1);

// ---------------------------------------------------------------------------
// Tree-walking helpers (no factory dependency)
// ---------------------------------------------------------------------------
//
// `cloneNode`, `replaceVariable`, and `partialDerivative` have moved to
// `core/tree-walks.ts` because they need the cons factories from
// `core/context.ts`. Keeping them out of this file lets `tree.ts` stay
// at the bottom of the dependency graph.

export function childrenOfNumNode(node: NumNode): NumNode[] {
  const kind = node.kind;
  if (kind === KIND_FOREIGN) {
    return [...(node as ForeignFn).inputs];
  }
  if (kind === KIND_SELECT) {
    const s = node as SelectOp;
    return [s.condition, s.ifNonZero, s.ifZero];
  }
  if (isUnaryKind(kind)) {
    return [(node as UnaryOp).original];
  }
  if (isBinaryKind(kind)) {
    const b = node as BinaryOp;
    return [b.left, b.right];
  }
  if (kind === KIND_DERIVATIVE) {
    return [(node as Derivative).variable];
  }
  return [];
}

/**
 * Returns a structural label for a NumNode — the node's own identity
 * (type + any embedded constants like variable names or literal values),
 * but NOT its children. Used by topo-fingerprinting.
 */
export function numNodeLabel(node: NumNode): string {
  const kind = node.kind;
  if (kind === KIND_LIT) return `LIT:${(node as LiteralNum).value}`;
  if (kind === KIND_VAR) return `VAR:${String((node as Variable).name)}`;
  if (isUnaryKind(kind)) return `UOP:${(node as UnaryOp).operation}`;
  if (isBinaryKind(kind)) return `BOP:${(node as BinaryOp).operation}`;
  if (kind === KIND_DERIVATIVE) {
    return `DER:${String((node as Derivative).variable.name)}`;
  }
  if (kind === KIND_FOREIGN) return "FOREIGN";
  if (kind === KIND_SELECT) return "SELECT";
  return "UNKNOWN";
}

const allVariables = function (node: NumNode): Set<VarName> {
  const variables = new Set<VarName>();

  visitFromLeaves(node, childrenOfNumNode, (n) => {
    if (n.kind === KIND_VAR) {
      variables.add((n as Variable).name);
    }
  });

  return variables;
};

export { allVariables };
