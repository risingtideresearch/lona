/**
 * Hash-cons factories for NumNode construction.
 *
 * All production code that builds NumNodes should go through these factories
 * instead of calling `new LiteralNum(…)` / `new BinaryOp(…)` directly. Two
 * structurally equal expressions built through the factories return the
 * *same* object, which means:
 *
 *   - Reference equality (`===`) is safe structural equality at all times,
 *     without the "assume compressed" invariant that foldConstants.ts relied
 *     on historically.
 *   - DAG sharing is automatic. Two invocations of `x.mul(x)` produce the
 *     same MUL node; a level-set that references the same sub-expression
 *     many times builds it once.
 *   - Transforms (replaceVariable, partialDerivative, simplify) become
 *     naturally memoizable via `WeakMap<NumNode, NumNode>` because equal
 *     inputs give equal outputs.
 *   - The old post-hoc "compress" pass is gone entirely — structural
 *     dedup now happens at construction time, so `simplify` is just the
 *     algebraic rewrite pass. `deserializeNumDAG` routes through these
 *     factories so deserialized trees compose safely with freshly built
 *     ones.
 *
 * All cache tables live on a `NumTreeContext`. The module keeps a mutable
 * "current context" that every top-level factory function delegates to, so
 * existing call sites don't need to thread a context argument through
 * every layer of Num arithmetic. Callers that want node isolation — e.g.
 * one DAG universe per tab, or a sandboxed test — can create a fresh
 * `NumTreeContext`, install it via `setContext` or `withContext`, and the
 * cached identities are entirely separate from the default.
 *
 * Cache shapes (on each context):
 *
 *   - Literals and variables live in strong `Map`s keyed by value/name.
 *     These are small in practice (a handful of vars, at most hundreds of
 *     distinct constants per context) and never evicted.
 *   - Operator nodes use nested `WeakMap`s keyed by their *operand* nodes,
 *     so the cache entry for a BinaryOp can be reclaimed as soon as either
 *     of its operand sub-DAGs is GC'd. No manual eviction needed.
 *   - ForeignFn cannot be structurally hash-consed (closures carry
 *     non-comparable state), so `foreignFnNode` always allocates.
 */
import {
  BinaryOp,
  DebugNode,
  Derivative,
  ForeignFn,
  KIND_LIT,
  Call,
  LiteralNum,
  NEG_ONE_NODE,
  NumNode,
  ONE_NODE,
  Param,
  Project,
  SelectOp,
  type Proc,
  TWO_NODE,
  UnaryOp,
  Variable,
  ZERO_NODE,
  type VarName,
} from "./tree";
import type { BinaryOperation, UnaryOperation } from "../types";

// ---------------------------------------------------------------------------
// Canonical operand ordering for commutative binary ops
// ---------------------------------------------------------------------------
//
// Every node produced by the factories below is stamped with a monotonic
// `sortId`. Because hash-consing guarantees one object per structural
// identity, the id assigned at first construction is stable for the node's
// lifetime. A total order on sortIds is enough to canonicalise the operand
// order of commutative binary ops, so that `ADD(a, b)` and `ADD(b, a)` end
// up at the same WeakMap entry and dedupe structurally.
//
// Semantics:
//   - Literals always sort before non-literals (so constants end up on the
//     `left` of canonical forms, which is the shape `foldConstants` already
//     pattern-matches on).
//   - Two literals are ordered by sortId (i.e. first-seen order within the
//     process). That's fine for dedup — both orderings always land in the
//     same slot.
//   - Everything else is ordered by sortId.
//
// Only ADD, MUL, MIN, MAX are commutative in this codebase. SUB, DIV, MOD,
// ATAN2 and COMPARE are not. AND/OR have short-circuit-like numeric
// semantics (`a===0 ? 0 : b` / `a===0 ? b : a`) that are not commutative
// either; they are intentionally excluded.
const COMMUTATIVE_OPS: ReadonlySet<BinaryOperation> = new Set([
  "ADD",
  "MUL",
  "MIN",
  "MAX",
]);

// sortId is stored externally in a WeakMap rather than as a class field on
// NumNode so that the NumNode hidden-class layout stays identical to what
// evaluators see. Entries are garbage-collected with their nodes.
const sortIds = new WeakMap<NumNode, number>();
let nextSortId = 1;
const assignSortId = (node: NumNode): void => {
  sortIds.set(node, nextSortId++);
};
const getSortId = (node: NumNode): number => sortIds.get(node) ?? 0;

// Called from `ensureLiteralTable` on the first `litNode` invocation to
// stamp the four well-known sentinels, which were built with
// `new LiteralNum(...)` at module init (before this module loaded) and
// therefore are not yet in the sortIds map.
const sentinelsStamped = { done: false };
const stampSentinels = (): void => {
  if (sentinelsStamped.done) return;
  assignSortId(ZERO_NODE);
  assignSortId(ONE_NODE);
  assignSortId(TWO_NODE);
  assignSortId(NEG_ONE_NODE);
  sentinelsStamped.done = true;
};

// Benchmark escape hatch: callers can flip this off to measure the effect
// of canonicalisation. Production code must leave it alone.
let canonicaliseCommutativeOperands = true;
export function setCanonicaliseCommutativeOperands(enabled: boolean): void {
  canonicaliseCommutativeOperands = enabled;
}

const shouldSwap = (
  op: BinaryOperation,
  left: NumNode,
  right: NumNode,
): boolean => {
  if (!canonicaliseCommutativeOperands) return false;
  if (!COMMUTATIVE_OPS.has(op)) return false;
  const leftIsLit = left.kind === KIND_LIT;
  const rightIsLit = right.kind === KIND_LIT;
  // Literals always sort first — biases the canonical form to
  // `BinaryOp(lit, nonLit)`, which is the shape foldConstants matches on.
  if (leftIsLit !== rightIsLit) return rightIsLit;
  return getSortId(left) > getSortId(right);
};

// ---------------------------------------------------------------------------
// NumTreeContext — holds every cons table for one isolated DAG universe
// ---------------------------------------------------------------------------

export class NumTreeContext {
  // Literals: strong `Map<number, LiteralNum>`, populated lazily so that
  // the circular import between tree.ts and this module doesn't hit
  // the temporal dead zone trying to read `ZERO_NODE` at module init.
  private readonly literalTable = new Map<number, LiteralNum>();
  private literalTableReady = false;

  // Special float values (`NaN`, `-0`) are intentionally *not* given
  // dedicated slots. JavaScript `Map` uses SameValueZero on number keys,
  // which treats `NaN === NaN` and `-0 === +0` — so `litNode(NaN)` caches
  // under the literal table just like any other value, and `litNode(-0)`
  // collapses onto the canonical `ZERO_NODE` seeded by `ensureLiteralTable`.
  // The latter is an intentional hash-consing decision: two mathematically
  // equal literals are the same node, and this codebase erases the only
  // place the sign-of-zero distinction would actually matter (division by
  // zero, which goes through `DIV_BY_ZERO_FALLBACK` in num.ts).

  private readonly variableTable = new Map<VarName, Variable>();

  // `unaryTable.get(op).get(operand) → UnaryOp`
  private readonly unaryTable = new Map<
    UnaryOperation,
    WeakMap<NumNode, UnaryOp>
  >();

  // `binaryTable.get(op).get(left).get(right) → BinaryOp`
  private readonly binaryTable = new Map<
    BinaryOperation,
    WeakMap<NumNode, WeakMap<NumNode, BinaryOp>>
  >();

  // `debugTable.get(original).get(debugString) → DebugNode`. Keyed on the
  // wrapped node first so entries are GC'd when the original dies.
  private readonly debugTable = new WeakMap<NumNode, Map<string, DebugNode>>();

  // `derivativeTable.get(variable) → Derivative`
  private readonly derivativeTable = new WeakMap<Variable, Derivative>();

  // `selectTable.get(condition).get(ifNonZero).get(ifZero) → SelectOp`
  private readonly selectTable = new WeakMap<
    NumNode,
    WeakMap<NumNode, WeakMap<NumNode, SelectOp>>
  >();

  private ensureLiteralTable(): void {
    if (this.literalTableReady) return;
    // Seed the well-known sentinels so `ctx.litNode(0) === ZERO_NODE`, etc.
    // These sentinels are *shared* across all contexts by design — they
    // represent semantic constants (0, 1, 2, -1) whose identity is a
    // codebase-wide invariant that existing reference checks rely on.
    //
    // The sentinels were built with `new LiteralNum(...)` at module init
    // (before this module had loaded), so stamp them with sortIds now in
    // the same monotonic space as every other node.
    stampSentinels();
    this.literalTable.set(0, ZERO_NODE);
    this.literalTable.set(1, ONE_NODE);
    this.literalTable.set(2, TWO_NODE);
    this.literalTable.set(-1, NEG_ONE_NODE);
    this.literalTableReady = true;
  }

  litNode(value: number): LiteralNum {
    this.ensureLiteralTable();
    const existing = this.literalTable.get(value);
    if (existing) return existing;
    const node = new LiteralNum(value);
    assignSortId(node);
    this.literalTable.set(value, node);
    return node;
  }

  varNode(name: VarName): Variable {
    const existing = this.variableTable.get(name);
    if (existing) return existing;
    const node = new Variable(name);
    assignSortId(node);
    this.variableTable.set(name, node);
    return node;
  }

  unaryNode(op: UnaryOperation, operand: NumNode): UnaryOp {
    let byOperand = this.unaryTable.get(op);
    if (!byOperand) {
      byOperand = new WeakMap();
      this.unaryTable.set(op, byOperand);
    }
    const existing = byOperand.get(operand);
    if (existing) return existing;
    const node = new UnaryOp(op, operand);
    assignSortId(node);
    byOperand.set(operand, node);
    return node;
  }

  binaryNode(op: BinaryOperation, left: NumNode, right: NumNode): BinaryOp {
    // Canonicalise commutative operand order so that, e.g., ADD(a, b) and
    // ADD(b, a) hit the same WeakMap entry and dedupe to one node.
    if (shouldSwap(op, left, right)) {
      const tmp = left;
      left = right;
      right = tmp;
    }
    let byLeft = this.binaryTable.get(op);
    if (!byLeft) {
      byLeft = new WeakMap();
      this.binaryTable.set(op, byLeft);
    }
    let byRight = byLeft.get(left);
    if (!byRight) {
      byRight = new WeakMap();
      byLeft.set(left, byRight);
    }
    const existing = byRight.get(right);
    if (existing) return existing;
    const node = new BinaryOp(op, left, right);
    assignSortId(node);
    byRight.set(right, node);
    return node;
  }

  debugNode(original: NumNode, debug: string): DebugNode {
    let byDebug = this.debugTable.get(original);
    if (!byDebug) {
      byDebug = new Map();
      this.debugTable.set(original, byDebug);
    }
    const existing = byDebug.get(debug);
    if (existing) return existing;
    const node = new DebugNode(original, debug);
    assignSortId(node);
    byDebug.set(debug, node);
    return node;
  }

  derivativeNode(variable: Variable): Derivative {
    const existing = this.derivativeTable.get(variable);
    if (existing) return existing;
    const node = new Derivative(variable);
    assignSortId(node);
    this.derivativeTable.set(variable, node);
    return node;
  }

  selectNode(
    condition: NumNode,
    ifNonZero: NumNode,
    ifZero: NumNode,
  ): SelectOp {
    let byThen = this.selectTable.get(condition);
    if (!byThen) {
      byThen = new WeakMap();
      this.selectTable.set(condition, byThen);
    }
    let byElse = byThen.get(ifNonZero);
    if (!byElse) {
      byElse = new WeakMap();
      byThen.set(ifNonZero, byElse);
    }
    const existing = byElse.get(ifZero);
    if (existing) return existing;
    const node = new SelectOp(condition, ifNonZero, ifZero);
    assignSortId(node);
    byElse.set(ifZero, node);
    return node;
  }

  // ForeignFn is intentionally not a method on the context — closures
  // can't be structurally hash-consed, so there is no table to scope.
  // See the module-level `foreignFnNode` below.
}

export const NUM_TREE_CONTEXT: NumTreeContext = new NumTreeContext();

// ---------------------------------------------------------------------------
// Top-level cons factory shims
// ---------------------------------------------------------------------------

export function litNode(value: number): LiteralNum {
  return NUM_TREE_CONTEXT.litNode(value);
}

export function varNode(name: VarName): Variable {
  return NUM_TREE_CONTEXT.varNode(name);
}

export function unaryNode(op: UnaryOperation, operand: NumNode): UnaryOp {
  return NUM_TREE_CONTEXT.unaryNode(op, operand);
}

export function binaryNode(
  op: BinaryOperation,
  left: NumNode,
  right: NumNode,
): BinaryOp {
  return NUM_TREE_CONTEXT.binaryNode(op, left, right);
}

export function debugNode(original: NumNode, debug: string): DebugNode {
  return NUM_TREE_CONTEXT.debugNode(original, debug);
}

/**
 * A fresh proc parameter slot. Deliberately NOT hash-consed: each call
 * allocates a distinct object, so two procs' param #0 never collide in the
 * cons cache (identity is what keeps distinct procs' bodies distinct). It IS
 * stamped in the sort-id space, so commutative canonicalization inside a proc
 * body (`shouldSwap`, which reads `getSortId` on operands) orders params
 * consistently — `p0.add(p1)` and `p1.add(p0)` canonicalize to the same node.
 */
export function paramNode(procTag: number, index: number): Param {
  const node = new Param(procTag, index);
  assignSortId(node);
  return node;
}

// Monotonic proc-tag source, shared by `defineProc` and proc deserialization so
// every proc definition (however constructed) gets a distinct tag.
let nextProcTag = 1;
export function newProcTag(): number {
  return nextProcTag++;
}

/**
 * A proc application node. Not hash-consed (v1 does not intern calls) but stamped
 * in the sort-id space for consistency. `Call` is never itself a commutative
 * operand, but its `Project`s are.
 */
export function callNode(proc: Proc, args: readonly NumNode[]): Call {
  const node = new Call(proc, args);
  assignSortId(node);
  return node;
}

/**
 * A projection (output `output`) of a call — the scalar node that flows in the
 * graph. Stamped so commutative canonicalization orders it consistently against
 * other operands. Validates the output index against the proc's output count.
 */
export function projectNode(call: Call, output: number): Project {
  const outputs = call.proc.body.length;
  if (!Number.isInteger(output) || output < 0 || output >= outputs) {
    throw new Error(
      `projectNode: output ${output} out of range [0, ${outputs}) for this call`,
    );
  }
  const node = new Project(call, output);
  assignSortId(node);
  return node;
}

export function derivativeNode(variable: Variable): Derivative {
  return NUM_TREE_CONTEXT.derivativeNode(variable);
}

export function selectNode(
  condition: NumNode,
  ifNonZero: NumNode,
  ifZero: NumNode,
): SelectOp {
  return NUM_TREE_CONTEXT.selectNode(condition, ifNonZero, ifZero);
}

// ForeignFn cannot be structurally hash-consed (closures carry opaque
// state), so this factory always allocates and isn't a cons method.

export function foreignFnNode(
  inputs: readonly NumNode[],
  evalFn: (inputValues: readonly number[]) => number,
  diffFn: (inputIndex: number) => NumNode,
): ForeignFn {
  return new ForeignFn(inputs, evalFn, diffFn);
}
