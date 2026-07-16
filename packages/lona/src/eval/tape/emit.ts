/**
 * Shared NumNode → tape emission. Both `compileTape` (which builds an
 * immutable `CompiledTape`) and `LiveTape` (which writes into growable
 * typed arrays) use this to walk a NumNode operand DAG in topological
 * order and dispatch on node kind.
 *
 * The strategy: a `TapeEmitter` owns the storage layout (whether
 * plain arrays for snapshot construction or typed arrays for in-place
 * live updates). This module owns the walk + per-kind dispatch.
 */
import {
  type NumNode,
  type Derivative,
  LiteralNum,
  Variable,
  UnaryOp,
  BinaryOp,
  ForeignFn,
  KIND_LIT,
  KIND_VAR,
  KIND_AND,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_NOT,
  KIND_OR,
  KIND_SELECT,
  SelectOp,
  isBinaryKind,
  isUnaryKind,
  type VarName,
} from "../../core/tree";

/**
 * Strategy for assigning tape slot indices to nodes. Implementations
 * own the underlying storage (opcodes/args/literals/varSlots arrays
 * or their typed-array equivalents).
 *
 * Each `emit*` returns the freshly-assigned slot index. `emitOperandSubgraph`
 * records that index in `nodeToIndex` and uses it to resolve operand
 * arguments for downstream consumers in the same walk.
 */
export interface TapeEmitter {
  emitLit(value: number): number;
  emitVar(name: VarName): number;
  emitUnary(op: number, operandIdx: number): number;
  emitBinary(op: number, leftIdx: number, rightIdx: number): number;
}

/** Reason an emit walk gave up partway through. */
export type EmitFailure = "foreign" | "derivative";

/** Outcome of {@link emitOperandSubgraph}. */
export type EmitResult =
  { ok: true; idx: number } | { ok: false; reason: EmitFailure };

/**
 * Optional handler for `KIND_DERIVATIVE` nodes. `compileTape` provides
 * one (derivatives are emitted as `OP_VAR` slots whose `argA` is fixed
 * up after the walk). `LiveTape` does not — derivatives bail.
 *
 * The handler returns the slot index it assigned for the derivative
 * node (typically by calling back into the emitter to push an OP_VAR
 * placeholder).
 */
export interface EmitOptions {
  onDerivative?: (node: Derivative) => number;
}

/**
 * Walk `root`'s operand DAG in topological (leaves-first) order.
 * For each node not yet in `nodeToIndex`, dispatch on kind and call
 * the appropriate `emit*` method on `emitter`. Records the returned
 * slot idx in `nodeToIndex`.
 *
 * Bails on foreign or derivative (when no `onDerivative` handler is
 * provided). Select nodes are lowered to the existing eager masking
 * expression `condition.and(ifNonZero).or(condition.not().and(ifZero))`.
 * Partial state may be left in `nodeToIndex` and the emitter's storage on
 * bail — callers that need rollback must handle it themselves.
 */
export function emitOperandSubgraph(
  root: NumNode,
  emitter: TapeEmitter,
  nodeToIndex: Map<NumNode, number>,
  opts: EmitOptions = {},
): EmitResult {
  // A dedicated post-order walk can prune an already-emitted node before
  // descending into it. This matters for multi-root tapes: each later root can
  // share most of its graph with earlier roots. The generic DAG traversal used
  // here previously allocated a fresh visited Set and walked those shared
  // subgraphs again for every root.
  //
  // Keep node and phase in parallel arrays rather than allocating one stack
  // entry object per visit. Children are pushed directly from node fields, so
  // the hot walk also avoids `childrenOfNumNode()` array allocations.
  const nodes: NumNode[] = [root];
  const expanded: boolean[] = [false];
  let bail: EmitFailure | null = null;

  const pushIfNeeded = (node: NumNode): void => {
    if (nodeToIndex.has(node)) return;
    nodes.push(node);
    expanded.push(false);
  };

  while (nodes.length > 0) {
    const top = nodes.length - 1;
    const n = nodes[top]!;

    if (nodeToIndex.has(n)) {
      nodes.pop();
      expanded.pop();
      continue;
    }

    const kind = n.kind;
    if (!expanded[top]) {
      expanded[top] = true;

      // Push in reverse evaluation order because the last item is visited
      // first. This preserves the old leaves-first left-to-right tape order.
      if (kind === KIND_FOREIGN) {
        const inputs = (n as ForeignFn).inputs;
        for (let i = inputs.length - 1; i >= 0; i--) pushIfNeeded(inputs[i]!);
      } else if (kind === KIND_SELECT) {
        const s = n as SelectOp;
        pushIfNeeded(s.ifZero);
        pushIfNeeded(s.ifNonZero);
        pushIfNeeded(s.condition);
      } else if (isUnaryKind(kind)) {
        pushIfNeeded((n as UnaryOp).original);
      } else if (isBinaryKind(kind)) {
        const b = n as BinaryOp;
        pushIfNeeded(b.right);
        pushIfNeeded(b.left);
      } else if (kind === KIND_DERIVATIVE) {
        pushIfNeeded((n as Derivative).variable);
      }
      continue;
    }

    let idx: number;
    if (kind === KIND_FOREIGN) {
      bail = "foreign";
      break;
    } else if (kind === KIND_LIT) {
      idx = emitter.emitLit((n as LiteralNum).value);
    } else if (kind === KIND_VAR) {
      idx = emitter.emitVar((n as Variable).name);
    } else if (kind === KIND_DERIVATIVE) {
      if (!opts.onDerivative) {
        bail = "derivative";
        break;
      }
      idx = opts.onDerivative(n as Derivative);
    } else if (isUnaryKind(kind)) {
      const childIdx = nodeToIndex.get((n as UnaryOp).original)!;
      idx = emitter.emitUnary(kind, childIdx);
    } else if (isBinaryKind(kind)) {
      const b = n as BinaryOp;
      const leftIdx = nodeToIndex.get(b.left)!;
      const rightIdx = nodeToIndex.get(b.right)!;
      idx = emitter.emitBinary(kind, leftIdx, rightIdx);
    } else if (kind === KIND_SELECT) {
      const s = n as SelectOp;
      const conditionIdx = nodeToIndex.get(s.condition)!;
      const ifNonZeroIdx = nodeToIndex.get(s.ifNonZero)!;
      const ifZeroIdx = nodeToIndex.get(s.ifZero)!;
      const lhs = emitter.emitBinary(KIND_AND, conditionIdx, ifNonZeroIdx);
      const notCondition = emitter.emitUnary(KIND_NOT, conditionIdx);
      const rhs = emitter.emitBinary(KIND_AND, notCondition, ifZeroIdx);
      idx = emitter.emitBinary(KIND_OR, lhs, rhs);
    } else {
      bail = "foreign";
      break;
    }

    nodeToIndex.set(n, idx);
    nodes.pop();
    expanded.pop();
  }

  if (bail) return { ok: false, reason: bail };
  return { ok: true, idx: nodeToIndex.get(root)! };
}
