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
import { visitFromLeaves } from "../../dag/traversal";
import {
  type NumNode,
  type Derivative,
  LiteralNum,
  Variable,
  UnaryOp,
  BinaryOp,
  childrenOfNumNode,
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
  let bail: EmitFailure | null = null;

  visitFromLeaves(root, childrenOfNumNode, (n) => {
    if (bail) return;
    if (nodeToIndex.has(n)) return;

    const kind = n.kind;
    if (kind === KIND_FOREIGN) {
      bail = "foreign";
      return;
    }

    let idx: number;
    if (kind === KIND_LIT) {
      idx = emitter.emitLit((n as LiteralNum).value);
    } else if (kind === KIND_VAR) {
      idx = emitter.emitVar((n as Variable).name);
    } else if (kind === KIND_DERIVATIVE) {
      if (!opts.onDerivative) {
        bail = "derivative";
        return;
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
      return;
    }
    nodeToIndex.set(n, idx);
  });

  if (bail) return { ok: false, reason: bail };
  return { ok: true, idx: nodeToIndex.get(root)! };
}
