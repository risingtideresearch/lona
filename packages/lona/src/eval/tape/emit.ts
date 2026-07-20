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
  KIND_CALL,
  KIND_PROJECT,
  Call,
  Project,
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

/** Reason an emit walk gave up partway through. `proc` means a `Call`/`Project`
 *  was reached but the caller supplied no `callEmissions` cache (procs are
 *  only lowered by `compileTape`; other emitters reject them). */
export type EmitFailure = "foreign" | "derivative" | "proc";

/** Outcome of {@link emitOperandSubgraph}. */
export type EmitResult =
  { ok: true; idx: number } | { ok: false; reason: EmitFailure };

/**
 * Per-`Call` emission record, owned by one compilation (a single `compileTape`
 * call, or a persistent `LiveTape` instance). NOT stored on the `Call` object —
 * the same graph can be compiled into multiple tapes with different slots.
 *
 * A call is emitted eagerly and completely on first contact: its body is emitted
 * once and every output root's tape slot is recorded here. Each `Project` is
 * then a pure alias to `outputSlots[output]` — no tape op is appended for it.
 * The body-node → slot map is *not* retained (it is scratch during emission);
 * because all outputs emit on first contact, later projections need only slots.
 */
export interface CallEmission {
  readonly outputSlots: readonly number[];
  readonly ok: EmitResult;
}

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
  /**
   * Per-compilation `Call` → {@link CallEmission} cache. Required to lower
   * `Call`/`Project` nodes; when absent, a proc-bearing graph bails with
   * `"proc"`. `compileTape` supplies a fresh map per compilation; a `LiveTape`
   * that supports procs supplies a persistent per-instance map so incremental
   * interning of a second projection reuses the emitted body.
   */
  callEmissions?: Map<Call, CallEmission>;
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

  // A `Call` has no scalar tape slot, so its "already processed" state lives in
  // the `callEmissions` cache, not `nodeToIndex`. Everything else resolves by
  // its assigned slot.
  const isResolved = (node: NumNode): boolean =>
    node.kind === KIND_CALL
      ? (opts.callEmissions?.has(node as Call) ?? false)
      : nodeToIndex.has(node);

  const pushIfNeeded = (node: NumNode): void => {
    if (isResolved(node)) return;
    nodes.push(node);
    expanded.push(false);
  };

  while (nodes.length > 0) {
    const top = nodes.length - 1;
    const n = nodes[top]!;

    if (isResolved(n)) {
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
      } else if (kind === KIND_PROJECT) {
        pushIfNeeded((n as Project).call);
      } else if (kind === KIND_CALL) {
        const args = (n as Call).args;
        for (let i = args.length - 1; i >= 0; i--) pushIfNeeded(args[i]!);
      }
      continue;
    }

    // A Call has no scalar slot: emit its body once (shared across projections)
    // and record output slots in the cache. Popped without touching nodeToIndex.
    if (kind === KIND_CALL) {
      if (!opts.callEmissions) {
        bail = "proc";
        break;
      }
      const emission = emitCallBody(n as Call, emitter, nodeToIndex, opts);
      if (!emission.ok.ok) {
        bail = emission.ok.reason;
        break;
      }
      nodes.pop();
      expanded.pop();
      continue;
    }

    let idx: number;
    if (kind === KIND_FOREIGN) {
      bail = "foreign";
      break;
    } else if (kind === KIND_PROJECT) {
      // A projection is a pure alias to one of its call's output slots — no op
      // is appended. The call was emitted just above (it is a pushed child).
      const pr = n as Project;
      const emission = opts.callEmissions?.get(pr.call);
      if (!emission) {
        bail = "proc";
        break;
      }
      if (!emission.ok.ok) {
        bail = emission.ok.reason;
        break;
      }
      const slot = emission.outputSlots[pr.output];
      if (slot === undefined) {
        // Malformed projection (output out of range) — reject rather than store
        // an undefined slot as a tape index.
        bail = "proc";
        break;
      }
      idx = slot;
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

/**
 * The index map used while emitting one call's body. Params are pre-bound to the
 * caller's argument slots (plain local entries). Literals route to the shared
 * `outer` map so a body constant emits ONE tape slot across all calls of the
 * proc (a `LiteralNum` has global identity; its tape slot is value-only). All
 * other body nodes stay local, so they re-emit per call — which is the intended
 * unroll, since different calls bind different arguments.
 */
class BodyScope extends Map<NumNode, number> {
  constructor(private readonly outer: Map<NumNode, number>) {
    super();
  }
  override has(n: NumNode): boolean {
    return super.has(n) || (n.kind === KIND_LIT && this.outer.has(n));
  }
  override get(n: NumNode): number | undefined {
    const local = super.get(n);
    if (local !== undefined) return local;
    return n.kind === KIND_LIT ? this.outer.get(n) : undefined;
  }
  override set(n: NumNode, idx: number): this {
    if (n.kind === KIND_LIT) this.outer.set(n, idx);
    else super.set(n, idx);
    return this;
  }
}

/**
 * Emit a `Call`'s body once, recording every output's tape slot. Idempotent:
 * returns the cached {@link CallEmission} if the call was already emitted (this
 * is what lets a second `Project`, or a later `LiveTape.ensureInterned`, reuse
 * the body). The call's arguments must already be emitted in `outer`.
 */
function emitCallBody(
  call: Call,
  emitter: TapeEmitter,
  outer: Map<NumNode, number>,
  opts: EmitOptions,
): CallEmission {
  const cache = opts.callEmissions!;
  const cached = cache.get(call);
  if (cached) return cached;

  const scope = new BodyScope(outer);
  const params = call.proc.params;
  for (let i = 0; i < params.length; i++) {
    // Args are children of the Call, already emitted into `outer`.
    scope.set(params[i]!, outer.get(call.args[i]!)!);
  }

  const outputSlots: number[] = [];
  let ok: EmitResult = { ok: true, idx: -1 };
  for (const root of call.proc.body) {
    const res = emitOperandSubgraph(root, emitter, scope, opts);
    if (!res.ok) {
      ok = res;
      break;
    }
    outputSlots.push(res.idx);
  }

  const emission: CallEmission = { outputSlots, ok };
  cache.set(call, emission);
  return emission;
}
