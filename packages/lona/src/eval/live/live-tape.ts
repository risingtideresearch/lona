/**
 * Live tape — stateful, incremental evaluator for a NumNode DAG.
 *
 * Composition (not inheritance):
 *   - LiveTape owns a private {@link GrowableTape} for tape construction.
 *   - LiveTape implements {@link TapeEmitter} itself: each emit method
 *     delegates structural append to the inner tape, then layers per-slot
 *     live-eval bookkeeping (slotValues, slotEpoch, valueEpoch) on top.
 *   - The walker {@link emitOperandSubgraph} dispatches into LiveTape's
 *     emit methods, so the operand subgraph is interned through the
 *     inner tape AND the live state is set up in one pass.
 *
 * Per-slot state:
 *
 *   - slotEpoch[i] === epoch  → slot i's value is fresh for the current
 *     `epoch`. Fast-path read is just this check.
 *   - valueEpoch[i]           → highest epoch at which slot i's value
 *     actually changed (under `Object.is`). Downstream consumers use
 *     this to decide whether their caches need redoing.
 *
 * `epoch` bumps on every effective `setVariable`. A no-op write does not.
 *
 * `getValue(idx)` is "if stale, catchUp; return slotValues[idx]". The
 * default sweep is `wasmInterpSweep`. `jsInterpSweep` is a drop-in.
 *
 * Variables can be set before any node referencing them has been
 * interned — the value is stashed and applied on first intern.
 *
 * Foreign and derivative nodes cannot live on the tape; `ensureInterned`
 * returns {@link INTERN_FAILED} for subgraphs containing them.
 */
import { type Call, type NumNode, type VarName } from "../../core/tree";
import {
  GrowableTape,
  INTERN_FAILED,
  TAPE_INITIAL_CAPACITY,
  emitOperandSubgraph,
  type CallEmission,
  type TapeEmitter,
} from "../tape";
import { wasmInterpSweep } from "./sweep";

export { INTERN_FAILED } from "../tape";

// ---------------------------------------------------------------------------
// Pluggable sweep backend
// ---------------------------------------------------------------------------

/**
 * Read-only surface that LiveTape exposes to a {@link LiveTapeSweepFn}.
 *
 * The sweep mutates the typed arrays in place (`slotValues`, `slotEpoch`,
 * `valueEpoch`) — that's the contract — but does not mutate scalar fields.
 * The new `firstStaleSlot` is communicated as the return value of the
 * sweep function, not by writing to the context.
 *
 * Post-call invariants on the typed arrays:
 *   - `slotValues[i]` holds the current value of every slot in [0, length).
 *   - `slotEpoch[i] === epoch` for every `i` in [0, length).
 *   - `valueEpoch[i]` is updated to `epoch` for every slot whose value
 *     actually changed (under `Object.is`) since the prior call.
 *
 * `firstStaleSlot` (the input field) is a hint — the lowest known stale
 * slot index. Sweeps that walk the whole tape can ignore it; sweeps
 * that skip clean prefixes use it as a starting point.
 *
 * `tapeVersion` increases monotonically each time the tape grows;
 * sweep backends use it as a cache key for derived state.
 */
export interface LiveTapeSweepContext {
  readonly tapeVersion: number;
  readonly length: number;
  readonly opcodes: Uint8Array;
  readonly argA: Int32Array;
  readonly argB: Int32Array;
  readonly literalCount: number;
  readonly literals: Float64Array;
  readonly numVarSlots: number;
  readonly variableValues: Float64Array;
  readonly slotValues: Float64Array;
  readonly slotEpoch: Int32Array;
  readonly valueEpoch: Int32Array;
  readonly epoch: number;
  readonly firstStaleSlot: number;
}

/**
 * Pluggable full-sweep implementation. Returns the new
 * `firstStaleSlot` (typically `ctx.length` after a complete sweep).
 */
export type LiveTapeSweepFn = (ctx: LiveTapeSweepContext) => number;

// ---------------------------------------------------------------------------
// LiveTape
// ---------------------------------------------------------------------------

export class LiveTape implements TapeEmitter {
  /** Inner tape — owns structural state. Exposed read-only via getters. */
  private readonly tape = new GrowableTape();

  // Variable values (separate from the tape's varSlots/varNameToSlot,
  // which are structural).
  private variableValues = new Float64Array(TAPE_INITIAL_CAPACITY);
  private readonly pendingVarValues = new Map<VarName, number>();

  // Per-slot live state. Sized to match `tape.opcodes.length`.
  private slotValues = new Float64Array(TAPE_INITIAL_CAPACITY);
  private valueEpoch = new Int32Array(TAPE_INITIAL_CAPACITY);
  private slotEpoch = new Int32Array(TAPE_INITIAL_CAPACITY);

  // Per-`Call` emission records, persistent for this LiveTape instance so that
  // incrementally interning a second projection of an already-emitted call
  // reuses the emitted body (aliasing its output slot) instead of re-emitting.
  private readonly callEmissions = new Map<Call, CallEmission>();

  /** Bumped on every effective `setVariable`. Starts at 1 so the
   * Int32Array default of 0 means "stale". */
  private _epoch = 1;

  /**
   * Read-only accessor for the current epoch. Exposed for consumers
   * (e.g. `NumValueContext.epoch`) that need to detect whether
   * any value-changing `setVariable` has happened since a prior read.
   */
  get epoch(): number {
    return this._epoch;
  }

  /** Lower bound on the lowest stale slot index. */
  private firstStaleSlot = 0;

  private readonly sweepFn: LiveTapeSweepFn;

  /**
   * @param opts.sweepFn Override for the sweep implementation. Defaults
   *   to {@link wasmInterpSweep}.
   */
  constructor(opts: { sweepFn?: LiveTapeSweepFn } = {}) {
    this.sweepFn = opts.sweepFn ?? wasmInterpSweep;
  }

  // --------------------------------------------------------------------
  // Read-only accessors for the inner tape. Tests and consumers that
  // need to inspect tape state go through these.
  // --------------------------------------------------------------------

  get length(): number {
    return this.tape.length;
  }

  get nodeToIndex(): ReadonlyMap<NumNode, number> {
    return this.tape.nodeToIndex;
  }

  get varSlots(): readonly VarName[] {
    return this.tape.varSlots;
  }

  // --------------------------------------------------------------------
  // Public live-eval API
  // --------------------------------------------------------------------

  /**
   * Lazily intern `node` and any operand subgraph that is not yet on
   * the tape. Returns the slot index of `node`, or {@link INTERN_FAILED}
   * if the subgraph contains a foreign or derivative node.
   */
  ensureInterned(node: NumNode): number {
    const existing = this.tape.nodeToIndex.get(node);
    if (existing !== undefined) return existing;
    const result = emitOperandSubgraph(node, this, this.tape.nodeToIndex, {
      callEmissions: this.callEmissions,
    });
    return result.ok ? result.idx : INTERN_FAILED;
  }

  /**
   * Register a default value for a variable that has not yet been
   * interned. Idempotent (first registration wins). No-op if the
   * variable is already interned.
   */
  registerInitialVariableValue(name: VarName, value: number): void {
    if (this.tape.varNameToSlot.has(name)) return;
    if (this.pendingVarValues.has(name)) return;
    this.pendingVarValues.set(name, value);
  }

  /**
   * Set the current value of a variable. Stash if not yet interned
   * (overwrites prior `registerInitialVariableValue`). No-op writes
   * (Object.is-equal to current) do not bump the epoch.
   */
  setVariable(name: VarName, value: number): void {
    const slot = this.tape.varNameToSlot.get(name);
    if (slot === undefined) {
      this.pendingVarValues.set(name, value);
      return;
    }
    if (Object.is(this.variableValues[slot], value)) return;
    this.variableValues[slot] = value;
    this._epoch++;

    const tapeIdx = this.tape.varSlotToTapeIdx[slot]!;
    this.slotEpoch[tapeIdx] = 0;
    if (tapeIdx < this.firstStaleSlot) this.firstStaleSlot = tapeIdx;
  }

  /**
   * Read the current value of slot `idx`. O(1) on the hot path. On a
   * miss, runs a full sweep via the configured `sweepFn`.
   */
  getValue(idx: number): number {
    if (this.slotEpoch[idx] === this._epoch) return this.slotValues[idx]!;
    this.catchUp();
    return this.slotValues[idx]!;
  }

  /** Validate every stale slot via the configured sweep backend. */
  catchUp(): void {
    this.firstStaleSlot = this.sweepFn({
      tapeVersion: this.tape.version,
      length: this.tape.length,
      opcodes: this.tape.opcodes,
      argA: this.tape.argA,
      argB: this.tape.argB,
      literalCount: this.tape.numLiterals,
      literals: this.tape.literals,
      numVarSlots: this.tape.varSlots.length,
      variableValues: this.variableValues,
      slotValues: this.slotValues,
      slotEpoch: this.slotEpoch,
      valueEpoch: this.valueEpoch,
      epoch: this._epoch,
      firstStaleSlot: this.firstStaleSlot,
    });
  }

  // --------------------------------------------------------------------
  // TapeEmitter implementation. Each method delegates structural append
  // to the inner tape, then sets up per-slot live state. Called by
  // emitOperandSubgraph during ensureInterned.
  // --------------------------------------------------------------------

  emitLit(value: number): number {
    const idx = this.tape.emitLit(value);
    this.syncCapacity();
    this.slotValues[idx] = value;
    this.slotEpoch[idx] = this._epoch;
    this.valueEpoch[idx] = this._epoch;
    return idx;
  }

  emitVar(name: VarName): number {
    const isFreshSlot = !this.tape.varNameToSlot.has(name);
    const idx = this.tape.emitVar(name);
    this.syncCapacity();
    const slot = this.tape.varNameToSlot.get(name)!;
    if (isFreshSlot) {
      this.ensureVarValueCapacity(slot + 1);
      const initial = this.pendingVarValues.get(name) ?? 0;
      this.variableValues[slot] = initial;
      this.pendingVarValues.delete(name);
    }
    this.slotValues[idx] = this.variableValues[slot]!;
    this.slotEpoch[idx] = this._epoch;
    this.valueEpoch[idx] = this._epoch;
    return idx;
  }

  emitUnary(op: number, operandIdx: number): number {
    const idx = this.tape.emitUnary(op, operandIdx);
    this.syncCapacity();
    this.markStale(idx);
    return idx;
  }

  emitBinary(op: number, leftIdx: number, rightIdx: number): number {
    const idx = this.tape.emitBinary(op, leftIdx, rightIdx);
    this.syncCapacity();
    this.markStale(idx);
    return idx;
  }

  // --------------------------------------------------------------------
  // Storage internals
  // --------------------------------------------------------------------

  /**
   * After delegating an emit to the inner tape, the tape's structural
   * arrays may have grown. Resize the live arrays in lockstep so
   * `slotValues[idx]` is addressable for the slot we just emitted.
   */
  private syncCapacity(): void {
    const cap = this.tape.opcodes.length;
    if (this.slotValues.length === cap) return;
    const newSlotValues = new Float64Array(cap);
    newSlotValues.set(this.slotValues);
    this.slotValues = newSlotValues;
    const newValueEpoch = new Int32Array(cap);
    newValueEpoch.set(this.valueEpoch);
    this.valueEpoch = newValueEpoch;
    const newSlotEpoch = new Int32Array(cap);
    newSlotEpoch.set(this.slotEpoch);
    this.slotEpoch = newSlotEpoch;
  }

  private ensureVarValueCapacity(needed: number): void {
    if (needed <= this.variableValues.length) return;
    let cap = this.variableValues.length;
    while (cap < needed) cap *= 2;
    const next = new Float64Array(cap);
    next.set(this.variableValues);
    this.variableValues = next;
  }

  private markStale(idx: number): void {
    this.slotEpoch[idx] = 0;
    this.valueEpoch[idx] = 0;
    if (idx < this.firstStaleSlot) this.firstStaleSlot = idx;
  }
}
