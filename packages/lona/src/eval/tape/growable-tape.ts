/**
 * `GrowableTape` — a `CompiledTape`-shaped structure that supports
 * incremental append. The tape arrays match `CompiledTape`'s typed-array
 * layout exactly, so backend interpreters that consume `CompiledTape`
 * arrays (e.g. `interpretTapeValues`) can read directly from a
 * `GrowableTape` with no per-call conversion.
 *
 * Implements {@link TapeEmitter}, so {@link emitOperandSubgraph} can
 * walk a NumNode DAG and dispatch new nodes into the tape's storage.
 *
 * Self-contained: `GrowableTape` does not know about live values,
 * dirty bits, or sweeping. Higher-level concerns (notably `LiveTape`)
 * compose a `GrowableTape` and add their own per-slot bookkeeping.
 */
import { KIND_LIT, type NumNode, type VarName } from "../../core/tree";
import type { CompiledTape } from "./compiled-tape";
import { emitOperandSubgraph, type TapeEmitter } from "./emit";
import { OP_VAR } from "./opcodes";

/** Initial capacity for op arrays. */
export const TAPE_INITIAL_CAPACITY = 64;
/** Initial capacity for the literal pool. */
export const TAPE_INITIAL_LITERAL_CAPACITY = 16;

/** Returned by {@link GrowableTape.ensureInterned} for foreign/derivative subgraphs. */
export const INTERN_FAILED = -1;

export class GrowableTape implements TapeEmitter {
  // Tape structural state — same shape as CompiledTape, but the arrays
  // double on overflow. Valid entries live in [0, tapeLength) for the
  // op arrays and [0, literalCount) for the literal pool. Public for
  // composing classes (e.g. LiveTape) and for sweep backends.
  opcodes = new Uint8Array(TAPE_INITIAL_CAPACITY);
  argA = new Int32Array(TAPE_INITIAL_CAPACITY);
  argB = new Int32Array(TAPE_INITIAL_CAPACITY);
  literals = new Float64Array(TAPE_INITIAL_LITERAL_CAPACITY);
  private tapeLength = 0;
  private literalCount = 0;

  /** Names of variables in the order they were first interned. */
  readonly varSlots: VarName[] = [];
  readonly varNameToSlot = new Map<VarName, number>();
  /** Tape index of each Variable node, parallel to `varSlots`. */
  readonly varSlotToTapeIdx: number[] = [];

  /** Sparse — only populated for nodes that have been interned. */
  readonly nodeToIndex = new Map<NumNode, number>();

  /** Monotonic counter, bumps each time a slot is appended. */
  private tapeVersion = 0;

  /** Number of slots currently appended to the tape. */
  get length(): number {
    return this.tapeLength;
  }

  /** Number of literals appended to the literal pool. */
  get numLiterals(): number {
    return this.literalCount;
  }

  /** See {@link tapeVersion}. */
  get version(): number {
    return this.tapeVersion;
  }

  /**
   * Lazily intern `node` and any operand subgraph that is not yet on
   * the tape. Returns the slot index of `node`, or {@link INTERN_FAILED}
   * if the subgraph contains a foreign or derivative node.
   *
   * Callers that need to layer their own per-slot state on top of the
   * append (e.g. LiveTape) should not use this method — they should
   * pass their own emitter to {@link emitOperandSubgraph} directly.
   */
  ensureInterned(node: NumNode): number {
    const existing = this.nodeToIndex.get(node);
    if (existing !== undefined) return existing;
    const result = emitOperandSubgraph(node, this, this.nodeToIndex);
    return result.ok ? result.idx : INTERN_FAILED;
  }

  /**
   * Snapshot the current tape state into an immutable {@link CompiledTape}.
   */
  toCompiledTape(rootIndices: [number, ...number[]]): CompiledTape {
    return {
      opcodes: this.opcodes.slice(0, this.tapeLength),
      argA: this.argA.slice(0, this.tapeLength),
      argB: this.argB.slice(0, this.tapeLength),
      literals: this.literals.slice(0, this.literalCount),
      varSlots: [...this.varSlots],
      numVars: this.varSlots.length,
      rootIndices,
    };
  }

  // --------------------------------------------------------------------
  // TapeEmitter implementation. Pure structural append — no per-slot
  // bookkeeping. Composing classes that need bookkeeping wrap the
  // calls (rather than overriding) and add their own state.
  // --------------------------------------------------------------------

  emitLit(value: number): number {
    const idx = this.allocateSlot();
    const litIdx = this.allocateLiteralSlot(value);
    this.opcodes[idx] = KIND_LIT;
    this.argA[idx] = litIdx;
    this.argB[idx] = 0;
    return idx;
  }

  emitVar(name: VarName): number {
    const idx = this.allocateSlot();
    let slot = this.varNameToSlot.get(name);
    if (slot === undefined) {
      slot = this.varSlots.length;
      this.varSlots.push(name);
      this.varNameToSlot.set(name, slot);
    }
    this.varSlotToTapeIdx[slot] = idx;
    this.opcodes[idx] = OP_VAR;
    this.argA[idx] = slot;
    this.argB[idx] = 0;
    return idx;
  }

  emitUnary(op: number, operandIdx: number): number {
    const idx = this.allocateSlot();
    this.opcodes[idx] = op;
    this.argA[idx] = operandIdx;
    this.argB[idx] = 0;
    return idx;
  }

  emitBinary(op: number, leftIdx: number, rightIdx: number): number {
    const idx = this.allocateSlot();
    this.opcodes[idx] = op;
    this.argA[idx] = leftIdx;
    this.argB[idx] = rightIdx;
    return idx;
  }

  // --------------------------------------------------------------------
  // Storage internals
  // --------------------------------------------------------------------

  private allocateSlot(): number {
    const idx = this.tapeLength;
    this.ensureTapeCapacity(idx + 1);
    this.tapeLength = idx + 1;
    this.tapeVersion++;
    return idx;
  }

  private allocateLiteralSlot(value: number): number {
    const idx = this.literalCount;
    this.ensureLiteralCapacity(idx + 1);
    this.literals[idx] = value;
    this.literalCount = idx + 1;
    return idx;
  }

  private ensureTapeCapacity(needed: number): void {
    if (needed <= this.opcodes.length) return;
    let cap = this.opcodes.length;
    while (cap < needed) cap *= 2;
    const newOpcodes = new Uint8Array(cap);
    newOpcodes.set(this.opcodes);
    this.opcodes = newOpcodes;
    const newArgA = new Int32Array(cap);
    newArgA.set(this.argA);
    this.argA = newArgA;
    const newArgB = new Int32Array(cap);
    newArgB.set(this.argB);
    this.argB = newArgB;
  }

  private ensureLiteralCapacity(needed: number): void {
    if (needed <= this.literals.length) return;
    let cap = this.literals.length;
    while (cap < needed) cap *= 2;
    const next = new Float64Array(cap);
    next.set(this.literals);
    this.literals = next;
  }
}
