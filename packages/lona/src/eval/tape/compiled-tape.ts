/**
 * `CompiledTape` — the linearized intermediate representation used by every
 * eval backend. A tape is a sequence of `opcodes[i]` (one byte each) with
 * two argument arrays `argA[i]` / `argB[i]`, a pool of f64 literals, and a
 * variable-slot table.
 */
import type { VarName } from "../../core/tree";
import type { TapeAssertionKind } from "./assertions";

export interface TapeAssertion {
  id: number;
  tapeIndex: number;
  kind: TapeAssertionKind;
  source?: unknown;
}

export interface CompiledTape {
  opcodes: Uint8Array;
  argA: Int32Array;
  argB: Int32Array;
  literals: Float64Array;
  /** All variable and derivative names in slot order.
   *  Slots [0, numVars) are variables; slots [numVars, varSlots.length) are
   *  derivatives (keyed by the base variable name). */
  varSlots: VarName[];
  /** Number of regular variable slots (derivatives start at this index) */
  numVars: number;
  /** Indices of all root nodes. Always has at least one element. */
  rootIndices: [number, ...number[]];
  /** Optional metadata for assertion opcodes in specialized tapes. */
  assertions?: TapeAssertion[];
}
