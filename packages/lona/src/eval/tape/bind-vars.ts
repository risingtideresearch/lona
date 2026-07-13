/**
 * Shared helper: extract variable + derivative values from a user-supplied
 * Map into a flat Float64Array laid out by tape slot order.
 *
 * Every CPU backend needs this exact logic — inline it once so they share it.
 */
import type { VarName } from "../../core/tree";
import type { CompiledTape } from "./compiled-tape";

/**
 * Mutable, index-writable container. Compatible with Float64Array, Float32Array,
 * and plain number arrays — every CPU backend has a different destination type.
 */
export type NumberSink = { length: number; [i: number]: number };

/** Underlying slot-based variant — callers can pass tape layout fields directly. */
export function bindVarSlots(
  varSlots: readonly VarName[],
  numVars: number,
  vars: Map<VarName, number> | Map<string, number>,
  derivatives: Map<VarName, number> | Map<string, number> | undefined,
  dest: NumberSink,
): void {
  for (let v = 0; v < numVars; v++) {
    dest[v] = (vars as Map<VarName, number>).get(varSlots[v]!) ?? 0;
  }
  if (derivatives) {
    for (let v = numVars; v < varSlots.length; v++) {
      dest[v] = (derivatives as Map<VarName, number>).get(varSlots[v]!) ?? 0;
    }
  } else {
    for (let v = numVars; v < varSlots.length; v++) {
      dest[v] = 0;
    }
  }
}

export function bindVarMap(
  tape: CompiledTape,
  vars: Map<VarName, number> | Map<string, number>,
  derivatives: Map<VarName, number> | Map<string, number> | undefined,
  dest: NumberSink,
): void {
  bindVarSlots(tape.varSlots, tape.numVars, vars, derivatives, dest);
}
