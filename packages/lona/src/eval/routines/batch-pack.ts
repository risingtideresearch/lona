/**
 * Shared helpers for turning user-friendly per-variable batch inputs
 * (Record/Map of `name → number[]`) into the interleaved Float32Array layout
 * expected by vectorized backends.
 */
import type { VarName } from "../../core/tree";
import type { VarBatch } from "./types";

/**
 * Inspect a VarBatch and infer `numPoints` (max array length across all
 * variables). Returns 0 if the batch is empty.
 */
export function inferNumPoints(vars: VarBatch): number {
  let n = 0;
  if (vars instanceof Map) {
    for (const arr of vars.values()) {
      if (arr.length > n) n = arr.length;
    }
  } else {
    for (const key of Object.keys(vars)) {
      const arr = vars[key]!;
      if (arr.length > n) n = arr.length;
    }
  }
  return n;
}

/** Look up a variable's per-point array from a VarBatch, or null if absent. */
function lookupVar(vars: VarBatch, name: VarName): number[] | null {
  if (vars instanceof Map) {
    return (
      (vars as Map<VarName, number[]>).get(name) ??
      (vars as unknown as Map<string, number[]>).get(
        name as unknown as string,
      ) ??
      null
    );
  }
  return (vars as Record<string, number[]>)[name as unknown as string] ?? null;
}

/**
 * Pack per-variable batch arrays into an interleaved Float32Array laid out
 * as `[v0_p0, v1_p0, …, v0_p1, v1_p1, …]` — exactly what the GPU backends
 * consume as `varData`.
 *
 * Any variable not present in `vars` (or shorter than `numPoints`) is filled
 * with 0 for the missing slots.
 */
export function packVarBatch(
  varSlots: readonly VarName[],
  numVars: number,
  vars: VarBatch,
  numPoints: number,
): Float32Array {
  const data = new Float32Array(numPoints * numVars);
  for (let s = 0; s < numVars; s++) {
    const values = lookupVar(vars, varSlots[s]!);
    if (!values) continue;
    const limit = Math.min(values.length, numPoints);
    for (let p = 0; p < limit; p++) {
      data[p * numVars + s] = values[p]!;
    }
  }
  return data;
}

/** Resolve batch columns once so CPU loops can reuse a single row map. */
export function resolveVarBatchColumns(
  varSlots: readonly VarName[],
  numVars: number,
  vars: VarBatch,
): ReadonlyArray<number[] | null> {
  const columns = new Array<number[] | null>(numVars);
  for (let s = 0; s < numVars; s++) {
    columns[s] = lookupVar(vars, varSlots[s]!);
  }
  return columns;
}

/** Overwrite a reusable row map with the point values at index `p`. */
export function writeVarBatchRow(
  row: Map<VarName, number>,
  varSlots: readonly VarName[],
  numVars: number,
  columns: ReadonlyArray<number[] | null>,
  p: number,
): void {
  for (let s = 0; s < numVars; s++) {
    row.set(varSlots[s]!, columns[s]?.[p] ?? 0);
  }
}
