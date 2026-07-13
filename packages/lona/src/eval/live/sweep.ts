/**
 * Live-tape sweep machinery.
 *
 * - {@link TapeInterpreter}: signature shared by the js-interp and
 *   wasm-interp tape interpreters.
 * - {@link createInterpreterSweep}: turn any `TapeInterpreter` into a
 *   {@link LiveTapeSweepFn} that does the post-pass `valueEpoch`
 *   tracking on top.
 * - {@link jsInterpSweep}, {@link wasmInterpSweep}: pre-bound sweeps
 *   for the two CPU interpreters.
 */
import { interpretTapeValues as jsInterpretTapeValues } from "../routines/backends/js-interp/tape-eval";
import { interpretTapeValues as wasmInterpretTapeValues } from "../routines/backends/wasm-interp/tape-eval";
import type { LiveTapeSweepFn } from "./live-tape";

/**
 * Contract for a tape interpreter usable as a live-tape sweep backend.
 * Mirrors the signature of `interpretTapeValues` exported by both
 * `js-interp/tape-eval.ts` and `wasm-interp/tape-eval.ts`.
 */
export type TapeInterpreter = (
  opcodes: Uint8Array,
  argA: Int32Array,
  argB: Int32Array,
  literals: Float64Array,
  len: number,
  varValues: Float64Array,
  values: Float64Array,
) => void;

/**
 * Build a {@link LiveTapeSweepFn} from a {@link TapeInterpreter}.
 *
 * The returned sweep:
 *   1. Snapshots the prior `slotValues` (one Float64Array allocation
 *      per call — small relative to the sweep itself).
 *   2. Runs the interpreter to produce fresh slot values in place.
 *   3. Post-passes: bumps `valueEpoch[i]` only where the new value
 *      actually differs from the prior under `Object.is`. Sets every
 *      `slotEpoch[i] = epoch`.
 *   4. Returns the new `firstStaleSlot` (`ctx.length`).
 *
 * Step 3 preserves downstream cache short-circuiting — consumers that
 * key off `valueEpoch` see exactly the slots whose values changed.
 */
export function createInterpreterSweep(
  interpret: TapeInterpreter,
): LiveTapeSweepFn {
  return (ctx) => {
    const len = ctx.length;
    if (len === 0) return 0;
    const epoch = ctx.epoch;
    const slotValues = ctx.slotValues;
    const slotEpoch = ctx.slotEpoch;
    const valueEpoch = ctx.valueEpoch;

    const prev = new Float64Array(len);
    prev.set(slotValues.subarray(0, len));

    interpret(
      ctx.opcodes,
      ctx.argA,
      ctx.argB,
      ctx.literals,
      len,
      ctx.variableValues,
      slotValues,
    );

    for (let i = 0; i < len; i++) {
      if (!Object.is(prev[i], slotValues[i])) {
        valueEpoch[i] = epoch;
      }
      slotEpoch[i] = epoch;
    }
    return len;
  };
}

/** Default sweep — wraps the js-interp tape interpreter. */
export const jsInterpSweep = createInterpreterSweep(jsInterpretTapeValues);

/** Sweep that wraps the wasm-interp tape interpreter. */
export const wasmInterpSweep = createInterpreterSweep(wasmInterpretTapeValues);
