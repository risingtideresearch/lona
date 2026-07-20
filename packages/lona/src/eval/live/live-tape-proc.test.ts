import { describe, expect, test } from "vitest";
import { defineProc, callProc } from "../../core/proc";
import { variableNum } from "../../core/num";
import { LiveTape } from "./live-tape";

// Procs on the incremental (LiveTape) path: a Call is emitted once (eagerly, all
// outputs) and cached persistently on the instance, so interning a second
// projection later aliases an existing slot rather than re-emitting — and, since
// closed procs put variables only in arguments, a variable update propagates
// through the normal per-slot sweep to every projection.

describe("LiveTape — proc calls", () => {
  test("second projection aliases the body and reflects variable updates", () => {
    // proc(a) = [a*a, a*a + 10] — the two outputs share the `a*a` subexpression.
    const proc = defineProc(1, ([a]) => {
      const s = a!.mul(a!);
      return [s, s.add(10)];
    });
    const [p0, p1] = callProc(proc, [variableNum("lt_proc_x")]);

    const tape = new LiveTape();
    tape.registerInitialVariableValue("lt_proc_x", 3);

    const i0 = tape.ensureInterned(p0!.n);
    expect(i0).not.toBe(-1);
    expect(tape.getValue(i0)).toBe(9);

    const lenAfterFirst = tape.length;

    tape.setVariable("lt_proc_x", 4);
    expect(tape.getValue(i0)).toBe(16);

    // Interning the second projection must emit NOTHING new (pure alias)…
    const i1 = tape.ensureInterned(p1!.n);
    expect(tape.length).toBe(lenAfterFirst);
    expect(i1).not.toBe(i0);
    // …and it reflects the already-applied variable update.
    expect(tape.getValue(i1)).toBe(26); // 16 + 10
  });

  test("distinct calls of one proc are independent (share only literals)", () => {
    const proc = defineProc(1, ([a]) => a!.add(2.5));
    const [c0] = callProc(proc, [variableNum("lt_proc_y")]);
    const [c1] = callProc(proc, [variableNum("lt_proc_z")]);

    const tape = new LiveTape();
    tape.registerInitialVariableValue("lt_proc_y", 1);
    tape.registerInitialVariableValue("lt_proc_z", 10);

    const a = tape.ensureInterned(c0!.n);
    const b = tape.ensureInterned(c1!.n);
    expect(tape.getValue(a)).toBe(3.5);
    expect(tape.getValue(b)).toBe(12.5);

    tape.setVariable("lt_proc_y", 100);
    expect(tape.getValue(a)).toBe(102.5);
    expect(tape.getValue(b)).toBe(12.5); // the other call is unaffected
  });
});
