import { describe, expect, test } from "vitest";
import {
  NumValueContext,
  variable,
  asNumber,
  currentValueEpoch,
  setVariable,
  withContext,
  initDefaultContext,
  clearDefaultContext,
} from "./value-context";
import { variableNum } from "../core/num";

describe("Behavior of global functions relying on the current context", () => {
  test("asNumber on a folded literal returns the literal value", () => {
    withContext(new NumValueContext(), () => {
      const v = variable("x", 0);
      const e = v.mul(0).add(7); // folds to literal 7
      expect(asNumber(e)).toBe(7);
    });
  });

  test("asNumber on a raw number passes through", () => {
    withContext(new NumValueContext(), () => {
      expect(asNumber(42)).toBe(42);
      expect(asNumber(-3.5)).toBe(-3.5);
    });
  });

  test("asNumber reflects setVariable on a simple expression", () => {
    withContext(new NumValueContext(), () => {
      const v = variable("x", 3);
      expect(asNumber(v)).toBe(3);
      setVariable("x", 11);
      expect(asNumber(v)).toBe(11);
      setVariable("x", 0);
      expect(asNumber(v)).toBe(0);
    });
  });

  test("asNumber reflects setVariable on a deeper expression", () => {
    withContext(new NumValueContext(), () => {
      const x = variable("x", 2);
      const y = variable("y", 5);
      const e = x.mul(x).add(y); // x^2 + y
      expect(asNumber(e)).toBe(2 * 2 + 5);
      setVariable("x", 4);
      expect(asNumber(e)).toBe(4 * 4 + 5);
      setVariable("y", -1);
      expect(asNumber(e)).toBe(4 * 4 + -1);
    });
  });

  test("setVariable before any consumer interns the var still works", () => {
    withContext(new NumValueContext(), () => {
      // Set a variable that has never been seen by the live tape.
      setVariable("x", 99);
      const v = variable("x", 0); // initial 0 — but pre-intern setVariable wins
      expect(asNumber(v)).toBe(99);
    });
  });

  test("variable() registers initial value with the live tape", () => {
    withContext(new NumValueContext(), () => {
      const v = variable("x", 7.5);
      // No setVariable call. asNumber must see the registered initial value.
      expect(asNumber(v)).toBe(7.5);
    });
  });

  test("variable() registration is first-write-wins per name", () => {
    withContext(new NumValueContext(), () => {
      const a = variable("x", 100);
      const b = variable("x", 200); // ignored — same hash-consed var
      expect(a.n).toBe(b.n);
      expect(asNumber(a)).toBe(100);
      // setVariable still overwrites:
      setVariable("x", 200);
      expect(asNumber(a)).toBe(200);
    });
  });

  test("variable() initial value ignored if asNumber() called first", () => {
    const a = variableNum("x");
    withContext(new NumValueContext(), () => {
      expect(asNumber(a)).toBe(0); // auto-registers with initial value 0
      const b = variable("x", 200);
      expect(a.n).toBe(b.n);
      expect(asNumber(a)).toBe(0);
      // setVariable still overwrites:
      setVariable("x", 200);
      expect(asNumber(a)).toBe(200);
    });
  });

  test("no-op setVariable does not change downstream values", () => {
    withContext(new NumValueContext(), () => {
      const x = variable("x", 4);
      const e = x.mul(3);
      expect(asNumber(e)).toBe(12);
      // Same value — no effect, no error.
      setVariable("x", 4);
      expect(asNumber(e)).toBe(12);
      setVariable("x", 4);
      expect(asNumber(e)).toBe(12);
    });
  });

  test("repeated asNumber on an unchanged Num is consistent", () => {
    withContext(new NumValueContext(), () => {
      const x = variable("x", 8);
      const e = x.add(2).mul(x);
      const v1 = asNumber(e);
      const v2 = asNumber(e);
      const v3 = asNumber(e);
      expect(v1).toBe(80); // (8+2)*8
      expect(v2).toBe(v1);
      expect(v3).toBe(v1);
    });
  });

  test("asNumber matches Num.eval for unchanged expressions", () => {
    withContext(new NumValueContext(), () => {
      // The live tape (asNumber) and the compiled routine (Num.eval) are
      // independent code paths. They must agree on initial-value reads.
      const xVal = 2.5;
      const yVal = -1.25;
      const x = variable("x", xVal);
      const y = variable("y", yVal);
      const vars = { x: xVal, y: yVal };
      const exprs = [
        x,
        y,
        x.add(y),
        x.sub(y),
        x.mul(y),
        x.div(y),
        x.square(),
        x.sqrt(),
        x.add(y).mul(x.sub(y)),
        x.cos().add(y.sin()),
      ];
      for (const e of exprs) {
        expect(asNumber(e)).toBe(e.eval(vars));
      }
    });
  });

  test("currentValueEpoch advances on a value-changing setVariable", () => {
    withContext(new NumValueContext(), () => {
      const v = variable("x", 0);
      asNumber(v); // intern
      const e0 = currentValueEpoch();
      setVariable("x", 1);
      expect(currentValueEpoch()).toBeGreaterThan(e0);
    });
  });

  test("currentValueEpoch is stable across no-op setVariable", () => {
    withContext(new NumValueContext(), () => {
      const v = variable("x", 7);
      asNumber(v);
      const e0 = currentValueEpoch();
      setVariable("x", 7); // Object.is-equal — no effect
      expect(currentValueEpoch()).toBe(e0);
    });
  });

  test("currentValueEpoch follows the active context", () => {
    const outerCtx = new NumValueContext();
    const innerCtx = new NumValueContext();
    withContext(outerCtx, () => {
      const outerEpochBefore = currentValueEpoch();
      withContext(innerCtx, () => {
        // Build and intern a var inside innerCtx so the next
        // setVariable hits a real slot (and thus bumps the epoch).
        const v = variable("x", 0);
        asNumber(v);
        const e0 = currentValueEpoch();
        setVariable("x", 1);
        expect(currentValueEpoch()).toBeGreaterThan(e0);
      });
      // Outer context's epoch untouched.
      expect(currentValueEpoch()).toBe(outerEpochBefore);
    });
  });

  test("contexts are independent: changes in one do not affect another", () => {
    const outerCtx = new NumValueContext();
    const innerCtx = new NumValueContext();
    withContext(outerCtx, () => {
      const v = variable("x", 1);
      expect(asNumber(v)).toBe(1);
      // Set the variable on innerCtx — must not affect outerCtx.
      withContext(innerCtx, () => {
        innerCtx.setVariable("x", 999);
      });
      // Back in outerCtx — innerCtx changes did not affect us.
      expect(asNumber(v)).toBe(1);
      // setVariable on the active context still works.
      setVariable("x", 42);
      expect(asNumber(v)).toBe(42);
    });
  });

  test("default context is required for global functions outside a withContext block", () => {
    expect(() => variable("x", 0)).toThrow();
    expect(() => asNumber(5)).toThrow();
    expect(() => setVariable("x", 1)).toThrow();
    expect(() => currentValueEpoch()).toThrow();
  });

  test("initDefaultContext makes it possible to use global functions", () => {
    initDefaultContext();
    const v = variable("x", 1);
    expect(asNumber(v)).toBe(1);
    clearDefaultContext();
  });

  test("clearDefaultContext removes the default context.", () => {
    initDefaultContext();
    clearDefaultContext();
    expect(() => variable("x", 0)).toThrow();
  });

  test("initDefaultContext throws if called more than once without clearing", () => {
    initDefaultContext();
    expect(() => initDefaultContext()).toThrow();
    clearDefaultContext();
  });

  test("initDefaultContext can be called again after clearing, creating a new context", () => {
    initDefaultContext();
    const a = variable("x", 1);
    expect(asNumber(a)).toBe(1);
    clearDefaultContext();

    initDefaultContext();
    expect(asNumber(a)).toBe(0); // "x" auto-registered with initial value 0 in the new context
    const b = variable("x", 2); // initial value ignored — same hash-consed var
    expect(a.n).toBe(b.n);
    expect(asNumber(b)).toBe(0);
    clearDefaultContext();
  });

  test("initDefaultContext throws if called within a withContext block", () => {
    withContext(new NumValueContext(), () => {
      expect(() => initDefaultContext()).toThrow();
    });
    // But ok after
    initDefaultContext();
    clearDefaultContext();
  });

  test("clearDefaultContext throws if called within a withContext block", () => {
    initDefaultContext();
    withContext(new NumValueContext(), () => {
      expect(() => clearDefaultContext()).toThrow();
    });
    // But ok after
    clearDefaultContext();
  });
});
