import { describe, expect, test } from "vitest";
import { variableNum } from "../../core/num";
import { ifTruthyElse } from "../../api/ops";

import {
  compileValueRoutine,
  compileGradRoutine,
  type ValueRoutine,
  type MultiValueRoutine,
  type GradRoutine,
  type JacobianRoutine,
} from "./index";

const NumX = variableNum("x");
const NumY = variableNum("y");

describe("compileValueRoutine — value shape", () => {
  const node = NumX.add(NumY).mul(NumX).n;

  test.each([
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ] as const)("backend=%s: eval matches expected value", (backend) => {
    const r = compileValueRoutine([node], { backend });
    expect(r).not.toBeNull();
    expect(r!.shape).toBe("value");
    const v = (r as ValueRoutine).eval(
      new Map<string, number>([
        ["x", 2],
        ["y", 3],
      ]),
    );
    expect(v).toBeCloseTo((2 + 3) * 2, 10);
  });

  test("eval default backend", () => {
    const r = compileValueRoutine([node]);
    expect(r).not.toBeNull();
    expect(r!.shape).toBe("value");
  });

  test("evalAsync returns a Promise resolving to the value", async () => {
    const r = compileValueRoutine([node]) as ValueRoutine;
    const v = await r.evalAsync(
      new Map<string, number>([
        ["x", 1],
        ["y", 4],
      ]),
    );
    expect(v).toBeCloseTo(5, 10);
  });

  test("evalBatch on CPU backend loops eval", async () => {
    const r = compileValueRoutine([node], {
      backend: "js-interp",
    }) as ValueRoutine;
    const out = await r.evalBatch({ x: [1, 2, 3], y: [0, 0, 0] });
    expect(Array.from(out)).toEqual([1, 4, 9]);
  });

  test.each([
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ] as const)(
    "backend=%s: selectSpecialization traces and retraces on guard failure",
    (backend) => {
      const branchy = ifTruthyElse(
        NumX.lessThan(0),
        NumX.mul(NumX),
        NumY.mul(NumY),
      ).n;
      const r = compileValueRoutine([branchy], {
        backend,
        selectSpecialization: "trace",
      }) as ValueRoutine;

      expect(
        r.eval(
          new Map([
            ["x", -2],
            ["y", 5],
          ]),
        ),
      ).toBe(4);
      // Cross the select boundary. The guarded specialized tape should throw,
      // the routine should retrace, and the user should only see the value.
      expect(
        r.eval(
          new Map([
            ["x", 3],
            ["y", 5],
          ]),
        ),
      ).toBe(25);
      expect(
        r.eval(
          new Map([
            ["x", -4],
            ["y", 2],
          ]),
        ),
      ).toBe(16);
    },
  );
});

describe("compileValueRoutine — multi-value shape", () => {
  const a = NumX.add(NumY).n;
  const b = NumX.mul(NumY).n;

  test.each([
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ] as const)(
    "backend=%s: multi-root eval returns one value per root",
    (backend) => {
      const r = compileValueRoutine([a, b], { backend });
      expect(r).not.toBeNull();
      expect(r!.shape).toBe("multi-value");
      const mv = r as MultiValueRoutine;
      expect(mv.numRoots).toBe(2);
      const vals = mv.eval(
        new Map<string, number>([
          ["x", 3],
          ["y", 4],
        ]),
      );
      expect(vals).toEqual([7, 12]);
    },
  );

  test("evalBatch returns interleaved Float32Array", async () => {
    const r = compileValueRoutine([a, b]) as MultiValueRoutine;
    const out = await r.evalBatch({ x: [1, 2], y: [10, 20] });
    expect(Array.from(out)).toEqual([11, 10, 22, 40]);
  });
});

describe("compileGradRoutine — grad shape", () => {
  const node = NumX.mul(NumX).add(NumY.mul(NumY)).n;

  test.each(["js-interp", "wasm-interp"] as const)(
    "backend=%s: returns {val, gradient}",
    (backend) => {
      const r = compileGradRoutine([node], undefined, { backend });
      expect(r).not.toBeNull();
      expect(r!.shape).toBe("grad");
      const { val, gradient } = (r as GradRoutine).eval(
        new Map<string, number>([
          ["x", 2],
          ["y", 3],
        ]),
      );
      expect(val).toBeCloseTo(4 + 9, 10);
      // gradient order follows varSlots (x, y) because we didn't specify diffVars
      expect(gradient[0]).toBeCloseTo(4, 10); // ∂/∂x = 2x
      expect(gradient[1]).toBeCloseTo(6, 10); // ∂/∂y = 2y
    },
  );

  test.each(["js-interp", "wasm-interp", "wasm-codegen"] as const)(
    "backend=%s: selectSpecialization traces and retraces grad routines",
    (backend) => {
      const branchy = ifTruthyElse(
        NumX.lessThan(0),
        NumX.mul(NumX),
        NumY.mul(NumY),
      ).n;
      const r = compileGradRoutine([branchy], ["x", "y"], {
        backend,
        selectSpecialization: "trace",
      }) as GradRoutine;

      const a = r.eval(
        new Map([
          ["x", -2],
          ["y", 5],
        ]),
      );
      expect(a.val).toBe(4);
      expect(a.gradient[0]).toBeCloseTo(-4, 10);
      expect(a.gradient[1]).toBeCloseTo(0, 10);

      const b = r.eval(
        new Map([
          ["x", 3],
          ["y", 5],
        ]),
      );
      expect(b.val).toBe(25);
      expect(b.gradient[0]).toBeCloseTo(0, 10);
      expect(b.gradient[1]).toBeCloseTo(10, 10);
    },
  );
});

describe("compileGradRoutine — jacobian shape", () => {
  const a = NumX.mul(NumX).n;
  const b = NumX.mul(NumY).n;

  test("multi-root returns jacobian", () => {
    const r = compileGradRoutine([a, b]);
    expect(r).not.toBeNull();
    expect(r!.shape).toBe("jacobian");
    const { vals, jacobian } = (r as JacobianRoutine).eval(
      new Map<string, number>([
        ["x", 2],
        ["y", 5],
      ]),
    );
    expect(vals[0]).toBeCloseTo(4, 10);
    expect(vals[1]).toBeCloseTo(10, 10);
    expect(jacobian[0]![0]).toBeCloseTo(4, 10); // ∂a/∂x = 2x
    expect(jacobian[0]![1]).toBeCloseTo(0, 10); // ∂a/∂y = 0
    expect(jacobian[1]![0]).toBeCloseTo(5, 10); // ∂b/∂x = y
    expect(jacobian[1]![1]).toBeCloseTo(2, 10); // ∂b/∂y = x
  });

  test.each(["js-interp", "wasm-interp"] as const)(
    "backend=%s: selectSpecialization traces and retraces jacobian routines",
    (backend) => {
      const branchyA = ifTruthyElse(
        NumX.lessThan(0),
        NumX.mul(NumX),
        NumY.mul(NumY),
      ).n;
      const branchyB = NumX.mul(NumY).n;
      const r = compileGradRoutine([branchyA, branchyB], ["x", "y"], {
        backend,
        selectSpecialization: "trace",
      }) as JacobianRoutine;
      expect(r.shape).toBe("jacobian");
      expect(r.numRoots).toBe(2);

      // x < 0: branchyA = x², so ∂a/∂x = 2x, ∂a/∂y = 0.
      const a = r.eval(
        new Map([
          ["x", -2],
          ["y", 5],
        ]),
      );
      expect(a.vals[0]).toBe(4);
      expect(a.vals[1]).toBe(-10);
      expect(a.jacobian[0]![0]).toBeCloseTo(-4, 10); // ∂a/∂x = 2x
      expect(a.jacobian[0]![1]).toBeCloseTo(0, 10); // ∂a/∂y = 0
      expect(a.jacobian[1]![0]).toBeCloseTo(5, 10); // ∂b/∂x = y
      expect(a.jacobian[1]![1]).toBeCloseTo(-2, 10); // ∂b/∂y = x

      // Cross the select boundary (x ≥ 0): branchyA = y², so ∂a/∂x = 0,
      // ∂a/∂y = 2y. The guarded tape throws, the routine retraces.
      const b = r.eval(
        new Map([
          ["x", 3],
          ["y", 5],
        ]),
      );
      expect(b.vals[0]).toBe(25);
      expect(b.vals[1]).toBe(15);
      expect(b.jacobian[0]![0]).toBeCloseTo(0, 10); // ∂a/∂x = 0
      expect(b.jacobian[0]![1]).toBeCloseTo(10, 10); // ∂a/∂y = 2y
      expect(b.jacobian[1]![0]).toBeCloseTo(5, 10); // ∂b/∂x = y
      expect(b.jacobian[1]![1]).toBeCloseTo(3, 10); // ∂b/∂y = x
    },
  );

  test("full-trace mode also specializes jacobian routines", () => {
    const branchyA = ifTruthyElse(
      NumX.lessThan(0),
      NumX.mul(NumX),
      NumY.mul(NumY),
    ).n;
    const branchyB = NumX.mul(NumY).n;
    const r = compileGradRoutine([branchyA, branchyB], ["x", "y"], {
      backend: "wasm-interp",
      selectSpecialization: "full-trace",
    }) as JacobianRoutine;
    expect(r.shape).toBe("jacobian");

    const a = r.eval(
      new Map([
        ["x", -2],
        ["y", 5],
      ]),
    );
    expect(a.vals[0]).toBe(4);
    expect(a.jacobian[0]![0]).toBeCloseTo(-4, 10);

    const b = r.eval(
      new Map([
        ["x", 3],
        ["y", 5],
      ]),
    );
    expect(b.vals[0]).toBe(25);
    expect(b.jacobian[0]![1]).toBeCloseTo(10, 10);
  });

  test("selectSpecialization rejects backends without a sync-jacobian path", () => {
    const a = NumX.mul(NumX).n;
    const b = NumX.mul(NumY).n;
    expect(() =>
      compileGradRoutine([a, b], undefined, {
        backend: "wasm-codegen",
        selectSpecialization: "trace",
      }),
    ).toThrow(/jacobian backend 'wasm-codegen'/);
  });
});

describe("compileGradRoutine — symbolic backends", () => {
  const node = NumX.mul(NumX).add(NumY.mul(NumY)).n;

  test.each(["js-codegen-sym", "wasm-interp-sym", "wasm-codegen-sym"] as const)(
    "backend=%s: symbolic grad matches AD reference",
    (backend) => {
      const ref = compileGradRoutine([node])!;
      expect(ref).not.toBeNull();
      const sym = compileGradRoutine([node], undefined, { backend });
      expect(sym).not.toBeNull();
      expect(sym!.shape).toBe("grad");

      const vars = new Map<string, number>([
        ["x", 2],
        ["y", 3],
      ]);
      const refResult = (ref as GradRoutine).eval(vars);
      const symResult = (sym as GradRoutine).eval(vars);
      expect(symResult.val).toBeCloseTo(refResult.val, 10);
      expect(symResult.gradient[0]).toBeCloseTo(refResult.gradient[0]!, 10);
      expect(symResult.gradient[1]).toBeCloseTo(refResult.gradient[1]!, 10);
    },
  );

  test("symbolic jacobian matches AD reference", () => {
    const a = NumX.mul(NumX).n;
    const b = NumX.mul(NumY).n;
    const ref = compileGradRoutine([a, b])! as JacobianRoutine;
    const sym = compileGradRoutine([a, b], undefined, {
      backend: "wasm-codegen-sym",
    })! as JacobianRoutine;
    expect(sym).not.toBeNull();
    expect(sym.shape).toBe("jacobian");

    const vars = new Map<string, number>([
      ["x", 2],
      ["y", 5],
    ]);
    const refR = ref.eval(vars);
    const symR = sym.eval(vars);
    expect(symR.vals[0]).toBeCloseTo(refR.vals[0]!, 10);
    expect(symR.vals[1]).toBeCloseTo(refR.vals[1]!, 10);
    for (let r = 0; r < 2; r++) {
      for (let d = 0; d < 2; d++) {
        expect(symR.jacobian[r]![d]).toBeCloseTo(refR.jacobian[r]![d]!, 10);
      }
    }
  });
});
