import { describe, test, expect } from "vitest";
import type { VarName } from "../core/tree";
import { variableNum } from "../core/num";
import { compileTape } from "./tape";
import { compileForwardAutodiff } from "./routines/backends/js-interp/tape-eval";
import { compileWasmForwardAutodiff } from "./routines/backends/wasm-interp/tape-eval";
import { compileWasmGradFromTape } from "./routines/backends/wasm-codegen/codegen";
import {
  compileGradRoutine,
  compileJvpRoutineFromTape,
  type GradRoutine,
} from "./routines";

const x = variableNum("x" as VarName);
const y = variableNum("y" as VarName);
const diffVars: VarName[] = ["x" as VarName, "y" as VarName];

function makeVars(xv: number, yv: number): Map<VarName, number> {
  return new Map<VarName, number>([
    ["x" as VarName, xv],
    ["y" as VarName, yv],
  ]);
}

const testPoints: [number, number][] = [
  [0, 0],
  [1, 2],
  [-3, 5],
  [0.5, -1.5],
  [10, -7],
];

// Reference gradient routine — uses wasm-interp (the old `NumTape` default).
function refRoutine(node: import("../core/tree").NumNode): GradRoutine {
  return compileGradRoutine([node], diffVars, {
    backend: "wasm-interp",
  })! as GradRoutine;
}

// f(x,y) = x^2 + 3*x*y + sin(y)
// df/dx = 2x + 3y
// df/dy = 3x + cos(y)
const expr = x.mul(x).add(x.mul(y).mul(3)).add(y.sin());

describe("Forward-mode autodiff (JS tape)", () => {
  test.each([
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ] as const)(
    "backend=%s applies arbitrary seeds to multiple roots",
    (backend) => {
      const tape = compileTape([x.mul(y).n, x.add(y).n])!;
      const jvp = compileJvpRoutineFromTape(tape, 2, { backend });
      const result = jvp.evalPacked(
        new Float64Array([2, 3]),
        // x carries [4, 5], y carries [6, 7].
        new Float64Array([4, 5, 6, 7]),
      );

      expect(result.vals).toEqual([6, 5]);
      expect(result.tangents[0]).toEqual([24, 29]);
      expect(result.tangents[1]![0]).toBeCloseTo(10, 10);
      expect(result.tangents[1]![1]).toBeCloseTo(12, 10);
    },
  );

  test("validates packed seed dimensions", () => {
    const tape = compileTape([x.add(y).n])!;
    const jvp = compileJvpRoutineFromTape(tape, 2, {
      backend: "js-interp",
    });
    expect(() =>
      jvp.evalPacked(new Float64Array([2, 3]), new Float64Array([1])),
    ).toThrow(/expected 4 seeds/);
  });

  test("matches symbolic gradient", () => {
    const tape = compileTape([expr.n])!;
    const fwdFn = compileForwardAutodiff(tape, diffVars);
    const ref = refRoutine(expr.n);

    for (const [xv, yv] of testPoints) {
      const vars = makeVars(xv, yv);
      const fwd = fwdFn(vars);
      const sym = ref.eval(vars);

      expect(fwd.val).toBeCloseTo(sym.val, 10);
      expect(fwd.gradient[0]).toBeCloseTo(sym.gradient[0]!, 10);
      expect(fwd.gradient[1]).toBeCloseTo(sym.gradient[1]!, 10);
    }
  });

  test("known values", () => {
    const tape = compileTape([expr.n])!;
    const fn = compileForwardAutodiff(tape, diffVars);

    const r = fn(makeVars(2, 3));
    // f(2,3) = 4 + 18 + sin(3) = 22 + sin(3)
    expect(r.val).toBeCloseTo(22 + Math.sin(3), 10);
    // df/dx(2,3) = 4 + 9 = 13
    expect(r.gradient[0]).toBeCloseTo(13, 10);
    // df/dy(2,3) = 6 + cos(3)
    expect(r.gradient[1]).toBeCloseTo(6 + Math.cos(3), 10);
  });
});

describe("Forward-mode autodiff (WASM tape)", () => {
  test("matches symbolic gradient", () => {
    const tape = compileTape([expr.n])!;
    const wasmFn = compileWasmForwardAutodiff(tape, diffVars);
    const ref = refRoutine(expr.n);

    for (const [xv, yv] of testPoints) {
      const vars = makeVars(xv, yv);
      const wasm = wasmFn(vars);
      const sym = ref.eval(vars);

      expect(wasm.val).toBeCloseTo(sym.val, 10);
      expect(wasm.gradient[0]).toBeCloseTo(sym.gradient[0]!, 10);
      expect(wasm.gradient[1]).toBeCloseTo(sym.gradient[1]!, 10);
    }
  });

  test("matches JS forward autodiff", () => {
    const tape = compileTape([expr.n])!;
    const jsFn = compileForwardAutodiff(tape, diffVars);
    const wasmFn = compileWasmForwardAutodiff(tape, diffVars);

    for (const [xv, yv] of testPoints) {
      const vars = makeVars(xv, yv);
      const js = jsFn(vars);
      const wasm = wasmFn(vars);

      expect(wasm.val).toBeCloseTo(js.val, 10);
      expect(wasm.gradient[0]).toBeCloseTo(js.gradient[0]!, 10);
      expect(wasm.gradient[1]).toBeCloseTo(js.gradient[1]!, 10);
    }
  });
});

describe("Forward autodiff: more ops", () => {
  // Test various ops: div, sqrt, max, min, abs, exp, log
  // max/min excluded: tie-breaking at equal values gives valid but different subgradients
  const exprs = [
    { name: "div", expr: x.div(y.add(1)) },
    { name: "sqrt", expr: x.mul(x).add(y.mul(y)).add(1).sqrt() },
    { name: "abs", expr: x.sub(y).abs() },
    { name: "exp", expr: x.mul(0.1).exp() },
    { name: "log", expr: x.mul(x).add(1).log() },
  ];

  for (const { name, expr: e } of exprs) {
    test(`${name}: JS fwd matches symbolic`, () => {
      const tape = compileTape([e.n])!;
      const fwdFn = compileForwardAutodiff(tape, diffVars);
      const ref = refRoutine(e.n);

      for (const [xv, yv] of testPoints) {
        const vars = makeVars(xv, yv);
        const fwd = fwdFn(vars);
        const sym = ref.eval(vars);
        expect(fwd.val).toBeCloseTo(sym.val, 8);
        expect(fwd.gradient[0]).toBeCloseTo(sym.gradient[0]!, 8);
        expect(fwd.gradient[1]).toBeCloseTo(sym.gradient[1]!, 8);
      }
    });

    test(`${name}: WASM fwd matches symbolic`, () => {
      const tape = compileTape([e.n])!;
      const wasmFn = compileWasmForwardAutodiff(tape, diffVars);
      const ref = refRoutine(e.n);

      for (const [xv, yv] of testPoints) {
        const vars = makeVars(xv, yv);
        const wasm = wasmFn(vars);
        const sym = ref.eval(vars);
        expect(wasm.val).toBeCloseTo(sym.val, 8);
        expect(wasm.gradient[0]).toBeCloseTo(sym.gradient[0]!, 8);
        expect(wasm.gradient[1]).toBeCloseTo(sym.gradient[1]!, 8);
      }
    });
  }
});

describe("Forward-mode autodiff (WASM codegen)", () => {
  test("matches symbolic gradient", () => {
    const fn = compileWasmGradFromTape(compileTape([expr.n])!, diffVars)!;
    const ref = refRoutine(expr.n);

    for (const [xv, yv] of testPoints) {
      const vars = makeVars(xv, yv);
      const result = fn(vars);
      const sym = ref.eval(vars);
      expect(result.val).toBeCloseTo(sym.val, 10);
      expect(result.gradient[0]).toBeCloseTo(sym.gradient[0]!, 10);
      expect(result.gradient[1]).toBeCloseTo(sym.gradient[1]!, 10);
    }
  });

  for (const { name, expr: e } of [
    { name: "div", expr: x.div(y.add(1)) },
    { name: "sqrt", expr: x.mul(x).add(y.mul(y)).add(1).sqrt() },
    { name: "abs", expr: x.sub(y).abs() },
    { name: "exp", expr: x.mul(0.1).exp() },
    { name: "log", expr: x.mul(x).add(1).log() },
  ]) {
    test(`${name}: matches symbolic`, () => {
      const fn = compileWasmGradFromTape(compileTape([e.n])!, diffVars)!;
      const ref = refRoutine(e.n);

      for (const [xv, yv] of testPoints) {
        const vars = makeVars(xv, yv);
        const result = fn(vars);
        const sym = ref.eval(vars);
        expect(result.val).toBeCloseTo(sym.val, 8);
        expect(result.gradient[0]).toBeCloseTo(sym.gradient[0]!, 8);
        expect(result.gradient[1]).toBeCloseTo(sym.gradient[1]!, 8);
      }
    });
  }
});
