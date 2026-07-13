import { describe, test, expect } from "vitest";
import { compileTape, compileTapeFromSerialized, evalTape } from "./tape-eval";
import { simpleEval } from "../../../eval-value";
import { Num, variableNum } from "../../../../core/num";
import {
  LiteralNum,
  Variable,
  UnaryOp,
  BinaryOp,
  ForeignFn,
  Derivative,
  VarName,
} from "../../../../core/tree";
import { serializeNumDAG } from "../../../../core/tree-serialization";

describe("compileTape", () => {
  test("returns null for ForeignFn nodes", () => {
    const foreign = new ForeignFn(
      [new LiteralNum(1)],
      (vals) => vals[0]!,
      () => new LiteralNum(0),
    );
    expect(compileTape([foreign])).toBeNull();
  });

  test("compiles Derivative nodes", () => {
    const deriv = new Derivative(new Variable("x"));
    const tape = compileTape([deriv])!;
    expect(tape).not.toBeNull();
    expect(tape.numVars).toBe(1);
    expect(tape.varSlots).toEqual(["x", "x"]);
    expect(evalTape(tape, new Map([["x", 3]]), new Map([["x", 7]]))[0]).toBe(7);
  });

  test("compiles a literal", () => {
    const tape = compileTape([new LiteralNum(42)])!;
    expect(tape).not.toBeNull();
    expect(evalTape(tape, new Map())[0]).toBe(42);
  });

  test("compiles a variable x", () => {
    const tape = compileTape([new Variable("x")])!;
    expect(evalTape(tape, new Map<VarName, number>([["x", 7]]))[0]).toBe(7);
  });

  test("compiles a variable y", () => {
    const tape = compileTape([new Variable("y")])!;
    expect(evalTape(tape, new Map<VarName, number>([["y", 13]]))[0]).toBe(13);
  });

  test("compiles a variable z", () => {
    const tape = compileTape([new Variable("z")])!;
    expect(evalTape(tape, new Map<VarName, number>([["z", 5]]))[0]).toBe(5);
  });

  test("compiles a named variable", () => {
    const tape = compileTape([new Variable("w")])!;
    const vars = new Map<VarName, number>([["w", 99]]);
    expect(evalTape(tape, vars)[0]).toBe(99);
  });

  test("compiles binary ADD", () => {
    const root = new BinaryOp("ADD", new Variable("x"), new Variable("y"));
    const tape = compileTape([root])!;
    expect(
      evalTape(
        tape,
        new Map<VarName, number>([
          ["x", 3],
          ["y", 4],
        ]),
      )[0],
    ).toBe(7);
  });

  test("compiles unary NEG", () => {
    const root = new UnaryOp("NEG", new Variable("x"));
    const tape = compileTape([root])!;
    expect(evalTape(tape, new Map<VarName, number>([["x", 5]]))[0]).toBe(-5);
  });

  test("compiles a deeper expression: sqrt(x^2 + y^2)", () => {
    const xVar = new Variable("x");
    const yVar = new Variable("y");
    const x2 = new BinaryOp("MUL", xVar, xVar);
    const y2 = new BinaryOp("MUL", yVar, yVar);
    const sum = new BinaryOp("ADD", x2, y2);
    const root = new UnaryOp("SQRT", sum);

    const tape = compileTape([root])!;
    expect(
      evalTape(
        tape,
        new Map<VarName, number>([
          ["x", 3],
          ["y", 4],
        ]),
      )[0],
    ).toBeCloseTo(5, 10);
  });

  test("handles DAG sharing (same node referenced twice)", () => {
    const xVar = new Variable("x");
    // x * x — same node used for both sides
    const root = new BinaryOp("MUL", xVar, xVar);
    const tape = compileTape([root])!;
    expect(evalTape(tape, new Map<VarName, number>([["x", 7]]))[0]).toBe(49);
  });

  test("DIV by zero returns fallback", () => {
    const root = new BinaryOp("DIV", new LiteralNum(1), new Variable("x"));
    const tape = compileTape([root])!;
    expect(evalTape(tape, new Map<VarName, number>([["x", 0]]))[0]).toBe(1e50);
  });
});

describe("compileTapeFromSerialized", () => {
  test("produces same results as compileTape", () => {
    const xVar = new Variable("x");
    const yVar = new Variable("y");
    const expr = new BinaryOp(
      "ADD",
      new UnaryOp("SIN", xVar),
      new BinaryOp("MUL", yVar, new LiteralNum(2)),
    );

    const tapeLive = compileTape([expr])!;
    const serialized = serializeNumDAG(expr);
    const tapeSerialized = compileTapeFromSerialized(serialized)!;

    for (let x = -5; x <= 5; x++) {
      for (let y = -5; y <= 5; y++) {
        const vars = new Map<VarName, number>([
          ["x", x],
          ["y", y],
        ]);
        expect(evalTape(tapeSerialized, vars)[0]).toBe(
          evalTape(tapeLive, vars)[0],
        );
      }
    }
  });
});

describe("parity with simpleEval", () => {
  test("Num.eval matches simpleEval for a complex expression", () => {
    const x = variableNum("x");
    const y = variableNum("y");

    // Build a non-trivial expression
    const expr = x.mul(x).add(y.mul(y)).sqrt().sub(x.abs()).max(y.neg());
    const simplified = expr.simplify();

    const bindings = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
    ]);

    for (let xi = -10; xi <= 10; xi += 2.5) {
      for (let yi = -10; yi <= 10; yi += 2.5) {
        bindings.set("x", xi);
        bindings.set("y", yi);

        const expected = simpleEval(simplified.n, bindings, true); // force old path
        const actual = simplified.eval(bindings);
        expect(actual).toBe(expected);
      }
    }
  });

  test("all unary operations produce correct results", () => {
    const ops: Array<[string, (n: Num) => Num, number]> = [
      ["sqrt", (n) => n.abs().sqrt(), 4],
      ["cbrt", (n) => n.cbrt(), 8],
      ["neg", (n) => n.neg(), 3],
      ["abs", (n) => n.abs(), -5],
      ["sin", (n) => n.sin(), 1],
      ["cos", (n) => n.cos(), 1],
      ["tan", (n) => n.tan(), 0.5],
      ["asin", (n) => n.asin(), 0.5],
      ["acos", (n) => n.acos(), 0.5],
      ["atan", (n) => n.atan(), 1],
      ["exp", (n) => n.exp(), 1],
      ["log", (n) => n.abs().add(1).log(), 2],
      ["log1p", (n) => n.abs().log1p(), 2],
      ["tanh", (n) => n.tanh(), 1],
      ["sign", (n) => n.sign(), -3],
    ];

    const x = variableNum("x");
    const bindings = new Map<VarName, number>([["x", 0]]);

    for (const [, buildFn, testVal] of ops) {
      const expr = buildFn(x);
      bindings.set("x", testVal);

      const expected = simpleEval(expr.n, bindings, true);
      const tape = compileTape([expr.n])!;
      const actual = evalTape(
        tape,
        new Map<VarName, number>([["x", testVal]]),
      )[0];

      expect(actual).toBe(expected);
    }
  });

  test("all binary operations produce correct results", () => {
    const ops: Array<[string, (a: Num, b: Num) => Num]> = [
      ["add", (a, b) => a.add(b)],
      ["sub", (a, b) => a.sub(b)],
      ["mul", (a, b) => a.mul(b)],
      ["div", (a, b) => a.div(b)],
      ["min", (a, b) => a.min(b)],
      ["max", (a, b) => a.max(b)],
      ["compare", (a, b) => a.compare(b)],
    ];

    const x = variableNum("x");
    const y = variableNum("y");
    const bindings = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
    ]);

    for (const [, buildFn] of ops) {
      const expr = buildFn(x, y);
      bindings.set("x", 7);
      bindings.set("y", 3);

      const expected = simpleEval(expr.n, bindings, true);
      const tape = compileTape([expr.n])!;
      const actual = evalTape(
        tape,
        new Map<VarName, number>([
          ["x", 7],
          ["y", 3],
        ]),
      )[0];

      expect(actual).toBe(expected);
    }
  });
});
