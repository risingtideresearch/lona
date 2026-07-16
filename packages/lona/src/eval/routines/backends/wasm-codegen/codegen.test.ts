import { describe, test, expect } from "vitest";
import { compileWasmFromTape, compileWasmGradFromTape } from "./codegen";
import { compileTape, compileTapeFromSerialized } from "../../../tape";
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

describe("compileWasm", () => {
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
    const fn = compileWasmFromTape(compileTape([deriv])!)!;
    expect(fn).not.toBeNull();
    expect(fn(new Map(), new Map([["x", 7]]))[0]).toBe(7);
  });

  test("compiles a literal", () => {
    const fn = compileWasmFromTape(compileTape([new LiteralNum(42)])!)!;
    expect(fn).not.toBeNull();
    expect(fn(new Map())[0]).toBe(42);
  });

  test("compiles a variable x", () => {
    const fn = compileWasmFromTape(compileTape([new Variable("x")])!)!;
    expect(fn(new Map<VarName, number>([["x", 7]]))[0]).toBe(7);
  });

  test("compiles a variable y", () => {
    const fn = compileWasmFromTape(compileTape([new Variable("y")])!)!;
    expect(fn(new Map<VarName, number>([["y", 13]]))[0]).toBe(13);
  });

  test("compiles a variable z", () => {
    const fn = compileWasmFromTape(compileTape([new Variable("z")])!)!;
    expect(fn(new Map<VarName, number>([["z", 5]]))[0]).toBe(5);
  });

  test("compiles a named variable", () => {
    const fn = compileWasmFromTape(compileTape([new Variable("w")])!)!;
    const vars = new Map<VarName, number>([["w", 99]]);
    expect(fn(vars)[0]).toBe(99);
  });

  test("compiles binary ADD", () => {
    const root = new BinaryOp("ADD", new Variable("x"), new Variable("y"));
    const fn = compileWasmFromTape(compileTape([root])!)!;
    expect(
      fn(
        new Map<VarName, number>([
          ["x", 3],
          ["y", 4],
        ]),
      )[0],
    ).toBe(7);
  });

  test("compiles unary NEG", () => {
    const root = new UnaryOp("NEG", new Variable("x"));
    const fn = compileWasmFromTape(compileTape([root])!)!;
    expect(fn(new Map<VarName, number>([["x", 5]]))[0]).toBe(-5);
  });

  test("compiles a deeper expression: sqrt(x^2 + y^2)", () => {
    const xVar = new Variable("x");
    const yVar = new Variable("y");
    const x2 = new BinaryOp("MUL", xVar, xVar);
    const y2 = new BinaryOp("MUL", yVar, yVar);
    const sum = new BinaryOp("ADD", x2, y2);
    const root = new UnaryOp("SQRT", sum);

    const fn = compileWasmFromTape(compileTape([root])!)!;
    expect(
      fn(
        new Map<VarName, number>([
          ["x", 3],
          ["y", 4],
        ]),
      )[0],
    ).toBeCloseTo(5, 10);
  });

  test("handles DAG sharing (same node referenced twice)", () => {
    const xVar = new Variable("x");
    const root = new BinaryOp("MUL", xVar, xVar);
    const fn = compileWasmFromTape(compileTape([root])!)!;
    expect(fn(new Map<VarName, number>([["x", 7]]))[0]).toBe(49);
  });

  test("DIV by zero returns fallback", () => {
    const root = new BinaryOp("DIV", new LiteralNum(1), new Variable("x"));
    const fn = compileWasmFromTape(compileTape([root])!)!;
    expect(fn(new Map<VarName, number>([["x", 0]]))[0]).toBe(1e50);
  });
});

describe("compileWasmFromSerialized", () => {
  test("produces same results as compileWasm", () => {
    const xVar = new Variable("x");
    const yVar = new Variable("y");
    const expr = new BinaryOp(
      "ADD",
      new UnaryOp("SIN", xVar),
      new BinaryOp("MUL", yVar, new LiteralNum(2)),
    );

    const fnLive = compileWasmFromTape(compileTape([expr])!)!;
    const serialized = serializeNumDAG(expr);
    const fnSerialized = compileWasmFromTape(
      compileTapeFromSerialized(serialized)!,
    )!;

    for (let x = -5; x <= 5; x++) {
      for (let y = -5; y <= 5; y++) {
        const vars = new Map<VarName, number>([
          ["x", x],
          ["y", y],
        ]);
        expect(fnSerialized(vars)[0]).toBe(fnLive(vars)[0]);
      }
    }
  });
});

describe("multi-chunk splitting", () => {
  // All of the tests below pass `maxChunkBytes: 1` to force every tape node
  // into its own chunk, so every arg read goes through the cross-chunk
  // escape memory. That exercises the chunked emission path on small DAGs
  // we can verify by hand.

  test("literal survives chunking", () => {
    const fn = compileWasmFromTape(compileTape([new LiteralNum(42)])!, {
      maxChunkBytes: 1,
    })!;
    expect(fn).not.toBeNull();
    expect(fn(new Map())[0]).toBe(42);
  });

  test("variable survives chunking", () => {
    const fn = compileWasmFromTape(compileTape([new Variable("x")])!, {
      maxChunkBytes: 1,
    })!;
    expect(fn(new Map<VarName, number>([["x", 9]]))[0]).toBe(9);
  });

  test("cross-chunk ADD works", () => {
    const root = new BinaryOp("ADD", new Variable("x"), new Variable("y"));
    const fn = compileWasmFromTape(compileTape([root])!, { maxChunkBytes: 1 })!;
    expect(
      fn(
        new Map<VarName, number>([
          ["x", 3],
          ["y", 4],
        ]),
      )[0],
    ).toBe(7);
  });

  test("cross-chunk sqrt(x^2 + y^2) matches simpleEval", () => {
    const xVar = new Variable("x");
    const yVar = new Variable("y");
    const expr = new UnaryOp(
      "SQRT",
      new BinaryOp(
        "ADD",
        new BinaryOp("MUL", xVar, xVar),
        new BinaryOp("MUL", yVar, yVar),
      ),
    );
    const fn = compileWasmFromTape(compileTape([expr])!, { maxChunkBytes: 1 })!;
    const ref = compileWasmFromTape(compileTape([expr])!)!;
    for (let xi = -4; xi <= 4; xi++) {
      for (let yi = -4; yi <= 4; yi++) {
        const vars = new Map<VarName, number>([
          ["x", xi],
          ["y", yi],
        ]);
        expect(fn(vars)[0]).toBe(ref(vars)[0]);
      }
    }
  });

  test("DAG sharing across chunks reuses escape slot", () => {
    // xVar is used twice — in chunked mode both reads should go through the
    // same escape slot because the producer (first OP_VAR node) dies only
    // once its final reader runs, and the pre-chunk producer is outside
    // the chunks that read it.
    const xVar = new Variable("x");
    const root = new BinaryOp("MUL", xVar, xVar);
    const fn = compileWasmFromTape(compileTape([root])!, { maxChunkBytes: 1 })!;
    expect(fn(new Map<VarName, number>([["x", 7]]))[0]).toBe(49);
  });

  test("cross-chunk handles complex ops (DIV, SIGN, imports)", () => {
    const x = variableNum("x");
    const expr = x.mul(x).add(1).sqrt().div(x.abs().add(1)).mul(x.sign()).sin();
    const refFn = compileWasmFromTape(compileTape([expr.n])!)!;
    const fn = compileWasmFromTape(compileTape([expr.n])!, {
      maxChunkBytes: 1,
    })!;
    for (let xi = -5; xi <= 5; xi += 0.5) {
      const vars = new Map<VarName, number>([["x", xi]]);
      expect(fn(vars)[0]).toBe(refFn(vars)[0]);
    }
  });

  test("forced chunking on a deep linear chain", () => {
    // Build x + 1 + 2 + 3 + ... + 20. The raw DAG is a tall chain, which
    // forces many cross-chunk reads — a good stress case for the escape
    // load/store path.
    let expr: Num = variableNum("x");
    for (let i = 1; i <= 20; i++) expr = expr.add(i);
    const refFn = compileWasmFromTape(compileTape([expr.n])!)!;
    const fn = compileWasmFromTape(compileTape([expr.n])!, {
      maxChunkBytes: 1,
    })!;
    for (let xi = -3; xi <= 3; xi++) {
      const vars = new Map<VarName, number>([["x", xi]]);
      expect(fn(vars)[0]).toBe(refFn(vars)[0]);
    }
  });

  test("local-count limit splits independently of byte limit", () => {
    let expr: Num = variableNum("x");
    for (let i = 1; i <= 20; i++) expr = expr.add(i);
    const tape = compileTape([expr.n])!;
    const refFn = compileWasmFromTape(tape)!;
    // One variable parameter leaves room for only two node locals.
    const fn = compileWasmFromTape(tape, {
      maxChunkBytes: 1_000_000,
      maxFunctionLocals: 3,
    });
    const vars = new Map<VarName, number>([["x", 2]]);
    expect(fn(vars)).toEqual(refFn(vars));
  });

  test("gradient local count includes the dual stride", () => {
    let expr: Num = variableNum("x");
    for (let i = 1; i <= 12; i++) expr = expr.mul(1.01).add(i);
    const tape = compileTape([expr.n])!;
    const ref = compileWasmGradFromTape(tape, ["x"]);
    // One parameter plus two locals (primal + derivative) per node.
    const chunked = compileWasmGradFromTape(tape, ["x"], {
      maxChunkBytes: 1_000_000,
      maxFunctionLocals: 5,
    });
    expect(chunked(new Map([["x", 3]]))).toEqual(ref(new Map([["x", 3]])));
  });

  test("reports an impossible local budget before invoking WebAssembly", () => {
    const tape = compileTape([new Variable("x")])!;
    expect(() => compileWasmFromTape(tape, { maxFunctionLocals: 1 })).toThrow(
      /cannot fit one tape node.*1 parameters.*1 total-local limit/,
    );
  });
});

describe("large tape regression", () => {
  test("default local budget compiles a dense large tape", () => {
    // Keep the normal suite at 120k nodes; opt into the original 360k
    // reproduction when validating scaling changes explicitly.
    const iterations = process.env.LONA_TEST_SCALING === "1" ? 60_000 : 20_000;
    const t = new Variable("t");
    let root: Variable | UnaryOp | BinaryOp = t;
    let expected = 0.3;
    for (let i = 0; i < iterations; i++) {
      const value = 1 + i * 1e-7;
      const c = new LiteralNum(value);
      root = new UnaryOp(
        "COS",
        new BinaryOp(
          "MUL",
          new UnaryOp(
            "SIN",
            new BinaryOp("ADD", new BinaryOp("MUL", root, t), c),
          ),
          c,
        ),
      );
      expected = Math.cos(Math.sin(expected * 0.3 + value) * value);
    }

    const tape = compileTape([root])!;
    expect(tape.opcodes.length).toBe(iterations * 6 + 1);
    const fn = compileWasmFromTape(tape);
    expect(fn(new Map([["t", 0.3]]))[0]).toBeCloseTo(expected, 14);
  }, 30_000);
});

describe("parity with simpleEval", () => {
  test("complex expression matches simpleEval", () => {
    const x = variableNum("x");
    const y = variableNum("y");

    const expr = x.mul(x).add(y.mul(y)).sqrt().sub(x.abs()).max(y.neg());
    const simplified = expr.simplify();

    const bindings = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
    ]);

    const fn = compileWasmFromTape(compileTape([simplified.n])!)!;

    for (let xi = -10; xi <= 10; xi += 2.5) {
      for (let yi = -10; yi <= 10; yi += 2.5) {
        bindings.set("x", xi);
        bindings.set("y", yi);

        const expected = simpleEval(simplified.n, bindings, true);
        const actual = fn(
          new Map<VarName, number>([
            ["x", xi],
            ["y", yi],
          ]),
        )[0];
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
      const fn = compileWasmFromTape(compileTape([expr.n])!)!;
      const actual = fn(new Map<VarName, number>([["x", testVal]]))[0];

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
      const fn = compileWasmFromTape(compileTape([expr.n])!)!;
      const actual = fn(
        new Map<VarName, number>([
          ["x", 7],
          ["y", 3],
        ]),
      )[0];

      expect(actual).toBe(expected);
    }
  });
});
