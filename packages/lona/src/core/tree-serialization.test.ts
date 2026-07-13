import { describe, test, expect } from "vitest";
import {
  serializeNumDAG,
  deserializeNumDAG,
  type SerializedNumDAG,
} from "./tree-serialization";
import {
  NumNode,
  LiteralNum,
  Variable,
  UnaryOp,
  DebugNode,
  BinaryOp,
  Derivative,
  ForeignFn,
  SelectOp,
  ZERO_NODE,
} from "./tree";

function roundTrip(node: NumNode) {
  const serialized = serializeNumDAG(node);
  const json = JSON.stringify(serialized);
  const deserialized = deserializeNumDAG(JSON.parse(json));
  return deserialized;
}

describe("NumNode DAG serialization", () => {
  test("round-trips a literal", () => {
    const node = new LiteralNum(42);
    const result = roundTrip(node);
    expect(result).toBeInstanceOf(LiteralNum);
    expect((result as LiteralNum).value).toBe(42);
  });

  test("round-trips a variable", () => {
    const node = new Variable("x");
    const result = roundTrip(node);
    expect(result).toBeInstanceOf(Variable);
    expect((result as Variable).name).toBe("x");
  });

  test("round-trips a unary op", () => {
    const node = new UnaryOp("NEG", new LiteralNum(5));
    const result = roundTrip(node);
    expect(result).toBeInstanceOf(UnaryOp);
    expect((result as UnaryOp).operation).toBe("NEG");
    expect(((result as UnaryOp).original as LiteralNum).value).toBe(5);
  });

  test("round-trips a debug node", () => {
    const node = new DebugNode(new LiteralNum(7), "test-debug");
    const result = roundTrip(node);
    expect(result).toBeInstanceOf(DebugNode);
    expect((result as DebugNode).debug).toBe("test-debug");
  });

  test("round-trips a binary op", () => {
    const node = new BinaryOp("ADD", new LiteralNum(3), new LiteralNum(4));
    const result = roundTrip(node);
    expect(result).toBeInstanceOf(BinaryOp);
    expect((result as BinaryOp).operation).toBe("ADD");
  });

  test("round-trips a derivative node", () => {
    const v = new Variable("x");
    const node = new Derivative(v);
    const result = roundTrip(node);
    expect(result).toBeInstanceOf(Derivative);
    expect((result as Derivative).variable).toBeInstanceOf(Variable);
    expect(((result as Derivative).variable as Variable).name).toBe("x");
  });

  test("round-trips a select node", () => {
    const node = new SelectOp(
      new Variable("c"),
      new LiteralNum(3),
      new LiteralNum(4),
    );
    const result = roundTrip(node);
    expect(result).toBeInstanceOf(SelectOp);
    expect((result as SelectOp).condition).toBeInstanceOf(Variable);
    expect((result as SelectOp).ifNonZero).toBeInstanceOf(LiteralNum);
    expect((result as SelectOp).ifZero).toBeInstanceOf(LiteralNum);
  });

  test("round-trips a complex expression tree", () => {
    // (x + 3) * neg(x)
    const x = new Variable("x");
    const three = new LiteralNum(3);
    const sum = new BinaryOp("ADD", x, three);
    const neg = new UnaryOp("NEG", x);
    const product = new BinaryOp("MUL", sum, neg);

    const serialized = serializeNumDAG(product);
    // x is shared in the DAG — it should only appear once
    const varNodes = serialized.nodes.filter((n) => n.type === "VAR");
    expect(varNodes).toHaveLength(1);

    const result = deserializeNumDAG(serialized);
    expect(result).toBeInstanceOf(BinaryOp);
    const mul = result as BinaryOp;
    expect(mul.operation).toBe("MUL");

    // Verify structural integrity after round-trip. MUL is commutative, so
    // `deserializeNumDAG` (which routes through the hash-cons factory) may
    // canonicalise the operand order — find the ADD/NEG by shape rather
    // than assuming a fixed position.
    const children = [mul.left, mul.right];
    const add = children.find((c) => c instanceof BinaryOp) as BinaryOp;
    const negOp = children.find((c) => c instanceof UnaryOp) as UnaryOp;
    expect(add.operation).toBe("ADD");
    expect(negOp.operation).toBe("NEG");
  });

  test("preserves DAG sharing after round-trip through JSON", () => {
    const x = new Variable("x");
    const sum = new BinaryOp("ADD", x, x);

    const serialized = serializeNumDAG(sum);
    const json = JSON.stringify(serialized);
    const data: SerializedNumDAG = JSON.parse(json);

    // x referenced by same id in both left and right
    expect(data.nodes.filter((n) => n.type === "VAR")).toHaveLength(1);
    const binNode = data.nodes[data.root] as { left: number; right: number };
    expect(binNode.left).toBe(binNode.right);
  });

  test("throws on ForeignFn", () => {
    const node = new ForeignFn(
      [new LiteralNum(1)],
      ([v]) => v!,
      () => ZERO_NODE,
    );
    expect(() => serializeNumDAG(node)).toThrow("ForeignFn");
  });

  test("handles symbol-named variables", () => {
    const sym = Symbol("myVar");
    const node = new Variable(sym);
    const serialized = serializeNumDAG(node);
    expect(serialized.nodes[0]).toEqual({ type: "VAR", name: "myVar" });
  });

  test("round-trips without JSON for special float values", () => {
    // serialize/deserialize without JSON to verify the structure itself works.
    // Note: `-0` is intentionally *not* in this list. The deserializer routes
    // through `litNode`, which collapses `-0` onto the canonical `ZERO_NODE`
    // — see `tree-cons.ts` for the rationale.
    for (const value of [Infinity, -Infinity, 0]) {
      const node = new LiteralNum(value);
      const serialized = serializeNumDAG(node);
      const result = deserializeNumDAG(serialized);
      expect(Object.is((result as LiteralNum).value, value)).toBe(true);
    }
  });

  test("round-trips through JSON for normal numbers", () => {
    for (const value of [0, 1, -1, 42, 3.14, 1e10]) {
      const node = new LiteralNum(value);
      const result = roundTrip(node);
      expect((result as LiteralNum).value).toBe(value);
    }
  });
});
