import { describe, test, expect } from "vitest";
import { binaryNode, litNode, unaryNode, varNode } from "../../core/tree-cons";
import {
  BinaryOp,
  UnaryOp,
  LiteralNum,
  ZERO_NODE,
  ONE_NODE,
  KIND_LIT,
  KIND_ADD,
  KIND_MUL,
  KIND_MIN,
  KIND_MAX,
  KIND_NEG,
  KIND_SUB,
  type NumNode,
} from "../../core/tree";
import { simplify } from "../../api/simplify";

describe("foldConstants — new rules unlocked by canonicalisation", () => {
  test("NEG(SUB(a,b)) -> SUB(b,a)", () => {
    const a = varNode("a");
    const b = varNode("b");
    const expr = unaryNode("NEG", binaryNode("SUB", a, b));
    const out = simplify(expr);
    expect(out.kind).toBe(KIND_SUB);
    const s = out as BinaryOp;
    expect(s.left).toBe(b);
    expect(s.right).toBe(a);
  });

  test("0 - x -> -x", () => {
    const x = varNode("x");
    const out = simplify(binaryNode("SUB", ZERO_NODE, x));
    expect(out.kind).toBe(KIND_NEG);
    expect((out as UnaryOp).original).toBe(x);
  });

  test("0 - (-y) -> y (double negation cancels)", () => {
    const y = varNode("y");
    const out = simplify(binaryNode("SUB", ZERO_NODE, unaryNode("NEG", y)));
    expect(out).toBe(y);
  });

  test("x - (-y) -> x + y", () => {
    const x = varNode("x");
    const y = varNode("y");
    const out = simplify(binaryNode("SUB", x, unaryNode("NEG", y)));
    expect(out.kind).toBe(KIND_ADD);
    const bin = out as BinaryOp;
    const operands = new Set<NumNode>([bin.left, bin.right]);
    expect(operands.has(x)).toBe(true);
    expect(operands.has(y)).toBe(true);
  });

  test("l1 * (l2 * x) -> (l1*l2) * x", () => {
    const x = varNode("x");
    // Build the outer MUL directly so canonicalisation doesn't collapse it.
    const inner = binaryNode("MUL", litNode(3), x); // lit-first (canonical)
    const outer = binaryNode("MUL", litNode(4), inner);
    const out = simplify(outer);
    expect(out.kind).toBe(KIND_MUL);
    const bin = out as BinaryOp;
    expect(bin.left.kind).toBe(KIND_LIT);
    expect((bin.left as LiteralNum).value).toBe(12);
    expect(bin.right).toBe(x);
  });

  test("l1 * (l2 * x) with l1*l2 == 0 folds to 0", () => {
    const x = varNode("zx");
    // Use a non-sentinel literal so we exercise the combined-literal path.
    const inner = binaryNode("MUL", litNode(3), x);
    const outer = binaryNode("MUL", litNode(0), inner);
    const out = simplify(outer);
    expect(out).toBe(ZERO_NODE);
  });

  test("l1 * (l2 * x) with l1*l2 == 1 folds to x", () => {
    const x = varNode("ox");
    const inner = binaryNode("MUL", litNode(2), x);
    const outer = binaryNode("MUL", litNode(0.5), inner);
    const out = simplify(outer);
    expect(out).toBe(x);
  });

  test("(a * x) / x -> a", () => {
    const a = varNode("a");
    const x = varNode("x");
    const mul = binaryNode("MUL", a, x);
    const out = simplify(binaryNode("DIV", mul, x));
    expect(out).toBe(a);
  });

  test("(x * a) / x -> a (shared factor in the other position)", () => {
    const a = varNode("ab");
    const x = varNode("xb");
    // Force the MUL operand order by hand: canonicalisation places
    // operands in sortId order, so both orderings collapse to the same
    // node regardless. The rule still needs to recognize the shared
    // factor in either position of the inner MUL.
    const mul = binaryNode("MUL", x, a);
    const out = simplify(binaryNode("DIV", mul, x));
    expect(out).toBe(a);
  });

  test("min(x, min(x, y)) -> min(x, y) (absorption)", () => {
    const x = varNode("x");
    const y = varNode("y");
    const inner = binaryNode("MIN", x, y);
    const outer = binaryNode("MIN", x, inner);
    const out = simplify(outer);
    expect(out).toBe(inner);
  });

  test("max(x, max(x, y)) -> max(x, y) (absorption)", () => {
    const x = varNode("x");
    const y = varNode("y");
    const inner = binaryNode("MAX", x, y);
    const outer = binaryNode("MAX", x, inner);
    const out = simplify(outer);
    expect(out).toBe(inner);
  });

  test("min(l1, min(l2, x)) -> min(min(l1,l2), x) (literal associativity)", () => {
    const x = varNode("x");
    const inner = binaryNode("MIN", litNode(5), x);
    const outer = binaryNode("MIN", litNode(3), inner);
    const out = simplify(outer);
    expect(out.kind).toBe(KIND_MIN);
    const bin = out as BinaryOp;
    expect(bin.left.kind).toBe(KIND_LIT);
    expect((bin.left as LiteralNum).value).toBe(3);
    expect(bin.right).toBe(x);
  });

  test("max(l1, max(l2, x)) -> max(max(l1,l2), x) (literal associativity)", () => {
    const x = varNode("x");
    const inner = binaryNode("MAX", litNode(5), x);
    const outer = binaryNode("MAX", litNode(3), inner);
    const out = simplify(outer);
    expect(out.kind).toBe(KIND_MAX);
    const bin = out as BinaryOp;
    expect(bin.left.kind).toBe(KIND_LIT);
    expect((bin.left as LiteralNum).value).toBe(5);
    expect(bin.right).toBe(x);
  });

  test("compare(x, x) -> 0", () => {
    const x = varNode("x");
    const out = simplify(binaryNode("COMPARE", x, x));
    expect(out).toBe(ZERO_NODE);
  });

  test("existing rules still hold: x + 0 = x, x * 1 = x", () => {
    const x = varNode("x");
    expect(simplify(binaryNode("ADD", ZERO_NODE, x))).toBe(x);
    expect(simplify(binaryNode("MUL", ONE_NODE, x))).toBe(x);
  });

  // -------------------------------------------------------------------
  // Cascading folds — rewrites must re-enter the fold dispatcher so
  // that `l1 + (l2 + x)` with `l1 + l2 == 0` collapses all the way
  // to `x`, instead of stopping at the intermediate `ADD(0, x)`.
  // -------------------------------------------------------------------

  test("ADD literal associativity cascades when the sum is zero", () => {
    const x = varNode("x");
    // 3 + (-3 + x)  -->  (3 + -3) + x  -->  0 + x  -->  x
    const inner = binaryNode("ADD", litNode(-3), x);
    const outer = binaryNode("ADD", litNode(3), inner);
    expect(simplify(outer)).toBe(x);
  });

  test("ADD literal associativity cascades twice", () => {
    const x = varNode("x");
    // 1 + (2 + (-3 + x))  -->  should fully collapse to x
    const l1 = binaryNode("ADD", litNode(-3), x);
    const l2 = binaryNode("ADD", litNode(2), l1);
    const l3 = binaryNode("ADD", litNode(1), l2);
    expect(simplify(l3)).toBe(x);
  });

  test("MUL literal associativity cascades when the product is one", () => {
    const x = varNode("x");
    // 2 * (0.5 * x)  -->  (2 * 0.5) * x  -->  1 * x  -->  x
    const inner = binaryNode("MUL", litNode(0.5), x);
    const outer = binaryNode("MUL", litNode(2), inner);
    expect(simplify(outer)).toBe(x);
  });

  test("MUL literal associativity cascades when the product is zero", () => {
    const x = varNode("x");
    // 0 * (5 * x)  -->  (0 * 5) * x  -->  0 * x  -->  0
    // Note: 0 * (anything) also fires earlier, but this tests the path
    // where the outer rule matches first via the literal-associativity
    // branch.
    const inner = binaryNode("MUL", litNode(5), x);
    const outer = binaryNode("MUL", litNode(0), inner);
    expect(simplify(outer)).toBe(ZERO_NODE);
  });

  test("x - (-y) cascades when y == 0", () => {
    const x = varNode("x");
    // x - (-(0))  -->  x + 0  -->  x
    const negZero = unaryNode("NEG", ZERO_NODE);
    const expr = binaryNode("SUB", x, negZero);
    expect(simplify(expr)).toBe(x);
  });

  test("NEG of SUB cascades when the swapped shape simplifies", () => {
    const x = varNode("x");
    // -(x - x)  -->  x - x reversed is still x - x  -->  0
    // The NEG-through-SUB rewrite must re-fold for this to collapse.
    const expr = unaryNode("NEG", binaryNode("SUB", x, x));
    expect(simplify(expr)).toBe(ZERO_NODE);
  });

  test("0 - (-y) via the fold path cascades to y", () => {
    const y = varNode("y");
    // The 0-x rule emits NEG(-y), which must cascade via foldUnary's
    // double-neg rule to just y.
    const expr = binaryNode("SUB", ZERO_NODE, unaryNode("NEG", y));
    expect(simplify(expr)).toBe(y);
  });

  test("min literal associativity can cascade", () => {
    const x = varNode("x");
    // min(5, min(3, x)) -> min(3, x), and min(3, min(5, x)) also -> min(3, x)
    const inner1 = binaryNode("MIN", litNode(5), x);
    const outer1 = binaryNode("MIN", litNode(3), inner1);
    const out1 = simplify(outer1) as BinaryOp;
    expect(out1.kind).toBe(KIND_MIN);
    expect((out1.left as LiteralNum).value).toBe(3);
    expect(out1.right).toBe(x);
  });
});
