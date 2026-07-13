/**
 * Tests for the algebraic identities that the Num smart constructors
 * perform at construction time (Layer 1), independent of the foldConstants
 * kernel (Layer 2). These assertions check *node identity* — the result
 * must short-circuit without allocating new binary/unary nodes.
 */
import { describe, test, expect } from "vitest";
import { Num, asNum, variableNum } from "./num";

import {
  LiteralNum,
  ZERO_NODE,
  ONE_NODE,
  KIND_ABS,
  KIND_MUL,
  KIND_NEG,
  KIND_DIV,
  KIND_COMPARE,
  KIND_LIT,
} from "./tree";

describe("Num smart constructors — new identity rules", () => {
  test("div: x / 1 = x", () => {
    const x = variableNum("x");
    expect(x.div(1)).toBe(x);
    expect(x.div(asNum(1))).toBe(x);
  });

  test("div: 0 / x = 0", () => {
    const x = variableNum("x");
    const result = asNum(0).div(x);
    expect(result.n).toBe(ZERO_NODE);
  });

  test("div: x / x = 1", () => {
    const x = variableNum("x");
    const result = x.div(x);
    expect(result.n).toBe(ONE_NODE);
  });

  test("div: (a * x) / x = a (shared factor eliminated)", () => {
    const a = variableNum("a");
    const x = variableNum("x");
    const ax = a.mul(x);
    const result = ax.div(x);
    // The result's node is either `a.n` or `x.n` depending on which
    // factor was matched; for a != x it must be a.
    expect(result.n).toBe(a.n);
  });

  test("neg: -(-x) = x (constructor unwraps NEG)", () => {
    const x = variableNum("x");
    // Build the inner NEG through the constructor.
    const negX = x.neg();
    expect(negX.n.kind).toBe(KIND_NEG);
    // Double-neg should not allocate a new node — it unwraps to `x.n`.
    const negNegX = negX.neg();
    expect(negNegX.n).toBe(x.n);
  });

  test("abs: abs(abs(x)) = abs(x) (no second ABS node)", () => {
    const x = variableNum("x");
    const ax = x.abs();
    expect(ax.n.kind).toBe(KIND_ABS);
    const aax = ax.abs();
    expect(aax).toBe(ax);
  });

  test("powi(1) returns this", () => {
    const x = variableNum("x");
    expect(x.powi(1)).toBe(x);
  });

  test("powi(2) delegates to square (single MUL, no chain)", () => {
    const x = variableNum("x");
    const p2 = x.powi(2);
    expect(p2.n.kind).toBe(KIND_MUL);
    // square builds MUL(x, x) directly.
    expect(p2.n).toBe(x.square().n);
  });

  test("square on literal folds", () => {
    const lit = asNum(4);
    const sq = lit.square();
    expect(sq.n.kind).toBe(KIND_LIT);
    const value = (sq.n as LiteralNum).value;
    expect(value).toBe(16);
  });

  test("square on ZERO is ZERO", () => {
    const result = asNum(0).square();
    expect(result.n).toBe(ZERO_NODE);
  });

  test("square on ONE is ONE", () => {
    const result = asNum(1).square();
    expect(result.n).toBe(ONE_NODE);
  });

  test("compare: x.compare(x) = 0", () => {
    const x = variableNum("x");
    const result = x.compare(x);
    expect(result.n).toBe(ZERO_NODE);
  });

  test("equals: x.equals(x) = 1", () => {
    const x = variableNum("x");
    const result = x.equals(x);
    expect(result.n).toBe(ONE_NODE);
  });

  test("non-equal operands still build real nodes", () => {
    // Sanity: don't short-circuit when operands differ.
    const x = variableNum("x");
    const y = variableNum("y");
    expect(x.compare(y).n.kind).toBe(KIND_COMPARE);
    expect(x.div(y).n.kind).toBe(KIND_DIV);
  });

  test("smart constructors are composable with foldConstants", () => {
    // End-to-end: a chain that stacks several Layer-1 rewrites.
    const x = variableNum("x");
    // ((-(-x)) / x) * 1 should collapse to 1 without any fold pass.
    const expr: Num = x.neg().neg().div(x).mul(1);
    expect(expr.n).toBe(ONE_NODE);
  });
});
