import { describe, it } from "vitest";
import { ex, exVar } from "../test-utils";
import { asNum, variableNum } from "../core/num";
import { nexpr } from "./expressions";

describe("nexpr - basic arithmetic", () => {
  it("evaluates constants with precedence", () => {
    ex(nexpr`2 + 3 * 4`).toBe(14);
    ex(nexpr`(2 + 3) * 4`).toBe(20);
  });

  it("supports unary minus", () => {
    ex(nexpr`-3 + 4`).toBe(1);
    ex(nexpr`-(3 + 4)`).toBe(-7);
  });

  it("supports power with ^ and ** (right associative)", () => {
    ex(nexpr`2 ^ 3`).toBe(8);
    ex(nexpr`2 ** 3`).toBe(8);
    ex(nexpr`2 ** 3 ** 2`).toBeCloseTo(512, 6); // 2 ** (3 ** 2)
  });

  it("applies exponent before unary minus like JS", () => {
    ex(nexpr`-2 ** 2`).toBe(-4);
    ex(nexpr`(-2) ** 2`).toBe(4);
  });

  it("supports sqrt function", () => {
    ex(nexpr`sqrt(9)`).toBe(3);
    ex(nexpr`sqrt(4 + 5)`).toBe(3);
  });

  it("supports log and exp functions", () => {
    ex(nexpr`exp(0)`).toBe(1);
    ex(nexpr`log(exp(1))`).toBeCloseTo(1, 6);
    ex(nexpr`exp(log(5))`).toBeCloseTo(5, 6);
    ex(nexpr`log(1 + 3)`).toBeCloseTo(Math.log(4), 6);
    ex(nexpr`log(sqrt(4))`).toBeCloseTo(Math.log(2), 6);
  });

  it("supports cbrt function", () => {
    ex(nexpr`cbrt(8)`).toBeCloseTo(2, 6);
    ex(nexpr`cbrt(27)`).toBeCloseTo(3, 6);
  });

  it("ident functions work with placeholders", () => {
    const x = variableNum("x");
    exVar(nexpr`exp(${x})`, { x: 1 }).toBeCloseTo(Math.E, 6);
    exVar(nexpr`log(${x})`, { x: Math.E }).toBeCloseTo(1, 6);
  });

  it("throws on unknown identifier", () => {
    ex(() => nexpr`foo(1)`).toThrow(/Unknown identifier/);
  });

  it("uses powi for **2 and **3 when applicable", () => {
    const x = variableNum("x");
    exVar(nexpr`${x} ** 2`, { x: 5 }).toBe(25);
    exVar(nexpr`${x} ** 3`, { x: 2 }).toBe(8);
    // ensure chaining still respects associativity and doesn't short-circuit
    ex(nexpr`2 ** 3 ** 2`).toBeCloseTo(512, 6);
  });

  it("works with placeholders (Nums)", () => {
    const x = variableNum("x");
    exVar(nexpr`${x} * 2 + 3`, { x: 5 }).toBe(13);
  });

  it("works with placeholders (numbers)", () => {
    ex(nexpr`${2} * (${3} + 4)`).toBe(14);
    const three = asNum(3);
    ex(nexpr`${three} * 2 + 1`).toBe(7);
  });

  it("mixes literals, numbers and Nums", () => {
    const a = variableNum("a");
    exVar(nexpr`1 + ${2} * (${a} + 3)`, { a: 4 }).toBe(1 + 2 * (4 + 3));
  });
});
