import { expect } from "vitest";
import { naiveEval, type Num } from "lona";

/** Evaluate a constant `Num` expression to a plain number for assertions. */
export function val(num: Num): number {
  return naiveEval(num.n, new Map());
}

/** `expect` wrapper that first evaluates a constant `Num` to a number. */
export function ex(num: Num) {
  return expect(naiveEval(num.n, new Map()));
}
