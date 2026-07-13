import { expect } from "vitest";
import { simpleEval, Num } from "lona";

export function ex(num: Num | (() => Num)) {
  if (typeof num === "function") {
    return expect(() => simpleEval(num().n));
  }
  return expect(simpleEval(num.n));
}
