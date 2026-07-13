import { expect } from "vitest";
import { simpleEval, naiveEval } from "./eval/eval-value";
import { Num } from "./core/num";

export function ex(num: Num | (() => Num)) {
  if (typeof num === "function") {
    return expect(() => simpleEval(num().n));
  }
  return expect(simpleEval(num.n));
}
export function exNaN(num: Num) {
  return expect(naiveEval(num.n, new Map()));
}
export function exVar(num: Num, vars: Record<string, number>) {
  const variables = new Map<string, number>();
  const derivatives = new Map<string, number>();
  for (const [key, value] of Object.entries(vars)) {
    if (key.startsWith("d_")) {
      derivatives.set(key.slice(2), value);
    } else {
      variables.set(key, value);
    }
  }
  return expect(
    naiveEval(num.n, variables, derivatives.size > 0 ? derivatives : undefined),
  );
}
