import { Num, ONE, asNum, binaryOpNum, selectOpNum } from "../core/num";
import type {
  Branch,
  CasesApi,
  Condition,
  WhenApi,
  WhenChainApi,
} from "./interfaces";
import { fullDerivative } from "./diff";
import { partialDerivative, replaceVariable } from "../core/tree-walks";

export function max(first: Num | number, ...others: Array<Num | number>): Num {
  let tree = asNum(first);
  others.forEach((n) => {
    tree = binaryOpNum("MAX", tree, asNum(n));
  });
  return tree;
}

export function min(first: Num | number, ...others: Array<Num | number>): Num {
  let tree = asNum(first);
  others.forEach((n) => {
    tree = binaryOpNum("MIN", tree, asNum(n));
  });
  return tree;
}

export function atan2(a: Num | number, b: Num | number): Num {
  return binaryOpNum("ATAN2", asNum(a), asNum(b));
}

export function select(
  condition: Num | number,
  ifNonZero: Num | number,
  ifZero: Num | number,
): Num {
  return selectOpNum(asNum(condition), asNum(ifNonZero), asNum(ifZero));
}

export function ifTruthyElse(
  condition: Num | number,
  if_non_zero: Num | number,
  if_zero: Num | number,
): Num {
  return select(condition, if_non_zero, if_zero);
}

export type NumBranch = Branch<Num>;
export type NumCondition = Condition<Num>;
export type NumWhen = WhenApi<Num>;
export type NumWhenChain = WhenChainApi<Num>;
export type NumCases = CasesApi<Num>;

type NumWhenCase = {
  condition: NumCondition;
  branch: NumBranch;
};

function resolveCondition(condition: NumCondition): Num | number {
  return typeof condition === "function" ? condition() : condition;
}

function buildWhen(cases: NumWhenCase[], fallback: NumBranch): Num {
  let result = asNum(fallback());
  for (let i = cases.length - 1; i >= 0; i--) {
    const c = cases[i]!;
    result = ifTruthyElse(resolveCondition(c.condition), c.branch(), result);
  }
  return result;
}

function makeWhen(cases: NumWhenCase[], condition: NumCondition): NumWhen {
  return {
    then: (ifNonZero: NumBranch): NumWhenChain =>
      makeWhenChain([...cases, { condition, branch: ifNonZero }]),
  };
}

function makeWhenChain(cases: NumWhenCase[]): NumWhenChain {
  const chooseFallback = (ifZero: NumBranch): Num => buildWhen(cases, ifZero);
  return {
    elseIf: (condition: NumCondition): NumWhen => makeWhen(cases, condition),
    else: chooseFallback,
  };
}

export function when(condition: NumCondition): NumWhen {
  return makeWhen([], condition);
}

type NumCase = {
  value: Num | number;
  branch: NumBranch;
};

function buildCases(
  selector: Num | number,
  caseDefs: NumCase[],
  fallback: NumBranch,
): Num {
  let result = asNum(fallback());
  const selected = asNum(selector);
  for (let i = caseDefs.length - 1; i >= 0; i--) {
    const c = caseDefs[i]!;
    result = ifTruthyElse(selected.equals(c.value), c.branch(), result);
  }
  return result;
}

function makeCases(selector: Num | number, caseDefs: NumCase[]): NumCases {
  const chooseFallback = (fallback: NumBranch): Num =>
    buildCases(selector, caseDefs, fallback);
  return {
    case: (value: Num | number, branch: NumBranch): NumCases =>
      makeCases(selector, [...caseDefs, { value, branch }]),
    default: chooseFallback,
  };
}

export function cases(selector: Num | number): NumCases {
  return makeCases(selector, []);
}

export function hypot(a: Num | number, b: Num | number): Num {
  return asNum(a).square().add(asNum(b).square()).sqrt();
}

export function clamp(
  a: Num | number,
  minVal: Num | number,
  maxVal: Num | number,
): Num {
  const bottomClamped = max(minVal, a);
  return min(maxVal, bottomClamped);
}

export function sigmoid(a: Num | number): Num {
  const v = asNum(a);

  const posExpr = ONE.div(v.neg().exp().add(ONE));
  const negExpr = v.exp().div(v.exp().add(ONE));

  const vGT0 = v.greaterThan(0);

  return ifTruthyElse(vGT0, posExpr, negExpr);
}

export function diff(num: Num): Num {
  return new Num(fullDerivative(num.n));
}

export function gradientAt(num: Num, point: [string, Num | number][]): Num[] {
  const diffNum = fullDerivative(num.n);
  const diffAtPoint = replaceVariable(
    diffNum,
    new Map(point.map(([k, v]) => [k, asNum(v).n])),
  );

  const grad = point.map(([k]) => {
    return new Num(partialDerivative(diffAtPoint, k));
  });
  return grad;
}
