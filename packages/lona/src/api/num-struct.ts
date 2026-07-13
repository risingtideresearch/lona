import { asNum, type Num } from "../core/num";
import { ifTruthyElse } from "./ops";

// ---------------------------------------------------------------------------
// NumStruct — a value that knows how to take itself apart into a flat tuple of
// `Num` scalars and rebuild a value of its own shape from one.
//
// Many higher-level structures (points, vectors, angles, Bézier segments, …)
// are ultimately a fixed-length tuple of `Num`s. Exposing that as two instance
// methods makes a whole family of operations generic over the type —
// select/`when`/`cases`, mapping a scalar op across every component, etc. — with
// no external dictionary to thread through call sites: the value carries the
// capability.
//
// `fromNums` is an *instance* method (a "value of my shape, from these parts"),
// not a static: TypeScript can't reach a class's statics through a generic type
// parameter, so an instance method is what makes the generics below work with no
// extra argument. The receiver is used only as a shape witness.
//
// Validity note: `selectStruct` is safe for *any* implementer, including
// constrained types (e.g. a unit-length angle), because each component is chosen
// under the SAME condition, so the result equals one of the two whole inputs.
// Arbitrary componentwise arithmetic (`zipNums` with a lerp) only preserves
// validity for free-coordinate types — use a type's own algebra otherwise.
// ---------------------------------------------------------------------------

/** A value that is isomorphic to a flat list of `Num` components. */
export interface NumStruct<T> {
  /** Decompose into scalar components. */
  toNums(): Num[];
  /** Rebuild a value of this shape from `toNums().length` components. */
  fromNums(parts: Num[]): T;
}

/**
 * `condition ? ifTrue : ifFalse`, applied per scalar component. Safe for any
 * `T`: the shared condition means each component resolves to the matching
 * input, so the result is exactly one of the two whole values.
 */
export function selectStruct<T extends NumStruct<T>>(
  condition: Num | number,
  ifTrue: T,
  ifFalse: T,
): T {
  const a = ifTrue.toNums();
  const b = ifFalse.toNums();
  return ifTrue.fromNums(a.map((ai, i) => ifTruthyElse(condition, ai, b[i])));
}

/** Apply a scalar function to every component of a structure. */
export function mapNums<T extends NumStruct<T>>(
  value: T,
  f: (n: Num, i: number) => Num,
): T {
  return value.fromNums(value.toNums().map((n, i) => f(n, i)));
}

/**
 * Combine two structures componentwise (e.g. a lerp). Only meaningful for
 * free-coordinate types — see the validity note above.
 */
export function zipNums<T extends NumStruct<T>>(
  a: T,
  b: T,
  f: (x: Num, y: Num, i: number) => Num,
): T {
  const xs = a.toNums();
  const ys = b.toNums();
  return a.fromNums(xs.map((x, i) => f(x, ys[i], i)));
}

/** Right-fold guarded values into nested `selectStruct`s: first truthy wins. */
function foldClauses<T extends NumStruct<T>>(
  clauses: ReadonlyArray<{ condition: Num | number; value: T }>,
  fallback: T,
): T {
  let result = fallback;
  for (let i = clauses.length - 1; i >= 0; i--) {
    result = selectStruct(clauses[i].condition, clauses[i].value, result);
  }
  return result;
}

/** Continuation after a `then(...)`: branch again or close with `else`. */
export interface WhenStructChainApi<T> {
  elseIf(condition: Num | number): { then(value: T): WhenStructChainApi<T> };
  else(value: T): T;
}

/**
 * Structure-valued `if/else-if/else`, mirroring lona's scalar `when`:
 *
 *   whenStruct(c1).then(a).elseIf(c2).then(b).else(c)
 *
 * picks `a` where `c1` is truthy, else `b` where `c2` is truthy, else `c`,
 * componentwise. `T` is inferred from the first `then`.
 */
export function whenStruct(condition: Num | number) {
  return {
    then<T extends NumStruct<T>>(value: T): WhenStructChainApi<T> {
      const clauses: { condition: Num | number; value: T }[] = [
        { condition, value },
      ];
      const chain: WhenStructChainApi<T> = {
        elseIf: (cond) => ({
          then: (v) => {
            clauses.push({ condition: cond, value: v });
            return chain;
          },
        }),
        else: (fallback) => foldClauses(clauses, fallback),
      };
      return chain;
    },
  };
}

/** Continuation of a `casesStruct(...)` chain: add a case or close with `default`. */
export interface CasesStructApi<T> {
  case(value: Num | number, result: T): CasesStructApi<T>;
  default(fallback: T): T;
}

/**
 * Structure-valued `switch`, mirroring lona's scalar `cases`:
 *
 *   casesStruct(selector).case(0, a).case(1, b).default(c)
 *
 * picks the result whose case value equals `selector`, else `default`. `T` is
 * inferred from the first `case`.
 */
export function casesStruct(selector: Num | number) {
  const sel = asNum(selector);
  return {
    case<T extends NumStruct<T>>(
      value: Num | number,
      result: T,
    ): CasesStructApi<T> {
      const clauses: { condition: Num; value: T }[] = [
        { condition: sel.equals(value), value: result },
      ];
      const api: CasesStructApi<T> = {
        case: (v, r) => {
          clauses.push({ condition: sel.equals(v), value: r });
          return api;
        },
        default: (fallback) => foldClauses(clauses, fallback),
      };
      return api;
    },
  };
}
