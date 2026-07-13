import type { VarName } from "../core/tree";
import type {
  Branch,
  CasesApi,
  Condition,
  NumericApi,
  WhenApi,
  WhenChainApi,
} from "./interfaces";

const DIV_BY_ZERO_FALLBACK = 1e50;
const SMOOTHABS_FACTOR = 10;
const SOFTPLUS_FACTOR = 50;

type DerivativeMap = ReadonlyMap<VarName, number>;

function cleanDerivativeMap(
  entries: Iterable<[VarName, number]>,
): Map<VarName, number> {
  const out = new Map<VarName, number>();
  for (const [name, value] of entries) {
    if (value !== 0) out.set(name, value);
  }
  return out;
}

function combineDerivativeMaps(
  a: DerivativeMap,
  b: DerivativeMap,
  aScale = 1,
  bScale = 1,
): Map<VarName, number> {
  const out = new Map<VarName, number>();
  if (aScale !== 0) {
    for (const [name, value] of a) {
      const scaled = value * aScale;
      if (scaled !== 0) out.set(name, scaled);
    }
  }
  if (bScale !== 0) {
    for (const [name, value] of b) {
      const next = (out.get(name) ?? 0) + value * bScale;
      if (next === 0) out.delete(name);
      else out.set(name, next);
    }
  }
  return out;
}

function scaleDerivativeMap(
  derivatives: DerivativeMap,
  scale: number,
): Map<VarName, number> {
  if (scale === 0) return new Map();
  return cleanDerivativeMap(
    Array.from(derivatives, ([name, value]) => [name, value * scale]),
  );
}

function literalValue(n: DirectNum | number): number {
  return typeof n === "number" ? n : n.value;
}

export function asNum(n: number | DirectNum): DirectNum {
  if (n instanceof DirectNum) return n;
  return new DirectNum(n);
}

export function variableNum(name: VarName, value = 0): DirectNum {
  return new DirectNum(value, [[name, 1]]);
}

export function variable(name: VarName, value = 0): DirectNum {
  return variableNum(name, value);
}

export function asNumber(n: number | DirectNum): number {
  return literalValue(n);
}

export class DirectNum implements NumericApi<DirectNum> {
  readonly value: number;
  readonly derivatives: DerivativeMap;

  constructor(
    value: number,
    derivatives: DerivativeMap | Iterable<[VarName, number]> = [],
  ) {
    this.value = value;
    this.derivatives = cleanDerivativeMap(derivatives);
  }

  derivative(name: VarName): number {
    return this.derivatives.get(name) ?? 0;
  }

  gradient(names: VarName[]): number[] {
    return names.map((name) => this.derivative(name));
  }

  add(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return new DirectNum(
      this.value + rhs.value,
      combineDerivativeMaps(this.derivatives, rhs.derivatives),
    );
  }

  sub(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return new DirectNum(
      this.value - rhs.value,
      combineDerivativeMaps(this.derivatives, rhs.derivatives, 1, -1),
    );
  }

  mul(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return new DirectNum(
      this.value * rhs.value,
      combineDerivativeMaps(
        this.derivatives,
        rhs.derivatives,
        rhs.value,
        this.value,
      ),
    );
  }

  powi(power: number): DirectNum {
    if (!Number.isInteger(power) || power <= 0) {
      throw new Error(`power must be a positive integer, ${power} recieved`);
    }
    if (power === 1) return this;
    if (power === 2) return this.square();
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let result: DirectNum = this;
    for (let i = 1; i < power; i++) {
      result = result.mul(this);
    }
    return result;
  }

  div(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    if (rhs.value === 0) return new DirectNum(DIV_BY_ZERO_FALLBACK);
    const invB2 = 1 / (rhs.value * rhs.value);
    return new DirectNum(
      this.value / rhs.value,
      combineDerivativeMaps(
        this.derivatives,
        rhs.derivatives,
        1 / rhs.value,
        -this.value * invB2,
      ),
    );
  }

  sqrt(): DirectNum {
    const value = Math.sqrt(this.value);
    const coeff = value !== 0 ? 0.5 / value : 0;
    return new DirectNum(value, scaleDerivativeMap(this.derivatives, coeff));
  }

  safeSqrt(): DirectNum {
    return this.max(0).sqrt();
  }

  cbrt(): DirectNum {
    const value = Math.cbrt(this.value);
    const coeff = value !== 0 ? 1 / (3 * value * value) : 0;
    return new DirectNum(value, scaleDerivativeMap(this.derivatives, coeff));
  }

  neg(): DirectNum {
    return new DirectNum(-this.value, scaleDerivativeMap(this.derivatives, -1));
  }

  inv(): DirectNum {
    return ONE.div(this);
  }

  sign(): DirectNum {
    return new DirectNum(Math.sign(this.value));
  }

  abs(): DirectNum {
    const sign = this.value > 0 ? 1 : this.value < 0 ? -1 : 0;
    return new DirectNum(
      Math.abs(this.value),
      scaleDerivativeMap(this.derivatives, sign),
    );
  }

  smoothabs(): DirectNum {
    return this.mul(this.mul(SMOOTHABS_FACTOR).tanh());
  }

  log1p(): DirectNum {
    return new DirectNum(
      Math.log1p(this.value),
      scaleDerivativeMap(this.derivatives, 1 / (1 + this.value)),
    );
  }

  softplus(): DirectNum {
    const factor = SOFTPLUS_FACTOR;
    const val = this.mul(factor);
    const amax = val.max(ZERO);
    return val.abs().neg().exp().log1p().add(amax).div(factor);
  }

  softminus(): DirectNum {
    return this.sub(this.softplus());
  }

  mod(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return new DirectNum(this.value % rhs.value, this.derivatives);
  }

  cos(): DirectNum {
    return new DirectNum(
      Math.cos(this.value),
      scaleDerivativeMap(this.derivatives, -Math.sin(this.value)),
    );
  }

  acos(): DirectNum {
    return new DirectNum(
      Math.acos(this.value),
      scaleDerivativeMap(
        this.derivatives,
        -1 / Math.sqrt(1 - this.value * this.value),
      ),
    );
  }

  sin(): DirectNum {
    return new DirectNum(
      Math.sin(this.value),
      scaleDerivativeMap(this.derivatives, Math.cos(this.value)),
    );
  }

  asin(): DirectNum {
    return new DirectNum(
      Math.asin(this.value),
      scaleDerivativeMap(
        this.derivatives,
        1 / Math.sqrt(1 - this.value * this.value),
      ),
    );
  }

  tan(): DirectNum {
    const cos = Math.cos(this.value);
    return new DirectNum(
      Math.tan(this.value),
      scaleDerivativeMap(this.derivatives, 1 / (cos * cos)),
    );
  }

  atan(): DirectNum {
    return new DirectNum(
      Math.atan(this.value),
      scaleDerivativeMap(this.derivatives, 1 / (1 + this.value * this.value)),
    );
  }

  exp(): DirectNum {
    const value = Math.exp(this.value);
    return new DirectNum(value, scaleDerivativeMap(this.derivatives, value));
  }

  tanh(): DirectNum {
    const value = Math.tanh(this.value);
    return new DirectNum(
      value,
      scaleDerivativeMap(this.derivatives, 1 - value * value),
    );
  }

  log(): DirectNum {
    return new DirectNum(
      Math.log(this.value),
      scaleDerivativeMap(this.derivatives, 1 / this.value),
    );
  }

  square(): DirectNum {
    return new DirectNum(
      this.value * this.value,
      scaleDerivativeMap(this.derivatives, 2 * this.value),
    );
  }

  compare(other: DirectNum | number): DirectNum {
    return new DirectNum(Math.sign(this.value - literalValue(other)));
  }

  and(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return this.value === 0 ? this : rhs;
  }

  or(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return this.value === 0 ? rhs : this;
  }

  not(): DirectNum {
    return this.value ? ZERO : ONE;
  }

  max(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return this.value >= rhs.value ? this : rhs;
  }

  min(other: DirectNum | number): DirectNum {
    const rhs = asNum(other);
    return this.value <= rhs.value ? this : rhs;
  }

  equals(other: DirectNum | number): DirectNum {
    return this.compare(other).not();
  }

  lessThan(other: DirectNum | number): DirectNum {
    return asNum(other).compare(this).max(ZERO);
  }

  lessThanOrEqual(other: DirectNum | number): DirectNum {
    return asNum(other).compare(this).add(ONE).min(ONE);
  }

  greaterThan(other: DirectNum | number): DirectNum {
    return asNum(other).lessThan(this);
  }

  greaterThanOrEqual(other: DirectNum | number): DirectNum {
    return asNum(other).lessThanOrEqual(this);
  }
}

export function max(
  first: DirectNum | number,
  ...others: Array<DirectNum | number>
): DirectNum {
  let result = asNum(first);
  for (const n of others) result = result.max(n);
  return result;
}

export function min(
  first: DirectNum | number,
  ...others: Array<DirectNum | number>
): DirectNum {
  let result = asNum(first);
  for (const n of others) result = result.min(n);
  return result;
}

export function atan2(a: DirectNum | number, b: DirectNum | number): DirectNum {
  const lhs = asNum(a);
  const rhs = asNum(b);
  const denom = lhs.value * lhs.value + rhs.value * rhs.value;
  if (denom === 0) return new DirectNum(Math.atan2(lhs.value, rhs.value));
  return new DirectNum(
    Math.atan2(lhs.value, rhs.value),
    combineDerivativeMaps(
      lhs.derivatives,
      rhs.derivatives,
      rhs.value / denom,
      -lhs.value / denom,
    ),
  );
}

export function ifTruthyElse(
  condition: DirectNum | number,
  ifNonZero: DirectNum | number,
  ifZero: DirectNum | number,
): DirectNum {
  const cond = asNum(condition);
  return cond.value === 0 ? asNum(ifZero) : asNum(ifNonZero);
}

export type DirectBranch = Branch<DirectNum>;
export type DirectCondition = Condition<DirectNum>;
export type DirectWhen = WhenApi<DirectNum>;
export type DirectWhenChain = WhenChainApi<DirectNum>;
export type DirectCases = CasesApi<DirectNum>;

function conditionValue(condition: DirectCondition): number {
  return literalValue(
    typeof condition === "function" ? condition() : condition,
  );
}

function whenChain(matched: boolean, value?: DirectNum): DirectWhenChain {
  const chooseFallback = (ifZero: DirectBranch): DirectNum =>
    matched ? value! : asNum(ifZero());

  return {
    elseIf: (condition: DirectCondition): DirectWhen => {
      if (matched) {
        return { then: () => whenChain(true, value) };
      }
      return when(condition);
    },
    else: chooseFallback,
  };
}

export function when(condition: DirectCondition): DirectWhen {
  const isTruthy = conditionValue(condition) !== 0;
  return {
    then: (ifNonZero: DirectBranch): DirectWhenChain =>
      isTruthy ? whenChain(true, asNum(ifNonZero())) : whenChain(false),
  };
}

type DirectCase = {
  value: DirectNum | number;
  branch: DirectBranch;
};

function makeCases(
  selector: DirectNum | number,
  caseDefs: DirectCase[],
): DirectCases {
  const chooseFallback = (fallback: DirectBranch): DirectNum => {
    const selected = asNum(selector).value;
    for (const c of caseDefs) {
      if (selected === literalValue(c.value)) {
        return asNum(c.branch());
      }
    }
    return asNum(fallback());
  };

  return {
    case: (value: DirectNum | number, branch: DirectBranch): DirectCases =>
      makeCases(selector, [...caseDefs, { value, branch }]),
    default: chooseFallback,
  };
}

export function cases(selector: DirectNum | number): DirectCases {
  return makeCases(selector, []);
}

export function hypot(a: DirectNum | number, b: DirectNum | number): DirectNum {
  return asNum(a).square().add(asNum(b).square()).sqrt();
}

export function clamp(
  a: DirectNum | number,
  minVal: DirectNum | number,
  maxVal: DirectNum | number,
): DirectNum {
  return max(minVal, a).min(maxVal);
}

export function sigmoid(a: DirectNum | number): DirectNum {
  const v = asNum(a);
  return when(v.greaterThan(0))
    .then(() => ONE.div(v.neg().exp().add(ONE)))
    .else(() => {
      const exp = v.exp();
      return exp.div(exp.add(ONE));
    });
}

export type NumLike = number | DirectNum;
export type DirectDerivativeBuilder = (...args: DirectNum[]) => DirectNum;

export type WrapNumFnOptions = {
  derivatives?: Record<string, DirectDerivativeBuilder>;
};

export type WrappedNumFn<T extends Array<NumLike>> = ((
  ...args: T
) => DirectNum) & {
  eval: (...args: number[]) => number;
  vars: () => string[];
  partial: (...args: Array<NumLike | undefined>) => WrappedNumFn<T>;
  partialNamed: (
    values: Record<string, NumLike | undefined>,
  ) => WrappedNumFn<T>;
  derivative: (nameOrIndex: string | number) => WrappedNumFn<T>;
  derivatives: () => Record<string, WrappedNumFn<T>>;
  gradient: () => Array<WrappedNumFn<T>>;
};

class WrappedDirectNumFn<T extends Array<NumLike>> {
  constructor(
    private readonly fn: (...args: DirectNum[]) => DirectNum,
    private readonly names: string[],
  ) {}

  private getName(nameOrIndex: string | number): string {
    if (typeof nameOrIndex === "number") {
      const name = this.names[nameOrIndex];
      if (!name) throw new Error(`Unknown derivative index ${nameOrIndex}`);
      return name;
    }
    return nameOrIndex;
  }

  call(...args: T): DirectNum {
    const values = this.names.map((name, index) => {
      const value = args[index];
      if (value instanceof DirectNum) return value;
      return variableNum(name, value ?? 0);
    });
    return this.fn(...values);
  }

  eval(...args: number[]): number {
    return this.call(...(args as T)).value;
  }

  vars(): string[] {
    return this.names.slice();
  }

  partial(...args: Array<NumLike | undefined>): WrappedNumFn<T> {
    const remainingNames: string[] = [];
    for (let i = 0; i < this.names.length; i++) {
      if (args[i] === undefined) {
        remainingNames.push(this.names[i]!);
      }
    }

    return this.spawnPartial(args, remainingNames);
  }

  partialNamed(values: Record<string, NumLike | undefined>): WrappedNumFn<T> {
    const args = this.names.map((name) => values[name]);
    const remainingNames: string[] = [];
    for (let i = 0; i < this.names.length; i++) {
      if (args[i] === undefined) {
        remainingNames.push(this.names[i]!);
      }
    }

    return this.spawnPartial(args, remainingNames);
  }

  private spawnPartial(
    args: Array<NumLike | undefined>,
    remainingNames: string[],
  ): WrappedNumFn<T> {
    return makeWrappedNumFn(
      new WrappedDirectNumFn<T>((...remainingArgs: DirectNum[]) => {
        const allArgs = new Array<DirectNum>(this.names.length);
        let nextRemaining = 0;
        for (let i = 0; i < this.names.length; i++) {
          const fixed = args[i];
          allArgs[i] =
            fixed === undefined
              ? remainingArgs[nextRemaining++]!
              : asNum(fixed);
        }
        return this.fn(...allArgs);
      }, remainingNames),
    );
  }

  derivative(nameOrIndex: string | number): WrappedNumFn<T> {
    const name = this.getName(nameOrIndex);
    if (!this.names.includes(name)) {
      throw new Error(`Unknown derivative '${name}'`);
    }
    return makeWrappedNumFn(
      new WrappedDirectNumFn<T>(
        (...args: DirectNum[]) =>
          new DirectNum(this.fn(...args).derivative(name)),
        this.names,
      ),
    );
  }

  derivatives(): Record<string, WrappedNumFn<T>> {
    const out: Record<string, WrappedNumFn<T>> = {};
    for (const name of this.names) out[name] = this.derivative(name);
    return out;
  }

  gradient(): Array<WrappedNumFn<T>> {
    return this.names.map((name) => this.derivative(name));
  }
}

function makeWrappedNumFn<T extends Array<NumLike>>(
  instance: WrappedDirectNumFn<T>,
): WrappedNumFn<T> {
  const wrapped = (...args: T) => instance.call(...args);
  wrapped.eval = (...args: number[]) => instance.eval(...args);
  wrapped.vars = () => instance.vars();
  wrapped.partial = (...args: Array<NumLike | undefined>) =>
    instance.partial(...args);
  wrapped.partialNamed = (values: Record<string, NumLike | undefined>) =>
    instance.partialNamed(values);
  wrapped.derivative = (nameOrIndex: string | number) =>
    instance.derivative(nameOrIndex);
  wrapped.derivatives = () => instance.derivatives();
  wrapped.gradient = () => instance.gradient();
  return wrapped;
}

export function wrapNumFn<T extends Array<NumLike>>(
  fn: (...args: DirectNum[]) => DirectNum,
  _options: WrapNumFnOptions = {},
): WrappedNumFn<T> {
  const names = Array.from({ length: fn.length }, (_, i) => `arg${i}`);
  return makeWrappedNumFn(new WrappedDirectNumFn<T>(fn, names));
}

export function wrapNumMethods<
  T extends object,
  K extends keyof T & string,
  Args extends unknown[] = unknown[],
>(
  cls: new (...args: Args) => T,
  ...methodNames: K[]
): (instance: T) => Record<K, WrappedNumFn<Array<NumLike>>> {
  void cls;
  return (instance: T) => {
    const out = {} as Record<K, WrappedNumFn<Array<NumLike>>>;
    for (const name of methodNames) {
      const method = instance[name];
      if (typeof method !== "function") {
        throw new Error(`Expected method '${name}' to be a function`);
      }
      out[name] = wrapNumFn(
        (method as (...args: DirectNum[]) => DirectNum).bind(instance),
      );
    }
    return out;
  };
}

export function wrapNumMethodClass<
  Args extends unknown[] = unknown[],
  K extends string = string,
>(
  BaseClass: new (...args: Args) => object,
  ...methodNames: K[]
): new (...args: Args) => object & Record<K, WrappedNumFn<Array<NumLike>>> {
  const Wrapped = class extends (BaseClass as new (...args: Args) => object) {
    constructor(...args: Args) {
      super(...args);
      for (const name of methodNames) {
        const method = (this as Record<string, unknown>)[name];
        if (typeof method !== "function") {
          throw new Error(`Expected method '${name}' to be a function`);
        }
        (this as Record<string, unknown>)[name] = wrapNumFn(
          (method as (...args: DirectNum[]) => DirectNum).bind(this),
        );
      }
    }
  };

  return Wrapped as new (
    ...args: Args
  ) => object & Record<K, WrappedNumFn<Array<NumLike>>>;
}

export * from "./direct-routine";

export const ZERO = new DirectNum(0);
export const NEG_ONE = new DirectNum(-1);
export const ONE = new DirectNum(1);
export const TWO = new DirectNum(2);

export const numFactory = {
  optimized: (n: number) => asNum(n),
};

export const NumX = variableNum("x");
export const NumY = variableNum("y");
export const NumZ = variableNum("z");
