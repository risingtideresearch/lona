import type { NumericApi } from "../api/interfaces";
import type { BinaryOperation, UnaryOperation } from "../types";
import {
  NumNode,
  LiteralNum,
  UnaryOp,
  BinaryOp,
  ZERO_NODE,
  ONE_NODE,
  TWO_NODE,
  NEG_ONE_NODE,
  KIND_ABS,
  KIND_LIT,
  KIND_MUL,
  KIND_NEG,
  type VarName,
} from "./tree";
import {
  binaryNode,
  debugNode,
  litNode,
  selectNode,
  unaryNode,
  varNode,
} from "./tree-cons";
import { renderNodeAsDot } from "../eval/transforms/to-dot";
import { simplify } from "../api/simplify";
import { simpleEval } from "../eval/eval-value";
import {
  serializeNumDAG,
  deserializeNumDAG,
  type SerializedNumDAG,
} from "./tree-serialization";
import {
  compileValueRoutine,
  type ValueRoutine,
  type MultiValueRoutine,
  type CompileOpts,
} from "../eval/routines";

const DIV_BY_ZERO_FALLBACK = 1e50;
const SMOOTHABS_FACTOR = 10;
const SOFTPLUS_FACTOR = 50;

function literalValue(n: Num | number): number | undefined {
  if (typeof n === "number") return n;
  return n.n.kind === KIND_LIT ? (n.n as LiteralNum).value : undefined;
}

function bothLiteralValues(
  a: Num,
  b: Num | number,
): [number, number] | undefined {
  const av = literalValue(a);
  const bv = literalValue(b);
  return av !== undefined && bv !== undefined ? [av, bv] : undefined;
}

// Fast inline check: returns true if both operands are literals.
// Avoids the array allocation of bothLiteralValues.
function areBothLiterals(a: Num, b: Num | number): boolean {
  if (typeof b === "number") return a.n.kind === KIND_LIT;
  return a.n.kind === KIND_LIT && b.n.kind === KIND_LIT;
}

function makeLiteral(value: number): Num {
  if (value === 0) return ZERO;
  if (value === 1) return ONE;
  if (value === -1) return NEG_ONE;
  if (value === 2) return TWO;
  return new Num(litNode(value));
}

export function asNum(n: number | Num): Num {
  if (n instanceof Num) return n;
  return makeLiteral(n);
}

export function variableNum(name: VarName): Num {
  return new Num(varNode(name));
}

export function binaryOpNum(op: BinaryOperation, a: Num, b: Num): Num {
  return new Num(binaryNode(op, a.n, b.n));
}

export function unaryOpNum(op: UnaryOperation, a: Num): Num {
  return new Num(unaryNode(op, a.n));
}

export function selectOpNum(condition: Num, ifNonZero: Num, ifZero: Num): Num {
  if (condition.n.kind === KIND_LIT) {
    return (condition.n as LiteralNum).value !== 0 ? ifNonZero : ifZero;
  }
  if (ifNonZero.n === ifZero.n) return ifNonZero;
  return new Num(selectNode(condition.n, ifNonZero.n, ifZero.n));
}

export class Num implements NumericApi<Num> {
  readonly n: NumNode;
  private _routine?: ValueRoutine | MultiValueRoutine | null;

  constructor(n: NumNode) {
    this.n = n;
  }

  add(other: Num | number): Num {
    if (areBothLiterals(this, other)) {
      const [a, b] = bothLiteralValues(this, other)!;
      return makeLiteral(a + b);
    }

    if (this.n === ZERO_NODE) return asNum(other);
    if (typeof other === "number" ? other === 0 : other.n === ZERO_NODE)
      return this;

    return new Num(binaryNode("ADD", this.n, asNum(other).n));
  }

  sub(other: Num | number): Num {
    if (areBothLiterals(this, other)) {
      const [a, b] = bothLiteralValues(this, other)!;
      return makeLiteral(a - b);
    }

    if (typeof other === "number" ? other === 0 : other.n === ZERO_NODE)
      return this;
    if (this.n === ZERO_NODE) return asNum(other).neg();
    // x - x = 0 (compare underlying nodes — hash-consing guarantees
    // that structural equality implies reference equality, so this
    // catches cases where the same expression was built via different
    // Num wrappers).
    if (other instanceof Num && this.n === other.n) return ZERO;

    return new Num(binaryNode("SUB", this.n, asNum(other).n));
  }

  mul(other: Num | number): Num {
    if (areBothLiterals(this, other)) {
      const [a, b] = bothLiteralValues(this, other)!;
      return makeLiteral(a * b);
    }

    const otherIsZero =
      typeof other === "number" ? other === 0 : other.n === ZERO_NODE;
    if (this.n === ZERO_NODE || otherIsZero) return ZERO;
    if (this.n === ONE_NODE) return asNum(other);
    if (typeof other === "number" ? other === 1 : other.n === ONE_NODE)
      return this;

    return new Num(binaryNode("MUL", this.n, asNum(other).n));
  }

  powi(power: number): Num {
    if (!Number.isInteger(power) || power <= 0) {
      throw new Error(`power must be a positive integer, ${power} recieved`);
    }
    if (power === 1) return this;
    if (power === 2) return this.square();
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let result: Num = this;
    for (let i = 1; i < power; i++) {
      result = result.mul(this);
    }
    return result;
  }

  div(other: Num | number): Num {
    if (areBothLiterals(this, other)) {
      const [a, b] = bothLiteralValues(this, other)!;
      return makeLiteral(b ? a / b : DIV_BY_ZERO_FALLBACK);
    }

    // x / 1 = x
    if (typeof other === "number" ? other === 1 : other.n === ONE_NODE)
      return this;
    // 0 / x = 0 (assume x != 0, consistent with foldConstants)
    if (this.n === ZERO_NODE) return ZERO;
    // x / x = 1 (assume x != 0, consistent with foldConstants).
    // Node-level identity is enough thanks to hash-consing.
    if (other instanceof Num && this.n === other.n) return ONE;
    // (a * x) / x = a, (x * a) / x = a
    if (other instanceof Num && this.n.kind === KIND_MUL) {
      const mul = this.n as BinaryOp;
      if (mul.left === other.n) return new Num(mul.right);
      if (mul.right === other.n) return new Num(mul.left);
    }

    return new Num(binaryNode("DIV", this.n, asNum(other).n));
  }

  sqrt(): Num {
    if (this.n === ZERO_NODE) return ZERO;
    if (this.n === ONE_NODE) return ONE;

    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.sqrt(v));
    return new Num(unaryNode("SQRT", this.n));
  }

  safeSqrt(): Num {
    return this.max(0).sqrt();
  }

  cbrt(): Num {
    if (this.n === ZERO_NODE) return ZERO;
    if (this.n === ONE_NODE) return ONE;

    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.cbrt(v));
    return new Num(unaryNode("CBRT", this.n));
  }

  neg(): Num {
    if (this.n === ZERO_NODE) return ZERO;

    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(-v);
    // -(-x) = x (unwrap the inner NEG directly)
    if (this.n.kind === KIND_NEG) {
      return new Num((this.n as UnaryOp).original);
    }
    return new Num(unaryNode("NEG", this.n));
  }

  inv(): Num {
    if (this.n === ONE_NODE) return ONE;

    const v = literalValue(this);
    if (v !== undefined) {
      return makeLiteral(v ? 1 / v : DIV_BY_ZERO_FALLBACK);
    }

    return new Num(binaryNode("DIV", ONE_NODE, this.n));
  }

  sign(): Num {
    if (this.n === ONE_NODE) return ONE;
    if (this.n === ZERO_NODE) return ZERO;
    if (this.n === NEG_ONE_NODE) return NEG_ONE;

    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.sign(v));
    return new Num(unaryNode("SIGN", this.n));
  }

  abs(): Num {
    if (this.n === ONE_NODE) return ONE;
    if (this.n === ZERO_NODE) return ZERO;

    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.abs(v));
    // abs(abs(x)) = abs(x) — `this` is already non-negative by construction.
    if (this.n.kind === KIND_ABS) return this;
    return new Num(unaryNode("ABS", this.n));
  }

  smoothabs(): Num {
    const v = literalValue(this);
    if (v !== undefined)
      return makeLiteral(v * Math.tanh(v * SMOOTHABS_FACTOR));

    return this.mul(unaryOpNum("TANH", this.mul(SMOOTHABS_FACTOR)));
  }

  log1p(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.log1p(v));
    return new Num(unaryNode("LOG1P", this.n));
  }

  softplus(): Num {
    // This implementation is based on the jax implementation of softplus.
    // It uses the log-sum-exp trick to avoid numerical instability.
    const factor = SOFTPLUS_FACTOR;
    const val = this.mul(factor);
    const amax = binaryOpNum("MAX", val, asNum(0));
    return val.abs().neg().exp().log1p().add(amax).div(factor);
  }

  softminus(): Num {
    return this.sub(this.softplus());
  }

  mod(other: Num): Num {
    const lit = bothLiteralValues(this, other);
    if (lit) return makeLiteral(lit[0] % lit[1]);

    return new Num(binaryNode("MOD", this.n, asNum(other).n));
  }

  cos(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.cos(v));
    return new Num(unaryNode("COS", this.n));
  }

  acos(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.acos(v));
    return new Num(unaryNode("ACOS", this.n));
  }

  sin(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.sin(v));
    return new Num(unaryNode("SIN", this.n));
  }

  asin(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.asin(v));
    return new Num(unaryNode("ASIN", this.n));
  }

  tan(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.tan(v));
    return new Num(unaryNode("TAN", this.n));
  }

  atan(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.atan(v));
    return new Num(unaryNode("ATAN", this.n));
  }

  exp(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.exp(v));
    return new Num(unaryNode("EXP", this.n));
  }

  tanh(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.tanh(v));
    return new Num(unaryNode("TANH", this.n));
  }

  log(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(Math.log(v));
    return new Num(unaryNode("LOG", this.n));
  }

  square(): Num {
    const v = literalValue(this);
    if (v !== undefined) return makeLiteral(v * v);
    if (this.n === ZERO_NODE) return ZERO;
    if (this.n === ONE_NODE) return ONE;
    return new Num(binaryNode("MUL", this.n, this.n));
  }

  compare(other: Num | number): Num {
    if (other instanceof Num && this.n === other.n) return ZERO;

    const lit = bothLiteralValues(this, other);
    if (lit) return makeLiteral(Math.sign(lit[0] - lit[1]));

    return new Num(binaryNode("COMPARE", this.n, asNum(other).n));
  }

  and(other: Num | number): Num {
    const lit = bothLiteralValues(this, other);
    if (lit) return makeLiteral(lit[0] === 0 ? lit[0] : lit[1]);

    const otherIsZero =
      typeof other === "number" ? other === 0 : other.n === ZERO_NODE;
    if (this.n === ZERO_NODE || otherIsZero) return ZERO;

    return new Num(binaryNode("AND", this.n, asNum(other).n));
  }

  or(other: Num | number): Num {
    const lit = bothLiteralValues(this, other);
    if (lit) return makeLiteral(lit[0] === 0 ? lit[1] : lit[0]);

    if (this.n === ZERO_NODE) return asNum(other);

    return new Num(binaryNode("OR", this.n, asNum(other).n));
  }

  not(): Num {
    const v = literalValue(this);
    if (v !== undefined) return v ? ZERO : ONE;
    return new Num(unaryNode("NOT", this.n));
  }

  max(other: Num | number): Num {
    if (other instanceof Num && this.n === other.n) return this;

    const lit = bothLiteralValues(this, other);
    if (lit) return makeLiteral(Math.max(lit[0], lit[1]));

    return new Num(binaryNode("MAX", this.n, asNum(other).n));
  }

  min(other: Num | number): Num {
    if (other instanceof Num && this.n === other.n) return this;

    const lit = bothLiteralValues(this, other);
    if (lit) return makeLiteral(Math.min(lit[0], lit[1]));

    return new Num(binaryNode("MIN", this.n, asNum(other).n));
  }

  equals(other: Num | number): Num {
    if (other instanceof Num && this.n === other.n) return ONE;

    const lit = bothLiteralValues(this, other);
    if (lit) return lit[0] === lit[1] ? ONE : ZERO;

    return asNum(other).compare(this).not();
  }

  lessThan(other: Num | number): Num {
    const lit = bothLiteralValues(this, other);
    if (lit) return lit[0] < lit[1] ? ONE : ZERO;

    return asNum(other).compare(this).max(ZERO);
  }

  lessThanOrEqual(other: Num | number): Num {
    const lit = bothLiteralValues(this, other);
    if (lit) return lit[0] <= lit[1] ? ONE : ZERO;

    return asNum(other).compare(this).add(ONE).min(ONE);
  }

  greaterThan(other: Num | number): Num {
    const lit = bothLiteralValues(this, other);
    if (lit) return lit[0] > lit[1] ? ONE : ZERO;

    return asNum(other).lessThan(this);
  }

  greaterThanOrEqual(other: Num | number): Num {
    const lit = bothLiteralValues(this, other);
    if (lit) return lit[0] >= lit[1] ? ONE : ZERO;

    return asNum(other).lessThanOrEqual(this);
  }

  debug(info: string): Num {
    return new Num(debugNode(this.n, info));
  }

  simplify(): Num {
    return new Num(simplify(this.n));
  }

  asDot(): string {
    return renderNodeAsDot(this.n);
  }

  eval(
    variables:
      | Record<string, number>
      | Map<VarName, number>
      | Map<string, number> = new Map(),
    logDebug = false,
    derivatives?: Map<VarName, number>,
  ): number {
    const vars =
      variables instanceof Map
        ? (variables as Map<VarName, number>)
        : new Map<VarName, number>(Object.entries(variables));

    if (!logDebug) {
      if (this._routine === undefined) {
        this._routine = compileValueRoutine([this.n]);
      }
      if (this._routine !== null) {
        // Single-root input → ValueRoutine.eval returns number.
        return (this._routine as ValueRoutine).eval(vars, derivatives);
      }
    }

    return simpleEval(this.n, vars, logDebug, derivatives);
  }

  /**
   * Compile this expression into a routine for repeated evaluation.
   * Pass `{ backend: "gpu-codegen" }` for GPU (requires `await initGpu()` first).
   */
  compile(opts?: CompileOpts): ValueRoutine {
    const r = compileValueRoutine([this.n], opts);
    if (!r) throw new Error("Failed to compile expression");
    if (r.shape !== "value")
      throw new Error(`Expected single-root routine, got '${r.shape}'`);
    return r;
  }

  toJSON(): { dag: SerializedNumDAG } {
    return { dag: serializeNumDAG(this.n) };
  }

  static fromJSON(json: { dag: SerializedNumDAG }): Num {
    return new Num(deserializeNumDAG(json.dag));
  }
}

export const ZERO = new Num(ZERO_NODE);
export const NEG_ONE = new Num(NEG_ONE_NODE);
export const ONE = new Num(ONE_NODE);
export const TWO = new Num(TWO_NODE);

export const numFactory = {
  optimized: (n: number) => asNum(n),
};

export const NumX = new Num(varNode("x"));
export const NumY = new Num(varNode("y"));
export const NumZ = new Num(varNode("z"));
