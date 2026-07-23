import type { VarName } from "../../../../core/tree";
import { assertTapeValue, bindVarMap } from "../../../tape";
import {
  type CompiledTape,
  OP_LIT,
  OP_VAR,
  OP_SQRT,
  OP_CBRT,
  OP_COS,
  OP_ACOS,
  OP_ASIN,
  OP_TAN,
  OP_ATAN,
  OP_LOG,
  OP_EXP,
  OP_ABS,
  OP_NEG,
  OP_SIN,
  OP_SIGN,
  OP_NOT,
  OP_TANH,
  OP_LOG1P,
  OP_DEBUG,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_MOD,
  OP_ATAN2,
  OP_MIN,
  OP_MAX,
  OP_COMPARE,
  OP_AND,
  OP_OR,
  OP_ASSERT_ZERO,
  OP_ASSERT_NONZERO,
  DIV_BY_ZERO_FALLBACK,
} from "../../../tape";

export type CompiledMultiFn = (
  vars: Map<VarName, number>,
  derivatives?: Map<VarName, number>,
) => number[];

// ---------------------------------------------------------------------------
// Code generation helpers
// ---------------------------------------------------------------------------

function varRef(i: number): string {
  return `_${i}`;
}

/** Opcode → JS unary expression */
function unaryExpr(op: number, operand: string): string | null {
  switch (op) {
    case OP_SQRT:
      return `Math.sqrt(${operand})`;
    case OP_CBRT:
      return `Math.cbrt(${operand})`;
    case OP_COS:
      return `Math.cos(${operand})`;
    case OP_ACOS:
      return `Math.acos(${operand})`;
    case OP_ASIN:
      return `Math.asin(${operand})`;
    case OP_TAN:
      return `Math.tan(${operand})`;
    case OP_ATAN:
      return `Math.atan(${operand})`;
    case OP_LOG:
      return `Math.log(${operand})`;
    case OP_EXP:
      return `Math.exp(${operand})`;
    case OP_ABS:
      return `Math.abs(${operand})`;
    case OP_NEG:
      return `(-${operand})`;
    case OP_SIN:
      return `Math.sin(${operand})`;
    case OP_SIGN:
      return `Math.sign(${operand})`;
    case OP_NOT:
      return `(${operand}?0:1)`;
    case OP_TANH:
      return `Math.tanh(${operand})`;
    case OP_LOG1P:
      return `Math.log1p(${operand})`;
    case OP_DEBUG:
      return operand;
    case OP_ASSERT_ZERO:
      return `__assertZero(${operand})`;
    case OP_ASSERT_NONZERO:
      return `__assertNonzero(${operand})`;
  }
  return null;
}

/** Opcode → JS binary expression */
function binaryExpr(op: number, left: string, right: string): string | null {
  switch (op) {
    case OP_ADD:
      return `(${left}+${right})`;
    case OP_SUB:
      return `(${left}-${right})`;
    case OP_MUL:
      return `(${left}*${right})`;
    case OP_DIV:
      return `(${right}?${left}/${right}:${DIV_BY_ZERO_FALLBACK})`;
    case OP_MOD:
      return `(${left}%${right})`;
    case OP_ATAN2:
      return `Math.atan2(${left},${right})`;
    case OP_MIN:
      return `Math.min(${left},${right})`;
    case OP_MAX:
      return `Math.max(${left},${right})`;
    case OP_COMPARE:
      return `Math.sign(${left}-${right})`;
    case OP_AND:
      return `(${left}===0?${left}:${right})`;
    case OP_OR:
      return `(${left}===0?${right}:${left})`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// compileFunctionFromTape — from a CompiledTape
// ---------------------------------------------------------------------------

function wrapCompiledMultiFn(
  body: string,
  tape: CompiledTape,
  numRoots: number,
): CompiledMultiFn {
  const fn = new Function(
    "vv",
    "r",
    "__assertZero",
    "__assertNonzero",
    body,
  ) as (
    vv: Float64Array,
    r: number[],
    assertZero: (observed: number) => number,
    assertNonzero: (observed: number) => number,
  ) => number[];
  const vv = new Float64Array(tape.varSlots.length);
  const result = new Array<number>(numRoots);
  return (vars, derivatives) => {
    bindVarMap(tape, vars, derivatives, vv);
    return fn(
      vv,
      result,
      (observed) => assertTapeValue("zero", observed),
      (observed) => assertTapeValue("nonzero", observed),
    );
  };
}

export function compileFunctionFromTape(tape: CompiledTape): CompiledMultiFn {
  const { opcodes, argA, argB, literals, rootIndices } = tape;
  const len = opcodes.length;
  const numRoots = rootIndices.length;
  const lines: string[] = [];

  for (let i = 0; i < len; i++) {
    const op = opcodes[i]!;
    const v = varRef(i);

    if (op === OP_LIT) {
      lines.push(`const ${v}=${literals[argA[i]!]!};`);
    } else if (op === OP_VAR) {
      lines.push(`const ${v}=vv[${argA[i]!}];`);
    } else {
      const a = varRef(argA[i]!);
      const b = varRef(argB[i]!);
      const expr = unaryExpr(op, a) ?? binaryExpr(op, a, b);
      if (!expr) {
        // Unknown opcode — should not happen with a valid tape
        lines.push(`const ${v}=NaN;`);
      } else {
        lines.push(`const ${v}=${expr};`);
      }
    }
  }

  if (numRoots === 1) {
    lines.push(`r[0]=${varRef(rootIndices[0])};`);
  } else {
    for (let i = 0; i < numRoots; i++) {
      lines.push(`r[${i}]=${varRef(rootIndices[i]!)};`);
    }
  }
  lines.push(`return r;`);

  return wrapCompiledMultiFn(lines.join("\n"), tape, numRoots);
}

export type CompiledJvpFn = (
  values: Float64Array,
  seeds: Float64Array,
) => { vals: number[]; tangents: number[][] };

function unaryDerivativeExpr(
  op: number,
  operand: string,
  value: string,
  derivative: string,
): string | null {
  switch (op) {
    case OP_NEG:
      return `(-${derivative})`;
    case OP_ABS:
      return `((${operand}>0?1:${operand}<0?-1:0)*${derivative})`;
    case OP_SQRT:
      return `(${value}!==0?0.5/${value}*${derivative}:0)`;
    case OP_CBRT:
      return `(${value}!==0?${derivative}/(3*${value}*${value}):0)`;
    case OP_SIN:
      return `(Math.cos(${operand})*${derivative})`;
    case OP_COS:
      return `(-Math.sin(${operand})*${derivative})`;
    case OP_TAN:
      return `(${derivative}/(Math.cos(${operand})**2))`;
    case OP_ASIN:
      return `(${derivative}/Math.sqrt(1-${operand}*${operand}))`;
    case OP_ACOS:
      return `(-${derivative}/Math.sqrt(1-${operand}*${operand}))`;
    case OP_ATAN:
      return `(${derivative}/(1+${operand}*${operand}))`;
    case OP_EXP:
      return `(${value}*${derivative})`;
    case OP_LOG:
      return `(${derivative}/${operand})`;
    case OP_LOG1P:
      return `(${derivative}/(1+${operand}))`;
    case OP_TANH:
      return `((1-${value}*${value})*${derivative})`;
    case OP_DEBUG:
    case OP_ASSERT_ZERO:
    case OP_ASSERT_NONZERO:
      return derivative;
    case OP_SIGN:
    case OP_NOT:
      return "0";
  }
  return null;
}

function binaryDerivativeExpr(
  op: number,
  left: string,
  right: string,
  leftDerivative: string,
  rightDerivative: string,
): string | null {
  switch (op) {
    case OP_ADD:
      return `(${leftDerivative}+${rightDerivative})`;
    case OP_SUB:
      return `(${leftDerivative}-${rightDerivative})`;
    case OP_MUL:
      return `(${leftDerivative}*${right}+${left}*${rightDerivative})`;
    case OP_DIV:
      return `(${right}?(${leftDerivative}*${right}-${left}*${rightDerivative})/(${right}*${right}):0)`;
    case OP_MOD:
      return leftDerivative;
    case OP_ATAN2:
      return `((${left}*${left}+${right}*${right})?(${right}*${leftDerivative}-${left}*${rightDerivative})/(${left}*${left}+${right}*${right}):0)`;
    case OP_MIN:
      return `(${left}<=${right}?${leftDerivative}:${rightDerivative})`;
    case OP_MAX:
      return `(${left}>=${right}?${leftDerivative}:${rightDerivative})`;
    case OP_COMPARE:
      return "0";
    case OP_AND:
      return `(${left}===0?${leftDerivative}:${rightDerivative})`;
    case OP_OR:
      return `(${left}===0?${rightDerivative}:${leftDerivative})`;
  }
  return null;
}

/** Generate a direct multi-root JVP function from a fixed tape. */
export function compileJvpFunctionFromTape(
  tape: CompiledTape,
  numDirections: number,
): CompiledJvpFn {
  const { opcodes, argA, argB, literals, rootIndices, numVars } = tape;
  const lines: string[] = [];
  const derivativeRef = (node: number, direction = "q") =>
    `dd[${node}*${numDirections}+${direction}]`;

  for (let i = 0; i < opcodes.length; i++) {
    const op = opcodes[i]!;
    const value = varRef(i);
    if (op === OP_LIT) {
      lines.push(`const ${value}=${literals[argA[i]!]!};`);
      lines.push(`for(let q=0;q<${numDirections};q++)${derivativeRef(i)}=0;`);
      continue;
    }
    if (op === OP_VAR) {
      const slot = argA[i]!;
      lines.push(`const ${value}=${slot < numVars ? `vv[${slot}]` : "0"};`);
      lines.push(
        `for(let q=0;q<${numDirections};q++)${derivativeRef(i)}=${slot < numVars ? `ss[${slot}*${numDirections}+q]` : "0"};`,
      );
      continue;
    }
    const aIndex = argA[i]!;
    const bIndex = argB[i]!;
    const a = varRef(aIndex);
    const b = varRef(bIndex);
    const primal =
      op === OP_MIN
        ? `(${a}<=${b}?${a}:${b})`
        : op === OP_MAX
          ? `(${a}>=${b}?${a}:${b})`
          : (unaryExpr(op, a) ?? binaryExpr(op, a, b));
    if (!primal) throw new Error(`unsupported JVP opcode ${op}`);
    lines.push(`const ${value}=${primal};`);
    const derivative =
      unaryDerivativeExpr(op, a, value, derivativeRef(aIndex)) ??
      binaryDerivativeExpr(
        op,
        a,
        b,
        derivativeRef(aIndex),
        derivativeRef(bIndex),
      );
    if (!derivative) throw new Error(`unsupported JVP derivative opcode ${op}`);
    lines.push(
      `for(let q=0;q<${numDirections};q++)${derivativeRef(i)}=${derivative};`,
    );
  }
  for (let root = 0; root < rootIndices.length; root++) {
    lines.push(`rr[${root}]=${varRef(rootIndices[root]!)};`);
    lines.push(
      `for(let q=0;q<${numDirections};q++)tt[${root}][q]=${derivativeRef(rootIndices[root]!)};`,
    );
  }
  lines.push("return {vals:rr,tangents:tt};");

  const fn = new Function(
    "vv",
    "ss",
    "rr",
    "tt",
    "dd",
    "__assertZero",
    "__assertNonzero",
    lines.join("\n"),
  ) as (
    values: Float64Array,
    seeds: Float64Array,
    results: number[],
    tangents: number[][],
    duals: Float64Array,
    assertZero: (observed: number) => number,
    assertNonzero: (observed: number) => number,
  ) => { vals: number[]; tangents: number[][] };

  return (values, seeds) => {
    if (values.length !== numVars) {
      throw new Error(
        `seeded JVP expected ${numVars} values, got ${values.length}`,
      );
    }
    if (seeds.length !== numVars * numDirections) {
      throw new Error(
        `seeded JVP expected ${numVars * numDirections} seeds, got ${seeds.length}`,
      );
    }
    return fn(
      values,
      seeds,
      new Array<number>(rootIndices.length),
      Array.from(
        { length: rootIndices.length },
        () => new Array<number>(numDirections),
      ),
      new Float64Array(opcodes.length * numDirections),
      (observed) => assertTapeValue("zero", observed),
      (observed) => assertTapeValue("nonzero", observed),
    );
  };
}
