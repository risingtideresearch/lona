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
