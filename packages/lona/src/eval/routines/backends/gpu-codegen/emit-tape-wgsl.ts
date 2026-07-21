import {
  type CompiledTape,
  OP_ABS,
  OP_ACOS,
  OP_ADD,
  OP_AND,
  OP_ASIN,
  OP_ASSERT_NONZERO,
  OP_ASSERT_ZERO,
  OP_ATAN,
  OP_ATAN2,
  OP_CBRT,
  OP_COMPARE,
  OP_COS,
  OP_DEBUG,
  OP_DIV,
  OP_EXP,
  OP_LIT,
  OP_LOG,
  OP_LOG1P,
  OP_MAX,
  OP_MIN,
  OP_MOD,
  OP_MUL,
  OP_NEG,
  OP_NOT,
  OP_OR,
  OP_SIGN,
  OP_SIN,
  OP_SQRT,
  OP_SUB,
  OP_TAN,
  OP_TANH,
  OP_VAR,
} from "../../../tape";

export function wgslValueRef(index: number): string {
  return `_${index}`;
}

export function unaryTapeOpWgsl(op: number, operand: string): string | null {
  switch (op) {
    case OP_SQRT:
      return `sqrt(${operand})`;
    case OP_CBRT:
      return `sign(${operand}) * pow(abs(${operand}), 0.33333333)`;
    case OP_COS:
      return `cos(${operand})`;
    case OP_ACOS:
      return `acos(${operand})`;
    case OP_ASIN:
      return `asin(${operand})`;
    case OP_TAN:
      return `tan(${operand})`;
    case OP_ATAN:
      return `atan(${operand})`;
    case OP_LOG:
      return `log(${operand})`;
    case OP_EXP:
      return `exp(${operand})`;
    case OP_ABS:
      return `abs(${operand})`;
    case OP_NEG:
      return `(-${operand})`;
    case OP_SIN:
      return `sin(${operand})`;
    case OP_SIGN:
      return `sign(${operand})`;
    case OP_NOT:
      return `select(0.0, 1.0, ${operand} == 0.0)`;
    case OP_TANH:
      return `tanh(${operand})`;
    case OP_LOG1P:
      return `log(1.0 + ${operand})`;
    case OP_DEBUG:
    case OP_ASSERT_ZERO:
    case OP_ASSERT_NONZERO:
      return operand;
  }
  return null;
}

export function binaryTapeOpWgsl(
  op: number,
  left: string,
  right: string,
): string | null {
  switch (op) {
    case OP_ADD:
      return `(${left} + ${right})`;
    case OP_SUB:
      return `(${left} - ${right})`;
    case OP_MUL:
      return `(${left} * ${right})`;
    case OP_DIV:
      return `select(1e38, ${left} / ${right}, ${right} != 0.0)`;
    case OP_MOD:
      return `(${left} % ${right})`;
    case OP_ATAN2:
      return `atan2(${left}, ${right})`;
    case OP_MIN:
      return `min(${left}, ${right})`;
    case OP_MAX:
      return `max(${left}, ${right})`;
    case OP_COMPARE:
      return `sign(${left} - ${right})`;
    case OP_AND:
      return `select(${right}, ${left}, ${left} == 0.0)`;
    case OP_OR:
      return `select(${left}, ${right}, ${left} == 0.0)`;
  }
  return null;
}

export function formatWgslF32(value: number): string {
  if (!Number.isFinite(value)) {
    if (value === Infinity) return "1e38";
    if (value === -Infinity) return "-1e38";
    return "0.0";
  }
  const formatted = value.toString();
  return formatted.includes(".") ||
    formatted.includes("e") ||
    formatted.includes("E")
    ? formatted
    : `${formatted}.0`;
}

export interface EmittedTapeWgsl {
  readonly body: string;
  readonly roots: readonly string[];
}

/** Emit one invocation of a scalar tape without assuming a data layout. */
export function emitTapeWgsl(
  tape: CompiledTape,
  variable: (slot: number) => string,
): EmittedTapeWgsl {
  const lines: string[] = [];
  for (let index = 0; index < tape.opcodes.length; index++) {
    const op = tape.opcodes[index]!;
    const left = tape.argA[index]!;
    const right = tape.argB[index]!;
    let expression: string | null;
    if (op === OP_LIT) expression = formatWgslF32(tape.literals[left]!);
    else if (op === OP_VAR) expression = variable(left);
    else {
      expression =
        unaryTapeOpWgsl(op, wgslValueRef(left)) ??
        binaryTapeOpWgsl(op, wgslValueRef(left), wgslValueRef(right));
    }
    if (expression === null) {
      throw new Error(
        `GPU codegen does not support tape opcode ${op} at node ${index}`,
      );
    }
    lines.push(`  let ${wgslValueRef(index)} = ${expression};`);
  }
  return {
    body: lines.join("\n"),
    roots: tape.rootIndices.map(wgslValueRef),
  };
}
