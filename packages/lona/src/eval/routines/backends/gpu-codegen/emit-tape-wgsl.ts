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

export interface EmittedTapeJvpWgsl extends EmittedTapeWgsl {
  /** Root-major, direction-minor tangent expressions. */
  readonly tangentRoots: readonly (readonly string[])[];
}

export interface TapeJvpVariableWgsl {
  readonly value: string;
  tangent(direction: number): string;
}

function tangentRef(index: number, direction: number): string {
  return `_${index}_d${direction}`;
}

function unaryTangentWgsl(
  op: number,
  operand: string,
  value: string,
  tangent: string,
): string | null {
  switch (op) {
    case OP_NEG:
      return `(-${tangent})`;
    case OP_ABS:
      return `(select(select(0.0, -1.0, ${operand} < 0.0), 1.0, ${operand} > 0.0) * ${tangent})`;
    case OP_SQRT:
      return `select(0.0, 0.5 / ${value} * ${tangent}, ${value} != 0.0)`;
    case OP_CBRT:
      return `select(0.0, ${tangent} / (3.0 * ${value} * ${value}), ${value} != 0.0)`;
    case OP_SIN:
      return `(cos(${operand}) * ${tangent})`;
    case OP_COS:
      return `(-sin(${operand}) * ${tangent})`;
    case OP_TAN:
      return `(${tangent} / (cos(${operand}) * cos(${operand})))`;
    case OP_ASIN:
      return `(${tangent} / sqrt(1.0 - ${operand} * ${operand}))`;
    case OP_ACOS:
      return `(-${tangent} / sqrt(1.0 - ${operand} * ${operand}))`;
    case OP_ATAN:
      return `(${tangent} / (1.0 + ${operand} * ${operand}))`;
    case OP_EXP:
      return `(${value} * ${tangent})`;
    case OP_LOG:
      return `(${tangent} / ${operand})`;
    case OP_LOG1P:
      return `(${tangent} / (1.0 + ${operand}))`;
    case OP_TANH:
      return `((1.0 - ${value} * ${value}) * ${tangent})`;
    case OP_DEBUG:
    case OP_ASSERT_ZERO:
    case OP_ASSERT_NONZERO:
      return tangent;
    case OP_SIGN:
    case OP_NOT:
      return "0.0";
  }
  return null;
}

function binaryTangentWgsl(
  op: number,
  left: string,
  right: string,
  leftTangent: string,
  rightTangent: string,
): string | null {
  switch (op) {
    case OP_ADD:
      return `(${leftTangent} + ${rightTangent})`;
    case OP_SUB:
      return `(${leftTangent} - ${rightTangent})`;
    case OP_MUL:
      return `(${leftTangent} * ${right} + ${left} * ${rightTangent})`;
    case OP_DIV:
      return `select(0.0, (${leftTangent} * ${right} - ${left} * ${rightTangent}) / (${right} * ${right}), ${right} != 0.0)`;
    case OP_MOD:
      return leftTangent;
    case OP_ATAN2:
      return `select(0.0, (${right} * ${leftTangent} - ${left} * ${rightTangent}) / (${left} * ${left} + ${right} * ${right}), (${left} * ${left} + ${right} * ${right}) != 0.0)`;
    case OP_MIN:
      return `select(${rightTangent}, ${leftTangent}, ${left} <= ${right})`;
    case OP_MAX:
      return `select(${rightTangent}, ${leftTangent}, ${left} >= ${right})`;
    case OP_COMPARE:
      return "0.0";
    case OP_AND:
      return `select(${rightTangent}, ${leftTangent}, ${left} == 0.0)`;
    case OP_OR:
      return `select(${leftTangent}, ${rightTangent}, ${left} == 0.0)`;
  }
  return null;
}

/** Emit one invocation of a scalar tape without assuming a data layout. */
/** Emit one invocation of a seeded, multi-direction tape JVP. */
export function emitTapeJvpWgsl(
  tape: CompiledTape,
  numDirections: number,
  variable: (slot: number) => TapeJvpVariableWgsl,
): EmittedTapeJvpWgsl {
  const lines: string[] = [];
  for (let index = 0; index < tape.opcodes.length; index++) {
    const op = tape.opcodes[index]!;
    const left = tape.argA[index]!;
    const right = tape.argB[index]!;
    const binding = op === OP_VAR ? variable(left) : null;
    let expression: string | null;
    if (op === OP_LIT) expression = formatWgslF32(tape.literals[left]!);
    else if (binding) expression = binding.value;
    else {
      expression =
        op === OP_MIN
          ? `select(${wgslValueRef(right)}, ${wgslValueRef(left)}, ${wgslValueRef(left)} <= ${wgslValueRef(right)})`
          : op === OP_MAX
            ? `select(${wgslValueRef(right)}, ${wgslValueRef(left)}, ${wgslValueRef(left)} >= ${wgslValueRef(right)})`
            : (unaryTapeOpWgsl(op, wgslValueRef(left)) ??
              binaryTapeOpWgsl(op, wgslValueRef(left), wgslValueRef(right)));
    }
    if (expression === null) {
      throw new Error(
        `GPU JVP codegen does not support tape opcode ${op} at node ${index}`,
      );
    }
    lines.push(`  let ${wgslValueRef(index)} = ${expression};`);
    for (let direction = 0; direction < numDirections; direction++) {
      const tangent =
        op === OP_LIT
          ? "0.0"
          : binding
            ? binding.tangent(direction)
            : (unaryTangentWgsl(
                op,
                wgslValueRef(left),
                wgslValueRef(index),
                tangentRef(left, direction),
              ) ??
              binaryTangentWgsl(
                op,
                wgslValueRef(left),
                wgslValueRef(right),
                tangentRef(left, direction),
                tangentRef(right, direction),
              ));
      if (tangent === null) {
        throw new Error(
          `GPU JVP codegen does not support derivative opcode ${op} at node ${index}`,
        );
      }
      lines.push(`  let ${tangentRef(index, direction)} = ${tangent};`);
    }
  }
  return {
    body: lines.join("\n"),
    roots: tape.rootIndices.map(wgslValueRef),
    tangentRoots: tape.rootIndices.map((root) =>
      Array.from({ length: numDirections }, (_, direction) =>
        tangentRef(root, direction),
      ),
    ),
  };
}

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
