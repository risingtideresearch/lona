import { Decimal } from "decimal.js";

import type { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";
import { DebugNode, NumNode, type VarName } from "../../core/tree";

Decimal.set({ precision: 40 });

export class JSPrecisionEvalKernel implements NumEvalKernel<Decimal> {
  public readonly variablesValues: Map<VarName, number>;
  public logDebug = false;

  constructor(
    variablesValues: Map<VarName, number> | Map<string, number> = new Map(),
    logDebug = false,
  ) {
    this.variablesValues = new Map(variablesValues as Map<VarName, number>);
    this.logDebug = logDebug;
  }

  value(value: Decimal) {
    return value.toNumber();
  }
  literal(value: number) {
    return new Decimal(value);
  }
  variable(name: VarName) {
    if (!this.variablesValues.has(name)) {
      throw new Error(`Unknown variable: ${String(name)}`);
    }
    return new Decimal(this.variablesValues.get(name)!);
  }
  derivative(_name: VarName) {
    return new Decimal(0);
  }
  unaryOp(operation: UnaryOperation, operand: Decimal, node: NumNode): Decimal {
    if (operation === "SQRT") {
      return Decimal.sqrt(operand);
    }
    if (operation === "CBRT") {
      return Decimal.cbrt(operand);
    }
    if (operation === "COS") {
      return Decimal.cos(operand);
    }
    if (operation === "ACOS") {
      return Decimal.acos(operand);
    }
    if (operation === "ASIN") {
      return Decimal.asin(operand);
    }
    if (operation === "TAN") {
      return Decimal.tan(operand);
    }
    if (operation === "ATAN") {
      return Decimal.atan(operand);
    }
    if (operation === "LOG") {
      return Decimal.ln(operand);
    }
    if (operation === "EXP") {
      return Decimal.exp(operand);
    }
    if (operation === "ABS") {
      return Decimal.abs(operand);
    }
    if (operation === "NEG") {
      return operand.neg();
    }
    if (operation === "SIN") {
      return Decimal.sin(operand);
    }
    if (operation === "SIGN") {
      return new Decimal(Decimal.sign(operand));
    }
    if (operation === "NOT") {
      return operand.isZero() ? Decimal(1) : Decimal(0);
    }
    if (operation === "TANH") {
      return Decimal.tanh(operand);
    }
    if (operation === "LOG1P") {
      return Decimal.ln(operand.add(1));
    }
    if (operation === "DEBUG") {
      if (this.logDebug) {
        console.log((node as DebugNode).debug, this.value(operand));
      }
      return operand;
    }
    throw new Error(`Unknown unary operation: ${operation}`);
  }

  select(condition: Decimal, ifNonZero: Decimal, ifZero: Decimal): Decimal {
    return condition.isZero() ? ifZero : ifNonZero;
  }

  binaryOp(operation: BinaryOperation, left: Decimal, right: Decimal): Decimal {
    if (operation === "ADD") {
      return left.add(right);
    }
    if (operation === "SUB") {
      return left.sub(right);
    }
    if (operation === "MUL") {
      return left.mul(right);
    }
    if (operation === "DIV") {
      return !right.equals(0) ? left.div(right) : Decimal(1e50);
    }
    if (operation === "MOD") {
      return left.mod(right);
    }
    if (operation === "ATAN2") {
      return Decimal.atan2(left, right);
    }
    if (operation === "MIN") {
      return Decimal.min(left, right);
    }
    if (operation === "MAX") {
      return Decimal.max(left, right);
    }
    if (operation === "COMPARE") {
      return new Decimal(Decimal.sign(left.sub(right)));
    }
    if (operation === "AND") {
      return left.isZero() ? left : right;
    }
    if (operation === "OR") {
      return left.isZero() ? right : left;
    }
    throw new Error(`Unknown binary operation: ${operation}`);
  }
}
