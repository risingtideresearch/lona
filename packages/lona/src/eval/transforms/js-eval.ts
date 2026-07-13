import type { DebugNode, NumNode, VarName } from "../../core/tree";
import type { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";

export class JSEvalKernel implements NumEvalKernel<number> {
  public readonly variablesValues: Map<VarName, number>;
  public readonly derivativeValues: Map<VarName, number>;
  public logDebug = false;

  constructor(
    variablesValues: Map<VarName, number> | Map<string, number> = new Map(),
    logDebug = false,
    derivativeValues: Map<VarName, number> = new Map(),
  ) {
    this.variablesValues = new Map(variablesValues as Map<VarName, number>);
    this.derivativeValues = new Map(derivativeValues);
    this.logDebug = logDebug;
  }
  value(value: number) {
    return value;
  }
  literal(value: number) {
    return value;
  }
  variable(name: VarName) {
    if (!this.variablesValues.has(name)) {
      throw new Error(`Unknown variable: ${String(name)}`);
    }
    return this.variablesValues.get(name)!;
  }
  derivative(name: VarName) {
    return this.derivativeValues.get(name) ?? 0;
  }
  unaryOp(operation: UnaryOperation, operand: number, node: NumNode) {
    if (operation === "DEBUG") {
      if (this.logDebug) {
        console.log((node as DebugNode).debug, this.value(operand));
      }
      return operand;
    }
    return evaluateUnaryOp(operation, operand);
  }

  binaryOp(
    operation: BinaryOperation,
    left: number,
    right: number,
    _node: NumNode,
  ) {
    return evaluateBinaryOp(operation, left, right);
  }

  select(condition: number, ifNonZero: number, ifZero: number): number {
    return condition !== 0 ? ifNonZero : ifZero;
  }
}

export function evaluateUnaryOp(
  operation: UnaryOperation,
  operand: number,
): number {
  switch (operation) {
    case "SQRT":
      return Math.sqrt(operand);
    case "CBRT":
      return Math.cbrt(operand);
    case "COS":
      return Math.cos(operand);
    case "ACOS":
      return Math.acos(operand);
    case "ASIN":
      return Math.asin(operand);
    case "TAN":
      return Math.tan(operand);
    case "ATAN":
      return Math.atan(operand);
    case "LOG":
      return Math.log(operand);
    case "EXP":
      return Math.exp(operand);
    case "ABS":
      return Math.abs(operand);
    case "NEG":
      return -operand;
    case "SIN":
      return Math.sin(operand);
    case "SIGN":
      return Math.sign(operand);
    case "NOT":
      return operand ? 0 : 1;
    case "TANH":
      return Math.tanh(operand);
    case "LOG1P":
      return Math.log1p(operand);
    case "DEBUG":
      return operand;
  }
  throw new Error(`Unknown unary operation: ${operation}`);
}

export function evaluateBinaryOp(
  operation: BinaryOperation,
  left: number,
  right: number,
): number {
  switch (operation) {
    case "ADD":
      return left + right;
    case "SUB":
      return left - right;
    case "MUL":
      return left * right;
    case "DIV":
      return right ? left / right : 1e50;
    case "MOD":
      return left % right;
    case "ATAN2":
      return Math.atan2(left, right);
    case "MIN":
      return Math.min(left, right);
    case "MAX":
      return Math.max(left, right);
    case "COMPARE":
      return Math.sign(left - right);
    case "AND":
      return left === 0 ? left : right;
    case "OR":
      return left === 0 ? right : left;
  }
  throw new Error(`Unknown binary operation: ${operation}`);
}
