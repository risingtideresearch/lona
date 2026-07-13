import {
  type ForeignFn,
  type NumNode,
  type SelectOp,
  type VarName,
} from "./core/tree";

export type UnaryOperation =
  | "SQRT"
  | "CBRT"
  | "COS"
  | "ACOS"
  | "ASIN"
  | "TAN"
  | "ATAN"
  | "LOG"
  | "EXP"
  | "ABS"
  | "NEG"
  | "SIN"
  | "SIGN"
  | "NOT"
  | "TANH"
  | "LOG1P"
  | "DEBUG";

export type BinaryOperation =
  | "ADD"
  | "SUB"
  | "MUL"
  | "DIV"
  | "MOD"
  | "ATAN2"
  | "MIN"
  | "MAX"
  | "COMPARE"
  | "AND"
  | "OR";

export interface NumEvalKernel<T> {
  unaryOp(op: UnaryOperation, arg: T, node: NumNode): T;
  binaryOp(op: BinaryOperation, lhs: T, rhs: T, node: NumNode): T;
  variable(name: VarName, node: NumNode): T;
  derivative(name: VarName, node: NumNode): T;
  literal(value: number, node: NumNode): T;
  value(value: T): number;
  select?(condition: T, ifNonZero: T, ifZero: T, node: SelectOp): T;
  foreignFn?(inputs: T[], node: ForeignFn): T;
}
