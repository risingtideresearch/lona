import type { VarName } from "../core/tree";

export type NumericInput<TNum> = TNum | number;

export interface NumericApi<TNum> {
  add(other: NumericInput<TNum>): TNum;
  sub(other: NumericInput<TNum>): TNum;
  mul(other: NumericInput<TNum>): TNum;
  div(other: NumericInput<TNum>): TNum;
  mod(other: TNum): TNum;
  powi(power: number): TNum;

  sqrt(): TNum;
  safeSqrt(): TNum;
  cbrt(): TNum;
  neg(): TNum;
  inv(): TNum;
  sign(): TNum;
  abs(): TNum;
  smoothabs(): TNum;
  log1p(): TNum;
  softplus(): TNum;
  softminus(): TNum;
  cos(): TNum;
  acos(): TNum;
  sin(): TNum;
  asin(): TNum;
  tan(): TNum;
  atan(): TNum;
  exp(): TNum;
  tanh(): TNum;
  log(): TNum;
  square(): TNum;

  compare(other: NumericInput<TNum>): TNum;
  and(other: NumericInput<TNum>): TNum;
  or(other: NumericInput<TNum>): TNum;
  not(): TNum;
  max(other: NumericInput<TNum>): TNum;
  min(other: NumericInput<TNum>): TNum;
  equals(other: NumericInput<TNum>): TNum;
  lessThan(other: NumericInput<TNum>): TNum;
  lessThanOrEqual(other: NumericInput<TNum>): TNum;
  greaterThan(other: NumericInput<TNum>): TNum;
  greaterThanOrEqual(other: NumericInput<TNum>): TNum;
}

export type Branch<TNum> = () => NumericInput<TNum>;
export type Condition<TNum> = NumericInput<TNum> | (() => NumericInput<TNum>);

export interface WhenApi<TNum> {
  then(ifNonZero: Branch<TNum>): WhenChainApi<TNum>;
}

export interface WhenChainApi<TNum> {
  elseIf(condition: Condition<TNum>): WhenApi<TNum>;
  else(ifZero: Branch<TNum>): TNum;
}

export interface CasesApi<TNum> {
  case(value: NumericInput<TNum>, branch: Branch<TNum>): CasesApi<TNum>;
  default(fallback: Branch<TNum>): TNum;
}

export interface RoutineBuildContext<TNum> {
  variable(name: VarName): TNum;
  asNum(value: NumericInput<TNum>): TNum;
  when(condition: Condition<TNum>): WhenApi<TNum>;
}

export type RoutineBuildResult<TNum> =
  NumericInput<TNum> | readonly NumericInput<TNum>[];

export type RoutineBuilder<TNum, TResult extends RoutineBuildResult<TNum>> = (
  ctx: RoutineBuildContext<TNum>,
) => TResult;

export type RoutineDiffVars = readonly VarName[] | "all";
