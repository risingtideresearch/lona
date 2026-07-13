/**
 * Single-root one-shot value evaluation. Compile-and-call wrappers around the
 * routine system; prefer `compileValueRoutine` from `../routines` for
 * compile-once-call-many use cases.
 */
import { JSEvalKernel } from "./transforms/js-eval";
import { NumNode, type VarName } from "../core/tree";
import { BinaryOperation, UnaryOperation } from "../types";
import { genericEval } from "./transforms/generic-eval";
import { compileValueRoutine, type ValueRoutine } from "./routines";

class NaNReportingKernel extends JSEvalKernel {
  private reportNaN(message: string, _node?: NumNode) {
    throw new Error(message);
  }

  unaryOp(operation: UnaryOperation, operand: number, node: NumNode) {
    const result = super.unaryOp(operation, operand, node);
    if (Number.isNaN(result)) {
      this.reportNaN(`NaN in unary op: ${operation}(${operand})`, node);
    }
    return result;
  }

  binaryOp(
    operation: BinaryOperation,
    lhs: number,
    rhs: number,
    node: NumNode,
  ) {
    const result = super.binaryOp(operation, lhs, rhs, node);
    if (Number.isNaN(result)) {
      this.reportNaN(`NaN in binary op: ${lhs} ${operation} ${rhs}`, node);
    }
    return result;
  }
}

export const naiveEval = (
  node: NumNode,
  variables: Map<VarName, number> | Map<string, number>,
  derivatives?: Map<VarName, number>,
): number => {
  const kernel = new JSEvalKernel(variables, false, derivatives);
  return genericEval(node, kernel);
};

const routineCache = new WeakMap<NumNode, ValueRoutine | null>();

export function simpleEval(
  node: NumNode,
  variables: Map<VarName, number> | Map<string, number> = new Map(),
  logDebug = false,
  derivatives?: Map<VarName, number>,
): number {
  if (!logDebug) {
    let routine = routineCache.get(node);
    if (routine === undefined) {
      const r = compileValueRoutine([node]);
      routine = r?.shape === "value" ? (r as ValueRoutine) : null;
      routineCache.set(node, routine);
    }
    if (routine !== null) {
      return routine.eval(variables, derivatives);
    }
  }

  const kernel = new NaNReportingKernel(variables, logDebug, derivatives);
  return genericEval(node, kernel);
}
