import { genericEval } from "../eval/transforms/generic-eval";
import { DiffEvalKernel } from "../eval/transforms/diff";
import { NumNode, type VarName } from "../core/tree";
import { partialDerivative } from "../core/tree-walks";

export function fullDerivative(node: NumNode): NumNode {
  return genericEval(node, new DiffEvalKernel());
}

export function gradient(node: NumNode, variables: VarName[]): NumNode[] {
  const diffRoot = fullDerivative(node);
  return variables.map((v) => partialDerivative(diffRoot, v));
}

export function jacobian(nodes: NumNode[], variables: VarName[]): NumNode[][] {
  return nodes.map((node) => gradient(node, variables));
}
