import { visitFromLeaves } from "../../dag/traversal";
import {
  NumNode,
  childrenOfNumNode,
  LiteralNum,
  Variable,
  Derivative,
  UnaryOp,
  BinaryOp,
  ForeignFn,
  SelectOp,
  KIND_LIT,
  KIND_VAR,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_SELECT,
  isUnaryKind,
  isBinaryKind,
} from "../../core/tree";
import { NumEvalKernel } from "../../types";

export function genericEval<T>(root: NumNode, kernel: NumEvalKernel<T>): T {
  const evaledNodes = new Map<NumNode, T>();

  visitFromLeaves(root, childrenOfNumNode, (node) => {
    let evaled;
    const kind = node.kind;

    if (kind === KIND_LIT) {
      const n = node as LiteralNum;
      evaled = kernel.literal(n.value, n);
    } else if (kind === KIND_VAR) {
      const n = node as Variable;
      evaled = kernel.variable(n.name, n);
    } else if (kind === KIND_DERIVATIVE) {
      const n = node as Derivative;
      evaled = kernel.derivative(n.variable.name, n);
    } else if (isUnaryKind(kind)) {
      const n = node as UnaryOp;
      const operand = evaledNodes.get(n.original)!;
      evaled = kernel.unaryOp(n.operation, operand, n);
    } else if (isBinaryKind(kind)) {
      const n = node as BinaryOp;
      const left = evaledNodes.get(n.left)!;
      const right = evaledNodes.get(n.right)!;
      evaled = kernel.binaryOp(n.operation, left, right, n);
    } else if (kind === KIND_SELECT) {
      if (!kernel.select) {
        throw new Error("Kernel does not support Select nodes");
      }
      const n = node as SelectOp;
      evaled = kernel.select(
        evaledNodes.get(n.condition)!,
        evaledNodes.get(n.ifNonZero)!,
        evaledNodes.get(n.ifZero)!,
        n,
      );
    } else if (kind === KIND_FOREIGN) {
      const n = node as ForeignFn;
      const inputs = n.inputs.map((input) => evaledNodes.get(input)!);
      if (kernel.foreignFn) {
        evaled = kernel.foreignFn(inputs, n);
      } else {
        const inputValues = inputs.map((v) => kernel.value(v));
        evaled = kernel.literal(n.evalFn(inputValues), n);
      }
    } else {
      throw new Error(
        `Unknown node kind: ${kind} (${node?.operation} ${node.constructor.name})`,
      );
    }
    evaledNodes.set(node, evaled);
  });

  return evaledNodes.get(root)!;
}

export function genericMultiEval<T>(
  roots: NumNode[],
  kernel: NumEvalKernel<T>,
): T[] {
  const evaledNodes = new Map<NumNode, T>();

  // Visit all roots. Shared sub-DAGs are deduplicated by the evaledNodes map:
  // visitFromLeaves may re-traverse shared nodes, but the callback only
  // evaluates each node once (first root wins).
  for (const root of roots) {
    visitFromLeaves(root, childrenOfNumNode, (node) => {
      if (evaledNodes.has(node)) return;

      let evaled;
      const kind = node.kind;

      if (kind === KIND_LIT) {
        const n = node as LiteralNum;
        evaled = kernel.literal(n.value, n);
      } else if (kind === KIND_VAR) {
        const n = node as Variable;
        evaled = kernel.variable(n.name, n);
      } else if (kind === KIND_DERIVATIVE) {
        const n = node as Derivative;
        evaled = kernel.derivative(n.variable.name, n);
      } else if (isUnaryKind(kind)) {
        const n = node as UnaryOp;
        const operand = evaledNodes.get(n.original)!;
        evaled = kernel.unaryOp(n.operation, operand, n);
      } else if (isBinaryKind(kind)) {
        const n = node as BinaryOp;
        const left = evaledNodes.get(n.left)!;
        const right = evaledNodes.get(n.right)!;
        evaled = kernel.binaryOp(n.operation, left, right, n);
      } else if (kind === KIND_SELECT) {
        if (!kernel.select) {
          throw new Error("Kernel does not support Select nodes");
        }
        const n = node as SelectOp;
        evaled = kernel.select(
          evaledNodes.get(n.condition)!,
          evaledNodes.get(n.ifNonZero)!,
          evaledNodes.get(n.ifZero)!,
          n,
        );
      } else if (kind === KIND_FOREIGN) {
        const n = node as ForeignFn;
        const inputs = n.inputs.map((input) => evaledNodes.get(input)!);
        if (kernel.foreignFn) {
          evaled = kernel.foreignFn(inputs, n);
        } else {
          const inputValues = inputs.map((v) => kernel.value(v));
          evaled = kernel.literal(n.evalFn(inputValues), n);
        }
      } else {
        throw new Error(
          `Unknown node kind: ${kind} (${node?.operation} ${node.constructor.name})`,
        );
      }
      evaledNodes.set(node, evaled);
    });
  }

  return roots.map((r) => evaledNodes.get(r)!);
}
