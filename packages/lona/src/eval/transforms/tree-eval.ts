/**
 * `treeEval` — clone a NumNode DAG into a fresh tree with `.evalsTo` metadata
 * attached to every node. Unlike `genericEval` (which produces a single result
 * value), this transformation produces a DAG-shaped copy decorated with its
 * intermediate evaluation values.
 *
 * Used by the debug visualization pipeline (see `debug-utils.ts`).
 */
import { visitFromLeaves } from "../../dag/traversal";
import {
  NumNode,
  LiteralNum,
  Variable,
  Derivative,
  UnaryOp,
  BinaryOp,
  ForeignFn,
  DebugNode,
  SelectOp,
  childrenOfNumNode,
  KIND_LIT,
  KIND_VAR,
  KIND_DERIVATIVE,
  KIND_DEBUG,
  KIND_FOREIGN,
  KIND_SELECT,
  isBinaryKind,
  isUnaryKind,
} from "../../core/tree";
import { NumEvalKernel } from "../../types";
import { JSEvalKernel } from "./js-eval";

export function treeEval<T = number>(
  root: NumNode,
  kernel: NumEvalKernel<T> = new JSEvalKernel() as unknown as NumEvalKernel<T>,
): NumNode & { evalsTo: number } {
  const outNodes = new Map<NumNode, NumNode & { evalsTo: number }>();
  const evaledNodes = new Map<NumNode, T>();

  visitFromLeaves(root, childrenOfNumNode, (node) => {
    // `treeEval` intentionally allocates fresh nodes with `new` so each
    // call gets its own mutable `evalsTo` metadata attached — hash-cons
    // wrappers would share that metadata across evaluations. Dispatch
    // via `kind` for speed even though we can't avoid the allocations.
    let evaled;
    let extended: NumNode & { evalsTo: number };
    const kind = node.kind;

    if (kind === KIND_LIT) {
      const n = node as LiteralNum;
      evaled = kernel.literal(n.value, n);
      extended = Object.assign(new LiteralNum(n.value), {
        evalsTo: kernel.value(evaled),
      });
    } else if (kind === KIND_VAR) {
      const n = node as Variable;
      evaled = kernel.variable(n.name, n);
      extended = Object.assign(new Variable(n.name), {
        evalsTo: kernel.value(evaled),
      });
    } else if (kind === KIND_DERIVATIVE) {
      const n = node as Derivative;
      evaled = kernel.derivative(n.variable.name, n);
      extended = Object.assign(new Derivative(n.variable), {
        evalsTo: kernel.value(evaled),
      });
    } else if (isUnaryKind(kind)) {
      const n = node as UnaryOp;
      const operand = evaledNodes.get(n.original)!;
      evaled = kernel.unaryOp(n.operation, operand, n);

      if (kind === KIND_DEBUG) {
        extended = Object.assign(
          new DebugNode(outNodes.get(n.original)!, (n as DebugNode).debug),
          { evalsTo: kernel.value(evaled) },
        );
      } else {
        extended = Object.assign(
          new UnaryOp(n.operation, outNodes.get(n.original)!),
          { evalsTo: kernel.value(evaled) },
        );
      }
    } else if (isBinaryKind(kind)) {
      const n = node as BinaryOp;
      const left = evaledNodes.get(n.left)!;
      const right = evaledNodes.get(n.right)!;

      evaled = kernel.binaryOp(n.operation, left, right, n);
      extended = Object.assign(
        new BinaryOp(
          n.operation,
          outNodes.get(n.left)!,
          outNodes.get(n.right)!,
        ),
        { evalsTo: kernel.value(evaled) },
      );
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
      extended = Object.assign(
        new SelectOp(
          outNodes.get(n.condition)!,
          outNodes.get(n.ifNonZero)!,
          outNodes.get(n.ifZero)!,
        ),
        { evalsTo: kernel.value(evaled) },
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
      const newInputs = n.inputs.map((input) => outNodes.get(input)!);
      extended = Object.assign(new ForeignFn(newInputs, n.evalFn, n.diffFn), {
        evalsTo: kernel.value(evaled),
      });
    } else {
      throw new Error(
        `Unknown node kind: ${kind} (${node?.operation} ${node.constructor.name})`,
      );
    }

    outNodes.set(node, extended);
    evaledNodes.set(node, evaled);
  });

  return outNodes.get(root)!;
}
