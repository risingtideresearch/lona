import {
  BinaryOp,
  Derivative,
  ForeignFn,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_LIT,
  KIND_SELECT,
  KIND_VAR,
  SelectOp,
  Variable,
  UnaryOp,
  isBinaryKind,
  isUnaryKind,
  type NumNode,
} from "lona/internal";

export function validateKernelRoots(
  params: readonly Variable[],
  roots: readonly NumNode[],
  stageLabel: string,
): void {
  if (roots.length === 0)
    throw new Error(`${stageLabel} produced no Num roots`);

  const allowedParams = new Set<NumNode>(params);
  const visited = new Set<NumNode>();
  const stack = [...roots];

  while (stack.length > 0) {
    const node = stack.pop()!;
    if (visited.has(node)) continue;
    visited.add(node);

    const kind = node.kind;
    if (kind === KIND_LIT) continue;

    if (kind === KIND_VAR) {
      if (allowedParams.has(node)) continue;
      throw new Error(
        `${stageLabel} captured an external Num; declare it under \`using\``,
      );
    }
    if (kind === KIND_FOREIGN) {
      void (node as ForeignFn);
      throw new Error(
        `${stageLabel} contains a ForeignFn, which is unsupported`,
      );
    }
    if (kind === KIND_DERIVATIVE) {
      void (node as Derivative);
      throw new Error(
        `${stageLabel} contains a Derivative, which is unsupported`,
      );
    }

    if (kind === KIND_SELECT) {
      const select = node as SelectOp;
      stack.push(select.condition, select.ifNonZero, select.ifZero);
      continue;
    }
    if (isUnaryKind(kind)) {
      stack.push((node as UnaryOp).original);
      continue;
    }
    if (isBinaryKind(kind)) {
      const binary = node as BinaryOp;
      stack.push(binary.left, binary.right);
      continue;
    }

    throw new Error(
      `${stageLabel} contains unsupported node '${node.operation}' (kind ${kind})`,
    );
  }
}
