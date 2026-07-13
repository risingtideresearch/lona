export function stackTraversal<Node>(
  root: Node,
  childrenFcn: (node: Node) => Node[],
  {
    preCallBack,
    postCallBack,
  }: {
    preCallBack?: (node: Node) => void;
    postCallBack?: (node: Node) => void;
  } = {},
): void {
  // Use a standard set to track visited nodes for pre-order processing
  const visitedNodes = new Set<Node>();

  // Use a stack with a flag to distinguish between pre and post processing
  type StackEntry = { node: Node; postProcess: boolean };
  const stack: StackEntry[] = [{ node: root, postProcess: false }];

  while (stack.length > 0) {
    const entry = stack.pop()!;

    if (entry.postProcess) {
      // This node is on the stack for post-order processing
      if (postCallBack) {
        postCallBack(entry.node);
      }
      continue;
    }

    if (visitedNodes.has(entry.node)) continue;
    visitedNodes.add(entry.node);

    // Pre-order processing
    if (preCallBack) {
      preCallBack(entry.node);
    }

    // If post-order callback exists, push node back for later processing
    if (postCallBack) {
      stack.push({ node: entry.node, postProcess: true });
    }

    // Push children onto stack in reverse order to maintain original traversal order
    const children = childrenFcn(entry.node);
    for (let i = children.length - 1; i >= 0; i--) {
      if (!visitedNodes.has(children[i])) {
        stack.push({ node: children[i], postProcess: false });
      }
    }
  }
}

export function visitFromLeaves<Node>(
  root: Node,
  childrenFcn: (n: Node) => Node[],
  visitFcn: (n: Node) => void,
): void {
  return stackTraversal(root, childrenFcn, {
    postCallBack: visitFcn,
  });
}

export function visitFromRoot<Node>(
  root: Node,
  childrenFcn: (n: Node) => Node[],
  visitFcn: (n: Node) => void,
): void {
  return stackTraversal(root, childrenFcn, {
    preCallBack: visitFcn,
  });
}

export function mapDAG<Node, T>(
  root: Node,
  childrenFcn: (n: Node) => Node[],
  mapFcn: (n: Node, childrenValues: T[]) => T,
): Map<Node, T> {
  const valueMap = new Map<Node, T>();

  stackTraversal(root, childrenFcn, {
    postCallBack: (node) => {
      const children = childrenFcn(node);
      const childrenValues = children.map((c) => valueMap.get(c)!);
      valueMap.set(node, mapFcn(node, childrenValues));
    },
  });

  return valueMap;
}
