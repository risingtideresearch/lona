/**
 * Tree-walking utilities that build new trees through the current
 * {@link NumContext}'s cons tables.
 *
 * Lives outside `tree.ts` so that `tree.ts` stays at the bottom of the
 * dependency graph (pure node definitions, no current-context routing).
 * Putting these here also makes the routing explicit: every operation
 * that produces NumNodes goes through the cons factories in
 * `context.ts`, never around them.
 */
import { visitFromLeaves } from "../dag/traversal";
import {
  binaryNode,
  derivativeNode,
  foreignFnNode,
  litNode,
  selectNode,
  unaryNode,
  varNode,
} from "./tree-cons";
import {
  BinaryOp,
  Derivative,
  ForeignFn,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_LIT,
  KIND_VAR,
  LiteralNum,
  NumNode,
  ONE_NODE,
  SelectOp,
  UnaryOp,
  Variable,
  ZERO_NODE,
  childrenOfNumNode,
  KIND_SELECT,
  isBinaryKind,
  isUnaryKind,
  type VarName,
} from "./tree";

// Reconstruct a node through the hash-cons factories. With hash-consing this
// is effectively an identity function for any input built by the factories
// in the first place — but it still walks subtrees so that trees constructed
// with raw `new` (e.g. in tests) are normalised when they reach a transform
// that expects `===` structural equality.
const cloneNode = (node: NumNode): NumNode => {
  const kind = node.kind;
  if (kind === KIND_LIT) {
    return litNode((node as LiteralNum).value);
  }
  if (kind === KIND_VAR) {
    return varNode((node as Variable).name);
  }
  if (kind === KIND_DERIVATIVE) {
    return derivativeNode((node as Derivative).variable);
  }
  if (isUnaryKind(kind)) {
    const u = node as UnaryOp;
    return unaryNode(u.operation, cloneNode(u.original));
  }
  if (isBinaryKind(kind)) {
    const b = node as BinaryOp;
    return binaryNode(b.operation, cloneNode(b.left), cloneNode(b.right));
  }
  if (kind === KIND_FOREIGN) {
    const f = node as ForeignFn;
    return foreignFnNode(f.inputs.map(cloneNode), f.evalFn, f.diffFn);
  }
  if (kind === KIND_SELECT) {
    const s = node as SelectOp;
    return selectNode(
      cloneNode(s.condition),
      cloneNode(s.ifNonZero),
      cloneNode(s.ifZero),
    );
  }
  throw new Error(`Unknown node kind: ${kind}`);
};

export function replaceVariable(
  node: NumNode,
  variables: Map<VarName, number | NumNode>,
): NumNode;
export function replaceVariable(
  node: NumNode,
  variables: Map<string, number | NumNode>,
): NumNode;
export function replaceVariable(
  node: NumNode,
  variables: Map<VarName, number | NumNode> | Map<string, number | NumNode>,
): NumNode {
  const variablesMap = variables as Map<VarName, number | NumNode>;
  const modifiedNodes = new Map<NumNode, NumNode>();

  visitFromLeaves(node, childrenOfNumNode, (n) => {
    const kind = n.kind;
    let outNode: NumNode;

    if (kind === KIND_VAR) {
      const value = variablesMap.get((n as Variable).name);
      if (value === undefined) {
        outNode = n;
      } else if (value instanceof NumNode) {
        outNode = value;
      } else {
        outNode = litNode(value);
      }
    } else if (isUnaryKind(kind)) {
      const u = n as UnaryOp;
      const newOriginal = modifiedNodes.get(u.original)!;
      outNode =
        newOriginal === u.original ? n : unaryNode(u.operation, newOriginal);
    } else if (isBinaryKind(kind)) {
      const b = n as BinaryOp;
      const newLeft = modifiedNodes.get(b.left)!;
      const newRight = modifiedNodes.get(b.right)!;
      outNode =
        newLeft === b.left && newRight === b.right
          ? n
          : binaryNode(b.operation, newLeft, newRight);
    } else if (kind === KIND_SELECT) {
      const s = n as SelectOp;
      const newCondition = modifiedNodes.get(s.condition)!;
      const newIfNonZero = modifiedNodes.get(s.ifNonZero)!;
      const newIfZero = modifiedNodes.get(s.ifZero)!;
      outNode =
        newCondition === s.condition &&
        newIfNonZero === s.ifNonZero &&
        newIfZero === s.ifZero
          ? n
          : selectNode(newCondition, newIfNonZero, newIfZero);
    } else if (kind === KIND_FOREIGN) {
      const f = n as ForeignFn;
      const newInputs = f.inputs.map((input) => modifiedNodes.get(input)!);
      const changed = newInputs.some((ni, i) => ni !== f.inputs[i]);
      outNode = changed ? foreignFnNode(newInputs, f.evalFn, f.diffFn) : n;
    } else {
      // KIND_LIT, KIND_DERIVATIVE, or anything else: defer to cloneNode,
      // which in the hash-consed world is effectively an identity.
      outNode = cloneNode(n);
    }

    modifiedNodes.set(n, outNode);
  });

  return modifiedNodes.get(node)!;
}

export const partialDerivative = (
  node: NumNode,
  variable: VarName,
): NumNode => {
  const memo = new Map<NumNode, NumNode>();

  const go = (node: NumNode): NumNode => {
    const cached = memo.get(node);
    if (cached !== undefined) return cached;

    let result: NumNode;
    const kind = node.kind;
    if (kind === KIND_DERIVATIVE) {
      result =
        (node as Derivative).variable.name === variable ? ONE_NODE : ZERO_NODE;
    } else if (isUnaryKind(kind)) {
      const u = node as UnaryOp;
      const operand = go(u.original);
      result = unaryNode(u.operation, operand);
    } else if (isBinaryKind(kind)) {
      const b = node as BinaryOp;
      const left = go(b.left);
      const right = go(b.right);
      result = binaryNode(b.operation, left, right);
    } else if (kind === KIND_SELECT) {
      const s = node as SelectOp;
      const condition = go(s.condition);
      const ifNonZero = go(s.ifNonZero);
      const ifZero = go(s.ifZero);
      result =
        condition === s.condition &&
        ifNonZero === s.ifNonZero &&
        ifZero === s.ifZero
          ? node
          : selectNode(condition, ifNonZero, ifZero);
    } else if (kind === KIND_FOREIGN) {
      const f = node as ForeignFn;
      const newInputs = f.inputs.map((input) => go(input));
      const changed = newInputs.some((ni, i) => ni !== f.inputs[i]);
      result = changed ? foreignFnNode(newInputs, f.evalFn, f.diffFn) : node;
    } else {
      result = node;
    }

    memo.set(node, result);
    return result;
  };

  return go(node);
};
