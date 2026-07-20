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
  Call,
  Derivative,
  ForeignFn,
  KIND_CALL,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_LIT,
  KIND_PARAM,
  KIND_PROJECT,
  KIND_VAR,
  LiteralNum,
  NumNode,
  ONE_NODE,
  Project,
  SelectOp,
  UnaryOp,
  Variable,
  ZERO_NODE,
  childrenOfNumNode,
  KIND_SELECT,
  isBinaryKind,
  isUnaryKind,
  type Proc,
  type VarName,
} from "./tree";

/** Rebuild `node` through the hash-cons factories, using `child` to (recursively)
 *  produce each operand. Shared by `cloneNode`, `inlineProcs`, and proc body
 *  substitution. Rejects proc nodes — callers that may see them must handle
 *  Call/Project/Param before delegating here. */
const rebuild = (node: NumNode, child: (n: NumNode) => NumNode): NumNode => {
  const kind = node.kind;
  if (kind === KIND_LIT) return litNode((node as LiteralNum).value);
  if (kind === KIND_VAR) return varNode((node as Variable).name);
  if (kind === KIND_DERIVATIVE) {
    return derivativeNode((node as Derivative).variable);
  }
  if (isUnaryKind(kind)) {
    const u = node as UnaryOp;
    return unaryNode(u.operation, child(u.original));
  }
  if (isBinaryKind(kind)) {
    const b = node as BinaryOp;
    return binaryNode(b.operation, child(b.left), child(b.right));
  }
  if (kind === KIND_SELECT) {
    const s = node as SelectOp;
    return selectNode(child(s.condition), child(s.ifNonZero), child(s.ifZero));
  }
  if (kind === KIND_FOREIGN) {
    const f = node as ForeignFn;
    return foreignFnNode(f.inputs.map(child), f.evalFn, f.diffFn);
  }
  if (kind === KIND_CALL || kind === KIND_PROJECT || kind === KIND_PARAM) {
    throw new Error(
      "This transform does not support proc nodes (Call/Project/Param). " +
        "Inline them first with inlineProcs().",
    );
  }
  throw new Error(`Unknown node kind: ${kind}`);
};

// Reconstruct a node through the hash-cons factories. With hash-consing this
// is effectively an identity function for any input built by the factories
// in the first place — but it still walks subtrees so that trees constructed
// with raw `new` (e.g. in tests) are normalised when they reach a transform
// that expects `===` structural equality.
const cloneNode = (node: NumNode): NumNode => rebuild(node, cloneNode);

/**
 * Desugar procs: replace every `Project(Call(proc, args), k)` with `proc.body[k]`
 * rebuilt with the proc's params substituted by the (inlined) argument nodes.
 * Multi-root with a single shared memo, so subexpressions shared across roots —
 * and repeated calls — collapse rather than duplicating. This RE-EXPANDS to a
 * plain DAG: it is a correctness fallback so proc-unaware transforms
 * (simplify / partialDerivative / serialization) can run, NOT a memory path.
 */
export function inlineProcs(roots: readonly NumNode[]): NumNode[] {
  const memo = new Map<NumNode, NumNode>();

  const go = (node: NumNode): NumNode => {
    const cached = memo.get(node);
    if (cached !== undefined) return cached;

    const kind = node.kind;
    let result: NumNode;
    if (kind === KIND_PROJECT) {
      const pr = node as Project;
      const args = pr.call.args.map(go);
      result = substituteBody(pr.call.proc, pr.output, args);
    } else if (kind === KIND_CALL) {
      throw new Error("inlineProcs: a Call must be reached through a Project");
    } else if (kind === KIND_PARAM) {
      throw new Error("inlineProcs: a Param escaped its proc body");
    } else {
      result = rebuild(node, go);
    }

    memo.set(node, result);
    return result;
  };

  return roots.map(go);
}

/** Substitute a proc's params with `args` in its `output`-th body root, rebuilt
 *  through the cons factories. Bodies are validated flat (no nested procs), so
 *  `rebuild` never meets a Call/Project here. */
function substituteBody(
  proc: Proc,
  output: number,
  args: readonly NumNode[],
): NumNode {
  const sub = new Map<NumNode, NumNode>();
  proc.params.forEach((p, i) => sub.set(p, args[i]!));
  const go = (n: NumNode): NumNode => {
    const cached = sub.get(n);
    if (cached !== undefined) return cached; // params resolve here
    const r = rebuild(n, go);
    sub.set(n, r);
    return r;
  };
  return go(proc.body[output]!);
}

/**
 * Number of distinct nodes reachable from `roots`, **including each referenced
 * proc's body once**. Ordinary `childrenOfNumNode` deliberately excludes proc
 * bodies (they are shared via `Proc`, not the call graph), so a plain reachable
 * count omits them; this is the honest "how big is this graph" metric for the
 * rolled DAG.
 */
export function countReachableIncludingProcBodies(
  roots: readonly NumNode[],
): number {
  const seen = new Set<NumNode>();
  const procs = new Set<Proc>();

  const walk = (start: readonly NumNode[]): void => {
    const stack: NumNode[] = [...start];
    while (stack.length > 0) {
      const n = stack.pop()!;
      if (seen.has(n)) continue;
      seen.add(n);
      if (n.kind === KIND_CALL) procs.add((n as Call).proc);
      for (const c of childrenOfNumNode(n)) stack.push(c);
    }
  };

  walk(roots);
  for (const proc of procs) walk(proc.body);
  return seen.size;
}

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
    } else if (
      kind === KIND_CALL ||
      kind === KIND_PROJECT ||
      kind === KIND_PARAM
    ) {
      // Would otherwise be silently treated as an unchanging leaf → wrong.
      throw new Error(
        "partialDerivative: proc nodes (Call/Project/Param) are not " +
          "differentiable in place; inline them first with inlineProcs().",
      );
    } else {
      result = node;
    }

    memo.set(node, result);
    return result;
  };

  return go(node);
};
