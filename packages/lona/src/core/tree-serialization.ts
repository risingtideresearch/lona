import { visitFromLeaves } from "../dag/traversal";
import type { UnaryOperation, BinaryOperation } from "../types";
import {
  NumNode,
  LiteralNum,
  Variable,
  UnaryOp,
  DebugNode,
  BinaryOp,
  Call,
  Derivative,
  Param,
  Project,
  SelectOp,
  childrenOfNumNode,
  KIND_LIT,
  KIND_VAR,
  KIND_DEBUG,
  KIND_CALL,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_PARAM,
  KIND_PROJECT,
  KIND_SELECT,
  isBinaryKind,
  isUnaryKind,
  type Proc,
} from "./tree";
import {
  binaryNode,
  callNode,
  debugNode,
  derivativeNode,
  litNode,
  newProcTag,
  paramNode,
  projectNode,
  selectNode,
  unaryNode,
  varNode,
} from "./tree-cons";
import { validateProcBody } from "./proc-validate";

// --- Serialized format types ---

type SerializedLiteral = { type: "LIT"; value: number };
type SerializedVariable = { type: "VAR"; name: string };
type SerializedUnaryOp = { type: "UNA"; op: UnaryOperation; input: number };
type SerializedDebugNode = {
  type: "DBG";
  input: number;
  debug: string;
};
type SerializedBinaryOp = {
  type: "BIN";
  op: BinaryOperation;
  left: number;
  right: number;
};
type SerializedDerivative = { type: "DRV"; variable: number };
type SerializedSelect = {
  type: "SEL";
  condition: number;
  ifNonZero: number;
  ifZero: number;
};
// Proc nodes. `PAR` appears only inside a proc body; `CALL`/`PRJ` only in the
// outer graph. `CALL.proc` indexes `SerializedNumDAG.procs`; `CALL.args` and
// `PRJ.call` index the node list they live in.
type SerializedParam = { type: "PAR"; index: number };
type SerializedCall = { type: "CALL"; proc: number; args: number[] };
type SerializedProject = { type: "PRJ"; call: number; output: number };

type SerializedNode =
  | SerializedLiteral
  | SerializedVariable
  | SerializedUnaryOp
  | SerializedDebugNode
  | SerializedBinaryOp
  | SerializedDerivative
  | SerializedSelect
  | SerializedParam
  | SerializedCall
  | SerializedProject;

/** A proc definition: a flat body (its own topo-ordered node list, with `PAR`
 *  leaves) and the body-node index of each output. */
export type SerializedProc = {
  arity: number;
  nodes: SerializedNode[];
  outputs: number[];
};

export type SerializedNumDAG = {
  nodes: SerializedNode[];
  root: number;
  /** Proc definitions referenced by `CALL` nodes. Absent if the DAG uses none. */
  procs?: SerializedProc[];
};

// --- Serialize ---

/**
 * Serializes a NumNode DAG into a JSON-friendly structure.
 * Nodes are stored in topological order (leaves first) with references by index.
 * DAG sharing is preserved: identical object references get the same index.
 *
 * Throws if the DAG contains ForeignFn nodes (which hold closures).
 * Symbol-valued variable names are converted to their description string.
 */
export function serializeNumDAG(root: NumNode): SerializedNumDAG {
  const procToId = new Map<Proc, number>();
  const procs: SerializedProc[] = [];

  // Register a proc (idempotent), serializing its flat body into `procs`.
  const registerProc = (proc: Proc): number => {
    const existing = procToId.get(proc);
    if (existing !== undefined) return existing;
    const id = procs.length;
    procToId.set(proc, id);
    // reserve the slot before serializing the body so the index is stable.
    procs.push({ arity: 0, nodes: [], outputs: [] });
    procs[id] = serializeBody(proc, registerProc);
    return id;
  };

  const { nodes, id } = serializeGraph(root, registerProc);
  return {
    nodes,
    root: id,
    ...(procs.length > 0 ? { procs } : {}),
  };
}

/** Serialize one node given a resolver for its operands' ids and a proc
 *  registrar (used only for `CALL`). Shared by the outer graph and proc bodies. */
function serializeOne(
  node: NumNode,
  idOf: (child: NumNode) => number,
  registerProc: (proc: Proc) => number,
): SerializedNode {
  const kind = node.kind;
  if (kind === KIND_DEBUG) {
    const n = node as DebugNode;
    return { type: "DBG", input: idOf(n.original), debug: n.debug };
  }
  if (isUnaryKind(kind)) {
    const n = node as UnaryOp;
    return { type: "UNA", op: n.operation, input: idOf(n.original) };
  }
  if (isBinaryKind(kind)) {
    const n = node as BinaryOp;
    return {
      type: "BIN",
      op: n.operation,
      left: idOf(n.left),
      right: idOf(n.right),
    };
  }
  if (kind === KIND_LIT) {
    return { type: "LIT", value: (node as LiteralNum).value };
  }
  if (kind === KIND_VAR) {
    const name = (node as Variable).name;
    return {
      type: "VAR",
      name: typeof name === "symbol" ? (name.description ?? "") : name,
    };
  }
  if (kind === KIND_DERIVATIVE) {
    return { type: "DRV", variable: idOf((node as Derivative).variable) };
  }
  if (kind === KIND_SELECT) {
    const n = node as SelectOp;
    return {
      type: "SEL",
      condition: idOf(n.condition),
      ifNonZero: idOf(n.ifNonZero),
      ifZero: idOf(n.ifZero),
    };
  }
  if (kind === KIND_PARAM) {
    return { type: "PAR", index: (node as Param).index };
  }
  if (kind === KIND_CALL) {
    const n = node as Call;
    return {
      type: "CALL",
      proc: registerProc(n.proc),
      args: n.args.map(idOf),
    };
  }
  if (kind === KIND_PROJECT) {
    const n = node as Project;
    return { type: "PRJ", call: idOf(n.call), output: n.output };
  }
  if (kind === KIND_FOREIGN) {
    throw new Error("Cannot serialize ForeignFn nodes (they contain closures)");
  }
  throw new Error(`Unknown NumNode kind: ${kind} (${node.operation})`);
}

/** Serialize a reachable graph (outer DAG or one body root set) into a topo-
 *  ordered node list, returning the list and the root's id. */
function serializeGraph(
  root: NumNode,
  registerProc: (proc: Proc) => number,
): { nodes: SerializedNode[]; id: number } {
  const nodeToId = new Map<NumNode, number>();
  const nodes: SerializedNode[] = [];
  visitFromLeaves(root, childrenOfNumNode, (node) => {
    if (nodeToId.has(node)) return;
    const id = nodes.length;
    nodeToId.set(node, id);
    nodes.push(serializeOne(node, (c) => nodeToId.get(c)!, registerProc));
  });
  return { nodes, id: nodeToId.get(root)! };
}

/** Serialize a proc's flat body: all output roots share one node list; `PAR`
 *  leaves stand in for parameters. */
function serializeBody(
  proc: Proc,
  registerProc: (proc: Proc) => number,
): SerializedProc {
  const nodeToId = new Map<NumNode, number>();
  const nodes: SerializedNode[] = [];
  for (const outRoot of proc.body) {
    visitFromLeaves(outRoot, childrenOfNumNode, (node) => {
      if (nodeToId.has(node)) return;
      nodeToId.set(node, nodes.length);
      nodes.push(serializeOne(node, (c) => nodeToId.get(c)!, registerProc));
    });
  }
  return {
    arity: proc.arity,
    nodes,
    outputs: proc.body.map((r) => nodeToId.get(r)!),
  };
}

// --- Deserialize ---

/**
 * Reconstructs a NumNode DAG from its serialized form.
 * Returns the root NumNode.
 */
export function deserializeNumDAG(data: SerializedNumDAG): NumNode {
  const procs = (data.procs ?? []).map(deserializeProc);
  const built = buildNodes(data.nodes, procs);
  return built[data.root]!;
}

/** Rebuild a serialized node list (outer graph or a proc body) in order. When
 *  `params` is provided (a proc body), `PAR` resolves to `params[index]`. */
function buildNodes(
  serialized: SerializedNode[],
  procs: Proc[],
  params?: readonly Param[],
): NumNode[] {
  const built: NumNode[] = [];
  for (const entry of serialized) {
    switch (entry.type) {
      case "LIT":
        built.push(litNode(entry.value));
        break;
      case "VAR":
        built.push(varNode(entry.name));
        break;
      case "UNA":
        built.push(unaryNode(entry.op, built[entry.input]!));
        break;
      case "DBG":
        built.push(debugNode(built[entry.input]!, entry.debug));
        break;
      case "BIN":
        built.push(
          binaryNode(entry.op, built[entry.left]!, built[entry.right]!),
        );
        break;
      case "DRV":
        built.push(derivativeNode(built[entry.variable]! as Variable));
        break;
      case "SEL":
        built.push(
          selectNode(
            built[entry.condition]!,
            built[entry.ifNonZero]!,
            built[entry.ifZero]!,
          ),
        );
        break;
      case "PAR": {
        const p = params?.[entry.index];
        if (p === undefined)
          throw new Error(
            `deserialize: PAR node (index ${entry.index}) outside a proc body`,
          );
        built.push(p);
        break;
      }
      case "CALL": {
        const proc = procs[entry.proc];
        if (proc === undefined)
          throw new Error(
            `deserialize: CALL references unknown proc ${entry.proc}`,
          );
        built.push(
          callNode(
            proc,
            entry.args.map((i) => built[i]!),
          ),
        );
        break;
      }
      case "PRJ":
        built.push(projectNode(built[entry.call] as Call, entry.output));
        break;
      default:
        throw new Error(
          `Unknown serialized node type: ${(entry as SerializedNode & { type: string }).type}`,
        );
    }
  }
  return built;
}

/** Rebuild a proc definition: fresh params, flat body via `buildNodes`,
 *  re-validated (so a tampered payload is rejected exactly like `defineProc`). */
function deserializeProc(sp: SerializedProc): Proc {
  const tag = newProcTag();
  const params = Object.freeze(
    Array.from({ length: sp.arity }, (_, i) => paramNode(tag, i)),
  );
  // A body has no nested procs (validated), so it needs no proc table.
  const built = buildNodes(sp.nodes, [], params);
  const body = Object.freeze(sp.outputs.map((id) => built[id]!));
  validateProcBody(params, body);
  return { tag, arity: sp.arity, params, body };
}
