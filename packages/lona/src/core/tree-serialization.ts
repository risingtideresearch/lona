import { visitFromLeaves } from "../dag/traversal";
import type { UnaryOperation, BinaryOperation } from "../types";
import {
  NumNode,
  LiteralNum,
  Variable,
  UnaryOp,
  DebugNode,
  BinaryOp,
  Derivative,
  SelectOp,
  childrenOfNumNode,
  KIND_LIT,
  KIND_VAR,
  KIND_DEBUG,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_SELECT,
  isBinaryKind,
  isUnaryKind,
} from "./tree";
import {
  binaryNode,
  debugNode,
  derivativeNode,
  litNode,
  selectNode,
  unaryNode,
  varNode,
} from "./tree-cons";

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

type SerializedNode =
  | SerializedLiteral
  | SerializedVariable
  | SerializedUnaryOp
  | SerializedDebugNode
  | SerializedBinaryOp
  | SerializedDerivative
  | SerializedSelect;

export type SerializedNumDAG = {
  nodes: SerializedNode[];
  root: number;
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
  const nodeToId = new Map<NumNode, number>();
  const nodes: SerializedNode[] = [];

  visitFromLeaves(root, childrenOfNumNode, (node) => {
    if (nodeToId.has(node)) return;

    const id = nodes.length;
    nodeToId.set(node, id);
    const kind = node.kind;

    if (kind === KIND_DEBUG) {
      const n = node as DebugNode;
      nodes.push({
        type: "DBG",
        input: nodeToId.get(n.original)!,
        debug: n.debug,
      });
    } else if (isUnaryKind(kind)) {
      const n = node as UnaryOp;
      nodes.push({
        type: "UNA",
        op: n.operation,
        input: nodeToId.get(n.original)!,
      });
    } else if (isBinaryKind(kind)) {
      const n = node as BinaryOp;
      nodes.push({
        type: "BIN",
        op: n.operation,
        left: nodeToId.get(n.left)!,
        right: nodeToId.get(n.right)!,
      });
    } else if (kind === KIND_LIT) {
      nodes.push({ type: "LIT", value: (node as LiteralNum).value });
    } else if (kind === KIND_VAR) {
      const name = (node as Variable).name;
      nodes.push({
        type: "VAR",
        name: typeof name === "symbol" ? (name.description ?? "") : name,
      });
    } else if (kind === KIND_DERIVATIVE) {
      nodes.push({
        type: "DRV",
        variable: nodeToId.get((node as Derivative).variable)!,
      });
    } else if (kind === KIND_SELECT) {
      const n = node as SelectOp;
      nodes.push({
        type: "SEL",
        condition: nodeToId.get(n.condition)!,
        ifNonZero: nodeToId.get(n.ifNonZero)!,
        ifZero: nodeToId.get(n.ifZero)!,
      });
    } else if (kind === KIND_FOREIGN) {
      throw new Error(
        "Cannot serialize ForeignFn nodes (they contain closures)",
      );
    } else {
      throw new Error(`Unknown NumNode kind: ${kind} (${node.operation})`);
    }
  });

  return { nodes, root: nodeToId.get(root)! };
}

// --- Deserialize ---

/**
 * Reconstructs a NumNode DAG from its serialized form.
 * Returns the root NumNode.
 */
export function deserializeNumDAG(data: SerializedNumDAG): NumNode {
  const built: NumNode[] = [];

  for (const entry of data.nodes) {
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
      default:
        throw new Error(
          `Unknown serialized node type: ${(entry as SerializedNode & { type: string }).type}`,
        );
    }
  }

  return built[data.root]!;
}
