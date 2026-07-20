/**
 * Tape compilation — transform a NumNode DAG (live object graph or serialized
 * form) into a `CompiledTape`.
 *
 * The walker + per-kind dispatch live in `./emit.ts` (shared with
 * `LiveTape`). Live NumNode DAGs are written directly into growable typed
 * storage; serialized DAG compilation retains its simpler array-based path.
 */
import { Call, NumNode, type VarName } from "../../core/tree";
import type { SerializedNumDAG } from "../../core/tree-serialization";
import type { CompiledTape, TapeAssertion } from "./compiled-tape";
import type { TapeAssertionKind } from "./assertions";
import { emitOperandSubgraph, type CallEmission } from "./emit";
import { GrowableTape } from "./growable-tape";
import {
  OP_AND,
  OP_DEBUG,
  OP_LIT,
  OP_NOT,
  OP_OR,
  OP_VAR,
  OP_ASSERT_ZERO,
  OP_ASSERT_NONZERO,
  UNARY_OP_MAP,
  BINARY_OP_MAP,
} from "./opcodes";

// ---------------------------------------------------------------------------
// compileTape — from one or more live NumNode DAG roots
// ---------------------------------------------------------------------------

export interface TapeGuardPreludeAssertion {
  node: NumNode;
  kind: TapeAssertionKind;
  id?: number;
  source?: unknown;
}

export interface CompileTapeOpts {
  /**
   * Optional guard prelude emitted before value roots. Assertions are evaluated
   * for their side effect (throwing in assertion-aware scalar backends) and are
   * not included in `rootIndices`.
   */
  guardPrelude?: TapeGuardPreludeAssertion[];
}

export function compileTape(
  roots: NumNode[],
  opts: CompileTapeOpts = {},
): CompiledTape | null {
  if (roots.length === 0) return null;

  const guards = opts.guardPrelude ?? [];

  const builder = new GrowableTape();
  const { nodeToIndex, varSlots } = builder;
  const drvPendingSlots: { tapeIdx: number; baseName: VarName }[] = [];
  const assertions: TapeAssertion[] = [];

  // One Call → emission cache for the whole compilation: a Call referenced by
  // several roots (or via several projections) emits its body exactly once.
  const callEmissions = new Map<Call, CallEmission>();

  const emitSubgraph = (root: NumNode) =>
    emitOperandSubgraph(root, builder, nodeToIndex, {
      callEmissions,
      onDerivative: (node) => {
        // OP_VAR with a placeholder slot, fixed after regular variables have
        // all been assigned. `emitUnary` has exactly the required op/a/zero
        // layout and keeps the append inside typed storage.
        const idx = builder.emitUnary(OP_VAR, -1);
        drvPendingSlots.push({ tapeIdx: idx, baseName: node.variable.name });
        return idx;
      },
    });

  for (let i = 0; i < guards.length; i++) {
    const guard = guards[i]!;
    const result = emitSubgraph(guard.node);
    if (!result.ok) return null;

    const id = guard.id ?? i;
    const tapeIndex = builder.emitBinary(
      guard.kind === "zero" ? OP_ASSERT_ZERO : OP_ASSERT_NONZERO,
      result.idx,
      id,
    );
    assertions.push({
      id,
      tapeIndex,
      kind: guard.kind,
      source: guard.source,
    });
  }

  for (const root of roots) {
    const result = emitSubgraph(root);
    if (!result.ok) return null;
  }

  const numVars = varSlots.length;
  const drvNameToSlot = new Map<VarName, number>();
  for (const { tapeIdx, baseName } of drvPendingSlots) {
    let slot = drvNameToSlot.get(baseName);
    if (slot === undefined) {
      slot = varSlots.length;
      varSlots.push(baseName);
      drvNameToSlot.set(baseName, slot);
    }
    builder.argA[tapeIdx] = slot;
  }

  return {
    // Exact-length views avoid one final full-tape copy. The backing buffers
    // retain geometric spare capacity, but that is substantially smaller than
    // keeping both the builder buffers and an exact snapshot alive here.
    opcodes: builder.opcodes.subarray(0, builder.length),
    argA: builder.argA.subarray(0, builder.length),
    argB: builder.argB.subarray(0, builder.length),
    literals: builder.literals.subarray(0, builder.numLiterals),
    varSlots,
    numVars,
    rootIndices: [
      nodeToIndex.get(roots[0]!)!,
      ...roots.slice(1).map((r) => nodeToIndex.get(r)!),
    ],
    ...(assertions.length > 0 ? { assertions } : {}),
  };
}

// ---------------------------------------------------------------------------
// compileTapeFromSerialized — from SerializedNumDAG (avoids object graph)
// ---------------------------------------------------------------------------

export function compileTapeFromSerialized(
  data: SerializedNumDAG,
): CompiledTape | null {
  const nodes = data.nodes;
  const len = nodes.length;

  const opcodes: number[] = [];
  const argA: number[] = [];
  const argB: number[] = [];
  const litValues: number[] = [];
  const varSlots: VarName[] = [];
  const varNameToSlot = new Map<VarName, number>();
  const drvPending: { idx: number; varIdx: number }[] = [];
  const nodeToTapeIdx = new Int32Array(len);

  const emit = (op: number, a: number, b = 0): number => {
    const idx = opcodes.length;
    opcodes.push(op);
    argA.push(a);
    argB.push(b);
    return idx;
  };

  for (let i = 0; i < len; i++) {
    const entry = nodes[i]!;
    let idx: number;
    switch (entry.type) {
      case "LIT":
        idx = emit(OP_LIT, litValues.length);
        litValues.push(entry.value);
        break;
      case "VAR": {
        let slot = varNameToSlot.get(entry.name);
        if (slot === undefined) {
          slot = varSlots.length;
          varSlots.push(entry.name);
          varNameToSlot.set(entry.name, slot);
        }
        idx = emit(OP_VAR, slot);
        break;
      }
      case "UNA": {
        const op = UNARY_OP_MAP[entry.op];
        if (op === undefined) return null;
        idx = emit(op, nodeToTapeIdx[entry.input]!);
        break;
      }
      case "DBG":
        idx = emit(OP_DEBUG, nodeToTapeIdx[entry.input]!);
        break;
      case "BIN": {
        const op = BINARY_OP_MAP[entry.op];
        if (op === undefined) return null;
        idx = emit(op, nodeToTapeIdx[entry.left]!, nodeToTapeIdx[entry.right]!);
        break;
      }
      case "DRV":
        idx = emit(OP_VAR, -1);
        // Slot assigned after all variables are collected.
        drvPending.push({ idx, varIdx: entry.variable });
        break;
      case "SEL": {
        const conditionIdx = nodeToTapeIdx[entry.condition]!;
        const ifNonZeroIdx = nodeToTapeIdx[entry.ifNonZero]!;
        const ifZeroIdx = nodeToTapeIdx[entry.ifZero]!;
        const lhs = emit(OP_AND, conditionIdx, ifNonZeroIdx);
        const notCondition = emit(OP_NOT, conditionIdx);
        const rhs = emit(OP_AND, notCondition, ifZeroIdx);
        idx = emit(OP_OR, lhs, rhs);
        break;
      }
      default:
        return null;
    }
    nodeToTapeIdx[i] = idx;
  }

  // Assign derivative slots after all variable slots.
  const numVars = varSlots.length;
  const drvNameToSlot = new Map<VarName, number>();
  for (const { idx, varIdx } of drvPending) {
    const varEntry = nodes[varIdx]!;
    if (varEntry.type !== "VAR") return null;
    const baseName = varEntry.name;
    let slot = drvNameToSlot.get(baseName);
    if (slot === undefined) {
      slot = varSlots.length;
      varSlots.push(baseName);
      drvNameToSlot.set(baseName, slot);
    }
    argA[idx] = slot;
  }

  return {
    opcodes: new Uint8Array(opcodes),
    argA: new Int32Array(argA),
    argB: new Int32Array(argB),
    literals: new Float64Array(litValues),
    varSlots,
    numVars,
    rootIndices: [nodeToTapeIdx[data.root]!],
  };
}
