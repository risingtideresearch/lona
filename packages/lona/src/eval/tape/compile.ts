/**
 * Tape compilation — transform a NumNode DAG (live object graph or serialized
 * form) into a `CompiledTape`.
 *
 * The walker + per-kind dispatch live in `./emit.ts` (shared with
 * `LiveTape`). This file owns the snapshot-style storage strategy:
 * push to plain arrays during the walk, then convert to typed arrays
 * for the final `CompiledTape` result.
 */
import { NumNode, KIND_LIT, type VarName } from "../../core/tree";
import type { SerializedNumDAG } from "../../core/tree-serialization";
import type { CompiledTape, TapeAssertion } from "./compiled-tape";
import type { TapeAssertionKind } from "./assertions";
import { emitOperandSubgraph, type TapeEmitter } from "./emit";
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

  const opcodes: number[] = [];
  const argA: number[] = [];
  const argB: number[] = [];
  const literals: number[] = [];
  const varSlots: VarName[] = [];
  const varNameToSlot = new Map<VarName, number>();
  const drvPendingSlots: { tapeIdx: number; baseName: VarName }[] = [];
  const assertions: TapeAssertion[] = [];

  const emitter: TapeEmitter = {
    emitLit(value) {
      const idx = opcodes.length;
      opcodes.push(KIND_LIT);
      argA.push(literals.length);
      argB.push(0);
      literals.push(value);
      return idx;
    },
    emitVar(name) {
      const idx = opcodes.length;
      let slot = varNameToSlot.get(name);
      if (slot === undefined) {
        slot = varSlots.length;
        varSlots.push(name);
        varNameToSlot.set(name, slot);
      }
      opcodes.push(OP_VAR);
      argA.push(slot);
      argB.push(0);
      return idx;
    },
    emitUnary(op, operandIdx) {
      const idx = opcodes.length;
      opcodes.push(op);
      argA.push(operandIdx);
      argB.push(0);
      return idx;
    },
    emitBinary(op, leftIdx, rightIdx) {
      const idx = opcodes.length;
      opcodes.push(op);
      argA.push(leftIdx);
      argB.push(rightIdx);
      return idx;
    },
  };

  const nodeToIndex = new Map<NumNode, number>();
  const emitSubgraph = (root: NumNode) =>
    emitOperandSubgraph(root, emitter, nodeToIndex, {
      onDerivative: (node) => {
        const idx = opcodes.length;
        opcodes.push(OP_VAR);
        argA.push(-1); // placeholder, filled in after the walk
        argB.push(0);
        drvPendingSlots.push({ tapeIdx: idx, baseName: node.variable.name });
        return idx;
      },
    });

  for (let i = 0; i < guards.length; i++) {
    const guard = guards[i]!;
    const result = emitSubgraph(guard.node);
    if (!result.ok) return null;

    const tapeIndex = opcodes.length;
    const id = guard.id ?? i;
    opcodes.push(guard.kind === "zero" ? OP_ASSERT_ZERO : OP_ASSERT_NONZERO);
    argA.push(result.idx);
    argB.push(id);
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
    argA[tapeIdx] = slot;
  }

  return {
    opcodes: new Uint8Array(opcodes),
    argA: new Int32Array(argA),
    argB: new Int32Array(argB),
    literals: new Float64Array(literals),
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
