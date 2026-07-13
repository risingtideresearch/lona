import type { BinaryOperation, UnaryOperation } from "../../types";
import {
  BinaryOp,
  DebugNode,
  Derivative,
  KIND_DEBUG,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_LIT,
  KIND_SELECT,
  KIND_VAR,
  type NumNode,
  SelectOp,
  UnaryOp,
  type VarName,
  childrenOfNumNode,
  isBinaryKind,
  isUnaryKind,
} from "../../core/tree";
import {
  binaryNode,
  debugNode,
  derivativeNode,
  unaryNode,
} from "../../core/tree-cons";
import { visitFromLeaves } from "../../dag/traversal";
import { compileTape, type TapeGuardPreludeAssertion } from "./compile";
import { bindVarMap } from "./bind-vars";
import type { CompiledTape } from "./compiled-tape";
import { interpretTapeValues } from "../routines/backends/wasm-interp/tape-eval";

export interface SelectTraceTape {
  tape: CompiledTape;
  nodeToTapeIdx: WeakMap<NumNode, number>;
}

export interface SelectSpecializationResult {
  roots: NumNode[];
  guardPrelude: TapeGuardPreludeAssertion[];
  selectDecisions: number;
}

interface SubgraphSpecialization {
  value: NumNode;
  guardPrelude: TapeGuardPreludeAssertion[];
  selectDecisions: number;
}

/**
 * Compile a normal tape and build the `NumNode -> tape index` map needed to
 * interpret trace values back at the DAG level.
 */
export function compileSelectTraceTape(
  roots: NumNode[],
): SelectTraceTape | null {
  const tape = compileTape(roots);
  if (!tape) return null;

  const nodeToTapeIdx = new WeakMap<NumNode, number>();
  let tapeIdx = 0;
  for (const root of roots) {
    visitFromLeaves(root, childrenOfNumNode, (node) => {
      if (nodeToTapeIdx.has(node)) return;

      if (node.kind === KIND_SELECT) {
        // compileTape lowers select to:
        //   and(condition, ifNonZero)
        //   not(condition)
        //   and(not(condition), ifZero)
        //   or(lhs, rhs)
        // The select node maps to the final OR value.
        tapeIdx += 4;
        nodeToTapeIdx.set(node, tapeIdx - 1);
      } else {
        tapeIdx += 1;
        nodeToTapeIdx.set(node, tapeIdx - 1);
      }
    });
  }

  if (tapeIdx !== tape.opcodes.length) {
    throw new Error(
      `Trace tape map mismatch: mapped ${tapeIdx} ops, tape has ${tape.opcodes.length}`,
    );
  }

  return { tape, nodeToTapeIdx };
}

/** Evaluate a tape and keep every intermediate value for trace specialization. */
export function traceTapeValues(
  tape: CompiledTape,
  bindings: Map<VarName, number> | Map<string, number>,
  derivatives?: Map<VarName, number> | Map<string, number>,
): Float64Array {
  const values = new Float64Array(tape.opcodes.length);
  const varValues = new Float64Array(tape.varSlots.length);
  bindVarMap(tape, bindings, derivatives, varValues);
  interpretTapeValues(
    tape.opcodes,
    tape.argA,
    tape.argB,
    tape.literals,
    tape.opcodes.length,
    varValues,
    values,
  );
  return values;
}

function uniqueGuardPrelude(
  guards: TapeGuardPreludeAssertion[],
): TapeGuardPreludeAssertion[] {
  const seenSources = new WeakSet<object>();
  const result: TapeGuardPreludeAssertion[] = [];

  for (const guard of guards) {
    const source = guard.source;
    if (
      (typeof source === "object" && source !== null) ||
      typeof source === "function"
    ) {
      if (seenSources.has(source)) continue;
      seenSources.add(source);
    }
    result.push(guard);
  }

  return result;
}

/**
 * Specialize `roots` by pruning `SelectOp` branches according to a tape trace.
 *
 * The returned roots contain no select decisions that were reached during the
 * specialization walk. The returned `guardPrelude` contains assertion nodes
 * that prove the traced select decisions still hold, intended to be passed to
 * `compileTape(specialized.roots, { guardPrelude })`.
 */
export function specializeSelectsFromTrace(
  roots: NumNode[],
  trace: SelectTraceTape,
  traceValues: Float64Array,
): SelectSpecializationResult {
  const memo = new WeakMap<NumNode, SubgraphSpecialization>();
  let nextAssertionId = 0;

  const specialize = (node: NumNode): SubgraphSpecialization => {
    const cached = memo.get(node);
    if (cached) return cached;

    let result: SubgraphSpecialization;
    if (node.kind === KIND_SELECT) {
      const select = node as SelectOp;
      const condition = specialize(select.condition);
      const conditionTapeIdx = trace.nodeToTapeIdx.get(select.condition);
      if (conditionTapeIdx === undefined) {
        throw new Error("Trace is missing select condition value");
      }
      const conditionPassed = traceValues[conditionTapeIdx] !== 0;
      const branch = specialize(
        conditionPassed ? select.ifNonZero : select.ifZero,
      );
      const assertion: TapeGuardPreludeAssertion = {
        node: condition.value,
        kind: conditionPassed ? "nonzero" : "zero",
        id: nextAssertionId++,
        source: select,
      };
      result = {
        value: branch.value,
        guardPrelude: [
          ...condition.guardPrelude,
          assertion,
          ...branch.guardPrelude,
        ],
        selectDecisions: 1 + condition.selectDecisions + branch.selectDecisions,
      };
    } else if (node.kind === KIND_LIT || node.kind === KIND_VAR) {
      result = { value: node, guardPrelude: [], selectDecisions: 0 };
    } else if (isUnaryKind(node.kind)) {
      const unary = node as UnaryOp;
      const original = specialize(unary.original);
      result = {
        value:
          node.kind === KIND_DEBUG
            ? debugNode(original.value, (node as DebugNode).debug)
            : unaryNode(unary.operation as UnaryOperation, original.value),
        guardPrelude: original.guardPrelude,
        selectDecisions: original.selectDecisions,
      };
    } else if (isBinaryKind(node.kind)) {
      const binary = node as BinaryOp;
      const left = specialize(binary.left);
      const right = specialize(binary.right);
      result = {
        value: binaryNode(
          binary.operation as BinaryOperation,
          left.value,
          right.value,
        ),
        guardPrelude: [...left.guardPrelude, ...right.guardPrelude],
        selectDecisions: left.selectDecisions + right.selectDecisions,
      };
    } else if (node.kind === KIND_DERIVATIVE) {
      result = {
        value: derivativeNode((node as Derivative).variable),
        guardPrelude: [],
        selectDecisions: 0,
      };
    } else if (node.kind === KIND_FOREIGN) {
      throw new Error("Cannot specialize foreign nodes");
    } else {
      throw new Error(`Unsupported node kind ${node.kind}`);
    }

    memo.set(node, result);
    return result;
  };

  const subgraphs = roots.map((root) => specialize(root));
  return {
    roots: subgraphs.map((s) => s.value),
    guardPrelude: uniqueGuardPrelude(subgraphs.flatMap((s) => s.guardPrelude)),
    selectDecisions: subgraphs.reduce((sum, s) => sum + s.selectDecisions, 0),
  };
}

/** Convenience wrapper: trace once with `bindings`, then specialize selects. */
export function traceAndSpecializeSelects(
  roots: NumNode[],
  bindings: Map<VarName, number> | Map<string, number>,
  derivatives?: Map<VarName, number> | Map<string, number>,
): (SelectSpecializationResult & { trace: SelectTraceTape }) | null {
  const trace = compileSelectTraceTape(roots);
  if (!trace) return null;
  const traceValues = traceTapeValues(trace.tape, bindings, derivatives);
  return {
    ...specializeSelectsFromTrace(roots, trace, traceValues),
    trace,
  };
}
