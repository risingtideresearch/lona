import { describe, expect, test } from "vitest";
import fs from "node:fs";
import path from "node:path";
import {
  KIND_SELECT,
  type NumNode,
  type VarName,
  childrenOfNumNode,
} from "../../core/tree";
import {
  deserializeNumDAG,
  type SerializedNumDAG,
} from "../../core/tree-serialization";
import { visitFromLeaves } from "../../dag/traversal";
import { compileTape } from "./compile";
import {
  compileSelectTraceTape,
  specializeSelectsFromTrace,
  traceTapeValues,
} from "./select-specialization";
import { evalTape } from "../routines/backends/js-interp/tape-eval";

function fixturePath(filename: string): string {
  return path.resolve(import.meta.dirname, "../../../bench/fixtures", filename);
}

function loadSelectGridfinityRoot(): NumNode {
  const raw = fs.readFileSync(
    fixturePath("gridfinity-3d-select-numtree.json"),
    "utf-8",
  );
  const dag = JSON.parse(raw).select.dag as SerializedNumDAG;
  return deserializeNumDAG(dag);
}

function countNodes(root: NumNode): number {
  const seen = new Set<NumNode>();
  visitFromLeaves(root, childrenOfNumNode, (node) => seen.add(node));
  return seen.size;
}

function countSelects(root: NumNode): number {
  let count = 0;
  visitFromLeaves(root, childrenOfNumNode, (node) => {
    if (node.kind === KIND_SELECT) count++;
  });
  return count;
}

function vars(x: number, y: number, z: number): Map<VarName, number> {
  return new Map<VarName, number>([
    ["x", x],
    ["y", y],
    ["z", z],
  ]);
}

function percent(numerator: number, denominator: number): number {
  return Number(((100 * numerator) / denominator).toFixed(2));
}

function reportRow(
  root: NumNode,
  label: string,
  x: number,
  y: number,
  z: number,
) {
  const bindings = vars(x, y, z);
  const trace = compileSelectTraceTape([root])!;
  const traceValues = traceTapeValues(trace.tape, bindings);
  const specialized = specializeSelectsFromTrace([root], trace, traceValues);
  const specializedRoot = specialized.roots[0]!;
  const valueTape = compileTape([specializedRoot])!;
  const guardedTape = compileTape([specializedRoot], {
    guardPrelude: specialized.guardPrelude,
  })!;

  const fullValue = evalTape(trace.tape, bindings)[0]!;
  const specializedValue = evalTape(valueTape, bindings)[0]!;
  const guardedValue = evalTape(guardedTape, bindings)[0]!;
  const fullOps = trace.tape.opcodes.length;
  const specializedOps = valueTape.opcodes.length;
  const guardedOps = guardedTape.opcodes.length;

  return {
    tracePoint: label,
    x,
    y,
    z,
    fullValue: Number(fullValue.toFixed(6)),
    specializedValue: Number(specializedValue.toFixed(6)),
    guardedValue: Number(guardedValue.toFixed(6)),
    fullDagNodes: countNodes(root),
    specializedDagNodes: countNodes(specializedRoot),
    originalSelects: countSelects(root),
    specializedSelects: countSelects(specializedRoot),
    selectDecisions: specialized.selectDecisions,
    guardAssertions: specialized.guardPrelude.length,
    fullOps,
    specializedOps,
    guardedOps,
    removedOps: fullOps - specializedOps,
    removedPct: percent(fullOps - specializedOps, fullOps),
    guardedOpsDelta: guardedOps - fullOps,
    guardedOpsDeltaPct: percent(guardedOps - fullOps, fullOps),
  };
}

describe("DAG-level select specialization from tape traces", () => {
  const root = loadSelectGridfinityRoot();

  test("fixture deserializes to fonseranes ifTruthyElse/select nodes", () => {
    expect(countNodes(root)).toBe(12731);
    expect(countSelects(root)).toBe(831);
  });

  test("uses tape trace values to prune selects on the Num DAG", () => {
    const rows = [
      reportRow(root, "center", 0, 0, 10),
      reportRow(root, "near-center", 1e-3, -1e-3, 10 + 1e-3),
      reportRow(root, "min-corner", -45, -25, -5),
      reportRow(root, "max-corner", 45, 25, 25),
      reportRow(root, "bottom-center", 0, 0, -5),
      reportRow(root, "top-center", 0, 0, 25),
    ];

    console.table(rows);

    for (const row of rows) {
      expect(row.specializedValue).toBe(row.fullValue);
      expect(row.guardedValue).toBe(row.fullValue);
      expect(row.specializedSelects).toBe(0);
      expect(row.specializedOps).toBeLessThan(row.fullOps);
      expect(row.guardedOps).toBeLessThan(row.fullOps);
      expect(row.guardAssertions).toBeGreaterThan(0);
      expect(row.guardAssertions).toBeLessThanOrEqual(row.selectDecisions);
    }
  });
});
