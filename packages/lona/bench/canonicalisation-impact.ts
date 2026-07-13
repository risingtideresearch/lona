/**
 * Benchmark: how much does canonical commutative-operand ordering reduce
 * node count on real-world fixtures?
 *
 * Loads each fixture twice into a fresh NumTreeContext:
 *   1. With canonicalisation disabled — gives the baseline node count
 *      that hash-consing alone produces.
 *   2. With canonicalisation enabled — shows how many additional nodes
 *      collapse because `op(a, b)` and `op(b, a)` now share an entry.
 *
 * The count is derived from `sortId`, which is incremented once per
 * distinct allocated node across all factories, so the delta between
 * start and end is exactly the number of fresh allocations.
 *
 * Run:
 *   node --import tsx bench/canonicalisation-impact.ts
 */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import {
  NumTreeContext,
  setCanonicaliseCommutativeOperands,
} from "../src/core/tree-cons";
import { NumContext, setContext } from "../src/core/context";
import { deserializeNumDAG } from "../src/core/tree-serialization";
import { simplify } from "../src/api/simplify";
import { visitFromLeaves } from "../src/dag/traversal";
import { childrenOfNumNode, type NumNode } from "../src/core/tree";

const fixturesDir = path.resolve(import.meta.dirname, "fixtures");

type FixtureSpec = {
  label: string;
  file: string;
  // Key path into the JSON to reach the SerializedNumDAG.
  dagPath: string[];
};

const FIXTURES: FixtureSpec[] = [
  {
    label: "gridfinity-3d (raw)",
    file: "gridfinity-3d-raw-numtree.json",
    dagPath: ["raw", "dag"],
  },
  {
    label: "gridfinity-3d (pre-simplified)",
    file: "gridfinity-3d-raw-numtree.json",
    dagPath: ["simplify", "dag"],
  },
  {
    label: "gridfinity-3d (bundled)",
    file: "gridfinity-3d-numtree.json",
    dagPath: ["compressSimplify", "dag"],
  },
  {
    label: "ducktape-3d",
    file: "ducktape-3d-numtree.json",
    dagPath: ["compressSimplify", "dag"],
  },
  {
    label: "logo-3d",
    file: "logo-3d-numtree.json",
    dagPath: ["compressSimplify", "dag"],
  },
  {
    label: "gridfinity-slice-2d",
    file: "gridfinity-slice-numtree.json",
    dagPath: ["compressSimplify", "dag"],
  },
];

function loadDag(spec: FixtureSpec): {
  serializedNodeCount: number;
  dag: unknown;
} {
  const raw = fs.readFileSync(path.join(fixturesDir, spec.file), "utf-8");
  const json = JSON.parse(raw);
  let cursor: any = json;
  for (const k of spec.dagPath) cursor = cursor[k];
  return { serializedNodeCount: cursor.nodes.length, dag: cursor };
}

function countReachable(root: NumNode): number {
  let count = 0;
  visitFromLeaves(root, childrenOfNumNode, () => {
    count += 1;
  });
  return count;
}

function runOne(spec: FixtureSpec): void {
  const { serializedNodeCount, dag } = loadDag(spec);

  const measureDeserialize = (canonical: boolean) => {
    setContext(new NumContext(new NumTreeContext()));
    setCanonicaliseCommutativeOperands(canonical);

    const t0 = performance.now();
    const root = deserializeNumDAG(dag as any);
    const deserMs = performance.now() - t0;

    const reachable = countReachable(root);

    const t1 = performance.now();
    const simplified = simplify(root);
    const simpMs = performance.now() - t1;
    const reachableAfterSimplify = countReachable(simplified);

    setCanonicaliseCommutativeOperands(true);
    return { reachable, reachableAfterSimplify, deserMs, simpMs };
  };

  const off = measureDeserialize(false);
  const on = measureDeserialize(true);

  const reduction = (1 - on.reachable / off.reachable) * 100;
  const reductionAfter =
    (1 - on.reachableAfterSimplify / off.reachableAfterSimplify) * 100;

  console.log(`\n=== ${spec.label} ===`);
  console.log(
    `  serialized:               ${serializedNodeCount.toLocaleString().padStart(9)}`,
  );
  console.log(
    `  reachable (cons only):    ${off.reachable.toLocaleString().padStart(9)}` +
      `   deserialize ${off.deserMs.toFixed(1).padStart(6)} ms`,
  );
  console.log(
    `  reachable (cons+canon):   ${on.reachable.toLocaleString().padStart(9)}` +
      `   deserialize ${on.deserMs.toFixed(1).padStart(6)} ms  ` +
      `Δ ${reduction.toFixed(2)}%`,
  );
  console.log(
    `  after simplify (cons):    ${off.reachableAfterSimplify
      .toLocaleString()
      .padStart(9)}   simplify    ${off.simpMs.toFixed(1).padStart(6)} ms`,
  );
  console.log(
    `  after simplify (canon):   ${on.reachableAfterSimplify
      .toLocaleString()
      .padStart(9)}   simplify    ${on.simpMs.toFixed(1).padStart(6)} ms  ` +
      `Δ ${reductionAfter.toFixed(2)}%`,
  );
}

console.log("Canonical commutative operand ordering — impact on DAG size");
console.log("============================================================");

for (const spec of FIXTURES) {
  try {
    runOne(spec);
  } catch (e) {
    console.log(`\n=== ${spec.label} ===\n  SKIPPED: ${(e as Error).message}`);
  }
}

// ---------------------------------------------------------------------------
// Synthetic microbench: fresh construction with randomized operand order
// ---------------------------------------------------------------------------

import { varNode, binaryNode } from "../src/core/context";

function syntheticFreshConstruction(): void {
  // Build a bunch of ADDs with randomly-ordered operands. Without
  // canonicalisation, each order produces a distinct node. With it, the
  // two orderings collapse.
  const numVars = 50;
  const numPairs = 2_000;

  const measure = (canonical: boolean) => {
    setContext(new NumContext(new NumTreeContext()));
    setCanonicaliseCommutativeOperands(canonical);

    const vars = Array.from({ length: numVars }, (_, i) => varNode(`v${i}`));
    // Deterministic pseudo-random pairs
    let seed = 1;
    const rand = () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };

    const seen = new Set<unknown>();
    for (let i = 0; i < numPairs; i++) {
      const a = vars[Math.floor(rand() * numVars)]!;
      const b = vars[Math.floor(rand() * numVars)]!;
      if (a === b) continue;
      // Build both orders explicitly
      seen.add(binaryNode("ADD", a, b));
      seen.add(binaryNode("ADD", b, a));
      seen.add(binaryNode("MUL", a, b));
      seen.add(binaryNode("MUL", b, a));
    }
    setCanonicaliseCommutativeOperands(true);
    return seen.size;
  };

  const off = measure(false);
  const on = measure(true);
  console.log(`\n=== synthetic: both orders of random (ADD|MUL)(a,b) ===`);
  console.log(`  ${numPairs} random pairs × {ADD, MUL} × {(a,b), (b,a)}`);
  console.log(
    `  distinct nodes (cons only):   ${off.toLocaleString().padStart(6)}`,
  );
  console.log(
    `  distinct nodes (cons+canon):  ${on.toLocaleString().padStart(6)}`,
  );
  console.log(`  reduction: ${((1 - on / off) * 100).toFixed(2)}%`);
}

syntheticFreshConstruction();

console.log("");
