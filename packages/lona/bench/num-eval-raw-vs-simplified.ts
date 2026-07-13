/**
 * Benchmark: wasm vs wasmTape evaluation on the gridfinity level-set DAG,
 * comparing the raw num tree against the compress+simplify'd tree.
 *
 * The raw fixture has ~600k DAG nodes vs ~18k after compress+simplify, so
 * this measures both the compile cost and the steady-state eval cost of
 * skipping in-num simplification.
 *
 * Fixture is produced by:
 *   cd packages/fonseranes && node --import tsx bench/extract-gridfinity-raw.ts
 *
 * Run:
 *   node --import tsx bench/num-eval-raw-vs-simplified.ts
 */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import type { NumNode, VarName } from "../src/core/tree";
import {
  deserializeNumDAG,
  type SerializedNumDAG,
} from "../src/core/tree-serialization";
import {
  compileValueRoutine,
  type BackendName,
  type ValueRoutine,
} from "../src/eval/routines";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const EVAL_COUNTS = [5_000, 30_000];
const FIXTURE = "gridfinity-3d-raw-numtree.json";

type Bounds3D = {
  x: [number, number];
  y: [number, number];
  z: [number, number];
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function timeGrid(
  evalPoint: (x: number, y: number, z: number) => void,
  bounds: Bounds3D,
  totalEvals: number,
): number {
  const n = Math.max(2, Math.ceil(Math.cbrt(totalEvals)));
  const stepX = (bounds.x[1] - bounds.x[0]) / (n - 1);
  const stepY = (bounds.y[1] - bounds.y[0]) / (n - 1);
  const stepZ = (bounds.z[1] - bounds.z[0]) / (n - 1);

  let count = 0;
  const t0 = performance.now();
  outer: for (let k = 0; k < n; k++) {
    const zv = bounds.z[0] + k * stepZ;
    for (let j = 0; j < n; j++) {
      const yv = bounds.y[0] + j * stepY;
      for (let i = 0; i < n; i++) {
        evalPoint(bounds.x[0] + i * stepX, yv, zv);
        if (++count >= totalEvals) break outer;
      }
    }
  }
  return performance.now() - t0;
}

type Strategy = {
  name: "wasm-codegen" | "wasm-interp";
  backend: BackendName;
};

const STRATEGIES: Strategy[] = [
  { name: "wasm-codegen", backend: "wasm-codegen" },
  { name: "wasm-interp", backend: "wasm-interp" },
];

function compileStrategy(
  root: NumNode,
  backend: BackendName,
): (vars: Map<VarName, number>) => number {
  const routine = compileValueRoutine([root], { backend }) as ValueRoutine;
  return (vars) => routine.eval(vars);
}

type Row = {
  variant: "raw" | "simplify";
  strategy: "wasm-codegen" | "wasm-interp";
  evals: number;
  nodes: number;
  /**
   * Cost of producing the num tree itself (from the live DAG of the
   * expression). For the `raw` variant this is `shape.expr.levelSet(…)`
   * alone; for `simplify` it includes `.simplify()` on top, which is
   * what a caller actually pays in practice.
   */
  buildMs: number;
  compileMs: number;
  evalMs: number;
  /** buildMs + compileMs + evalMs — end-to-end wall clock for this batch. */
  totalMs: number;
  "µs/eval": number;
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const fixturesDir = path.resolve(import.meta.dirname, "fixtures");
const raw = fs.readFileSync(path.join(fixturesDir, FIXTURE), "utf-8");
const fixture = JSON.parse(raw) as {
  bounds: Bounds3D;
  buildMs?: { raw: number; simplify: number };
  raw: { dag: SerializedNumDAG };
  simplify: { dag: SerializedNumDAG };
};
const bounds = fixture.bounds;

const buildMs = fixture.buildMs ?? { raw: 0, simplify: 0 };

const VARIANTS = [
  { name: "raw" as const, dag: fixture.raw.dag, buildMs: buildMs.raw },
  {
    name: "simplify" as const,
    dag: fixture.simplify.dag,
    buildMs: buildMs.simplify,
  },
];

console.log(
  "Gridfinity raw-vs-simplified benchmark — wasm-codegen & wasm-interp only\n",
);
console.log(`  raw DAG:       ${fixture.raw.dag.nodes.length} nodes`);
console.log(`  simplify DAG:  ${fixture.simplify.dag.nodes.length} nodes`);
if (fixture.buildMs) {
  console.log(
    `  num build cost: raw=${fixture.buildMs.raw.toFixed(1)}ms, simplify=${fixture.buildMs.simplify.toFixed(1)}ms`,
  );
}
console.log(`  eval counts:          ${EVAL_COUNTS.join(", ")}\n`);

const rows: Row[] = [];

for (const variant of VARIANTS) {
  console.log(`${"=".repeat(60)}`);
  console.log(`variant: ${variant.name} (${variant.dag.nodes.length} nodes)`);
  console.log(`${"=".repeat(60)}`);

  const variantRoot = deserializeNumDAG(variant.dag);

  for (const strategy of STRATEGIES) {
    // Compile once, warm it, and measure the combined cost.
    // The raw DAG is large enough that wasm-codegen exceeds the
    // V8 per-function size limit (~7.6MB of bytecode), so catch and skip.
    const compileT0 = performance.now();
    let fn: (vars: Map<VarName, number>) => number;
    try {
      fn = compileStrategy(variantRoot, strategy.backend);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`\n  ${strategy.name}: SKIPPED — ${msg}`);
      for (const batch of EVAL_COUNTS) {
        rows.push({
          variant: variant.name,
          strategy: strategy.name,
          evals: batch,
          nodes: variant.dag.nodes.length,
          buildMs: Number(variant.buildMs.toFixed(2)),
          compileMs: NaN,
          evalMs: NaN,
          totalMs: NaN,
          "µs/eval": NaN,
        });
      }
      continue;
    }
    const warmup = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
      ["z", 0],
    ]);
    fn(warmup); // force V8 Turbofan to finish for wasm
    const compileMs = performance.now() - compileT0;

    console.log(
      `\n  ${strategy.name} (cold compile: ${compileMs.toFixed(1)}ms):`,
    );

    for (const batch of EVAL_COUNTS) {
      const vars = new Map<VarName, number>([
        ["x", 0],
        ["y", 0],
        ["z", 0],
      ]);
      const evalMs = timeGrid(
        (x, y, z) => {
          vars.set("x", x);
          vars.set("y", y);
          vars.set("z", z);
          fn(vars);
        },
        bounds,
        batch,
      );
      const totalMs = variant.buildMs + compileMs + evalMs;
      const usPerEval = (totalMs / batch) * 1000;
      rows.push({
        variant: variant.name,
        strategy: strategy.name,
        evals: batch,
        nodes: variant.dag.nodes.length,
        buildMs: Number(variant.buildMs.toFixed(2)),
        compileMs: Number(compileMs.toFixed(2)),
        evalMs: Number(evalMs.toFixed(2)),
        totalMs: Number(totalMs.toFixed(2)),
        "µs/eval": Number(usPerEval.toFixed(2)),
      });
      console.log(
        `    ${String(batch).padStart(6)} evals → ${variant.buildMs.toFixed(1).padStart(7)}ms build + ${compileMs.toFixed(1).padStart(7)}ms compile + ${evalMs.toFixed(1).padStart(8)}ms eval = ${totalMs.toFixed(1).padStart(8)}ms  (${usPerEval.toFixed(2)} µs/eval)`,
      );
    }
  }
  console.log();
}

console.log("--- Summary ---");
console.table(rows);

// Side-by-side totalMs comparison per (strategy, evals), including the
// cost of actually producing each num tree (raw vs simplify).
console.log(
  "\n--- Raw vs simplify (build + compile + eval, lower is better) ---",
);
const summary: Array<{
  strategy: string;
  evals: number;
  "raw build": number;
  "raw compile": number;
  "raw eval": number;
  "raw total": number;
  "simplify build": number;
  "simplify compile": number;
  "simplify eval": number;
  "simplify total": number;
  "simplify speedup": string;
}> = [];
for (const strategy of STRATEGIES) {
  for (const batch of EVAL_COUNTS) {
    const rawRow = rows.find(
      (r) =>
        r.variant === "raw" &&
        r.strategy === strategy.name &&
        r.evals === batch,
    )!;
    const simplifyRow = rows.find(
      (r) =>
        r.variant === "simplify" &&
        r.strategy === strategy.name &&
        r.evals === batch,
    )!;
    const speedup = Number.isFinite(rawRow.totalMs)
      ? (rawRow.totalMs / simplifyRow.totalMs).toFixed(2) + "x"
      : "—";
    summary.push({
      strategy: strategy.name,
      evals: batch,
      "raw build": rawRow.buildMs,
      "raw compile": rawRow.compileMs,
      "raw eval": Number.isFinite(rawRow.evalMs)
        ? Number(rawRow.evalMs.toFixed(1))
        : NaN,
      "raw total": rawRow.totalMs,
      "simplify build": simplifyRow.buildMs,
      "simplify compile": simplifyRow.compileMs,
      "simplify eval": Number(simplifyRow.evalMs.toFixed(1)),
      "simplify total": simplifyRow.totalMs,
      "simplify speedup": speedup,
    });
  }
}
console.table(summary);
