/**
 * Benchmark for 3D Num tree evaluation across real-world shapes.
 *
 * Fixtures extracted from fonseranes via:
 *   cd packages/fonseranes && node --import tsx bench/extract-num-tree.ts
 *
 * Run:
 *   node --import tsx bench/num-eval-3d.ts
 *
 * Env vars:
 *   BENCH_GRID_SIZE  — points per axis for fast strategies (default 30)
 *   BENCH_WARMUP     — warmup iterations (default 2)
 *   BENCH_RUNS       — timed iterations (default 5)
 */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { deserializeNumDAG } from "../src/core/tree-serialization";
import { simpleEval } from "../src/eval/eval-value";
import {
  compileValueRoutine,
  type BackendName,
  type ValueRoutine,
} from "../src/eval/routines";
import type { NumNode, VarName } from "../src/core/tree";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

function envNumber(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n))
    throw new Error(`Invalid numeric env var ${name}: ${raw}`);
  return n;
}

const GRID_SIZE = envNumber("BENCH_GRID_SIZE", 30);
const WARMUP = envNumber("BENCH_WARMUP", 2);
const RUNS = envNumber("BENCH_RUNS", 5);

// genericEval is orders of magnitude slower — use a smaller grid to keep
// total runtime reasonable, then normalise to per-eval cost for comparison.
const GENERIC_GRID_SIZE = envNumber("BENCH_GENERIC_GRID_SIZE", 8);

// ---------------------------------------------------------------------------
// Fixture definitions
// ---------------------------------------------------------------------------

type Bounds3D = {
  x: [number, number];
  y: [number, number];
  z: [number, number];
};

const SHAPES = [
  { name: "gridfinity", file: "gridfinity-3d-numtree.json" },
  { name: "ducktape", file: "ducktape-3d-numtree.json" },
  { name: "logo", file: "logo-3d-numtree.json" },
];

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

type BenchStats = {
  name: string;
  totalMs: number;
  perEvalUs: number;
  gridSize: number;
};

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1]! + sorted[mid]!) / 2
    : sorted[mid]!;
}

function runBenchmark(
  name: string,
  fn: () => void,
  gridSize: number,
  warmup: number,
  runs: number,
): BenchStats {
  for (let i = 0; i < warmup; i++) fn();
  const samplesMs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    samplesMs.push(performance.now() - t0);
  }
  const totalMs = median(samplesMs);
  const evals = gridSize ** 3;
  return {
    name,
    totalMs,
    perEvalUs: (totalMs / evals) * 1000,
    gridSize,
  };
}

// ---------------------------------------------------------------------------
// Grid evaluation helpers
// ---------------------------------------------------------------------------

function makeGridLoop(
  bounds: Bounds3D,
  gridSize: number,
  evalPoint: (x: number, y: number, z: number) => void,
): () => void {
  const stepX = (bounds.x[1] - bounds.x[0]) / (gridSize - 1);
  const stepY = (bounds.y[1] - bounds.y[0]) / (gridSize - 1);
  const stepZ = (bounds.z[1] - bounds.z[0]) / (gridSize - 1);
  return () => {
    for (let k = 0; k < gridSize; k++) {
      const z = bounds.z[0] + k * stepZ;
      for (let j = 0; j < gridSize; j++) {
        const y = bounds.y[0] + j * stepY;
        for (let i = 0; i < gridSize; i++) {
          evalPoint(bounds.x[0] + i * stepX, y, z);
        }
      }
    }
  };
}

function makeBackendLoop(
  root: NumNode,
  backend: BackendName,
  bounds: Bounds3D,
): () => void {
  const routine = compileValueRoutine([root], { backend }) as ValueRoutine;
  const vars = new Map<VarName, number>([
    ["x", 0],
    ["y", 0],
    ["z", 0],
  ]);
  return makeGridLoop(bounds, GRID_SIZE, (x, y, z) => {
    vars.set("x", x);
    vars.set("y", y);
    vars.set("z", z);
    routine.eval(vars);
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log(`3D Num eval benchmark`);
console.log(`  Fast strategies: ${GRID_SIZE}^3 = ${GRID_SIZE ** 3} evals`);
console.log(
  `  genericEval:     ${GENERIC_GRID_SIZE}^3 = ${GENERIC_GRID_SIZE ** 3} evals`,
);
console.log(`  Warmup: ${WARMUP}, Runs: ${RUNS}\n`);

const fixturesDir = path.resolve(import.meta.dirname, "fixtures");

for (const shape of SHAPES) {
  const raw = fs.readFileSync(path.join(fixturesDir, shape.file), "utf-8");
  const fixture = JSON.parse(raw);
  const dag = fixture.compressSimplify.dag;
  const bounds: Bounds3D = fixture.bounds;

  const nodeCount = dag.nodes.length;
  console.log(`${"=".repeat(60)}`);
  console.log(`${shape.name} — ${nodeCount} DAG nodes`);
  console.log(`Bounds: x=[${bounds.x}] y=[${bounds.y}] z=[${bounds.z}]`);
  console.log(`${"=".repeat(60)}`);

  const root = deserializeNumDAG(dag);

  // --- genericEval (smaller grid) ---
  const bindings = new Map<VarName, number>([
    ["x", 0],
    ["y", 0],
    ["z", 0],
  ]);
  const genericLoop = makeGridLoop(bounds, GENERIC_GRID_SIZE, (x, y, z) => {
    bindings.set("x", x);
    bindings.set("y", y);
    bindings.set("z", z);
    simpleEval(root, bindings, true);
  });

  const strategies: [string, () => void, number][] = [
    ["genericEval", genericLoop, GENERIC_GRID_SIZE],
    [
      "routine js-interp",
      makeBackendLoop(root, "js-interp", bounds),
      GRID_SIZE,
    ],
    [
      "routine js-codegen",
      makeBackendLoop(root, "js-codegen", bounds),
      GRID_SIZE,
    ],
    [
      "routine wasm-interp",
      makeBackendLoop(root, "wasm-interp", bounds),
      GRID_SIZE,
    ],
    [
      "routine wasm-codegen",
      makeBackendLoop(root, "wasm-codegen", bounds),
      GRID_SIZE,
    ],
  ];

  const stats: BenchStats[] = [];
  for (const [name, fn, grid] of strategies) {
    process.stdout.write(`  ${name} (${grid}^3)...`);
    const stat = runBenchmark(name, fn, grid, WARMUP, RUNS);
    stats.push(stat);
    console.log(
      ` ${stat.totalMs.toFixed(1)}ms total, ${stat.perEvalUs.toFixed(2)}µs/eval`,
    );
  }

  const baselineUs = stats[0]!.perEvalUs;
  console.log(`\n  --- ${shape.name}: per-eval cost ---`);
  console.table(
    stats.map((s) => ({
      strategy: s.name,
      "µs/eval": Number(s.perEvalUs.toFixed(2)),
      "evals/sec": Number((1_000_000 / s.perEvalUs).toFixed(0)),
      speedup: (baselineUs / s.perEvalUs).toFixed(1) + "x",
    })),
  );

  // Parity check across backends against the genericEval reference.
  const backends: BackendName[] = [
    "js-interp",
    "js-codegen",
    "wasm-interp",
    "wasm-codegen",
  ];
  const diffs: Record<string, number> = Object.fromEntries(
    backends.map((b) => [b, 0]),
  );
  const routines = new Map<BackendName, ValueRoutine>();
  for (const b of backends) {
    routines.set(
      b,
      compileValueRoutine([root], { backend: b }) as ValueRoutine,
    );
  }
  const CHECK_N = 5;
  const cStepX = (bounds.x[1] - bounds.x[0]) / (CHECK_N - 1);
  const cStepY = (bounds.y[1] - bounds.y[0]) / (CHECK_N - 1);
  const cStepZ = (bounds.z[1] - bounds.z[0]) / (CHECK_N - 1);
  for (let k = 0; k < CHECK_N; k++) {
    const z = bounds.z[0] + k * cStepZ;
    bindings.set("z", z);
    for (let j = 0; j < CHECK_N; j++) {
      const y = bounds.y[0] + j * cStepY;
      bindings.set("y", y);
      for (let i = 0; i < CHECK_N; i++) {
        const x = bounds.x[0] + i * cStepX;
        bindings.set("x", x);
        const ref = simpleEval(root, bindings, true);
        const checkVars = new Map<VarName, number>([
          ["x", x],
          ["y", y],
          ["z", z],
        ]);
        for (const b of backends) {
          diffs[b] = Math.max(
            diffs[b]!,
            Math.abs(ref - routines.get(b)!.eval(checkVars)),
          );
        }
      }
    }
  }
  console.log(
    `  Parity (${CHECK_N}^3): ${backends.map((b) => `${b}=${diffs[b]}`).join(", ")}\n`,
  );
}
