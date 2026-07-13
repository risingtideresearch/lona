/**
 * Benchmark for Num tree evaluation using a real-world gridfinity slice DAG.
 *
 * The fixture was extracted from fonseranes via:
 *   node --import tsx bench/extract-num-tree.ts
 *
 * Run:
 *   node --import tsx bench/num-eval.ts
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
import type { VarName } from "../src/core/tree";

// ---------------------------------------------------------------------------
// Load fixture
// ---------------------------------------------------------------------------

const fixturePath = path.resolve(
  import.meta.dirname,
  "fixtures/gridfinity-slice-numtree.json",
);
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf-8"));

const simplifiedRoot = deserializeNumDAG(fixture.compressSimplify.dag);

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

const GRID_SIZE = envNumber("BENCH_GRID_SIZE", 100);
const WARMUP = envNumber("BENCH_WARMUP", 2);
const RUNS = envNumber("BENCH_RUNS", 5);

const MIN_X = -45,
  MAX_X = 45,
  MIN_Y = -25,
  MAX_Y = 25;

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

type BenchStats = {
  name: string;
  samplesMs: number[];
  meanMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
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
  config: { warmup?: number; runs?: number } = {},
): BenchStats {
  const warmup = config.warmup ?? 2;
  const runs = config.runs ?? 5;
  for (let i = 0; i < warmup; i++) fn();
  const samplesMs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    samplesMs.push(performance.now() - t0);
  }
  const meanMs = samplesMs.reduce((a, b) => a + b, 0) / samplesMs.length;
  return {
    name,
    samplesMs,
    meanMs,
    medianMs: median(samplesMs),
    minMs: Math.min(...samplesMs),
    maxMs: Math.max(...samplesMs),
  };
}

// ---------------------------------------------------------------------------
// Evaluation strategies — each compares a different routine backend
// ---------------------------------------------------------------------------

function evalGrid(fn: (vars: Map<VarName, number>) => number): number[] {
  const stepX = (MAX_X - MIN_X) / (GRID_SIZE - 1);
  const stepY = (MAX_Y - MIN_Y) / (GRID_SIZE - 1);
  const vars = new Map<VarName, number>([
    ["x", 0],
    ["y", 0],
  ]);

  const values: number[] = [];
  let idx = 0;
  for (let j = 0; j < GRID_SIZE; j++) {
    vars.set("y", MIN_Y + j * stepY);
    for (let i = 0; i < GRID_SIZE; i++, idx++) {
      vars.set("x", MIN_X + i * stepX);
      values[idx] = fn(vars);
    }
  }
  return values;
}

/** Old path: genericEval via visitFromLeaves (logDebug=true bypasses the cache) */
function evalViaGenericEval(): number[] {
  return evalGrid((vars) => simpleEval(simplifiedRoot, vars, true));
}

/** Routine with a pinned backend. */
function evalViaBackend(backend: BackendName): () => number[] {
  return () => {
    const routine = compileValueRoutine([simplifiedRoot], {
      backend,
    }) as ValueRoutine;
    return evalGrid((vars) => routine.eval(vars));
  };
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

console.log(
  `Benchmarking num eval: ${GRID_SIZE}x${GRID_SIZE} grid = ${GRID_SIZE * GRID_SIZE} evaluations`,
);
console.log(`DAG nodes: ${fixture.compressSimplify.dag.nodes.length}`);
console.log(`Warmup: ${WARMUP}, Runs: ${RUNS}\n`);

const benchmarks: [string, () => number[]][] = [
  ["genericEval (old)", evalViaGenericEval],
  ["routine js-interp", evalViaBackend("js-interp")],
  ["routine js-codegen", evalViaBackend("js-codegen")],
  ["routine wasm-interp", evalViaBackend("wasm-interp")],
  ["routine wasm-codegen", evalViaBackend("wasm-codegen")],
];

const stats: BenchStats[] = [];
for (const [name, fn] of benchmarks) {
  console.log(`Benchmarking ${name}...`);
  const stat = runBenchmark(name, fn, { warmup: WARMUP, runs: RUNS });
  stats.push(stat);
  console.log(`  mean=${stat.meanMs.toFixed(3)}ms`);
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

console.log("\n--- Results ---");
console.table(
  stats.map((s) => ({
    name: s.name,
    meanMs: Number(s.meanMs.toFixed(3)),
    medianMs: Number(s.medianMs.toFixed(3)),
    minMs: Number(s.minMs.toFixed(3)),
    maxMs: Number(s.maxMs.toFixed(3)),
    evalsPerSec: Number(((GRID_SIZE * GRID_SIZE * 1000) / s.meanMs).toFixed(0)),
  })),
);

const baseline = stats[0]!.meanMs;
console.log("\n--- Speedup vs genericEval ---");
console.table(
  stats.map((s) => ({
    name: s.name,
    speedup: Number((baseline / s.meanMs).toFixed(2)) + "x",
  })),
);

// Parity check
const reference = evalViaGenericEval();
console.log("\n--- Parity check ---");
for (const [name, fn] of benchmarks.slice(1)) {
  const results = fn();
  let maxDiff = 0;
  for (let i = 0; i < reference.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(reference[i]! - results[i]!));
  }
  console.log(`${name}: max difference = ${maxDiff}`);
}
