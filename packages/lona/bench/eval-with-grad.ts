/**
 * Benchmark for evalWithGrad — value + gradient evaluation on the
 * gridfinity 3D NumNode tree.
 *
 * Compares every grad backend (AD and symbolic) via the routine abstraction.
 *
 * Run:
 *   node --import tsx bench/eval-with-grad.ts
 *
 * Env vars:
 *   BENCH_WARMUP — warmup iterations (default 2)
 *   BENCH_RUNS   — timed iterations (default 5)
 */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { deserializeNumDAG } from "../src/core/tree-serialization";
import type { NumNode, VarName } from "../src/core/tree";
import {
  compileGradRoutine,
  initGpu,
  destroyGpu,
  type BackendName,
  type GradRoutine,
} from "../src/eval/routines";

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

const WARMUP = envNumber("BENCH_WARMUP", 2);
const RUNS = envNumber("BENCH_RUNS", 5);
const POINT_COUNTS = [1, 10, 100, 1_000, 5_000, 10_000];

const VAR_NAMES: VarName[] = ["x" as VarName, "y" as VarName, "z" as VarName];

type Bounds3D = {
  x: [number, number];
  y: [number, number];
  z: [number, number];
};

// ---------------------------------------------------------------------------
// Every grad backend to bench
// ---------------------------------------------------------------------------

const AD_BACKENDS: BackendName[] = [
  "js-interp",
  "wasm-interp",
  "wasm-codegen",
  "gpu-interp",
];

const SYM_BACKENDS: BackendName[] = [
  "js-codegen-sym",
  "wasm-interp-sym",
  "wasm-codegen-sym",
  "gpu-codegen-sym",
];

const ALL_BACKENDS = [...AD_BACKENDS, ...SYM_BACKENDS];

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1]! + sorted[mid]!) / 2
    : sorted[mid]!;
}

type BenchResult = { name: string; totalMs: number; perEvalUs: number };

async function runBenchmark(
  name: string,
  fn: () => void | Promise<void>,
  numPoints: number,
  warmup: number,
  runs: number,
): Promise<BenchResult> {
  for (let i = 0; i < warmup; i++) await fn();
  const samplesMs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await fn();
    samplesMs.push(performance.now() - t0);
  }
  const totalMs = median(samplesMs);
  return { name, totalMs, perEvalUs: (totalMs / numPoints) * 1000 };
}

// ---------------------------------------------------------------------------
// Grid generation
// ---------------------------------------------------------------------------

function makeGridPoints(
  bounds: Bounds3D,
  numPoints: number,
): { xs: number[]; ys: number[]; zs: number[] } {
  const n = Math.max(2, Math.ceil(Math.cbrt(numPoints)));
  const stepX = (bounds.x[1] - bounds.x[0]) / (n - 1);
  const stepY = (bounds.y[1] - bounds.y[0]) / (n - 1);
  const stepZ = (bounds.z[1] - bounds.z[0]) / (n - 1);

  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  for (let k = 0; k < n; k++) {
    const z = bounds.z[0] + k * stepZ;
    for (let j = 0; j < n; j++) {
      const y = bounds.y[0] + j * stepY;
      for (let i = 0; i < n; i++) {
        xs.push(bounds.x[0] + i * stepX);
        ys.push(y);
        zs.push(z);
        if (xs.length >= numPoints) return { xs, ys, zs };
      }
    }
  }
  return { xs, ys, zs };
}

// ---------------------------------------------------------------------------
// Strategy types
// ---------------------------------------------------------------------------

type Strategy = {
  name: string;
  evalBatch: (xs: number[], ys: number[], zs: number[]) => void | Promise<void>;
  compileMs: number;
  dispose?: () => void;
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

(async () => {
  console.log(`evalWithGrad benchmark (gridfinity 3D)`);
  console.log(`  Warmup: ${WARMUP}, Runs: ${RUNS}`);
  console.log(`  Point counts: ${POINT_COUNTS.join(", ")}`);
  console.log(
    `  Backends: ${ALL_BACKENDS.length} (${AD_BACKENDS.length} AD + ${SYM_BACKENDS.length} symbolic)\n`,
  );

  const fixturesDir = path.resolve(import.meta.dirname, "fixtures");
  const raw = fs.readFileSync(
    path.join(fixturesDir, "gridfinity-3d-numtree.json"),
    "utf-8",
  );
  const fixture = JSON.parse(raw);
  const dag = fixture.compressSimplify.dag;
  const bounds: Bounds3D = fixture.bounds;
  const root = deserializeNumDAG(dag);

  console.log(`DAG: ${dag.nodes.length} nodes`);
  console.log(`Bounds: x=[${bounds.x}] y=[${bounds.y}] z=[${bounds.z}]`);

  // Pre-initialize GPU device so compilation can be synchronous.
  const gpuAvailable = !!(await initGpu());
  if (!gpuAvailable)
    console.log("  GPU not available — GPU backends will be skipped");

  // ---------------------------------------------------------------------------
  // Compile every backend
  // ---------------------------------------------------------------------------

  const compiled: Strategy[] = [];

  console.log(`\n--- Compilation ---`);
  for (const backend of ALL_BACKENDS) {
    const t0 = performance.now();
    let routine: GradRoutine | null = null;
    try {
      routine = compileGradRoutine([root], VAR_NAMES, {
        backend,
      }) as GradRoutine | null;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  ${backend}: FAILED — ${msg.slice(0, 80)}`);
      continue;
    }
    const compileMs = performance.now() - t0;

    if (!routine) {
      console.log(`  ${backend}: not available`);
      continue;
    }

    // Force JIT / warm the routine
    try {
      await routine.evalAsync(
        new Map<VarName, number>([
          ["x" as VarName, 0],
          ["y" as VarName, 0],
          ["z" as VarName, 0],
        ]),
      );
    } catch {
      // GPU routines might fail evalAsync on batch-of-1 if GPU unavailable
    }

    const isGpuBackend = backend.startsWith("gpu-");
    const vars = new Map<VarName, number>();

    let evalBatch: Strategy["evalBatch"];
    if (isGpuBackend) {
      // GPU: use evalBatch for batch dispatch
      evalBatch = async (xs, ys, zs) => {
        await routine!.evalBatch({ x: xs, y: ys, z: zs }, xs.length);
      };
    } else {
      // CPU: loop single-point eval
      evalBatch = (xs, ys, zs) => {
        for (let p = 0; p < xs.length; p++) {
          vars.set("x" as VarName, xs[p]!);
          vars.set("y" as VarName, ys[p]!);
          vars.set("z" as VarName, zs[p]!);
          routine!.eval(vars);
        }
      };
    }

    compiled.push({
      name: backend,
      compileMs,
      evalBatch,
      dispose: routine.dispose,
    });
    console.log(`  ${backend}: ${compileMs.toFixed(1)}ms`);
  }

  // ---------------------------------------------------------------------------
  // Evaluation benchmarks
  // ---------------------------------------------------------------------------

  for (const numPoints of POINT_COUNTS) {
    const { xs, ys, zs } = makeGridPoints(bounds, numPoints);
    const actualPoints = xs.length;

    console.log(`\n${"=".repeat(60)}`);
    console.log(`${actualPoints} points (requested ${numPoints})`);
    console.log(`${"=".repeat(60)}`);

    const results: BenchResult[] = [];

    for (const strategy of compiled) {
      process.stdout.write(`  ${strategy.name}...`);
      const stat = await runBenchmark(
        strategy.name,
        () => strategy.evalBatch(xs, ys, zs),
        actualPoints,
        WARMUP,
        RUNS,
      );
      results.push(stat);
      console.log(
        ` ${stat.totalMs.toFixed(1)}ms total, ${stat.perEvalUs.toFixed(2)}µs/eval`,
      );
    }

    if (results.length > 0) {
      const baseline = results[0]!;
      const baselineCompile = compiled[0]!.compileMs;
      console.log(
        `\n  --- per-eval cost (compile + ${actualPoints} evals) ---`,
      );
      console.table(
        results.map((s) => {
          const strategy = compiled.find((c) => c.name === s.name)!;
          const total = strategy.compileMs + s.totalMs;
          return {
            strategy: s.name,
            "compile ms": Number(strategy.compileMs.toFixed(1)),
            "eval ms": Number(s.totalMs.toFixed(1)),
            "total ms": Number(total.toFixed(1)),
            "µs/eval": Number(s.perEvalUs.toFixed(2)),
            "vs_first (eval)":
              s === baseline
                ? "baseline"
                : (baseline.perEvalUs / s.perEvalUs).toFixed(2) + "x",
            "vs_first (total)":
              s === baseline
                ? "baseline"
                : ((baselineCompile + baseline.totalMs) / total).toFixed(2) +
                  "x",
          };
        }),
      );
    }
  }

  // Cleanup
  for (const strategy of compiled) strategy.dispose?.();
  await destroyGpu();
})();
