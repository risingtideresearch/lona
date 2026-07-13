/**
 * GPU vs CPU benchmark for 3D Num tree evaluation.
 *
 * Compares WebGPU compute shader batch evaluation against CPU strategies
 * via the routine API with a pinned backend.
 *
 * Run:
 *   node --import tsx bench/num-eval-3d-gpu.ts
 *
 * Env vars:
 *   BENCH_GRID_SIZE  — points per axis (default 30, so 30^3 = 27000 evals)
 *   BENCH_WARMUP     — warmup iterations (default 2)
 *   BENCH_RUNS       — timed iterations (default 5)
 */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import {
  compileValueRoutine,
  destroyGpu,
  type BackendName,
  type ValueRoutine,
} from "../src/eval/routines";
import { deserializeNumDAG } from "../src/core/tree-serialization";
import type { VarName } from "../src/core/tree";

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

const CPU_BACKENDS: BackendName[] = [
  "js-interp",
  "wasm-codegen",
  "wasm-interp",
];
const GPU_BACKENDS: BackendName[] = ["gpu-interp", "gpu-codegen"];

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

async function runBenchmarkAsync(
  name: string,
  fn: () => Promise<void>,
  warmup: number,
  runs: number,
): Promise<{ name: string; medianMs: number }> {
  for (let i = 0; i < warmup; i++) await fn();
  const samplesMs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await fn();
    samplesMs.push(performance.now() - t0);
  }
  return { name, medianMs: median(samplesMs) };
}

function runBenchmarkSync(
  name: string,
  fn: () => void,
  warmup: number,
  runs: number,
): { name: string; medianMs: number } {
  for (let i = 0; i < warmup; i++) fn();
  const samplesMs: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    samplesMs.push(performance.now() - t0);
  }
  return { name, medianMs: median(samplesMs) };
}

function makeGridVarBatch(
  bounds: Bounds3D,
  gridSize: number,
): { vars: Record<string, number[]>; numPoints: number } {
  const numPoints = gridSize ** 3;
  const stepX = (bounds.x[1] - bounds.x[0]) / (gridSize - 1);
  const stepY = (bounds.y[1] - bounds.y[0]) / (gridSize - 1);
  const stepZ = (bounds.z[1] - bounds.z[0]) / (gridSize - 1);
  const x = new Array<number>(numPoints);
  const y = new Array<number>(numPoints);
  const z = new Array<number>(numPoints);
  let p = 0;
  for (let k = 0; k < gridSize; k++) {
    const zv = bounds.z[0] + k * stepZ;
    for (let j = 0; j < gridSize; j++) {
      const yv = bounds.y[0] + j * stepY;
      for (let i = 0; i < gridSize; i++) {
        x[p] = bounds.x[0] + i * stepX;
        y[p] = yv;
        z[p] = zv;
        p++;
      }
    }
  }
  return { vars: { x, y, z }, numPoints };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const totalEvals = GRID_SIZE ** 3;

async function main() {
  console.log(
    `GPU vs CPU benchmark: ${GRID_SIZE}^3 = ${totalEvals} points per shape`,
  );
  console.log(`Warmup: ${WARMUP}, Runs: ${RUNS}\n`);

  const fixturesDir = path.resolve(import.meta.dirname, "fixtures");

  for (const shape of SHAPES) {
    const raw = fs.readFileSync(path.join(fixturesDir, shape.file), "utf-8");
    const fixture = JSON.parse(raw);
    const dag = fixture.compressSimplify.dag;
    const bounds: Bounds3D = fixture.bounds;
    const root = deserializeNumDAG(dag);

    console.log(`${"=".repeat(60)}`);
    console.log(`${shape.name} — ${dag.nodes.length} DAG nodes`);
    console.log(`${"=".repeat(60)}`);

    const stepX = (bounds.x[1] - bounds.x[0]) / (GRID_SIZE - 1);
    const stepY = (bounds.y[1] - bounds.y[0]) / (GRID_SIZE - 1);
    const stepZ = (bounds.z[1] - bounds.z[0]) / (GRID_SIZE - 1);

    const stats: { name: string; medianMs: number }[] = [];

    // --- CPU: routines with pinned backend ---
    for (const backend of CPU_BACKENDS) {
      const routine = compileValueRoutine([root], {
        backend,
      }) as ValueRoutine | null;
      if (!routine) {
        console.log(`  ${backend}: not available`);
        continue;
      }
      const vars = new Map<VarName, number>([
        ["x", 0],
        ["y", 0],
        ["z", 0],
      ]);
      const loop = () => {
        for (let k = 0; k < GRID_SIZE; k++) {
          vars.set("z", bounds.z[0] + k * stepZ);
          for (let j = 0; j < GRID_SIZE; j++) {
            vars.set("y", bounds.y[0] + j * stepY);
            for (let i = 0; i < GRID_SIZE; i++) {
              vars.set("x", bounds.x[0] + i * stepX);
              routine.eval(vars);
            }
          }
        }
      };
      process.stdout.write(`  ${backend}...`);
      const stat = runBenchmarkSync(backend, loop, WARMUP, RUNS);
      stats.push(stat);
      console.log(` ${stat.medianMs.toFixed(1)}ms`);
    }

    // --- GPU: routine ---
    for (const backend of GPU_BACKENDS) {
      let routine: ValueRoutine | null = null;
      try {
        routine = compileValueRoutine([root], { backend }) as ValueRoutine;
      } catch {
        /* GPU not available */
      }
      if (!routine) {
        console.log(`  ${backend}: not available`);
        continue;
      }
      const { vars, numPoints } = makeGridVarBatch(bounds, GRID_SIZE);
      process.stdout.write(`  ${backend}...`);
      const stat = await runBenchmarkAsync(
        backend,
        () => routine.evalBatch(vars, numPoints).then(() => {}),
        WARMUP,
        RUNS,
      );
      stats.push(stat);
      console.log(` ${stat.medianMs.toFixed(1)}ms`);
      routine.dispose?.();
    }

    // Results
    const baseMs = stats[0]?.medianMs ?? 1;
    console.log(
      `\n  --- ${shape.name}: total time for ${totalEvals} points ---`,
    );
    console.table(
      stats.map((s) => ({
        strategy: s.name,
        totalMs: Number(s.medianMs.toFixed(1)),
        "µs/eval": Number(((s.medianMs / totalEvals) * 1000).toFixed(2)),
        speedup: (baseMs / s.medianMs).toFixed(1) + "x",
      })),
    );

    console.log();
  }

  await destroyGpu();
}

main().catch(console.error);
