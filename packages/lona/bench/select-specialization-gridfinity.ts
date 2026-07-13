/**
 * Benchmark: gridfinity 3D level-set value routines with and without
 * selectSpecialization:"trace" and selectSpecialization:"full-trace".
 *
 * Uses the select-preserving fixture extracted from fonseranes. Unlike the
 * general num-eval benchmark, this intentionally does not include genericEval:
 * it compares routine backends only.
 *
 * Run:
 *   cd packages/lona
 *   node --import tsx bench/select-specialization-gridfinity.ts
 *
 * Optional env:
 *   BENCH_EVAL_COUNTS=5000,30000
 *   BENCH_BACKENDS=wasm-codegen,wasm-interp,js-codegen,js-interp
 *
 * Variants:
 *   normal                         no select specialization
 *   selectSpecialization:trace     dynamic selected-path trace
 *   selectSpecialization:full-trace old full-tape trace
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

type Bounds3D = {
  x: [number, number];
  y: [number, number];
  z: [number, number];
};

type Fixture = {
  description: string;
  bounds: Bounds3D;
  select: { dag: SerializedNumDAG };
};

type VariantName =
  "normal" | "selectSpecialization:trace" | "selectSpecialization:full-trace";

type Strategy = {
  backend: BackendName;
  variant: VariantName;
};

type Row = {
  backend: BackendName;
  variant: VariantName;
  evals: number;
  nodes: number;
  compileMs: number;
  warmupMs: number;
  evalMs: number;
  totalMs: number;
  "µs/eval": number;
  maxDiff: number | null;
};

const DEFAULT_EVAL_COUNTS = [5_000, 30_000];
const DEFAULT_BACKENDS: BackendName[] = [
  "wasm-codegen",
  "wasm-interp",
  "js-codegen",
  "js-interp",
];

function envList(name: string): string[] | null {
  const raw = process.env[name];
  if (!raw) return null;
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function evalCounts(): number[] {
  const raw = envList("BENCH_EVAL_COUNTS");
  if (!raw) return DEFAULT_EVAL_COUNTS;
  const counts = raw.map((s) => Number(s));
  for (const count of counts) {
    if (!Number.isInteger(count) || count <= 0) {
      throw new Error(`Invalid BENCH_EVAL_COUNTS entry: ${count}`);
    }
  }
  return counts;
}

function backends(): BackendName[] {
  return (
    (envList("BENCH_BACKENDS") as BackendName[] | null) ?? DEFAULT_BACKENDS
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fixturePath(filename: string): string {
  return path.resolve(import.meta.dirname, "fixtures", filename);
}

function timeGrid(
  evalPoint: (x: number, y: number, z: number) => number,
  bounds: Bounds3D,
  totalEvals: number,
): { ms: number; values: number[] } {
  const n = Math.max(2, Math.ceil(Math.cbrt(totalEvals)));
  const stepX = (bounds.x[1] - bounds.x[0]) / (n - 1);
  const stepY = (bounds.y[1] - bounds.y[0]) / (n - 1);
  const stepZ = (bounds.z[1] - bounds.z[0]) / (n - 1);

  const values = new Array<number>(totalEvals);
  let count = 0;
  const t0 = performance.now();
  outer: for (let k = 0; k < n; k++) {
    const zv = bounds.z[0] + k * stepZ;
    for (let j = 0; j < n; j++) {
      const yv = bounds.y[0] + j * stepY;
      for (let i = 0; i < n; i++) {
        values[count] = evalPoint(bounds.x[0] + i * stepX, yv, zv);
        if (++count >= totalEvals) break outer;
      }
    }
  }
  return { ms: performance.now() - t0, values };
}

function compileStrategy(
  root: NumNode,
  strategy: Strategy,
): { routine: ValueRoutine; compileMs: number } {
  const t0 = performance.now();
  const routine = compileValueRoutine([root], {
    backend: strategy.backend,
    selectSpecialization:
      strategy.variant === "selectSpecialization:trace"
        ? "trace"
        : strategy.variant === "selectSpecialization:full-trace"
          ? "full-trace"
          : false,
  }) as ValueRoutine;
  return { routine, compileMs: performance.now() - t0 };
}

function evalRoutineGrid(
  routine: ValueRoutine,
  bounds: Bounds3D,
  totalEvals: number,
): { ms: number; values: number[] } {
  const vars = new Map<VarName, number>([
    ["x", 0],
    ["y", 0],
    ["z", 0],
  ]);
  return timeGrid(
    (x, y, z) => {
      vars.set("x", x);
      vars.set("y", y);
      vars.set("z", z);
      return routine.eval(vars);
    },
    bounds,
    totalEvals,
  );
}

function maxAbsDiff(a: number[], b: number[]): number {
  let max = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    max = Math.max(max, Math.abs(a[i]! - b[i]!));
  }
  return max;
}

function fmt(n: number): string {
  return Number.isFinite(n) ? n.toFixed(2) : "NaN";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const fixture = JSON.parse(
  fs.readFileSync(fixturePath("gridfinity-3d-select-numtree.json"), "utf-8"),
) as Fixture;
const root = deserializeNumDAG(fixture.select.dag);
const counts = evalCounts();
const selectedBackends = backends();
const strategies: Strategy[] = selectedBackends.flatMap((backend) => [
  { backend, variant: "normal" as const },
  { backend, variant: "selectSpecialization:trace" as const },
  { backend, variant: "selectSpecialization:full-trace" as const },
]);
console.log(
  "Gridfinity select-specialization benchmark — value routines only\n",
);
console.log(`  fixture: ${fixture.description}`);
console.log(`  DAG nodes: ${fixture.select.dag.nodes.length}`);
console.log(`  eval counts: ${counts.join(", ")}`);
console.log(`  backends: ${selectedBackends.join(", ")}\n`);

const rows: Row[] = [];
const references = new Map<string, number[]>();

for (const strategy of strategies) {
  console.log(`${"=".repeat(72)}`);
  console.log(`${strategy.backend} / ${strategy.variant}`);
  console.log(`${"=".repeat(72)}`);

  let routine: ValueRoutine;
  let compileMs: number;
  try {
    ({ routine, compileMs } = compileStrategy(root, strategy));
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`  SKIPPED compile — ${msg}\n`);
    for (const evals of counts) {
      rows.push({
        backend: strategy.backend,
        variant: strategy.variant,
        evals,
        nodes: fixture.select.dag.nodes.length,
        compileMs: NaN,
        warmupMs: NaN,
        evalMs: NaN,
        totalMs: NaN,
        "µs/eval": NaN,
        maxDiff: null,
      });
    }
    continue;
  }

  // Warm one representative point. For selectSpecialization this pays the
  // first lazy trace/compile while still leaving grid-region retraces inside
  // the measured eval pass.
  const warmVars = new Map<VarName, number>([
    ["x", 0],
    ["y", 0],
    ["z", 10],
  ]);
  const warmupT0 = performance.now();
  routine.eval(warmVars);
  const warmupMs = performance.now() - warmupT0;

  console.log(
    `  compile=${fmt(compileMs)}ms, warmup=${fmt(warmupMs)}ms (1 eval)`,
  );

  for (const evals of counts) {
    const run = evalRoutineGrid(routine, fixture.bounds, evals);
    const referenceKey = `${strategy.backend}:${evals}`;
    if (strategy.variant === "normal") {
      references.set(referenceKey, run.values);
    }
    const reference = references.get(referenceKey);
    const maxDiff = reference ? maxAbsDiff(reference, run.values) : null;
    const totalMs = compileMs + warmupMs + run.ms;
    const usPerEval = (totalMs / evals) * 1000;
    rows.push({
      backend: strategy.backend,
      variant: strategy.variant,
      evals,
      nodes: fixture.select.dag.nodes.length,
      compileMs: Number(compileMs.toFixed(2)),
      warmupMs: Number(warmupMs.toFixed(2)),
      evalMs: Number(run.ms.toFixed(2)),
      totalMs: Number(totalMs.toFixed(2)),
      "µs/eval": Number(usPerEval.toFixed(2)),
      maxDiff: maxDiff === null ? null : Number(maxDiff.toExponential(3)),
    });
    console.log(
      `  ${String(evals).padStart(6)} evals → compile ${fmt(compileMs).padStart(8)}ms + warmup ${fmt(warmupMs).padStart(8)}ms + eval ${fmt(run.ms).padStart(8)}ms = ${fmt(totalMs).padStart(8)}ms (${usPerEval.toFixed(2)} µs/eval)`,
    );
  }
  routine.dispose?.();
  console.log();
}

console.log("--- Summary ---");
console.table(rows);

console.log(
  "\n--- selectSpecialization variants vs normal (same backend/evals) ---",
);
const comparison: Array<{
  backend: BackendName;
  variant: Exclude<VariantName, "normal">;
  evals: number;
  normalTotalMs: number;
  specializedTotalMs: number;
  speedup: string;
  evalOnlySpeedup: string;
  maxDiff: number | null;
}> = [];
for (const backend of selectedBackends) {
  for (const evals of counts) {
    const normal = rows.find(
      (r) =>
        r.backend === backend && r.variant === "normal" && r.evals === evals,
    );
    if (!normal) continue;
    for (const variant of [
      "selectSpecialization:trace",
      "selectSpecialization:full-trace",
    ] as const) {
      const specialized = rows.find(
        (r) =>
          r.backend === backend && r.variant === variant && r.evals === evals,
      );
      if (!specialized) continue;
      comparison.push({
        backend,
        variant,
        evals,
        normalTotalMs: normal.totalMs,
        specializedTotalMs: specialized.totalMs,
        speedup: `${Number((normal.totalMs / specialized.totalMs).toFixed(2))}x`,
        evalOnlySpeedup: `${Number((normal.evalMs / specialized.evalMs).toFixed(2))}x`,
        maxDiff: specialized.maxDiff,
      });
    }
  }
}
console.table(comparison);
