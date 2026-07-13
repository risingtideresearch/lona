/**
 * Benchmark: gridfinity 3D level-set value routines at one fixed point, with
 * and without selectSpecialization:"trace" and "full-trace".
 *
 * This benchmark intentionally warms the selected point first, so the measured
 * eval pass shows steady-state reuse of one guarded specialized routine. It is
 * the complement of select-specialization-gridfinity.ts, which sweeps many grid
 * regions and therefore measures retrace/recompile costs.
 *
 * Run:
 *   cd packages/lona
 *   node --import tsx bench/select-specialization-fixed-point.ts
 *
 * Optional env:
 *   BENCH_EVAL_COUNTS=5000,30000,100000
 *   BENCH_BACKENDS=wasm-codegen,wasm-interp,js-codegen,js-interp
 *   BENCH_POINT=0,0,10
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

type Point3D = { x: number; y: number; z: number };

type Row = {
  backend: BackendName;
  variant: VariantName;
  evals: number;
  point: string;
  nodes: number;
  compileMs: number;
  warmupMs: number;
  evalMs: number;
  totalMs: number;
  "µs/eval": number;
  value: number;
  maxDiff: number | null;
};

const DEFAULT_EVAL_COUNTS = [5_000, 30_000, 100_000];
const DEFAULT_BACKENDS: BackendName[] = [
  "wasm-codegen",
  "wasm-interp",
  "js-codegen",
  "js-interp",
];
const DEFAULT_POINT: Point3D = { x: 0, y: 0, z: 10 };

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

function fixedPoint(): Point3D {
  const raw = envList("BENCH_POINT");
  if (!raw) return DEFAULT_POINT;
  if (raw.length !== 3) {
    throw new Error("BENCH_POINT must have the form x,y,z");
  }
  const [x, y, z] = raw.map((s) => Number(s));
  if (![x, y, z].every(Number.isFinite)) {
    throw new Error(`Invalid BENCH_POINT: ${raw.join(",")}`);
  }
  return { x: x!, y: y!, z: z! };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fixturePath(filename: string): string {
  return path.resolve(import.meta.dirname, "fixtures", filename);
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

function varsForPoint(point: Point3D): Map<VarName, number> {
  return new Map<VarName, number>([
    ["x", point.x],
    ["y", point.y],
    ["z", point.z],
  ]);
}

function timeFixedPoint(
  routine: ValueRoutine,
  vars: Map<VarName, number>,
  totalEvals: number,
): { ms: number; value: number } {
  let value = NaN;
  const t0 = performance.now();
  for (let i = 0; i < totalEvals; i++) {
    value = routine.eval(vars);
  }
  return { ms: performance.now() - t0, value };
}

function fmt(n: number): string {
  return Number.isFinite(n) ? n.toFixed(2) : "NaN";
}

function fmtPoint(point: Point3D): string {
  return `(${point.x}, ${point.y}, ${point.z})`;
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
const point = fixedPoint();
const pointVars = varsForPoint(point);
const strategies: Strategy[] = selectedBackends.flatMap((backend) => [
  { backend, variant: "normal" as const },
  { backend, variant: "selectSpecialization:trace" as const },
  { backend, variant: "selectSpecialization:full-trace" as const },
]);

console.log(
  "Gridfinity select-specialization benchmark — fixed point steady state\n",
);
console.log(`  fixture: ${fixture.description}`);
console.log(`  DAG nodes: ${fixture.select.dag.nodes.length}`);
console.log(`  point: ${fmtPoint(point)}`);
console.log(`  eval counts: ${counts.join(", ")}`);
console.log(`  backends: ${selectedBackends.join(", ")}\n`);

const rows: Row[] = [];
const references = new Map<string, number>();

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
        point: fmtPoint(point),
        nodes: fixture.select.dag.nodes.length,
        compileMs: NaN,
        warmupMs: NaN,
        evalMs: NaN,
        totalMs: NaN,
        "µs/eval": NaN,
        value: NaN,
        maxDiff: null,
      });
    }
    continue;
  }

  // For selectSpecialization this pays the first lazy trace/compile at the
  // same point that will be repeatedly evaluated below. The measured eval pass
  // should therefore be guard-hit steady state, not retrace/recompile time.
  const warmupT0 = performance.now();
  routine.eval(pointVars);
  const warmupMs = performance.now() - warmupT0;

  console.log(
    `  compile=${fmt(compileMs)}ms, warmup=${fmt(warmupMs)}ms (1 eval at fixed point)`,
  );

  for (const evals of counts) {
    const run = timeFixedPoint(routine, pointVars, evals);
    const referenceKey = `${strategy.backend}:${evals}`;
    if (strategy.variant === "normal") {
      references.set(referenceKey, run.value);
    }
    const reference = references.get(referenceKey);
    const maxDiff =
      reference === undefined ? null : Math.abs(reference - run.value);
    const totalMs = compileMs + warmupMs + run.ms;
    const usPerEval = (totalMs / evals) * 1000;
    rows.push({
      backend: strategy.backend,
      variant: strategy.variant,
      evals,
      point: fmtPoint(point),
      nodes: fixture.select.dag.nodes.length,
      compileMs: Number(compileMs.toFixed(2)),
      warmupMs: Number(warmupMs.toFixed(2)),
      evalMs: Number(run.ms.toFixed(2)),
      totalMs: Number(totalMs.toFixed(2)),
      "µs/eval": Number(usPerEval.toFixed(2)),
      value: Number(run.value.toFixed(6)),
      maxDiff: maxDiff === null ? null : Number(maxDiff.toExponential(3)),
    });
    console.log(
      `  ${String(evals).padStart(7)} evals → compile ${fmt(compileMs).padStart(8)}ms + warmup ${fmt(warmupMs).padStart(8)}ms + eval ${fmt(run.ms).padStart(8)}ms = ${fmt(totalMs).padStart(8)}ms (${usPerEval.toFixed(2)} µs/eval)`,
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
  normalEvalMs: number;
  specializedEvalMs: number;
  evalOnlySpeedup: string;
  normalTotalMs: number;
  specializedTotalMs: number;
  totalSpeedup: string;
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
        normalEvalMs: normal.evalMs,
        specializedEvalMs: specialized.evalMs,
        evalOnlySpeedup: `${Number((normal.evalMs / specialized.evalMs).toFixed(2))}x`,
        normalTotalMs: normal.totalMs,
        specializedTotalMs: specialized.totalMs,
        totalSpeedup: `${Number((normal.totalMs / specialized.totalMs).toFixed(2))}x`,
        maxDiff: specialized.maxDiff,
      });
    }
  }
}
console.table(comparison);
