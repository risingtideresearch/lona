/**
 * Benchmark: how each backend scales with repeated evaluations.
 *
 * Each batch size gets a FRESH compiled routine so JIT tiering is measured
 * from a cold start every time. This isolates the effect of V8's
 * optimisation tiers at different eval counts.
 *
 * Run:
 *   node --import tsx bench/num-eval-jit-scaling.ts
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
import {
  deserializeNumDAG,
  type SerializedNumDAG,
} from "../src/core/tree-serialization";
import type { VarName } from "../src/core/tree";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Batch sizes: from cold start through JIT tiers
const BATCH_SIZES = [1, 10, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000];

const SHAPES = [
  { name: "gridfinity", file: "gridfinity-3d-numtree.json" },
  { name: "ducktape", file: "ducktape-3d-numtree.json" },
  { name: "logo", file: "logo-3d-numtree.json" },
];

const CPU_BACKENDS: BackendName[] = [
  "js-interp",
  "js-codegen",
  "wasm-interp",
  "wasm-codegen",
];
const GPU_BACKENDS: BackendName[] = ["gpu-interp", "gpu-codegen"];

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

function makeGridVarBatch(
  bounds: Bounds3D,
  totalEvals: number,
): Record<string, number[]> {
  const n = Math.max(2, Math.ceil(Math.cbrt(totalEvals)));
  const stepX = (bounds.x[1] - bounds.x[0]) / (n - 1);
  const stepY = (bounds.y[1] - bounds.y[0]) / (n - 1);
  const stepZ = (bounds.z[1] - bounds.z[0]) / (n - 1);

  const x = new Array<number>(totalEvals);
  const y = new Array<number>(totalEvals);
  const z = new Array<number>(totalEvals);
  let count = 0;
  outer: for (let k = 0; k < n; k++) {
    const zv = bounds.z[0] + k * stepZ;
    for (let j = 0; j < n; j++) {
      const yv = bounds.y[0] + j * stepY;
      for (let i = 0; i < n; i++) {
        x[count] = bounds.x[0] + i * stepX;
        y[count] = yv;
        z[count] = zv;
        if (++count >= totalEvals) break outer;
      }
    }
  }
  return { x, y, z };
}

/** Mutate the first literal in a serialized DAG to defeat caches. */
function cacheBust(dag: SerializedNumDAG): SerializedNumDAG {
  const copy = JSON.parse(JSON.stringify(dag)) as SerializedNumDAG;
  for (const n of copy.nodes) {
    if (n.type === "LIT") {
      n.value += Math.random() * 1e-25;
      break;
    }
  }
  return copy;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const fixturesDir = path.resolve(import.meta.dirname, "fixtures");

async function main() {
  console.log(
    "JIT scaling benchmark — per-eval cost (µs) at increasing batch sizes",
  );
  console.log("Each batch uses a FRESH compiled routine (cold JIT start)\n");
  console.log(`Batch sizes: ${BATCH_SIZES.join(", ")}\n`);

  for (const shape of SHAPES) {
    const raw = fs.readFileSync(path.join(fixturesDir, shape.file), "utf-8");
    const fixture = JSON.parse(raw);
    const dag = fixture.compressSimplify.dag as SerializedNumDAG;
    const bounds: Bounds3D = fixture.bounds;

    console.log(`${"=".repeat(60)}`);
    console.log(`${shape.name} — ${dag.nodes.length} nodes`);
    console.log(`${"=".repeat(60)}`);

    type Row = { batch: number; ms: number; "µs/eval": number };
    const perBackendResults = new Map<BackendName, Row[]>();

    const warmupVars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
      ["z", 0],
    ]);

    // --- CPU backends (via sync compileValueRoutine) ---
    for (const backend of CPU_BACKENDS) {
      const rows: Row[] = [];
      perBackendResults.set(backend, rows);

      // Cold compile once (with cache-busted DAG), reused for all batches.
      const coldRoot = deserializeNumDAG(cacheBust(dag));
      const t0 = performance.now();
      const routine = compileValueRoutine([coldRoot], {
        backend,
      }) as ValueRoutine | null;
      if (!routine) {
        console.log(`\n  ${backend}: not available`);
        for (const batch of BATCH_SIZES)
          rows.push({ batch, ms: 0, "µs/eval": 0 });
        continue;
      }
      routine.eval(warmupVars); // force tiering
      const coldMs = performance.now() - t0;
      console.log(`\n  ${backend} (cold compile: ${coldMs.toFixed(1)}ms):`);

      for (const batch of BATCH_SIZES) {
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
            routine.eval(vars);
          },
          bounds,
          batch,
        );
        const totalMs = coldMs + evalMs;
        const usPerEval = (totalMs / batch) * 1000;
        rows.push({
          batch,
          ms: Number(totalMs.toFixed(2)),
          "µs/eval": Number(usPerEval.toFixed(2)),
        });
        console.log(
          `    ${String(batch).padStart(9)} evals → ${evalMs.toFixed(1).padStart(7)}ms eval + ${coldMs.toFixed(1)}ms compile = ${totalMs.toFixed(1).padStart(7)}ms  (${usPerEval.toFixed(2)} µs/eval)`,
        );
      }
    }

    // --- GPU backends ---
    for (const backend of GPU_BACKENDS) {
      const rows: Row[] = [];
      perBackendResults.set(backend, rows);

      const coldRoot = deserializeNumDAG(cacheBust(dag));
      const t0 = performance.now();
      let routine: ValueRoutine | null = null;
      try {
        routine = compileValueRoutine([coldRoot], { backend }) as ValueRoutine;
      } catch {
        /* GPU not available */
      }
      if (!routine) {
        console.log(`\n  ${backend}: not available`);
        for (const batch of BATCH_SIZES)
          rows.push({ batch, ms: 0, "µs/eval": 0 });
        continue;
      }
      // Warmup: batch-of-1 to force any lazy compilation.
      await routine.evalBatch(makeGridVarBatch(bounds, 1), 1);
      const coldMs = performance.now() - t0;
      console.log(`\n  ${backend} (cold compile: ${coldMs.toFixed(1)}ms):`);

      for (const batch of BATCH_SIZES) {
        const vars = makeGridVarBatch(bounds, batch);
        const bt0 = performance.now();
        await routine.evalBatch(vars, batch);
        const evalMs = performance.now() - bt0;
        const totalMs = coldMs + evalMs;
        const usPerEval = (totalMs / batch) * 1000;
        rows.push({
          batch,
          ms: Number(totalMs.toFixed(2)),
          "µs/eval": Number(usPerEval.toFixed(2)),
        });
        console.log(
          `    ${String(batch).padStart(9)} evals → ${evalMs.toFixed(1).padStart(7)}ms eval + ${coldMs.toFixed(1)}ms compile = ${totalMs.toFixed(1).padStart(7)}ms  (${usPerEval.toFixed(2)} µs/eval)`,
        );
      }
      routine.dispose?.();
    }

    // Summary table
    console.log(`\n  --- ${shape.name}: side-by-side µs/eval ---`);
    console.table(
      BATCH_SIZES.map((batch, i) => {
        const row: Record<string, number> = { evals: batch };
        for (const b of [...CPU_BACKENDS, ...GPU_BACKENDS]) {
          row[b] = perBackendResults.get(b)![i]!["µs/eval"];
        }
        return row;
      }),
    );
    console.log();
  }

  await destroyGpu();
}

main().catch(console.error);
