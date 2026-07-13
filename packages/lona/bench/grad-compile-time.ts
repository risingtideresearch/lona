/**
 * Benchmark: compilation time for different gradient strategies on
 * the gridfinity 3D NumNode tree.
 *
 * Run:
 *   node --import tsx bench/grad-compile-time.ts
 */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { deserializeNumDAG } from "../src/core/tree-serialization";
import type { VarName } from "../src/core/tree";
import { partialDerivative } from "../src/core/tree-walks";
import { fullDerivative } from "../src/api/diff";
import {
  compileGradRoutine,
  compileValueRoutine,
  type BackendName,
  type GradRoutine,
  type MultiValueRoutine,
} from "../src/eval/routines";

const VAR_NAMES: VarName[] = ["x" as VarName, "y" as VarName, "z" as VarName];

// Load fixture
const fixturesDir = path.resolve(import.meta.dirname, "fixtures");
const fixture = JSON.parse(
  fs.readFileSync(
    path.join(fixturesDir, "gridfinity-3d-numtree.json"),
    "utf-8",
  ),
);
const dag = fixture.compressSimplify.dag;
const root = deserializeNumDAG(dag);

console.log(`Gradient compile-time benchmark (gridfinity 3D)`);
console.log(`DAG: ${dag.nodes.length} nodes\n`);

function time(label: string, fn: () => void): number {
  const t0 = performance.now();
  fn();
  const ms = performance.now() - t0;
  console.log(`  ${label}: ${ms.toFixed(1)}ms`);
  return ms;
}

// ---------------------------------------------------------------------------
// 1. Symbolic: fullDerivative + partialDerivative → multi-value routine
// ---------------------------------------------------------------------------
console.log("Symbolic (fullDerivative → compileValueRoutine wasm-codegen):");

let derivNodes: ReturnType<typeof partialDerivative>[] = [];
const derivMs = time("  fullDerivative + partialDerivative", () => {
  const diffRoot = fullDerivative(root);
  derivNodes = VAR_NAMES.map((v) => partialDerivative(diffRoot, v));
});

const allNodes = [root, ...derivNodes];
let symbolicRoutine: MultiValueRoutine | null = null;
const symbolicCompileMs = time(
  "  compileValueRoutine wasm-codegen (value + 3 derivs)",
  () => {
    symbolicRoutine = compileValueRoutine(allNodes, {
      backend: "wasm-codegen",
    }) as MultiValueRoutine | null;
  },
);
symbolicRoutine!.eval(new Map());
const symbolicJitMs = time("  Turbofan JIT (first call)", () => {
  symbolicRoutine!.eval(new Map());
});
console.log(
  `  TOTAL: ${(derivMs + symbolicCompileMs + symbolicJitMs).toFixed(1)}ms\n`,
);

// ---------------------------------------------------------------------------
// 2 / 3 / 4. Forward autodiff via compileGradRoutine (pick backend)
// ---------------------------------------------------------------------------
const GRAD_BACKENDS: { label: string; backend: BackendName }[] = [
  { label: "JS forward autodiff (tape)", backend: "js-interp" },
  { label: "WASM codegen grad (per-DAG)", backend: "wasm-codegen" },
];

for (const { label, backend } of GRAD_BACKENDS) {
  console.log(`${label} (routine backend=${backend}):`);
  let routine: GradRoutine | null = null;
  const compileMs = time("  compileGradRoutine", () => {
    routine = compileGradRoutine([root], VAR_NAMES, {
      backend,
    }) as GradRoutine | null;
  });
  if (!routine) {
    console.log("  (backend returned null; skipping)\n");
    continue;
  }
  const jitMs = time("  Turbofan JIT (first call)", () => {
    routine!.eval(new Map());
  });
  console.log(`  TOTAL: ${(compileMs + jitMs).toFixed(1)}ms\n`);
}
