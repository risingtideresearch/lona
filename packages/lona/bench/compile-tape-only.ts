/**
 * Direct microbenchmark for compileTape (NumNode -> CompiledTape).
 * Isolates the impact of the TapeEmitter refactor from any backend-specific
 * compile-time (wasm codegen, etc.).
 */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { deserializeNumDAG } from "../src/core/tree-serialization";
import { compileTape } from "../src/eval/tape";

const fixtures = [
  "gridfinity-3d-raw-numtree.json",
  "gridfinity-3d-numtree.json",
  "gridfinity-slice-numtree.json",
];

const WARMUP = 50;
const RUNS = 500;

function bench(label: string, fn: () => unknown): void {
  for (let i = 0; i < WARMUP; i++) fn();
  const samples: number[] = [];
  for (let r = 0; r < RUNS; r++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  const min = samples[0]!;
  const median = samples[Math.floor(RUNS / 2)]!;
  const p95 = samples[Math.floor(RUNS * 0.95)]!;
  const mean = samples.reduce((a, b) => a + b, 0) / RUNS;
  console.log(
    `  ${label.padEnd(40)} ` +
      `min=${min.toFixed(3)}ms  median=${median.toFixed(3)}ms  ` +
      `p95=${p95.toFixed(3)}ms  mean=${mean.toFixed(3)}ms`,
  );
}

console.log(`compileTape direct benchmark (warmup=${WARMUP}, runs=${RUNS})\n`);

for (const fixture of fixtures) {
  const data = JSON.parse(
    fs.readFileSync(
      path.resolve(import.meta.dirname, "fixtures", fixture),
      "utf-8",
    ),
  );
  const dag =
    data?.simplify?.dag ?? data?.compressSimplify?.dag ?? data?.dag ?? data;
  const root = deserializeNumDAG(dag);
  const nodeCount = JSON.stringify(dag).length;

  console.log(`${fixture} (~${(nodeCount / 1024).toFixed(0)}KB serialized)`);
  bench("compileTape([root])", () => compileTape([root]));
  console.log("");
}
