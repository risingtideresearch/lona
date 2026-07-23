/**
 * Correctness tests for num evaluation using real-world DAG fixtures
 * extracted from fonseranes shapes (gridfinity, duckTape, logo).
 *
 * Each test verifies that compiled tape and codegen evaluators produce
 * the same results as the reference genericEval path across a grid of
 * sample points.
 */
import { afterAll, describe, test, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { deserializeNumDAG } from "../core/tree-serialization";
import { simpleEval } from "./eval-value";
import { compileTape, compileTapeFromSerialized } from "./tape";
import { evalTape } from "./routines/backends/js-interp/tape-eval";
import { compileFunctionFromTape } from "./routines/backends/js-codegen/codegen";
import { compileWasmFromTape } from "./routines/backends/wasm-codegen/codegen";
import { compileWasmTapeFromTape } from "./routines/backends/wasm-interp/tape-eval";
import { compileGpuTapeFromTape } from "./routines/backends/gpu-interp/tape-eval";
import { compileGpuCodegenFromTape } from "./routines/backends/gpu-codegen/codegen";
import { variableNum } from "../core/num";
import { serializeNumDAG } from "../core/tree-serialization";
import type { NumNode, VarName } from "../core/tree";
import {
  compileGradRoutine,
  compileValueRoutine,
  type GradRoutine,
  type MultiValueRoutine,
} from "./routines";
import { compileForwardAutodiff } from "./routines/backends/js-interp/tape-eval";
import { compileWasmForwardAutodiff } from "./routines/backends/wasm-interp/tape-eval";
import { compileWasmGradFromTape } from "./routines/backends/wasm-codegen/codegen";
import { destroyGpu, initGpu } from "../main";

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

const shouldTestGpu = process.env.LONA_TEST_GPU === "1";
const gpuAvailable = shouldTestGpu && (await initGpu()) !== null;
const gpuTest = test.skipIf(!gpuAvailable);

afterAll(async () => {
  if (shouldTestGpu) await destroyGpu();
});

const fixturesDir = path.resolve(import.meta.dirname, "../../bench/fixtures");

function loadFixture(filename: string) {
  const raw = fs.readFileSync(path.join(fixturesDir, filename), "utf-8");
  return JSON.parse(raw);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Sample a 3D grid and evaluate with the reference (genericEval) path */
function referenceGrid3D(
  root: ReturnType<typeof deserializeNumDAG>,
  xs: number[],
  ys: number[],
  zs: number[],
): number[] {
  const bindings = new Map<VarName, number>([
    ["x", 0],
    ["y", 0],
    ["z", 0],
  ]);
  const results: number[] = [];
  for (const z of zs) {
    bindings.set("z", z);
    for (const y of ys) {
      bindings.set("y", y);
      for (const x of xs) {
        bindings.set("x", x);
        results.push(simpleEval(root, bindings, true)); // logDebug=true -> genericEval
      }
    }
  }
  return results;
}

function linspace(min: number, max: number, n: number): number[] {
  const step = (max - min) / (n - 1);
  return Array.from({ length: n }, (_, i) => min + i * step);
}

/** Build a Float32Array of varData for a 3D grid, given varSlots ordering */
function makeGridVarData3D(
  xs: number[],
  ys: number[],
  zs: number[],
  varSlots: VarName[],
): Float32Array {
  const numVars = varSlots.length;
  const numPoints = xs.length * ys.length * zs.length;
  const data = new Float32Array(numPoints * numVars);
  const xIdx = varSlots.indexOf("x" as VarName);
  const yIdx = varSlots.indexOf("y" as VarName);
  const zIdx = varSlots.indexOf("z" as VarName);
  let p = 0;
  for (const z of zs)
    for (const y of ys)
      for (const x of xs) {
        const base = p * numVars;
        if (xIdx >= 0) data[base + xIdx] = x;
        if (yIdx >= 0) data[base + yIdx] = y;
        if (zIdx >= 0) data[base + zIdx] = z;
        p++;
      }
  return data;
}

/** Build a Float32Array of varData for a 2D grid (z=0), given varSlots ordering */
function makeGridVarData2D(
  xs: number[],
  ys: number[],
  varSlots: VarName[],
): Float32Array {
  const numVars = varSlots.length;
  const numPoints = xs.length * ys.length;
  const data = new Float32Array(numPoints * numVars);
  const xIdx = varSlots.indexOf("x" as VarName);
  const yIdx = varSlots.indexOf("y" as VarName);
  let p = 0;
  for (const y of ys)
    for (const x of xs) {
      const base = p * numVars;
      if (xIdx >= 0) data[base + xIdx] = x;
      if (yIdx >= 0) data[base + yIdx] = y;
      p++;
    }
  return data;
}

/** Max absolute error between f32 GPU results and f64 CPU reference */
function maxAbsError(gpuResults: Float32Array, cpuReference: number[]): number {
  let maxErr = 0;
  for (let i = 0; i < cpuReference.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(gpuResults[i]! - cpuReference[i]!));
  }
  return maxErr;
}

// f32 precision tolerance -- GPU uses f32, CPU uses f64
const GPU_TOLERANCE = 1e-3;

// ---------------------------------------------------------------------------
// Test suites per fixture
// ---------------------------------------------------------------------------

const GRID_N = 5; // 5x5x5 = 125 sample points per fixture

describe.each([
  { name: "gridfinity-3d", file: "gridfinity-3d-numtree.json" },
  { name: "ducktape-3d", file: "ducktape-3d-numtree.json" },
  { name: "logo-3d", file: "logo-3d-numtree.json" },
])("$name", ({ file }) => {
  const fixture = loadFixture(file);
  const dag = fixture.compressSimplify.dag;
  const bounds = fixture.bounds as {
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
  const root = deserializeNumDAG(dag);

  const xs = linspace(bounds.x[0], bounds.x[1], GRID_N);
  const ys = linspace(bounds.y[0], bounds.y[1], GRID_N);
  const zs = linspace(bounds.z[0], bounds.z[1], GRID_N);

  const reference = referenceGrid3D(root, xs, ys, zs);

  test("compiledTape matches genericEval", () => {
    const tape = compileTapeFromSerialized(dag);
    expect(tape).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
      ["z", 0],
    ]);
    const results: number[] = [];
    for (const z of zs) {
      vars.set("z", z);
      for (const y of ys) {
        vars.set("y", y);
        for (const x of xs) {
          vars.set("x", x);
          results.push(evalTape(tape!, vars)[0]!);
        }
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  test("codegen matches genericEval", () => {
    const fn = compileFunctionFromTape(compileTapeFromSerialized(dag)!);
    expect(fn).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
      ["z", 0],
    ]);
    const results: number[] = [];
    for (const z of zs) {
      vars.set("z", z);
      for (const y of ys) {
        vars.set("y", y);
        for (const x of xs) {
          vars.set("x", x);
          results.push(fn!(vars)[0]!);
        }
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  test("wasm matches genericEval", () => {
    const fn = compileWasmFromTape(compileTapeFromSerialized(dag)!);
    expect(fn).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
      ["z", 0],
    ]);
    const results: number[] = [];
    for (const z of zs) {
      vars.set("z", z);
      for (const y of ys) {
        vars.set("y", y);
        for (const x of xs) {
          vars.set("x", x);
          results.push(fn!(vars)[0]!);
        }
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  test("wasmTape matches genericEval", () => {
    const fn = compileWasmTapeFromTape(compileTapeFromSerialized(dag)!);
    expect(fn).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
      ["z", 0],
    ]);
    const results: number[] = [];
    for (const z of zs) {
      vars.set("z", z);
      for (const y of ys) {
        vars.set("y", y);
        for (const x of xs) {
          vars.set("x", x);
          results.push(fn!(vars)[0]!);
        }
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  gpuTest("gpuTape approximately matches genericEval (f32)", async () => {
    const gpuEval = await compileGpuTapeFromTape(
      compileTapeFromSerialized(dag)!,
    );
    expect(gpuEval).not.toBeNull();

    const numPoints = xs.length * ys.length * zs.length;
    const varData = makeGridVarData3D(xs, ys, zs, gpuEval!.varSlots);
    const results = await gpuEval!.evalBatch(varData, numPoints);

    const err = maxAbsError(results, reference);
    expect(err).toBeLessThan(GPU_TOLERANCE);

    gpuEval!.destroy();
  });

  gpuTest("gpuCodegen approximately matches genericEval (f32)", async () => {
    const gpuEval = await compileGpuCodegenFromTape(
      compileTapeFromSerialized(dag)!,
    );
    expect(gpuEval).not.toBeNull();

    const numPoints = xs.length * ys.length * zs.length;
    const varData = makeGridVarData3D(xs, ys, zs, gpuEval!.varSlots);
    const results = await gpuEval!.evalBatch(varData, numPoints);

    const err = maxAbsError(results, reference);
    expect(err).toBeLessThan(GPU_TOLERANCE);

    gpuEval!.destroy();
  });
});

// ---------------------------------------------------------------------------
// 2D slice fixture (gridfinity slice)
// ---------------------------------------------------------------------------

describe("gridfinity-slice-2d", () => {
  const fixture = loadFixture("gridfinity-slice-numtree.json");
  const dag = fixture.compressSimplify.dag;
  const bounds = fixture.bounds as { x: [number, number]; y: [number, number] };
  const root = deserializeNumDAG(dag);

  const xs = linspace(bounds.x[0], bounds.x[1], GRID_N);
  const ys = linspace(bounds.y[0], bounds.y[1], GRID_N);

  const bindings = new Map<VarName, number>([
    ["x", 0],
    ["y", 0],
  ]);

  const reference: number[] = [];
  for (const y of ys) {
    bindings.set("y", y);
    for (const x of xs) {
      bindings.set("x", x);
      reference.push(simpleEval(root, bindings, true));
    }
  }

  test("compiledTape matches genericEval", () => {
    const tape = compileTapeFromSerialized(dag)!;
    expect(tape).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
    ]);
    const results: number[] = [];
    for (const y of ys) {
      vars.set("y", y);
      for (const x of xs) {
        vars.set("x", x);
        results.push(evalTape(tape, vars)[0]!);
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  test("codegen matches genericEval", () => {
    const fn = compileFunctionFromTape(compileTapeFromSerialized(dag)!)!;
    expect(fn).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
    ]);
    const results: number[] = [];
    for (const y of ys) {
      vars.set("y", y);
      for (const x of xs) {
        vars.set("x", x);
        results.push(fn(vars)[0]!);
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  test("wasm matches genericEval", () => {
    const fn = compileWasmFromTape(compileTapeFromSerialized(dag)!)!;
    expect(fn).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
    ]);
    const results: number[] = [];
    for (const y of ys) {
      vars.set("y", y);
      for (const x of xs) {
        vars.set("x", x);
        results.push(fn(vars)[0]!);
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  test("wasmTape matches genericEval", () => {
    const fn = compileWasmTapeFromTape(compileTapeFromSerialized(dag)!)!;
    expect(fn).not.toBeNull();

    const vars = new Map<VarName, number>([
      ["x", 0],
      ["y", 0],
    ]);
    const results: number[] = [];
    for (const y of ys) {
      vars.set("y", y);
      for (const x of xs) {
        vars.set("x", x);
        results.push(fn(vars)[0]!);
      }
    }

    for (let i = 0; i < reference.length; i++) {
      expect(results[i]).toBe(reference[i]);
    }
  });

  gpuTest("gpuTape approximately matches genericEval (f32)", async () => {
    const gpuEval = await compileGpuTapeFromTape(
      compileTapeFromSerialized(dag)!,
    );
    expect(gpuEval).not.toBeNull();

    const numPoints = xs.length * ys.length;
    const varData = makeGridVarData2D(xs, ys, gpuEval!.varSlots);
    const results = await gpuEval!.evalBatch(varData, numPoints);

    const err = maxAbsError(results, reference);
    expect(err).toBeLessThan(GPU_TOLERANCE);

    gpuEval!.destroy();
  });

  gpuTest("gpuCodegen approximately matches genericEval (f32)", async () => {
    const gpuEval = await compileGpuCodegenFromTape(
      compileTapeFromSerialized(dag)!,
    );
    expect(gpuEval).not.toBeNull();

    const numPoints = xs.length * ys.length;
    const varData = makeGridVarData2D(xs, ys, gpuEval!.varSlots);
    const results = await gpuEval!.evalBatch(varData, numPoints);

    const err = maxAbsError(results, reference);
    expect(err).toBeLessThan(GPU_TOLERANCE);

    gpuEval!.destroy();
  });
});

// ---------------------------------------------------------------------------
// Named variables (beyond x, y, z)
// ---------------------------------------------------------------------------

describe("named variables", () => {
  // Expression: x + w * y  -- uses named variable "w"
  const x = variableNum("x");
  const y = variableNum("y");
  const w = variableNum("w");
  const expr = x.add(w.mul(y));
  const dag = serializeNumDAG(expr.n);

  // Test points
  const testCases = [
    { x: 1, y: 2, w: 3 }, // 1 + 3*2 = 7
    { x: 0, y: 5, w: -2 }, // 0 + (-2)*5 = -10
    { x: 10, y: 0.5, w: 4 }, // 10 + 4*0.5 = 12
  ];

  const reference = testCases.map((c) =>
    simpleEval(
      expr.n,
      new Map<VarName, number>([
        ["x", c.x],
        ["y", c.y],
        ["w", c.w],
      ]),
      true,
    ),
  );

  test("compiledTape", () => {
    const tape = compileTape([expr.n])!;
    for (let i = 0; i < testCases.length; i++) {
      const c = testCases[i]!;
      const vars = new Map<VarName, number>([
        ["x", c.x],
        ["y", c.y],
        ["w", c.w],
      ]);
      expect(evalTape(tape, vars)[0]).toBe(reference[i]);
    }
  });

  test("codegen", () => {
    const fn = compileFunctionFromTape(compileTape([expr.n])!)!;
    for (let i = 0; i < testCases.length; i++) {
      const c = testCases[i]!;
      const vars = new Map<VarName, number>([
        ["x", c.x],
        ["y", c.y],
        ["w", c.w],
      ]);
      expect(fn(vars)[0]).toBe(reference[i]);
    }
  });

  test("wasm", () => {
    const fn = compileWasmFromTape(compileTape([expr.n])!)!;
    for (let i = 0; i < testCases.length; i++) {
      const c = testCases[i]!;
      const vars = new Map<VarName, number>([
        ["x", c.x],
        ["y", c.y],
        ["w", c.w],
      ]);
      expect(fn(vars)[0]).toBe(reference[i]);
    }
  });

  test("wasmTape", () => {
    const fn = compileWasmTapeFromTape(compileTape([expr.n])!)!;
    for (let i = 0; i < testCases.length; i++) {
      const c = testCases[i]!;
      const vars = new Map<VarName, number>([
        ["x", c.x],
        ["y", c.y],
        ["w", c.w],
      ]);
      expect(fn(vars)[0]).toBe(reference[i]);
    }
  });

  gpuTest("gpuTape", async () => {
    const gpuEval = await compileGpuTapeFromTape(
      compileTapeFromSerialized(dag)!,
    );
    expect(gpuEval).not.toBeNull();

    for (let i = 0; i < testCases.length; i++) {
      const c = testCases[i]!;
      const numVars = gpuEval!.varSlots.length;
      const varData = new Float32Array(numVars);
      for (let v = 0; v < numVars; v++) {
        const name = gpuEval!.varSlots[v]! as string;
        varData[v] = (c as Record<string, number>)[name] ?? 0;
      }
      const results = await gpuEval!.evalBatch(varData, 1);
      expect(results[0]).toBeCloseTo(reference[i]!, 3);
    }

    gpuEval!.destroy();
  });

  gpuTest("gpuCodegen", async () => {
    const gpuEval = await compileGpuCodegenFromTape(
      compileTapeFromSerialized(dag)!,
    );
    expect(gpuEval).not.toBeNull();

    for (let i = 0; i < testCases.length; i++) {
      const c = testCases[i]!;
      const numVars = gpuEval!.varSlots.length;
      const varData = new Float32Array(numVars);
      for (let v = 0; v < numVars; v++) {
        const name = gpuEval!.varSlots[v]! as string;
        varData[v] = (c as Record<string, number>)[name] ?? 0;
      }
      const results = await gpuEval!.evalBatch(varData, 1);
      expect(results[0]).toBeCloseTo(reference[i]!, 3);
    }

    gpuEval!.destroy();
  });
});

// ---------------------------------------------------------------------------
// gpu-interp multi-root value evaluation
// ---------------------------------------------------------------------------

describe("gpu-interp multi-root value evaluation", () => {
  // Three roots sharing variables x, y, w:
  //   rootA = x + w*y
  //   rootB = x*x - y
  //   rootC = w*x + y*y
  const x = variableNum("x");
  const y = variableNum("y");
  const w = variableNum("w");
  const rootA = x.add(w.mul(y));
  const rootB = x.mul(x).sub(y);
  const rootC = w.mul(x).add(y.mul(y));

  const points = [
    { x: 1, y: 2, w: 3 },
    { x: 0, y: 5, w: -2 },
    { x: 10, y: 0.5, w: 4 },
    { x: -3, y: -1, w: 2 },
    { x: 2.5, y: 7, w: -0.5 },
  ];

  /** CPU reference values for a single root across all sample points. */
  function referenceForRoot(root: NumNode): number[] {
    return points.map((p) =>
      simpleEval(
        root,
        new Map<VarName, number>([
          ["x", p.x],
          ["y", p.y],
          ["w", p.w],
        ]),
        true, // logDebug=true -> genericEval
      ),
    );
  }

  /** Point-major interleaved CPU reference: out[p * numRoots + r]. */
  function referenceInterleaved(roots: NumNode[]): number[] {
    const perRoot = roots.map(referenceForRoot);
    const out: number[] = [];
    for (let p = 0; p < points.length; p++) {
      for (let r = 0; r < roots.length; r++) out.push(perRoot[r]![p]!);
    }
    return out;
  }

  /** Pack the sample points into a VarBatch keyed by variable name. */
  function pointsVarBatch(): Record<string, number[]> {
    return {
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      w: points.map((p) => p.w),
    };
  }

  gpuTest(
    "evalBatch: multi-root output is point-major interleaved (f32)",
    async () => {
      const roots = [rootA.n, rootB.n, rootC.n];
      const routine = compileValueRoutine(roots, {
        backend: "gpu-interp",
      }) as MultiValueRoutine;
      expect(routine).not.toBeNull();
      expect(routine.shape).toBe("multi-value");
      expect(routine.numRoots).toBe(3);

      const results = await routine.evalBatch(pointsVarBatch(), points.length);
      expect(results.length).toBe(points.length * routine.numRoots);

      const reference = referenceInterleaved(roots);
      const err = maxAbsError(results, reference);
      expect(err).toBeLessThan(GPU_TOLERANCE);

      routine.dispose?.();
    },
  );

  gpuTest("evalAsync: one value per root for a single point", async () => {
    const roots = [rootA.n, rootB.n, rootC.n];
    const routine = compileValueRoutine(roots, {
      backend: "gpu-interp",
    }) as MultiValueRoutine;
    expect(routine).not.toBeNull();

    for (const p of points) {
      const vars = new Map<VarName, number>([
        ["x", p.x],
        ["y", p.y],
        ["w", p.w],
      ]);
      const result = await routine.evalAsync(vars);
      expect(result.length).toBe(roots.length);

      const reference = roots.map(
        (root) => simpleEval(root, vars, true), // logDebug=true -> genericEval
      );
      for (let r = 0; r < roots.length; r++) {
        expect(result[r]).toBeCloseTo(reference[r]!, 3);
      }
    }

    routine.dispose?.();
  });

  gpuTest(
    "duplicated roots: both output slots per point agree and are correct",
    async () => {
      // Same node twice, plus a distinct third root.
      const roots = [rootA.n, rootA.n, rootB.n];
      const tape = compileTape(roots)!;
      expect(tape).not.toBeNull();

      const gpuEval = await compileGpuTapeFromTape(tape);
      expect(gpuEval).not.toBeNull();

      const numVars = gpuEval!.varSlots.length;
      const numPoints = points.length;
      const varData = new Float32Array(numPoints * numVars);
      for (let p = 0; p < numPoints; p++) {
        for (let s = 0; s < numVars; s++) {
          const name = gpuEval!.varSlots[s]! as string;
          varData[p * numVars + s] =
            (points[p] as unknown as Record<string, number>)[name] ?? 0;
        }
      }

      const results = await gpuEval!.evalBatch(varData, numPoints);
      expect(results.length).toBe(numPoints * roots.length);

      const refA = referenceForRoot(rootA.n);
      const refB = referenceForRoot(rootB.n);
      for (let p = 0; p < numPoints; p++) {
        const slot0 = results[p * roots.length + 0]!;
        const slot1 = results[p * roots.length + 1]!;
        const slot2 = results[p * roots.length + 2]!;
        expect(slot1).toBe(slot0); // duplicated root -> identical f32 values
        expect(Math.abs(slot0 - refA[p]!)).toBeLessThan(GPU_TOLERANCE);
        expect(Math.abs(slot2 - refB[p]!)).toBeLessThan(GPU_TOLERANCE);
      }

      gpuEval!.destroy();
    },
  );

  gpuTest.each(["gpu-interp", "gpu-codegen"] as const)(
    "backend=%s gradient batch uses a device JVP kernel",
    async (backend) => {
      const routine = compileGradRoutine([rootA.n], ["x", "y", "w"], {
        backend,
      }) as GradRoutine;
      const results = await routine.evalBatch(pointsVarBatch(), points.length);
      expect(results).toHaveLength(points.length);
      for (let point = 0; point < points.length; point++) {
        const input = points[point]!;
        const result = results[point]!;
        expect(result.val).toBeCloseTo(input.x + input.w * input.y, 4);
        expect(result.gradient[0]).toBeCloseTo(1, 4);
        expect(result.gradient[1]).toBeCloseTo(input.w, 4);
        expect(result.gradient[2]).toBeCloseTo(input.y, 4);
      }
      routine.dispose?.();
    },
  );

  gpuTest(
    "single-root regression: evalBatch still returns numPoints f32s",
    async () => {
      const roots = [rootA.n];
      const tape = compileTape(roots)!;
      expect(tape).not.toBeNull();

      const gpuEval = await compileGpuTapeFromTape(tape);
      expect(gpuEval).not.toBeNull();

      const numVars = gpuEval!.varSlots.length;
      const numPoints = points.length;
      const varData = new Float32Array(numPoints * numVars);
      for (let p = 0; p < numPoints; p++) {
        for (let s = 0; s < numVars; s++) {
          const name = gpuEval!.varSlots[s]! as string;
          varData[p * numVars + s] =
            (points[p] as unknown as Record<string, number>)[name] ?? 0;
        }
      }

      const results = await gpuEval!.evalBatch(varData, numPoints);
      expect(results.length).toBe(numPoints); // guards against layout change (numPoints * numRoots with numRoots=1)

      const reference = referenceForRoot(rootA.n);
      const err = maxAbsError(results, reference);
      expect(err).toBeLessThan(GPU_TOLERANCE);

      gpuEval!.destroy();
    },
  );
});

// ---------------------------------------------------------------------------
// Gradient parity — JS tape vs WASM tape vs WASM codegen
// ---------------------------------------------------------------------------

const GRAD_GRID_N = 4; // 4^3 = 64 points per fixture
const DIFF_VARS: VarName[] = ["x" as VarName, "y" as VarName, "z" as VarName];

/** Build a flat array of grid points as Maps, reusable across tests. */
function makeGradGridPoints(bounds: {
  x: [number, number];
  y: [number, number];
  z: [number, number];
}): Map<VarName, number>[] {
  const xs = linspace(bounds.x[0], bounds.x[1], GRAD_GRID_N);
  const ys = linspace(bounds.y[0], bounds.y[1], GRAD_GRID_N);
  const zs = linspace(bounds.z[0], bounds.z[1], GRAD_GRID_N);
  const points: Map<VarName, number>[] = [];
  for (const z of zs)
    for (const y of ys)
      for (const x of xs)
        points.push(
          new Map<VarName, number>([
            ["x" as VarName, x],
            ["y" as VarName, y],
            ["z" as VarName, z],
          ]),
        );
  return points;
}

describe.each([
  { name: "gridfinity-3d", file: "gridfinity-3d-numtree.json" },
  { name: "ducktape-3d", file: "ducktape-3d-numtree.json" },
  { name: "logo-3d", file: "logo-3d-numtree.json" },
])("$name gradient parity", ({ file }) => {
  const fixture = loadFixture(file);
  const dag = fixture.compressSimplify.dag;
  const bounds = fixture.bounds as {
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
  const root = deserializeNumDAG(dag);
  const tape = compileTape([root])!;
  const points = makeGradGridPoints(bounds);

  // Precompute reference grid using JS tape forward autodiff
  const jsFn = compileForwardAutodiff(tape, DIFF_VARS);
  const refVals: number[] = [];
  const refGrads: number[][] = [];
  for (const pt of points) {
    const r = jsFn(pt);
    refVals.push(r.val);
    refGrads.push(r.gradient);
  }

  function expectMatchesRef(
    fn: (vars: Map<VarName, number>) => { val: number; gradient: number[] },
    precision: number,
  ) {
    for (let i = 0; i < points.length; i++) {
      const r = fn(points[i]!);
      expect(r.val).toBeCloseTo(refVals[i]!, precision);
      for (let d = 0; d < DIFF_VARS.length; d++) {
        expect(r.gradient[d]).toBeCloseTo(refGrads[i]![d]!, precision);
      }
    }
  }

  test("WASM tape matches JS tape", () => {
    expectMatchesRef(compileWasmForwardAutodiff(tape, DIFF_VARS), 12);
  });

  test("WASM codegen matches JS tape", () => {
    const wasmFn = compileWasmGradFromTape(compileTape([root])!, DIFF_VARS)!;
    expect(wasmFn).not.toBeNull();
    expectMatchesRef(wasmFn, 10);
  });

  test("GradRoutine (wasm-interp) matches JS tape", () => {
    const routine = compileGradRoutine([root], DIFF_VARS, {
      backend: "wasm-interp",
    })! as GradRoutine;
    expectMatchesRef((vars) => routine.eval(vars), 12);
  });

  test("GradRoutine (js-interp) matches JS tape", () => {
    const routine = compileGradRoutine([root], DIFF_VARS, {
      backend: "js-interp",
    })! as GradRoutine;
    expectMatchesRef((vars) => routine.eval(vars), 14);
  });
});
