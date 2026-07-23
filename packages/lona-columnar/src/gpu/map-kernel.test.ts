import { afterAll, describe, expect, test } from "vitest";
import type { NumStruct } from "lona";
import { Num, asNum, variableNum } from "lona";
import { varNode } from "lona/internal";
import { destroyGpu, initGpu } from "lona/internal";
import { column, columnarGradRoutine, columnarRoutine } from "../api";
import type { ColumnarParamInput } from "../ir";
import { packGpuMapInputs } from "./map-kernel";

class Quad implements NumStruct<Quad> {
  constructor(readonly values: readonly [Num, Num, Num, Num]) {}

  toNums(): Num[] {
    return [...this.values];
  }

  fromNums(values: Num[]): Quad {
    if (values.length !== 4) throw new Error("Quad requires four components");
    return new Quad(values as unknown as readonly [Num, Num, Num, Num]);
  }
}

const shouldTestGpu = process.env.LONA_TEST_GPU === "1";
const gpuAvailable = shouldTestGpu && (await initGpu()) !== null;
const gpuTest = test.skipIf(!gpuAvailable);

afterAll(async () => {
  if (shouldTestGpu) await destroyGpu();
});

describe("columnar GPU map", () => {
  test("packs rows, uniforms, and indexes in tape slot order", () => {
    const param = (index: number) => varNode(Symbol(`packing.param.${index}`));
    const inputs: ColumnarParamInput[] = [
      { param: param(3), binding: { kind: "index" } },
      {
        param: param(2),
        binding: { kind: "uniform", component: 0 },
      },
      {
        param: param(1),
        binding: { kind: "row", component: 1 },
      },
      {
        param: param(0),
        binding: { kind: "row", component: 0 },
      },
    ];

    expect(packGpuMapInputs(inputs, [10, 20, 30, 40], [2], 2, 2)).toEqual(
      new Float32Array([0, 2, 20, 10, 1, 2, 40, 30]),
    );
  });

  test("rejects malformed host source data before dispatch", () => {
    expect(() => packGpuMapInputs([], [1], [], 1, 2)).toThrow(
      /expected 2 source values/,
    );
  });

  gpuTest("propagates columnar JVP through GPU map and reduction", async () => {
    const x = variableNum("gpu-jvp-x");
    const y = variableNum("gpu-jvp-y");
    const scale = variableNum("gpu-jvp-scale");
    const routine = columnarGradRoutine(
      () =>
        column([x, y], { placement: "gpu", backend: "gpu-codegen" })
          .map({
            using: { scale },
            build: (value, { index, using }) =>
              value.mul(using.scale).add(index),
            placement: "gpu",
            backend: "gpu-codegen",
          })
          .sum({ placement: "gpu", backend: "gpu-codegen" })
          .output(),
      ["gpu-jvp-x", "gpu-jvp-y", "gpu-jvp-scale"],
      {
        backends: { cpu: "js-interp", gpu: "gpu-codegen" },
        autodiff: { tangentBlockSize: 2 },
      },
    );

    const result = await routine.evalAsync(
      new Map([
        ["gpu-jvp-x", 2],
        ["gpu-jvp-y", 3],
        ["gpu-jvp-scale", 4],
      ]),
    );
    expect(result).toMatchObject({ val: 21 });
    if (!("gradient" in result)) throw new Error("expected gradient");
    expect(result.gradient[0]).toBeCloseTo(4, 5);
    expect(result.gradient[1]).toBeCloseTo(4, 5);
    expect(result.gradient[2]).toBeCloseTo(5, 5);
    routine.dispose();
  });

  gpuTest(
    "propagates arbitrary seeds through the GPU interpreter",
    async () => {
      const x = variableNum("gpu-interp-jvp-x");
      const y = variableNum("gpu-interp-jvp-y");
      const z = variableNum("gpu-interp-jvp-z");
      const scale = variableNum("gpu-interp-jvp-scale");
      const routine = columnarGradRoutine(
        () =>
          column([x, y, z], {
            placement: "gpu",
            backend: "gpu-interp",
          })
            .map({
              using: { scale },
              build: (value, { index, using }) =>
                value.mul(using.scale).add(index),
              placement: "gpu",
              backend: "gpu-interp",
            })
            .sum({ placement: "gpu", backend: "gpu-interp" })
            .output(),
        [
          "gpu-interp-jvp-x",
          "gpu-interp-jvp-y",
          "gpu-interp-jvp-z",
          "gpu-interp-jvp-scale",
        ],
        {
          backends: { cpu: "js-interp", gpu: "gpu-interp" },
          autodiff: { tangentBlockSize: 2 },
        },
      );

      const result = await routine.evalAsync(
        new Map([
          ["gpu-interp-jvp-x", 2],
          ["gpu-interp-jvp-y", 3],
          ["gpu-interp-jvp-z", 5],
          ["gpu-interp-jvp-scale", 4],
        ]),
      );
      expect(result).toMatchObject({ val: 43 });
      if (!("gradient" in result)) throw new Error("expected gradient");
      expect(result.gradient).toEqual([4, 4, 4, 10]);
      routine.dispose();
    },
  );

  gpuTest(
    "includes a variable initial value in GPU JVP reduction",
    async () => {
      const x = variableNum("gpu-jvp-reduce-x");
      const y = variableNum("gpu-jvp-reduce-y");
      const initial = variableNum("gpu-jvp-reduce-initial");
      const routine = columnarGradRoutine(
        () =>
          column([x, y], { placement: "gpu", backend: "gpu-codegen" })
            .reduce((left, right) => left.add(right), initial, {
              associative: true,
              order: "tree",
              placement: "gpu",
              backend: "gpu-codegen",
            })
            .output(),
        ["gpu-jvp-reduce-x", "gpu-jvp-reduce-y", "gpu-jvp-reduce-initial"],
        { backends: { cpu: "js-interp", gpu: "gpu-codegen" } },
      );

      const result = await routine.evalAsync(
        new Map([
          ["gpu-jvp-reduce-x", 2],
          ["gpu-jvp-reduce-y", 3],
          ["gpu-jvp-reduce-initial", 4],
        ]),
      );
      expect(result).toMatchObject({ val: 9 });
      if (!("gradient" in result)) throw new Error("expected gradient");
      expect(result.gradient).toEqual([1, 1, 1]);
      routine.dispose();
    },
  );

  gpuTest("maps and reduces a multi-component column on device", async () => {
    const checkpoints: string[] = [];
    const routine = columnarRoutine(
      () =>
        column([
          new Quad([asNum(1), asNum(2), asNum(3), asNum(4)]),
          new Quad([asNum(5), asNum(6), asNum(7), asNum(8)]),
        ])
          .map(
            (quad) =>
              new Quad(
                quad.values.map((value) =>
                  value.mul(2),
                ) as unknown as readonly [Num, Num, Num, Num],
              ),
            { placement: "gpu" },
          )
          .sum({
            componentWise: true,
            placement: "gpu",
            order: "tree",
          })
          .output(),
      {
        backends: { cpu: "wasm-codegen", gpu: "gpu-codegen" },
        diagnosticCheckpoint: (checkpoint) => checkpoints.push(checkpoint),
      },
    );

    await expect(routine.evalAsync(new Map())).resolves.toEqual([
      12, 16, 20, 24,
    ]);
    expect(
      checkpoints.filter((checkpoint) => checkpoint.endsWith("host-to-device")),
    ).toHaveLength(1);
    expect(
      checkpoints.filter((checkpoint) => checkpoint.endsWith("device-to-host")),
    ).toHaveLength(1);
    expect(
      checkpoints.filter(
        (checkpoint) => checkpoint === "lona:columnar:gpu:submit",
      ),
    ).toHaveLength(1);
    routine.dispose();
  });

  gpuTest(
    "computes an arbitrary source DAG directly on the device",
    async () => {
      const x = variableNum("gpu-source-x");
      const y = variableNum("gpu-source-y");
      const routine = columnarRoutine(
        () =>
          column([x.add(1), y.mul(2)], { placement: "gpu" })
            .sum({ placement: "gpu" })
            .output(),
        { backends: { gpu: "gpu-codegen" } },
      );

      expect(routine.stages[0]).toMatchObject({
        kind: "source",
        placement: "gpu",
        backend: "gpu-codegen",
      });
      await expect(
        routine.evalAsync(
          new Map([
            ["gpu-source-x", 2],
            ["gpu-source-y", 5],
          ]),
        ),
      ).resolves.toBe(13);
      expect(routine.lastEvaluationStats?.readbackCount).toBe(1);
      expect(routine.lastEvaluationStats?.uploadedBytes).toBe(8);
      routine.dispose();
    },
  );

  gpuTest("supports all scalar GPU reduction built-ins", async () => {
    const cases = [
      ["sum", 9],
      ["product", 24],
      ["min", 2],
      ["max", 4],
    ] as const;
    for (const [operation, expected] of cases) {
      const source = column([asNum(2), asNum(3), asNum(4)]);
      const reduced =
        operation === "sum"
          ? source.sum({ placement: "gpu" })
          : operation === "product"
            ? source.product({ placement: "gpu" })
            : operation === "min"
              ? source.min({ placement: "gpu" })
              : source.max({ placement: "gpu" });
      const routine = columnarRoutine(() => reduced.output(), {
        backends: { cpu: "wasm-codegen" },
      });
      await expect(routine.evalAsync(new Map())).resolves.toBe(expected);
      routine.dispose();
    }
  });

  gpuTest(
    "runs an arbitrary associative GPU reduction with uniforms",
    async () => {
      const bias = variableNum("gpu-reduce-bias");
      const routine = columnarRoutine(
        () =>
          column([asNum(1), asNum(2), asNum(3)])
            .reduce({
              using: { bias },
              combine: (left, right, { using }) =>
                left.add(right).add(using.bias),
              initial: asNum(0),
              associative: true,
              order: "tree",
              placement: "gpu",
            })
            .output(([value]) => value!),
        { backends: { cpu: "wasm-codegen" } },
      );

      await expect(
        routine.evalAsync(new Map([["gpu-reduce-bias", 10]])),
      ).resolves.toBe(36);
      expect(routine.lastEvaluationStats).toMatchObject({
        uploadedBytes: 20,
        downloadedBytes: 4,
        transferredBytes: 24,
        dispatchCount: 2,
        readbackCount: 1,
      });
      routine.dispose();
    },
  );

  gpuTest("keeps chained GPU maps device-resident", async () => {
    const x = variableNum("gpu-columnar-x");
    const y = variableNum("gpu-columnar-y");
    const offset = variableNum("gpu-columnar-offset");
    const checkpoints: string[] = [];

    const gpuRoutine = columnarRoutine(
      () =>
        column([x, y, asNum(4)])
          .map({
            using: { offset },
            build: (value, { index, using }) =>
              value.add(using.offset).mul(index.add(1)),
            placement: "gpu",
          })
          .map({
            using: { offset },
            build: (value, { using }) => value.add(using.offset),
            placement: "gpu",
          })
          .reduce((left, right) => left.add(right), asNum(0), {
            associative: true,
            order: "left",
            placement: "cpu",
          })
          .output(([total]) => total!),
      {
        backends: { cpu: "wasm-codegen", gpu: "gpu-codegen" },
        diagnosticCheckpoint: (checkpoint) => checkpoints.push(checkpoint),
      },
    );

    const cpuRoutine = columnarRoutine(
      () =>
        column([x, y, asNum(4)])
          .map({
            using: { offset },
            build: (value, { index, using }) =>
              value.add(using.offset).mul(index.add(1)),
            placement: "cpu",
          })
          .map({
            using: { offset },
            build: (value, { using }) => value.add(using.offset),
            placement: "cpu",
          })
          .reduce((left, right) => left.add(right), asNum(0), {
            associative: true,
            order: "left",
          })
          .output(([total]) => total!),
      { backends: { cpu: "wasm-codegen" } },
    );

    const vars = new Map([
      ["gpu-columnar-x", 2],
      ["gpu-columnar-y", 5],
      ["gpu-columnar-offset", 10],
    ]);
    const gpuResult = await gpuRoutine.evalAsync(vars);
    const cpuResult = await cpuRoutine.evalAsync(vars);
    checkpoints.length = 0;
    await expect(gpuRoutine.evalAsync(vars)).resolves.toBeCloseTo(gpuResult, 5);

    expect(gpuResult).toBeCloseTo(cpuResult, 5);
    expect(gpuResult).toBe(114); // (12+10) + (30+10) + (42+10)
    expect(
      gpuRoutine.stages.filter((stage) => stage.backend === "gpu-codegen"),
    ).toHaveLength(2);
    expect(
      checkpoints.filter((checkpoint) => checkpoint.endsWith("host-to-device")),
    ).toHaveLength(2); // one column and one shared uniform upload
    expect(
      checkpoints.filter((checkpoint) => checkpoint.endsWith("device-to-host")),
    ).toHaveLength(1);
    expect(
      checkpoints.filter(
        (checkpoint) => checkpoint === "lona:columnar:gpu:submit",
      ),
    ).toHaveLength(1);
    expect(
      checkpoints.filter((checkpoint) =>
        checkpoint.startsWith("lona:columnar:buffer-reuse"),
      ).length,
    ).toBeGreaterThan(0);

    gpuRoutine.dispose();
    gpuRoutine.dispose();
    cpuRoutine.dispose();
  });
});
