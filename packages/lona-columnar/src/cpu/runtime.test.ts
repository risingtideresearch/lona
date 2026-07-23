import { describe, expect, test } from "vitest";
import type { NumStruct } from "lona";
import { Num, asNum, variableNum } from "lona";
import { isGpuAvailable } from "lona/internal";
import type { CpuBackendName } from "../types";
import { columnarGradRoutine, columnarRoutine, column } from "../api";

class Pair implements NumStruct<Pair> {
  constructor(
    readonly x: Num,
    readonly y: Num,
  ) {}

  toNums(): Num[] {
    return [this.x, this.y];
  }

  fromNums([x, y]: Num[]): Pair {
    if (!x || !y) throw new Error("Pair requires two components");
    return new Pair(x, y);
  }

  add(other: Pair): Pair {
    return new Pair(this.x.add(other.x), this.y.add(other.y));
  }
}

const CPU_BACKENDS: readonly CpuBackendName[] = [
  "js-interp",
  "js-codegen",
  "wasm-interp",
  "wasm-codegen",
];

describe("columnar CPU runtime", () => {
  test.each(CPU_BACKENDS)(
    "map -> reduce -> output executes on %s",
    async (backend) => {
      const x = variableNum("runtime-x");
      const y = variableNum("runtime-y");
      const offset = variableNum("runtime-offset");
      const scale = variableNum("runtime-scale");

      const routine = columnarRoutine(
        () =>
          column([x, y])
            .map({
              using: { offset },
              build: (value, { index, using }) =>
                value.add(using.offset).add(index),
              placement: "cpu",
            })
            .reduce((left, right) => left.add(right), asNum(0), {
              associative: true,
              order: "tree",
              placement: "cpu",
            })
            .output({
              using: { scale },
              build: ([total], { using }) => [total!.mul(using.scale)],
              placement: "cpu",
            }),
        { backends: { cpu: backend }, placement: "cpu" },
      );

      expect(routine.varSlots).toEqual([
        "runtime-x",
        "runtime-y",
        "runtime-offset",
        "runtime-scale",
      ]);
      expect(routine.numVars).toBe(4);
      const result = await routine.evalAsync(
        new Map([
          ["runtime-x", 2],
          ["runtime-y", 5],
          ["runtime-offset", 10],
          ["runtime-scale", 3],
        ]),
      );

      // map rows: 2+10+0=12, 5+10+1=16; sum=28; scale=3.
      expect(result).toEqual([84]);
      expect(routine.stages.map((stage) => stage.kind)).toEqual([
        "source",
        "map",
        "reduce",
        "output",
      ]);
      routine.dispose();
    },
  );

  test("outputs a column or reduced value without adding a CPU output stage", async () => {
    const values = columnarRoutine(
      () =>
        column([asNum(2), asNum(5)])
          .map((value) => value.add(1))
          .output(),
      { placement: "cpu", backends: { cpu: "js-interp" } },
    );
    await expect(values.evalAsync(new Map())).resolves.toEqual([3, 6]);
    expect(values.stages.map(({ kind }) => kind)).toEqual(["source", "map"]);

    const total = columnarRoutine(
      () =>
        column([asNum(2), asNum(5)])
          .sum()
          .output(),
      { placement: "cpu", backends: { cpu: "js-interp" } },
    );
    await expect(total.evalAsync(new Map())).resolves.toBe(7);
    expect(total.stages.map(({ kind }) => kind)).toEqual(["source", "reduce"]);
    values.dispose();
    total.dispose();
  });

  test("continues execution through a column returned by then", async () => {
    const x = variableNum("expanded-x");
    const y = variableNum("expanded-y");
    const routine = columnarRoutine(
      () =>
        column([x, y])
          .then(([a, b]) => column([a!.add(b!), b!.mul(2)]))
          .map((value) => value.add(1))
          .sum()
          .output(([total]) => total!),
      { backends: { cpu: "js-codegen" }, placement: "cpu" },
    );

    expect(
      await routine.evalAsync(
        new Map([
          ["expanded-x", 2],
          ["expanded-y", 5],
        ]),
      ),
    ).toBe(19);
    expect(routine.stages.map(({ kind }) => kind)).toEqual([
      "source",
      "then",
      "map",
      "reduce",
      "output",
    ]);
    routine.dispose();
  });

  test("executes chained NumStruct maps and decodes a struct result", async () => {
    const x = variableNum("pair-x");
    const y = variableNum("pair-y");
    const zero = new Pair(asNum(0), asNum(0));

    const routine = columnarRoutine(
      () =>
        column([new Pair(x, y), new Pair(y, x)])
          .map((pair, { index }) =>
            new Pair(pair.x.add(index), pair.y.sub(index)).add(zero),
          )
          .map((pair) => new Pair(pair.x.mul(2), pair.y.mul(3)))
          .reduce((left, right) => left.add(right), zero, {
            associative: true,
            order: "left",
          })
          .output(([total]) => total!),
      { backends: { cpu: "js-interp" }, placement: "cpu" },
    );

    const result = await routine.evalAsync(
      new Map([
        ["pair-x", 2],
        ["pair-y", 5],
      ]),
    );

    // rows before scaling: (2,5), (6,1); after scaling: (4,15), (12,3)
    expect(result).toEqual([16, 18]);
  });

  test("decodes an array of NumStruct outputs", async () => {
    const routine = columnarRoutine(
      () =>
        column([new Pair(asNum(1), asNum(2))]).output(([pair]) => [
          pair!,
          new Pair(pair!.x.add(10), pair!.y.add(20)),
        ]),
      { backends: { cpu: "js-interp" }, placement: "cpu" },
    );

    await expect(routine.evalAsync(new Map())).resolves.toEqual([
      [1, 2],
      [11, 22],
    ]);
  });

  test("empty reduction returns its initial value", async () => {
    const zero = new Pair(asNum(7), asNum(9));
    const routine = columnarRoutine(
      () =>
        column([], { shape: zero })
          .reduce((left, right) => left.add(right), zero, {
            associative: true,
            order: "tree",
          })
          .output(([result]) => result!),
      { backends: { cpu: "wasm-codegen" }, placement: "cpu" },
    );

    await expect(routine.evalAsync(new Map())).resolves.toEqual([7, 9]);
  });

  test("built-in reductions define identities and component-wise structs", async () => {
    const scalar = async (
      operation: "sum" | "product" | "min" | "max",
      values: readonly number[],
    ): Promise<number> => {
      const source =
        values.length === 0
          ? column([], { shape: asNum(0) })
          : column(values.map(asNum));
      const reduced =
        operation === "sum"
          ? source.sum()
          : operation === "product"
            ? source.product()
            : operation === "min"
              ? source.min()
              : source.max();
      const routine = columnarRoutine(
        () => reduced.output(([value]) => value!),
        { backends: { cpu: "js-interp" }, placement: "cpu" },
      );
      return routine.evalAsync(new Map());
    };

    await expect(scalar("sum", [])).resolves.toBe(0);
    await expect(scalar("product", [])).resolves.toBe(1);
    await expect(scalar("min", [])).resolves.toBe(Infinity);
    await expect(scalar("max", [])).resolves.toBe(-Infinity);
    await expect(scalar("sum", [2, 3, 4])).resolves.toBe(9);
    await expect(scalar("product", [2, 3, 4])).resolves.toBe(24);
    expect(await scalar("min", [Number.NaN, 1])).toBeNaN();

    const signedZero = async (operation: "min" | "max"): Promise<number> => {
      const left = variableNum("builtin-zero-left");
      const right = variableNum("builtin-zero-right");
      const source = column([left, right]);
      const reduced = operation === "min" ? source.min() : source.max();
      return columnarRoutine(() => reduced.output(([value]) => value!), {
        backends: { cpu: "js-interp" },
        placement: "cpu",
      }).evalAsync(
        new Map([
          ["builtin-zero-left", -0],
          ["builtin-zero-right", 0],
        ]),
      );
    };
    expect(Object.is(await signedZero("min"), -0)).toBe(true);
    expect(Object.is(await signedZero("max"), 0)).toBe(true);

    const pairShape = new Pair(asNum(0), asNum(0));
    const pairRoutine = columnarRoutine(
      () =>
        column([new Pair(asNum(1), asNum(2)), new Pair(asNum(3), asNum(4))])
          .sum({ componentWise: true })
          .output(([value]) => value!),
      { backends: { cpu: "wasm-codegen" }, placement: "cpu" },
    );
    await expect(pairRoutine.evalAsync(new Map())).resolves.toEqual([4, 6]);

    const emptyRoutine = columnarRoutine(
      () =>
        column([], { shape: pairShape })
          .sum({ componentWise: true })
          .output(([value]) => value!),
      { backends: { cpu: "js-interp" }, placement: "cpu" },
    );
    await expect(emptyRoutine.evalAsync(new Map())).resolves.toEqual([0, 0]);
  });

  test("left and tree reductions have explicit ordering", async () => {
    const source = [asNum(1), asNum(2), asNum(3)];
    const make = (order: "left" | "tree") =>
      columnarRoutine(
        () =>
          column(source)
            .reduce((left, right) => left.sub(right), asNum(0), {
              associative: order === "tree",
              order,
            })
            .output(([result]) => result!),
        { backends: { cpu: "js-interp" }, placement: "cpu" },
      );

    await expect(make("left").evalAsync(new Map())).resolves.toBe(-6);
    await expect(make("tree").evalAsync(new Map())).resolves.toBe(0);
  });

  test("hard placement is enforced without explicit transfer stages", async () => {
    const cpu = columnarRoutine(
      () =>
        column([asNum(2)])
          .map((value) => value.square(), { placement: "cpu" })
          .output(([value]) => value!, { placement: "cpu" }),
      { backends: { cpu: "js-interp" } },
    );
    await expect(cpu.evalAsync(new Map())).resolves.toBe(4);

    expect(() =>
      columnarRoutine(() =>
        column([asNum(1)]).output(([value]) => value!, {
          placement: "gpu",
        }),
      ),
    ).toThrow(/whole-column stages execute on CPU/);
  });

  test("map/reduce default to GPU-first auto and permit per-kind placement", async () => {
    const preferredMap = columnarRoutine(
      () =>
        column([asNum(2)])
          .map((value) => value.square())
          .output(([value]) => value!),
      { backends: { cpu: "js-interp" } },
    );
    const mapStage = preferredMap.stages.find((stage) => stage.kind === "map");
    expect(mapStage?.backend).toBe(
      isGpuAvailable() ? "gpu-codegen" : "js-interp",
    );
    await expect(preferredMap.evalAsync(new Map())).resolves.toBe(4);

    const ordered = columnarRoutine(
      () =>
        column([asNum(4), asNum(1)])
          .reduce((left, right) => left.sub(right), asNum(0), {
            associative: false,
            order: "left",
          })
          .output(([value]) => value!),
      { backends: { cpu: "js-interp" } },
    );
    expect(
      ordered.stages.find((stage) => stage.kind === "reduce")?.backend,
    ).toBe("js-interp");
    await expect(ordered.evalAsync(new Map())).resolves.toBe(-5);

    const forcedCpu = columnarRoutine(
      () =>
        column([asNum(3)])
          .map((value) => value.add(1))
          .sum()
          .output(([value]) => value!),
      {
        backends: { cpu: "js-interp" },
        placement: { map: "cpu", reduce: "cpu", output: "cpu" },
      },
    );
    expect(
      forcedCpu.stages
        .filter((stage) => stage.kind === "map" || stage.kind === "reduce")
        .map((stage) => stage.backend),
    ).toEqual(["js-interp", "js-interp"]);
    await expect(forcedCpu.evalAsync(new Map())).resolves.toBe(4);
  });

  test("stage backend overrides routine-level backend candidates", async () => {
    const routine = columnarRoutine(
      () =>
        column([asNum(2)], {
          placement: "cpu",
          backend: "js-interp",
        })
          .map((value) => value.square(), {
            placement: "cpu",
            backend: "js-codegen",
          })
          .output(([value]) => value!, {
            placement: "cpu",
            backend: "wasm-interp",
          }),
      {
        placement: "cpu",
        backends: { cpu: "wasm-codegen" },
      },
    );

    expect(routine.stages.map((stage) => stage.backend)).toEqual([
      "js-interp",
      "js-codegen",
      "wasm-interp",
    ]);
    await expect(routine.evalAsync(new Map())).resolves.toBe(4);
  });

  test("rejects a stage backend incompatible with hard placement", () => {
    expect(() =>
      columnarRoutine(
        () =>
          column([asNum(1)], {
            placement: "gpu",
            backend: "js-interp",
          }).output(),
        { backends: { cpu: "js-codegen" } },
      ),
    ).toThrow(/backend 'js-interp' requires cpu placement/);
  });

  test("auto target and backend preferences resolve by stage kind", async () => {
    const routine = columnarRoutine(
      () =>
        column([asNum(2), asNum(3)])
          .map((value) => value.square())
          .sum()
          .output(([value]) => value!),
      {
        placement: {
          map: "auto",
          reduce: "cpu",
          output: "cpu",
        },
        auto: {
          targets: { map: ["cpu", "gpu"] },
        },
        backends: {
          cpu: ["js-codegen", "js-interp"],
          gpu: ["gpu-codegen"],
        },
      },
    );

    expect(
      routine.stages.map(({ kind, placement, backend }) => ({
        kind,
        placement,
        backend,
      })),
    ).toEqual([
      { kind: "source", placement: "cpu", backend: "js-codegen" },
      { kind: "map", placement: "cpu", backend: "js-codegen" },
      { kind: "reduce", placement: "cpu", backend: "js-codegen" },
      { kind: "output", placement: "cpu", backend: "js-codegen" },
    ]);
    await expect(routine.evalAsync(new Map())).resolves.toBe(13);
  });

  test("reports evaluation costs and stage timings", async () => {
    const routine = columnarRoutine(
      () =>
        column([asNum(1), asNum(2)])
          .sum()
          .output(([value]) => value!),
      { backends: { cpu: "js-interp" }, placement: "cpu" },
    );
    expect(routine.lastEvaluationStats).toBeNull();
    await expect(routine.evalAsync(new Map())).resolves.toBe(3);
    expect(routine.lastEvaluationStats).toMatchObject({
      uploadedBytes: 0,
      downloadedBytes: 0,
      transferredBytes: 0,
      dispatchCount: 0,
      readbackCount: 0,
    });
    expect(routine.lastEvaluationStats?.stageTimings).toHaveLength(3);
    expect(
      routine.lastEvaluationStats!.evaluationMilliseconds,
    ).toBeGreaterThanOrEqual(0);
  });

  test("dispose is idempotent and prevents later evaluation", async () => {
    const routine = columnarRoutine(
      () => column([asNum(1)]).output(([value]) => value!),
      { backends: { cpu: "js-codegen" }, placement: "cpu" },
    );
    await expect(routine.evalAsync(new Map())).resolves.toBe(1);
    routine.dispose();
    routine.dispose();
    await expect(routine.evalAsync(new Map())).rejects.toThrow(/disposed/);
  });

  test("reports columnar compilation checkpoints", () => {
    const checkpoints: string[] = [];
    const routine = columnarRoutine(
      () =>
        column([asNum(1)])
          .map((value) => value.add(1))
          .output(([value]) => value!),
      {
        placement: "cpu",
        backends: { cpu: "js-interp" },
        diagnosticCheckpoint: (phase) => checkpoints.push(phase),
      },
    );

    expect(checkpoints).toEqual([
      "lona:columnar:compile:start",
      "lona:columnar:source:0:cpu:js-interp:compile",
      "lona:columnar:map:1:cpu:js-interp:compile",
      "lona:columnar:output:2:cpu:js-interp:compile",
      "lona:columnar:compile:done",
    ]);
    routine.dispose();
  });

  test("evaluates source-only columnar gradients with identity seeds", async () => {
    const x = variableNum("jvp-source-x");
    const y = variableNum("jvp-source-y");
    const routine = columnarGradRoutine(
      () => column([x.mul(y), x.add(y)]).output(),
      ["jvp-source-x", "jvp-source-y", "absent"],
      { placement: "cpu", backends: { cpu: "js-interp" } },
    );

    expect(routine.shape).toBe("jacobian");
    await expect(
      routine.evalAsync(
        new Map([
          ["jvp-source-x", 2],
          ["jvp-source-y", 3],
        ]),
      ),
    ).resolves.toEqual({
      vals: [6, 5],
      jacobian: [
        [3, 2, 0],
        [1, 1, 0],
      ],
    });
  });

  test("preserves structured flattening order across tangent blocks", async () => {
    const x = variableNum("jvp-structured-x");
    const y = variableNum("jvp-structured-y");
    const routine = columnarGradRoutine(
      () => column([new Pair(x, y), new Pair(x.add(y), x.mul(y))]).output(),
      ["jvp-structured-x", "jvp-structured-y", "jvp-structured-absent"],
      {
        placement: "cpu",
        backends: { cpu: "js-interp" },
        autodiff: { tangentBlockSize: 1 },
      },
    );

    await expect(
      routine.evalAsync(
        new Map([
          ["jvp-structured-x", 2],
          ["jvp-structured-y", 3],
        ]),
      ),
    ).resolves.toEqual({
      vals: [2, 3, 5, 6],
      jacobian: [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [3, 2, 0],
      ],
    });
    routine.dispose();
  });

  test("flattens structured callback outputs in collection/component order", async () => {
    const x = variableNum("jvp-callback-shape-x");
    const y = variableNum("jvp-callback-shape-y");
    const routine = columnarGradRoutine(
      () =>
        column([x, y]).output(([left, right]) => [
          new Pair(left!.add(right!), left!.mul(right!)),
          new Pair(left!, right!),
        ]),
      ["jvp-callback-shape-x", "jvp-callback-shape-y"],
      {
        placement: "cpu",
        backends: { cpu: "js-interp" },
        autodiff: { tangentBlockSize: 1 },
      },
    );

    await expect(
      routine.evalAsync(
        new Map([
          ["jvp-callback-shape-x", 2],
          ["jvp-callback-shape-y", 3],
        ]),
      ),
    ).resolves.toEqual({
      vals: [5, 6, 2, 3],
      jacobian: [
        [1, 1],
        [3, 2],
        [1, 0],
        [0, 1],
      ],
    });
    routine.dispose();
  });

  test("validates tangent block sizes", () => {
    const x = variableNum("jvp-invalid-block-x");
    expect(() =>
      columnarGradRoutine(() => column([x]).output(), ["jvp-invalid-block-x"], {
        autodiff: { tangentBlockSize: 0 },
      }),
    ).toThrow(/tangentBlockSize must be a positive integer/);
  });

  test("evaluates a single flattened source root as a gradient", async () => {
    const x = variableNum("jvp-single-x");
    const routine = columnarGradRoutine(
      () => column([x.square()]).output(),
      ["jvp-single-x"],
    );
    await expect(
      routine.evalAsync(new Map([["jvp-single-x", 4]])),
    ).resolves.toEqual({ val: 16, gradient: [8] });
  });

  test.each(CPU_BACKENDS)(
    "columnar autodiff propagates map/reduce/output tangents on %s",
    async (backend) => {
      const names = [
        `jvp-chain-x-${backend}`,
        `jvp-chain-y-${backend}`,
        `jvp-chain-scale-${backend}`,
        `jvp-chain-offset-${backend}`,
      ];
      const x = variableNum(names[0]!);
      const y = variableNum(names[1]!);
      const scale = variableNum(names[2]!);
      const offset = variableNum(names[3]!);
      const routine = columnarGradRoutine(
        () =>
          column([x, y])
            .map({
              using: { scale },
              build: (value, { index, using }) =>
                value.mul(using.scale).add(index),
              placement: "cpu",
              backend,
            })
            .sum({ placement: "cpu", backend })
            .output({
              using: { offset },
              build: ([total], { using }) => total!.mul(using.offset),
              placement: "cpu",
              backend,
            }),
        names,
        { placement: "cpu", backends: { cpu: backend } },
      );

      const result = await routine.evalAsync(
        new Map([
          [names[0]!, 2],
          [names[1]!, 3],
          [names[2]!, 4],
          [names[3]!, 5],
        ]),
      );
      expect(result).toMatchObject({ val: 105 });
      if (!("gradient" in result)) throw new Error("expected gradient");
      expect(result.gradient[0]).toBeCloseTo(20, 8);
      expect(result.gradient[1]).toBeCloseTo(20, 8);
      expect(result.gradient[2]).toBeCloseTo(25, 8);
      expect(result.gradient[3]).toBeCloseTo(21, 8);
    },
  );

  test("columnar autodiff includes a reduction initial value's tangents", async () => {
    const x = variableNum("jvp-reduce-x");
    const y = variableNum("jvp-reduce-y");
    const initial = variableNum("jvp-reduce-initial");
    const routine = columnarGradRoutine(
      () =>
        column([x, y])
          .reduce((left, right) => left.mul(right), initial, {
            associative: false,
            order: "left",
            placement: "cpu",
            backend: "js-interp",
          })
          .output(),
      ["jvp-reduce-x", "jvp-reduce-y", "jvp-reduce-initial"],
      { placement: "cpu", backends: { cpu: "js-interp" } },
    );
    await expect(
      routine.evalAsync(
        new Map([
          ["jvp-reduce-x", 2],
          ["jvp-reduce-y", 3],
          ["jvp-reduce-initial", 4],
        ]),
      ),
    ).resolves.toEqual({ val: 24, gradient: [12, 8, 6] });
  });

  test("columnar autodiff propagates through then and multiple outputs", async () => {
    const x = variableNum("jvp-then-x");
    const y = variableNum("jvp-then-y");
    const routine = columnarGradRoutine(
      () =>
        column([x, y])
          .then(([left, right]) =>
            column([left!.mul(right!), left!.add(right!)]),
          )
          .output(),
      ["jvp-then-x", "jvp-then-y"],
      { placement: "cpu", backends: { cpu: "js-interp" } },
    );
    await expect(
      routine.evalAsync(
        new Map([
          ["jvp-then-x", 2],
          ["jvp-then-y", 3],
        ]),
      ),
    ).resolves.toEqual({
      vals: [6, 5],
      jacobian: [
        [3, 2],
        [1, 1],
      ],
    });
  });

  test("debug/select operations survive parameter variable binding", async () => {
    const routine = columnarRoutine(
      () =>
        column([variableNum("select-x")])
          .map((value) =>
            value
              .greaterThan(0)
              .and(value.debug("positive"))
              .or(value.not().and(value.neg())),
          )
          .output(([value]) => value!),
      { backends: { cpu: "js-interp" }, placement: "cpu" },
    );

    await expect(routine.evalAsync(new Map([["select-x", 3]]))).resolves.toBe(
      3,
    );
  });
});
