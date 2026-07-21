import { describe, expect, test } from "vitest";
import type { NumStruct } from "lona";
import { Num, asNum, variableNum } from "lona";
import { isGpuAvailable } from "lona/internal";
import type { CpuBackendName } from "../types";
import { buildStructuredRoutine, column } from "../api";

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

describe("structured CPU runtime", () => {
  test.each(CPU_BACKENDS)(
    "map -> reduce -> toNums executes on %s",
    async (backend) => {
      const x = variableNum("runtime-x");
      const y = variableNum("runtime-y");
      const offset = variableNum("runtime-offset");
      const scale = variableNum("runtime-scale");

      const routine = buildStructuredRoutine(
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
            .toNums({
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
        "toNums",
      ]);
      routine.dispose();
    },
  );

  test("executes chained NumStruct maps and decodes a struct result", async () => {
    const x = variableNum("pair-x");
    const y = variableNum("pair-y");
    const zero = new Pair(asNum(0), asNum(0));

    const routine = buildStructuredRoutine(
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
          .toNums(([total]) => total!),
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
    const routine = buildStructuredRoutine(
      () =>
        column([new Pair(asNum(1), asNum(2))]).toNums(([pair]) => [
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
    const routine = buildStructuredRoutine(
      () =>
        column([], { shape: zero })
          .reduce((left, right) => left.add(right), zero, {
            associative: true,
            order: "tree",
          })
          .toNums(([result]) => result!),
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
      const routine = buildStructuredRoutine(
        () => reduced.toNums(([value]) => value!),
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
      return buildStructuredRoutine(() => reduced.toNums(([value]) => value!), {
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
    const pairRoutine = buildStructuredRoutine(
      () =>
        column([new Pair(asNum(1), asNum(2)), new Pair(asNum(3), asNum(4))])
          .sum({ componentWise: true })
          .toNums(([value]) => value!),
      { backends: { cpu: "wasm-codegen" }, placement: "cpu" },
    );
    await expect(pairRoutine.evalAsync(new Map())).resolves.toEqual([4, 6]);

    const emptyRoutine = buildStructuredRoutine(
      () =>
        column([], { shape: pairShape })
          .sum({ componentWise: true })
          .toNums(([value]) => value!),
      { backends: { cpu: "js-interp" }, placement: "cpu" },
    );
    await expect(emptyRoutine.evalAsync(new Map())).resolves.toEqual([0, 0]);
  });

  test("left and tree reductions have explicit ordering", async () => {
    const source = [asNum(1), asNum(2), asNum(3)];
    const make = (order: "left" | "tree") =>
      buildStructuredRoutine(
        () =>
          column(source)
            .reduce((left, right) => left.sub(right), asNum(0), {
              associative: order === "tree",
              order,
            })
            .toNums(([result]) => result!),
        { backends: { cpu: "js-interp" }, placement: "cpu" },
      );

    await expect(make("left").evalAsync(new Map())).resolves.toBe(-6);
    await expect(make("tree").evalAsync(new Map())).resolves.toBe(0);
  });

  test("hard placement is enforced without explicit transfer stages", async () => {
    const cpu = buildStructuredRoutine(
      () =>
        column([asNum(2)])
          .map((value) => value.square(), { placement: "cpu" })
          .toNums(([value]) => value!, { placement: "cpu" }),
      { backends: { cpu: "js-interp" } },
    );
    await expect(cpu.evalAsync(new Map())).resolves.toBe(4);

    expect(() =>
      buildStructuredRoutine(() =>
        column([asNum(1)]).toNums(([value]) => value!, {
          placement: "gpu",
        }),
      ),
    ).toThrow(/toNums stages execute on CPU/);
  });

  test("map/reduce default to GPU-first auto and permit per-kind placement", async () => {
    const preferredMap = buildStructuredRoutine(
      () =>
        column([asNum(2)])
          .map((value) => value.square())
          .toNums(([value]) => value!),
      { backends: { cpu: "js-interp" } },
    );
    const mapStage = preferredMap.stages.find((stage) => stage.kind === "map");
    expect(mapStage?.backend).toBe(
      isGpuAvailable() ? "gpu-codegen" : "js-interp",
    );
    await expect(preferredMap.evalAsync(new Map())).resolves.toBe(4);

    const ordered = buildStructuredRoutine(
      () =>
        column([asNum(4), asNum(1)])
          .reduce((left, right) => left.sub(right), asNum(0), {
            associative: false,
            order: "left",
          })
          .toNums(([value]) => value!),
      { backends: { cpu: "js-interp" } },
    );
    expect(
      ordered.stages.find((stage) => stage.kind === "reduce")?.backend,
    ).toBe("js-interp");
    await expect(ordered.evalAsync(new Map())).resolves.toBe(-5);

    const forcedCpu = buildStructuredRoutine(
      () =>
        column([asNum(3)])
          .map((value) => value.add(1))
          .sum()
          .toNums(([value]) => value!),
      {
        backends: { cpu: "js-interp" },
        placement: { map: "cpu", reduce: "cpu", toNums: "cpu" },
      },
    );
    expect(
      forcedCpu.stages
        .filter((stage) => stage.kind === "map" || stage.kind === "reduce")
        .map((stage) => stage.backend),
    ).toEqual(["js-interp", "js-interp"]);
    await expect(forcedCpu.evalAsync(new Map())).resolves.toBe(4);
  });

  test("auto target and backend preferences resolve by stage kind", async () => {
    const routine = buildStructuredRoutine(
      () =>
        column([asNum(2), asNum(3)])
          .map((value) => value.square())
          .sum()
          .toNums(([value]) => value!),
      {
        placement: {
          map: "auto",
          reduce: "cpu",
          toNums: "cpu",
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
      { kind: "source", placement: undefined, backend: undefined },
      { kind: "map", placement: "cpu", backend: "js-codegen" },
      { kind: "reduce", placement: "cpu", backend: "js-codegen" },
      { kind: "toNums", placement: "cpu", backend: "js-codegen" },
    ]);
    await expect(routine.evalAsync(new Map())).resolves.toBe(13);
  });

  test("reports evaluation costs and stage timings", async () => {
    const routine = buildStructuredRoutine(
      () =>
        column([asNum(1), asNum(2)])
          .sum()
          .toNums(([value]) => value!),
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
    const routine = buildStructuredRoutine(
      () => column([asNum(1)]).toNums(([value]) => value!),
      { backends: { cpu: "js-codegen" }, placement: "cpu" },
    );
    await expect(routine.evalAsync(new Map())).resolves.toBe(1);
    routine.dispose();
    routine.dispose();
    await expect(routine.evalAsync(new Map())).rejects.toThrow(/disposed/);
  });

  test("reports structured compilation checkpoints", () => {
    const checkpoints: string[] = [];
    const routine = buildStructuredRoutine(
      () =>
        column([asNum(1)])
          .map((value) => value.add(1))
          .toNums(([value]) => value!),
      {
        placement: "cpu",
        backends: { cpu: "js-interp" },
        diagnosticCheckpoint: (phase) => checkpoints.push(phase),
      },
    );

    expect(checkpoints).toEqual([
      "lona:structured:compile:start",
      "lona:structured:map:1:cpu:js-interp:compile",
      "lona:structured:to-nums:2:cpu:js-interp:compile",
      "lona:structured:compile:done",
    ]);
    routine.dispose();
  });

  test("debug/select operations survive parameter variable binding", async () => {
    const routine = buildStructuredRoutine(
      () =>
        column([variableNum("select-x")])
          .map((value) =>
            value
              .greaterThan(0)
              .and(value.debug("positive"))
              .or(value.not().and(value.neg())),
          )
          .toNums(([value]) => value!),
      { backends: { cpu: "js-interp" }, placement: "cpu" },
    );

    await expect(routine.evalAsync(new Map([["select-x", 3]]))).resolves.toBe(
      3,
    );
  });
});
