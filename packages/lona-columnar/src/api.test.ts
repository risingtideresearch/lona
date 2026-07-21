import { describe, expect, test } from "vitest";
import { Num, asNum, variableNum } from "lona";
import type { NumStruct } from "lona";
import { derivativeNode, foreignFnNode, litNode, varNode } from "lona/internal";
import { Variable } from "lona/internal";
import { column, getColumnDefinition, getStructuredDefinition } from "./api";

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

function lastStage<T extends { readonly kind: string }>(
  stages: readonly T[],
): T {
  const stage = stages[stages.length - 1];
  if (!stage) throw new Error("expected a stage");
  return stage;
}

describe("structured column Phase 1 builder", () => {
  test("builds a homogeneous scalar source stage", () => {
    const x = variableNum("x");
    const y = variableNum("y");
    const values = column([x, y]);
    const definition = getColumnDefinition(values);
    const source = definition.stages[0]!;

    expect(values.length).toBe(2);
    expect(source.kind).toBe("source");
    if (source.kind !== "source") throw new Error("expected source");
    expect(source.count).toBe(2);
    expect(source.shape.kind).toBe("num");
    expect(source.shape.width).toBe(1);
    expect(source.roots).toEqual([x.n, y.n]);
    expect(Object.isFrozen(source.roots)).toBe(true);
  });

  test("validates NumStruct width and supports an empty shape witness", () => {
    const zero = new Pair(asNum(0), asNum(0));
    const empty = column([], { shape: zero });
    const source = getColumnDefinition(empty).stages[0]!;

    expect(empty.length).toBe(0);
    if (source.kind !== "source") throw new Error("expected source");
    expect(source.shape.kind).toBe("struct");
    expect(source.shape.width).toBe(2);
    expect(source.roots).toHaveLength(0);

    const malformed = {
      toNums: () => [asNum(1)],
      fromNums: ([x]: Num[]) => new Pair(x!, asNum(0)),
    } as unknown as Pair;
    expect(() => column([zero, malformed])).toThrow(/components; expected 2/);
    expect(() => column([] as Num[])).toThrow(/at least one value/);
  });

  test("traces one map kernel over private Variable-backed NumStruct values", () => {
    const points = [
      new Pair(variableNum("x0"), variableNum("y0")),
      new Pair(variableNum("x1"), variableNum("y1")),
      new Pair(variableNum("x2"), variableNum("y2")),
    ];
    let traces = 0;
    const mapped = column(points).map((point, { index }) => {
      traces++;
      expect(point.x.n).toBeInstanceOf(Variable);
      expect(point.y.n).toBeInstanceOf(Variable);
      expect(index.n).toBeInstanceOf(Variable);
      expect((point.x.n as Variable).name).toBeTypeOf("symbol");
      expect(((point.x.n as Variable).name as symbol).description).toBe(
        "column.map.param.0",
      );
      expect(((point.y.n as Variable).name as symbol).description).toBe(
        "column.map.param.1",
      );
      expect(((index.n as Variable).name as symbol).description).toBe(
        "column.map.param.2",
      );
      return new Pair(point.x.add(index), point.y.mul(2));
    });
    const stage = lastStage(getColumnDefinition(mapped).stages);

    expect(traces).toBe(1);
    expect(mapped.length).toBe(3);
    expect(stage.kind).toBe("map");
    if (stage.kind !== "map") throw new Error("expected map");
    expect(stage.kernel.inputWidth).toBe(2);
    expect(stage.kernel.usingWidth).toBe(0);
    expect(stage.kernel.outputWidth).toBe(2);
    expect(stage.kernel.params).toHaveLength(3);
    expect(stage.kernel.inputs.map((input) => input.binding.kind)).toEqual([
      "row",
      "row",
      "index",
    ]);
    expect(stage.kernel.indexParam).toBe(stage.kernel.params[2]);
    expect(stage.kernel.roots).toHaveLength(2);
  });

  test("hash-conses repeated expressions within a kernel but isolates owners", () => {
    const build = (point: Pair): Pair => {
      const first = point.x.add(point.y);
      const repeated = point.x.add(point.y);
      expect(first.n).toBe(repeated.n);
      return new Pair(first, repeated);
    };

    const firstMap = column([new Pair(asNum(1), asNum(2))]).map(build);
    const secondMap = column([new Pair(asNum(1), asNum(2))]).map(build);
    const firstStage = lastStage(getColumnDefinition(firstMap).stages);
    const secondStage = lastStage(getColumnDefinition(secondMap).stages);
    if (firstStage.kind !== "map" || secondStage.kind !== "map") {
      throw new Error("expected map stages");
    }

    expect(firstStage.kernel.roots[0]).toBe(firstStage.kernel.roots[1]);
    expect(firstStage.kernel.owner).not.toBe(secondStage.kernel.owner);
    expect(firstStage.kernel.params[0]).not.toBe(secondStage.kernel.params[0]);
    expect(firstStage.kernel.roots[0]).not.toBe(secondStage.kernel.roots[0]);
  });

  test("preserves explicit scalar and NumStruct using dependencies", () => {
    const waterline = variableNum("waterline");
    const offset = new Pair(variableNum("ox"), variableNum("oy"));
    const mapped = column([new Pair(asNum(1), asNum(2))]).map({
      using: { waterline, offset },
      build: (point, { using }) => {
        expect(using.waterline.n).toBeInstanceOf(Variable);
        expect(using.offset.x.n).toBeInstanceOf(Variable);
        return new Pair(
          point.x.add(using.waterline).add(using.offset.x),
          point.y.add(using.offset.y),
        );
      },
      placement: "gpu",
    });
    const stage = lastStage(getColumnDefinition(mapped).stages);

    if (stage.kind !== "map") throw new Error("expected map");
    expect(stage.using.names).toEqual(["waterline", "offset"]);
    expect(stage.using.width).toBe(3);
    expect(stage.using.roots).toEqual([waterline.n, offset.x.n, offset.y.n]);
    expect(stage.kernel.usingWidth).toBe(3);
    expect(stage.kernel.inputs.map((input) => input.binding.kind)).toEqual([
      "row",
      "row",
      "uniform",
      "uniform",
      "uniform",
      "index",
    ]);
    expect(stage.requestedPlacement).toBe("gpu");
  });

  test("marks built-in reductions and requires component-wise structs", () => {
    const scalarStage = lastStage(
      getColumnDefinition(
        column([asNum(1), asNum(2)]).sum({ placement: "gpu" }),
      ).stages,
    );
    if (scalarStage.kind !== "reduce") throw new Error("expected reduce");
    expect(scalarStage.builtIn).toBe("sum");
    expect(scalarStage.order).toBe("tree");
    expect(scalarStage.initial).toHaveLength(1);

    const pair = new Pair(asNum(1), asNum(2));
    expect(() =>
      (column([pair]) as unknown as { sum(): unknown }).sum(),
    ).toThrow(/componentWise: true/);
    const pairStage = lastStage(
      getColumnDefinition(column([pair]).max({ componentWise: true })).stages,
    );
    if (pairStage.kind !== "reduce") throw new Error("expected reduce");
    expect(pairStage.builtIn).toBe("max");
    expect(pairStage.shape.width).toBe(2);
  });

  test("rejects undeclared captures and unsupported scalar nodes", () => {
    const external = variableNum("external");
    const source = column([asNum(1)]);

    expect(() => source.map((value) => value.add(external))).toThrow(
      /declare it under `using`/,
    );

    const foreign = new Num(
      foreignFnNode(
        [],
        () => 1,
        () => litNode(0),
      ),
    );
    expect(() => source.map(() => foreign)).toThrow(/ForeignFn/);

    const derivative = new Num(derivativeNode(varNode("x")));
    expect(() => source.map(() => derivative)).toThrow(/Derivative/);
  });

  test("rejects a parameter Variable captured from another stage", () => {
    const first = column([asNum(1)]).map((value) => value.add(1));
    const firstStage = lastStage(getColumnDefinition(first).stages);
    if (firstStage.kind !== "map") throw new Error("expected map");
    const foreignParam = new Num(firstStage.kernel.params[0]!);

    expect(() =>
      column([asNum(2)]).map((value) => value.add(foreignParam)),
    ).toThrow(/declare it under `using`/);
  });

  test("traces reduction shape, using dependencies, and placement contract", () => {
    const zero = new Pair(asNum(0), asNum(0));
    const scale = variableNum("scale");
    const source = column([
      new Pair(variableNum("x0"), variableNum("y0")),
      new Pair(variableNum("x1"), variableNum("y1")),
    ]);
    let traces = 0;
    const reduced = source.reduce({
      using: { scale },
      combine: (left, right, { using }) => {
        traces++;
        expect(using.scale.n).toBeInstanceOf(Variable);
        return left.add(right);
      },
      initial: zero,
      associative: true,
      order: "tree",
      placement: "gpu",
    });
    const stage = lastStage(getColumnDefinition(reduced).stages);

    expect(traces).toBe(1);
    expect(reduced.length).toBe(1);
    if (stage.kind !== "reduce") throw new Error("expected reduce");
    expect(stage.inputCount).toBe(2);
    expect(stage.kernel.inputWidth).toBe(4);
    expect(stage.kernel.usingWidth).toBe(1);
    expect(stage.kernel.params).toHaveLength(5);
    expect(stage.kernel.inputs.map((input) => input.binding.kind)).toEqual([
      "reduce-left",
      "reduce-left",
      "reduce-right",
      "reduce-right",
      "uniform",
    ]);
    expect(stage.initial).toEqual([zero.x.n, zero.y.n]);

    expect(() =>
      source.reduce((a, b) => a.add(b), zero, {
        associative: false,
        placement: "gpu",
      }),
    ).toThrow(/associative: true/);
  });

  test("toNums binds every row and external value into a terminal stage", () => {
    const density = variableNum("density");
    const points = column([
      new Pair(variableNum("x0"), variableNum("y0")),
      new Pair(variableNum("x1"), variableNum("y1")),
    ]);
    const output = points.toNums({
      using: { density },
      build: ([a, b], { using }) => [
        a!.x.add(b!.x).mul(using.density),
        a!.y.add(b!.y),
      ],
      placement: "cpu",
    });
    const definition = getStructuredDefinition(output);
    const stage = lastStage(definition.stages);

    expect(definition.stages.map(({ kind }) => kind)).toEqual([
      "source",
      "to-nums",
    ]);

    if (stage.kind !== "to-nums") throw new Error("expected to-nums");
    expect(stage.params).toHaveLength(4);
    expect(stage.inputs.map((input) => input.binding.kind)).toEqual([
      "materialized",
      "materialized",
      "materialized",
      "materialized",
      "uniform",
    ]);
    expect(stage.using.width).toBe(1);
    expect(stage.using.roots).toEqual([density.n]);
    expect(stage.roots).toHaveLength(2);
    expect(stage.resultShape.collection).toBe("array");
    expect(stage.resultShape.values).toHaveLength(2);
    expect(stage.requestedPlacement).toBe("cpu");
    expect(definition.outputStage).toBe(stage.id);
  });

  test("allows literal captures and rejects empty toNums output", () => {
    const literal = asNum(3);
    expect(() =>
      column([asNum(1)]).map((value) => value.add(literal)),
    ).not.toThrow();

    expect(() => column([asNum(1)]).toNums(() => [] as readonly Num[])).toThrow(
      /produced no values/,
    );
  });
});
