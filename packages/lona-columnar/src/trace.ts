import { Num } from "lona";
import { varNode } from "lona/internal";
import type { NumNode, Variable } from "lona/internal";
import type { ColumnValue, NumBuildResult, StageUsing } from "./types";
import type {
  ResultShape,
  ScalarBindings,
  StructuredKernel,
  StructuredParamInput,
} from "./ir";
import {
  flattenUsing,
  rebuildUsing,
  shapeOfValue,
  type FlatUsing,
  type ValueShape,
} from "./shape";
import { validateKernelRoots } from "./validate";

function makeParams(
  owner: symbol,
  start: number,
  count: number,
): { nodes: Variable[]; nums: Num[] } {
  const label = owner.description ?? "structured-stage";
  const nodes = Array.from({ length: count }, (_, index) =>
    varNode(Symbol(`${label}.param.${start + index}`)),
  );
  return { nodes, nums: nodes.map((node) => new Num(node)) };
}

export interface TracedUsing<TUsing extends StageUsing> {
  readonly bindings: ScalarBindings;
  readonly value: TUsing;
  readonly nextIndex: number;
}

export function traceUsing<TUsing extends StageUsing>(
  owner: symbol,
  using: TUsing,
  startIndex: number,
): TracedUsing<TUsing> {
  const flat: FlatUsing = flattenUsing(using);
  const params = makeParams(owner, startIndex, flat.roots.length);

  return {
    bindings: Object.freeze({
      names: flat.names,
      roots: Object.freeze(flat.roots.map((root) => root.n)),
      params: Object.freeze(params.nodes),
      shapes: flat.shapes,
      width: flat.roots.length,
    }),
    value: rebuildUsing<TUsing>(flat, params.nums),
    nextIndex: startIndex + flat.roots.length,
  };
}

export interface TracedMap<O extends ColumnValue> {
  readonly kernel: StructuredKernel;
  readonly using: ScalarBindings;
  readonly outputShape: ValueShape<O>;
}

export function traceMap<
  I extends ColumnValue,
  TUsing extends StageUsing,
  O extends ColumnValue,
>(
  inputShape: ValueShape<I>,
  using: TUsing,
  build: (value: I, index: Num, usingValue: TUsing) => O,
): TracedMap<O> {
  const owner = Symbol("column.map");
  const input = makeParams(owner, 0, inputShape.width);
  const inputValue = inputShape.rebuild(input.nums);
  const tracedUsing = traceUsing(owner, using, inputShape.width);
  const index = makeParams(owner, tracedUsing.nextIndex, 1);
  const output = build(inputValue, index.nums[0]!, tracedUsing.value);
  const outputShape = shapeOfValue(output, "column.map output");
  const roots = outputShape
    .flatten(output, "column.map output")
    .map((value) => value.n);

  const params = [
    ...input.nodes,
    ...tracedUsing.bindings.params,
    ...index.nodes,
  ];
  validateKernelRoots(params, roots, "column.map");

  const inputs: StructuredParamInput[] = [
    ...input.nodes.map((param, component) => ({
      param,
      binding: { kind: "row" as const, component },
    })),
    ...tracedUsing.bindings.params.map((param, component) => ({
      param,
      binding: { kind: "uniform" as const, component },
    })),
    { param: index.nodes[0]!, binding: { kind: "index" } },
  ];
  return {
    using: tracedUsing.bindings,
    outputShape,
    kernel: Object.freeze({
      owner,
      params: Object.freeze(params),
      inputs: Object.freeze(inputs),
      inputWidth: inputShape.width,
      usingWidth: tracedUsing.bindings.width,
      outputWidth: outputShape.width,
      indexParam: index.nodes[0],
      roots: Object.freeze(roots),
    }),
  };
}

export interface TracedReduce {
  readonly kernel: StructuredKernel;
  readonly using: ScalarBindings;
}

export function traceReduce<T extends ColumnValue, TUsing extends StageUsing>(
  shape: ValueShape<T>,
  using: TUsing,
  combine: (left: T, right: T, usingValue: TUsing) => T,
): TracedReduce {
  const owner = Symbol("column.reduce");
  const left = makeParams(owner, 0, shape.width);
  const right = makeParams(owner, shape.width, shape.width);
  const tracedUsing = traceUsing(owner, using, shape.width * 2);
  const output = combine(
    shape.rebuild(left.nums),
    shape.rebuild(right.nums),
    tracedUsing.value,
  );
  const roots = shape
    .flatten(output, "column.reduce output")
    .map((num) => num.n);
  const params = [
    ...left.nodes,
    ...right.nodes,
    ...tracedUsing.bindings.params,
  ];
  validateKernelRoots(params, roots, "column.reduce");

  const inputs: StructuredParamInput[] = [
    ...left.nodes.map((param, component) => ({
      param,
      binding: { kind: "reduce-left" as const, component },
    })),
    ...right.nodes.map((param, component) => ({
      param,
      binding: { kind: "reduce-right" as const, component },
    })),
    ...tracedUsing.bindings.params.map((param, component) => ({
      param,
      binding: { kind: "uniform" as const, component },
    })),
  ];

  return {
    using: tracedUsing.bindings,
    kernel: Object.freeze({
      owner,
      params: Object.freeze(params),
      inputs: Object.freeze(inputs),
      inputWidth: shape.width * 2,
      usingWidth: tracedUsing.bindings.width,
      outputWidth: shape.width,
      roots: Object.freeze(roots),
    }),
  };
}

export interface TracedToNums {
  readonly params: readonly Variable[];
  readonly inputs: readonly StructuredParamInput[];
  readonly using: ScalarBindings;
  readonly roots: readonly NumNode[];
  readonly resultShape: ResultShape;
}

function flattenBuildResult(result: NumBuildResult): {
  roots: NumNode[];
  shape: ResultShape;
} {
  const values: readonly ColumnValue[] = Array.isArray(result)
    ? (result as readonly ColumnValue[])
    : [result as ColumnValue];
  if (values.length === 0) throw new Error("column.toNums produced no values");

  const shapes: ValueShape[] = [];
  const roots: NumNode[] = [];
  for (let i = 0; i < values.length; i++) {
    const value = values[i]!;
    const shape = shapeOfValue(value, `column.toNums output ${i}`);
    shapes.push(shape);
    roots.push(
      ...shape
        .flatten(value, `column.toNums output ${i}`)
        .map((component) => component.n),
    );
  }

  return {
    roots,
    shape: Object.freeze({
      collection: Array.isArray(result) ? "array" : "single",
      values: Object.freeze(shapes),
    }),
  };
}

export function traceToNums<
  T extends ColumnValue,
  TUsing extends StageUsing,
  R extends NumBuildResult,
>(
  shape: ValueShape<T>,
  count: number,
  using: TUsing,
  build: (values: readonly T[], usingValue: TUsing) => R,
): TracedToNums {
  const owner = Symbol("column.toNums");
  const source = makeParams(owner, 0, count * shape.width);
  const values = Array.from({ length: count }, (_, row) =>
    shape.rebuild(
      source.nums.slice(row * shape.width, (row + 1) * shape.width),
    ),
  );
  const tracedUsing = traceUsing(owner, using, source.nodes.length);
  const result = flattenBuildResult(build(values, tracedUsing.value));
  const params = [...source.nodes, ...tracedUsing.bindings.params];
  validateKernelRoots(params, result.roots, "column.toNums");

  const inputs: StructuredParamInput[] = [
    ...source.nodes.map((param, component) => ({
      param,
      binding: { kind: "materialized" as const, component },
    })),
    ...tracedUsing.bindings.params.map((param, component) => ({
      param,
      binding: { kind: "uniform" as const, component },
    })),
  ];

  return {
    params: Object.freeze(source.nodes),
    inputs: Object.freeze(inputs),
    using: tracedUsing.bindings,
    roots: Object.freeze(result.roots),
    resultShape: result.shape,
  };
}
