import { asNum, type Num } from "lona";
import type { NumStruct } from "lona";
import { StructuredBuilder, type StructuredDefinition } from "./ir";
import { emptyUsing, shapeOfValue, type ValueShape } from "./shape";
import { traceMap, traceReduce, traceToNums } from "./trace";
import { compileStructuredRoutine } from "./runtime";
import {
  STRUCTURED_VALUE_RESULT,
  type BuiltInReductionOptions,
  type Column,
  type ColumnValue,
  type ComponentWiseReductionOptions,
  type EmptyColumnOptions,
  type EmptyUsing,
  type MapContext,
  type NumBuildResult,
  type ReducedColumn,
  type ReductionOptions,
  type StageOptions,
  type StagePlacement,
  type StageUsing,
  type StructuredCompileOptions,
  type StructuredRoutine,
  type StructuredValue,
  type UsingContext,
} from "./types";

interface MapSpec<
  T extends ColumnValue,
  TUsing extends StageUsing,
  U extends ColumnValue,
> {
  using: TUsing;
  build: (value: T, context: MapContext<TUsing>) => U;
  placement?: StagePlacement;
}

interface ReduceSpec<T extends ColumnValue, TUsing extends StageUsing> {
  using: TUsing;
  combine: (left: T, right: T, context: UsingContext<TUsing>) => T;
  initial: T;
  associative: boolean;
  order?: ReductionOptions["order"];
  placement?: StagePlacement;
}

interface ToNumsSpec<
  T extends ColumnValue,
  TUsing extends StageUsing,
  R extends NumBuildResult,
> {
  using: TUsing;
  build: (values: readonly T[], context: UsingContext<TUsing>) => R;
  placement?: StagePlacement;
}

class StructuredValueHandle<
  R extends NumBuildResult,
> implements StructuredValue<R> {
  declare readonly [STRUCTURED_VALUE_RESULT]: R;

  constructor(readonly definition: StructuredDefinition) {}
}

class ColumnHandle<T extends ColumnValue> implements Column<T> {
  constructor(
    private readonly builder: StructuredBuilder,
    private readonly stageId: number,
    readonly length: number,
    private readonly shape: ValueShape<T>,
  ) {}

  map<U extends ColumnValue>(
    build: (value: T, context: MapContext) => U,
    opts?: StageOptions,
  ): Column<U>;
  map<TUsing extends StageUsing, U extends ColumnValue>(
    spec: MapSpec<T, TUsing, U>,
  ): Column<U>;
  map<TUsing extends StageUsing, U extends ColumnValue>(
    buildOrSpec:
      | ((value: T, context: MapContext<EmptyUsing>) => U)
      | MapSpec<T, TUsing, U>,
    opts: StageOptions = {},
  ): Column<U> {
    let using: TUsing | EmptyUsing;
    let build: (value: T, context: MapContext<TUsing | EmptyUsing>) => U;
    let stageOptions: StageOptions;

    if (typeof buildOrSpec === "function") {
      using = emptyUsing<EmptyUsing>();
      build = buildOrSpec as (
        value: T,
        context: MapContext<TUsing | EmptyUsing>,
      ) => U;
      stageOptions = opts;
    } else {
      using = buildOrSpec.using;
      build = buildOrSpec.build as (
        value: T,
        context: MapContext<TUsing | EmptyUsing>,
      ) => U;
      stageOptions = buildOrSpec;
    }

    const traced = traceMap(this.shape, using, (value, index, usingValue) =>
      build(value, { index, using: usingValue }),
    );
    const stage = this.builder.add({
      kind: "map",
      source: this.stageId,
      count: this.length,
      inputShape: this.shape,
      outputShape: traced.outputShape,
      using: traced.using,
      kernel: traced.kernel,
      requestedPlacement: stageOptions.placement,
    });

    return new ColumnHandle(
      this.builder,
      stage.id,
      this.length,
      traced.outputShape,
    );
  }

  reduce(
    combine: (left: T, right: T) => T,
    initial: T,
    opts: ReductionOptions,
  ): ReducedColumn<T>;
  reduce<TUsing extends StageUsing>(
    spec: ReduceSpec<T, TUsing>,
  ): ReducedColumn<T>;
  reduce<TUsing extends StageUsing>(
    combineOrSpec: ((left: T, right: T) => T) | ReduceSpec<T, TUsing>,
    initial?: T,
    opts?: ReductionOptions,
  ): ReducedColumn<T> {
    let using: TUsing | EmptyUsing;
    let combine: (
      left: T,
      right: T,
      context: UsingContext<TUsing | EmptyUsing>,
    ) => T;
    let reductionInitial: T;
    let reductionOptions: ReductionOptions;

    if (typeof combineOrSpec === "function") {
      if (initial === undefined || opts === undefined) {
        throw new Error("column.reduce requires initial and options");
      }
      using = emptyUsing<EmptyUsing>();
      combine = (left, right) => combineOrSpec(left, right);
      reductionInitial = initial;
      reductionOptions = opts;
    } else {
      using = combineOrSpec.using;
      combine = combineOrSpec.combine as (
        left: T,
        right: T,
        context: UsingContext<TUsing | EmptyUsing>,
      ) => T;
      reductionInitial = combineOrSpec.initial;
      reductionOptions = combineOrSpec;
    }

    if (reductionOptions.placement === "gpu" && !reductionOptions.associative) {
      throw new Error("GPU column.reduce requires associative: true");
    }

    const initialRoots = this.shape
      .flatten(reductionInitial, "column.reduce initial")
      .map((value) => value.n);
    const traced = traceReduce(this.shape, using, (left, right, usingValue) =>
      combine(left, right, { using: usingValue }),
    );
    const stage = this.builder.add({
      kind: "reduce",
      source: this.stageId,
      inputCount: this.length,
      count: 1 as const,
      shape: this.shape,
      initial: Object.freeze(initialRoots),
      using: traced.using,
      kernel: traced.kernel,
      associative: reductionOptions.associative,
      order:
        reductionOptions.order ??
        (reductionOptions.associative ? "tree" : "left"),
      requestedPlacement: reductionOptions.placement,
    });

    return new ColumnHandle(
      this.builder,
      stage.id,
      1,
      this.shape,
    ) as ReducedColumn<T>;
  }

  private reduceBuiltIn(
    operation: "sum" | "product" | "min" | "max",
    opts: BuiltInReductionOptions | ComponentWiseReductionOptions = {},
  ): ReducedColumn<T> {
    if (this.shape.kind === "struct" && !("componentWise" in opts)) {
      throw new Error(
        `column.${operation} on a NumStruct requires componentWise: true`,
      );
    }
    const identity =
      operation === "sum"
        ? 0
        : operation === "product"
          ? 1
          : operation === "min"
            ? Infinity
            : -Infinity;
    const initialValue = this.shape.rebuild(
      Array.from({ length: this.shape.width }, () => asNum(identity)),
    );
    const using = emptyUsing<EmptyUsing>();
    const traced = traceReduce(this.shape, using, (left, right) => {
      const leftParts = this.shape.flatten(left);
      const rightParts = this.shape.flatten(right);
      return this.shape.rebuild(
        leftParts.map((value, index) => {
          const other = rightParts[index]!;
          switch (operation) {
            case "sum":
              return value.add(other);
            case "product":
              return value.mul(other);
            case "min":
              return value.min(other);
            case "max":
              return value.max(other);
          }
        }),
      );
    });
    const stage = this.builder.add({
      kind: "reduce",
      source: this.stageId,
      inputCount: this.length,
      count: 1 as const,
      shape: this.shape,
      initial: Object.freeze(
        this.shape.flatten(initialValue).map((value) => value.n),
      ),
      using: traced.using,
      kernel: traced.kernel,
      associative: true,
      order: opts.order ?? "tree",
      builtIn: operation,
      requestedPlacement: opts.placement,
    });
    return new ColumnHandle(
      this.builder,
      stage.id,
      1,
      this.shape,
    ) as ReducedColumn<T>;
  }

  sum(
    this: ColumnHandle<Num>,
    opts?: BuiltInReductionOptions,
  ): ReducedColumn<T>;
  sum(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  sum(
    opts: BuiltInReductionOptions | ComponentWiseReductionOptions = {},
  ): ReducedColumn<T> {
    return this.reduceBuiltIn("sum", opts);
  }

  product(
    this: ColumnHandle<Num>,
    opts?: BuiltInReductionOptions,
  ): ReducedColumn<T>;
  product(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  product(
    opts: BuiltInReductionOptions | ComponentWiseReductionOptions = {},
  ): ReducedColumn<T> {
    return this.reduceBuiltIn("product", opts);
  }

  min(
    this: ColumnHandle<Num>,
    opts?: BuiltInReductionOptions,
  ): ReducedColumn<T>;
  min(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  min(
    opts: BuiltInReductionOptions | ComponentWiseReductionOptions = {},
  ): ReducedColumn<T> {
    return this.reduceBuiltIn("min", opts);
  }

  max(
    this: ColumnHandle<Num>,
    opts?: BuiltInReductionOptions,
  ): ReducedColumn<T>;
  max(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  max(
    opts: BuiltInReductionOptions | ComponentWiseReductionOptions = {},
  ): ReducedColumn<T> {
    return this.reduceBuiltIn("max", opts);
  }

  toNums<R extends NumBuildResult>(
    build: (values: readonly T[]) => R,
    opts?: StageOptions,
  ): StructuredValue<R>;
  toNums<TUsing extends StageUsing, R extends NumBuildResult>(
    spec: ToNumsSpec<T, TUsing, R>,
  ): StructuredValue<R>;
  toNums<TUsing extends StageUsing, R extends NumBuildResult>(
    buildOrSpec: ((values: readonly T[]) => R) | ToNumsSpec<T, TUsing, R>,
    opts: StageOptions = {},
  ): StructuredValue<R> {
    let using: TUsing | EmptyUsing;
    let build: (
      values: readonly T[],
      context: UsingContext<TUsing | EmptyUsing>,
    ) => R;
    let stageOptions: StageOptions;

    if (typeof buildOrSpec === "function") {
      using = emptyUsing<EmptyUsing>();
      build = (values) => buildOrSpec(values);
      stageOptions = opts;
    } else {
      using = buildOrSpec.using;
      build = buildOrSpec.build as (
        values: readonly T[],
        context: UsingContext<TUsing | EmptyUsing>,
      ) => R;
      stageOptions = buildOrSpec;
    }

    const traced = traceToNums(
      this.shape,
      this.length,
      using,
      (values, usingValue) => build(values, { using: usingValue }),
    );
    const stage = this.builder.add({
      kind: "to-nums",
      source: this.stageId,
      sourceCount: this.length,
      sourceShape: this.shape,
      params: traced.params,
      inputs: traced.inputs,
      using: traced.using,
      roots: traced.roots,
      resultShape: traced.resultShape,
      requestedPlacement: stageOptions.placement,
    });

    return new StructuredValueHandle<R>(this.builder.definition(stage.id));
  }

  /** @internal Test and future compiler hook. */
  definition(): StructuredDefinition {
    return this.builder.definition(this.stageId);
  }
}

export function column(values: readonly Num[]): Column<Num>;
export function column<T extends NumStruct<T>>(values: readonly T[]): Column<T>;
export function column(
  values: readonly [],
  options: EmptyColumnOptions<Num>,
): Column<Num>;
export function column<T extends NumStruct<T>>(
  values: readonly [],
  options: EmptyColumnOptions<T>,
): Column<T>;
export function column<T extends ColumnValue>(
  values: readonly T[],
  options?: { readonly shape: ColumnValue },
): Column<T> {
  const copiedValues = Object.freeze([...values]);
  const witness = copiedValues[0] ?? options?.shape;
  if (witness === undefined) {
    throw new Error(
      "column requires at least one value or an empty-column shape",
    );
  }

  const shape = shapeOfValue(witness, "column shape");
  const roots = copiedValues.flatMap((value, index) =>
    shape.flatten(value, `column value ${index}`).map((num) => num.n),
  );
  const builder = new StructuredBuilder();
  const source = builder.add({
    kind: "source",
    count: copiedValues.length,
    shape,
    roots: Object.freeze(roots),
  });

  return new ColumnHandle(builder, source.id, copiedValues.length, shape);
}

/** @internal Inspect a partially-built column graph. */
export function getColumnDefinition<T extends ColumnValue>(
  value: Column<T>,
): StructuredDefinition {
  if (!(value instanceof ColumnHandle)) {
    throw new Error("Column was not created by lona column()");
  }
  return value.definition();
}

/** @internal Future compiler hook and structural test helper. */
export function getStructuredDefinition<R extends NumBuildResult>(
  value: StructuredValue<R>,
): StructuredDefinition {
  if (!(value instanceof StructuredValueHandle)) {
    throw new Error("StructuredValue was not created by lona column.toNums()");
  }
  return value.definition;
}

export function buildStructuredRoutine<R extends NumBuildResult>(
  build: () => StructuredValue<R>,
  opts?: StructuredCompileOptions,
): StructuredRoutine<R> {
  return compileStructuredRoutine<R>(getStructuredDefinition(build()), opts);
}
