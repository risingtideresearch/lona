import { asNum, Num } from "lona";
import type { NumStruct } from "lona";
import { ColumnarBuilder, type ColumnarDefinition } from "./ir";
import { emptyUsing, shapeOfValue, type ValueShape } from "./shape";
import { traceMap, traceReduce, traceWholeColumn } from "./trace";
import { compileColumnarRoutine } from "./runtime";
import {
  COLUMNAR_OUTPUT_RESULT,
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
  type SourceOptions,
  type StageOptions,
  type StagePlacement,
  type StageUsing,
  type ColumnarRoutineOptions,
  type ColumnarRoutine,
  type ColumnarOutput,
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
  backend?: StageOptions["backend"];
}

interface ReduceSpec<T extends ColumnValue, TUsing extends StageUsing> {
  using: TUsing;
  combine: (left: T, right: T, context: UsingContext<TUsing>) => T;
  initial: T;
  associative: boolean;
  order?: ReductionOptions["order"];
  placement?: StagePlacement;
  backend?: StageOptions["backend"];
}

interface OutputSpec<
  T extends ColumnValue,
  TUsing extends StageUsing,
  R extends NumBuildResult,
> {
  using: TUsing;
  build: (values: readonly T[], context: UsingContext<TUsing>) => R;
  placement?: StagePlacement;
  backend?: StageOptions["backend"];
}

interface ThenSpec<
  T extends ColumnValue,
  TUsing extends StageUsing,
  U extends ColumnValue,
> {
  using: TUsing;
  build: (values: readonly T[], context: UsingContext<TUsing>) => Column<U>;
  placement?: StagePlacement;
  backend?: StageOptions["backend"];
}

class ColumnarOutputHandle<
  R extends NumBuildResult,
> implements ColumnarOutput<R> {
  declare readonly [COLUMNAR_OUTPUT_RESULT]: R;

  constructor(readonly definition: ColumnarDefinition) {}
}

class ColumnHandle<T extends ColumnValue> implements Column<T> {
  constructor(
    private readonly builder: ColumnarBuilder,
    private readonly stageId: number,
    readonly length: number,
    private readonly shape: ValueShape<T>,
    private readonly outputCollection: "single" | "array" = "array",
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
      requestedBackend: stageOptions.backend,
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
      requestedBackend: reductionOptions.backend,
    });

    return new ColumnHandle(
      this.builder,
      stage.id,
      1,
      this.shape,
      "single",
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
      requestedBackend: opts.backend,
    });
    return new ColumnHandle(
      this.builder,
      stage.id,
      1,
      this.shape,
      "single",
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

  then<U extends ColumnValue>(
    build: (values: readonly T[]) => Column<U>,
    opts?: StageOptions,
  ): Column<U>;
  then<TUsing extends StageUsing, U extends ColumnValue>(
    spec: ThenSpec<T, TUsing, U>,
  ): Column<U>;
  then<TUsing extends StageUsing>(
    buildOrSpec:
      | ((values: readonly T[]) => Column<ColumnValue>)
      | ThenSpec<T, TUsing, ColumnValue>,
    opts: StageOptions = {},
  ): Column<ColumnValue> {
    let using: TUsing | EmptyUsing;
    let build: (
      values: readonly T[],
      context: UsingContext<TUsing | EmptyUsing>,
    ) => Column<ColumnValue>;
    let stageOptions: StageOptions;

    if (typeof buildOrSpec === "function") {
      using = emptyUsing<EmptyUsing>();
      build = (values) => buildOrSpec(values);
      stageOptions = opts;
    } else {
      using = buildOrSpec.using;
      build = buildOrSpec.build as typeof build;
      stageOptions = buildOrSpec;
    }

    let returnedColumn: ColumnHandle<ColumnValue> | undefined;
    const traced = traceWholeColumn(
      "column.then",
      this.shape,
      this.length,
      using,
      (values, usingValue) => {
        const result = build(values, { using: usingValue });
        if (!(result instanceof ColumnHandle)) {
          throw new Error("column.then callback must return column(...)");
        }
        returnedColumn = result as ColumnHandle<ColumnValue>;
        const source = returnedColumn.definition().stages[0];
        if (!source || source.kind !== "source") {
          throw new Error(
            "column.then callback column must start with a source",
          );
        }
        if (source.count === 0) {
          throw new Error("column.then produced an empty column");
        }
        return Array.from({ length: source.count }, (_, row) =>
          source.shape.rebuild(
            source.roots
              .slice(row * source.shape.width, (row + 1) * source.shape.width)
              .map((root) => new Num(root)),
          ),
        ) as readonly NumStruct<unknown>[];
      },
    );
    const stage = this.builder.add({
      kind: "then",
      source: this.stageId,
      sourceCount: this.length,
      sourceShape: this.shape,
      params: traced.params,
      inputs: traced.inputs,
      using: traced.using,
      roots: traced.roots,
      resultShape: traced.resultShape,
      requestedPlacement: stageOptions.placement,
      requestedBackend: stageOptions.backend,
    });

    const child = returnedColumn!;
    const outputStage = this.builder.adopt(child.definition(), stage.id);
    return new ColumnHandle(
      this.builder,
      outputStage,
      child.length,
      child.shape,
    );
  }

  output(this: ReducedColumn<T>): ColumnarOutput<T>;
  output(): ColumnarOutput<T | readonly T[]>;
  output<R extends NumBuildResult>(
    build: (values: readonly T[]) => R,
    opts?: StageOptions,
  ): ColumnarOutput<R>;
  output<TUsing extends StageUsing, R extends NumBuildResult>(
    spec: OutputSpec<T, TUsing, R>,
  ): ColumnarOutput<R>;
  output<TUsing extends StageUsing>(
    buildOrSpec?:
      | ((values: readonly T[]) => NumBuildResult)
      | OutputSpec<T, TUsing, NumBuildResult>,
    opts: StageOptions = {},
  ): ColumnarOutput<NumBuildResult> {
    if (buildOrSpec === undefined) {
      const values = Object.freeze(
        Array.from({ length: this.length }, () => this.shape),
      );
      const resultShape = Object.freeze({
        collection: this.outputCollection,
        values,
      });
      return new ColumnarOutputHandle<NumBuildResult>(
        this.builder.definition(this.stageId, resultShape),
      );
    }

    let using: TUsing | EmptyUsing;
    let build: (
      values: readonly T[],
      context: UsingContext<TUsing | EmptyUsing>,
    ) => NumBuildResult;
    let stageOptions: StageOptions;
    if (typeof buildOrSpec === "function") {
      using = emptyUsing<EmptyUsing>();
      build = (values) => buildOrSpec(values);
      stageOptions = opts;
    } else {
      using = buildOrSpec.using;
      build = buildOrSpec.build as typeof build;
      stageOptions = buildOrSpec;
    }

    const traced = traceWholeColumn(
      "column.output",
      this.shape,
      this.length,
      using,
      (values, usingValue) => build(values, { using: usingValue }),
    );
    const stage = this.builder.add({
      kind: "output",
      source: this.stageId,
      sourceCount: this.length,
      sourceShape: this.shape,
      params: traced.params,
      inputs: traced.inputs,
      using: traced.using,
      roots: traced.roots,
      resultShape: traced.resultShape,
      requestedPlacement: stageOptions.placement,
      requestedBackend: stageOptions.backend,
    });
    return new ColumnarOutputHandle<NumBuildResult>(
      this.builder.definition(stage.id, traced.resultShape),
    );
  }

  /** @internal Test and future compiler hook. */
  definition(): ColumnarDefinition {
    return this.builder.definition(this.stageId);
  }
}

export function column(
  values: readonly Num[],
  options?: SourceOptions,
): Column<Num>;
export function column<T extends NumStruct<T>>(
  values: readonly T[],
  options?: SourceOptions,
): Column<T>;
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
  options?: {
    readonly shape?: ColumnValue;
    readonly placement?: StagePlacement;
    readonly backend?: StageOptions["backend"];
  },
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
  const builder = new ColumnarBuilder();
  const source = builder.add({
    kind: "source",
    count: copiedValues.length,
    shape,
    roots: Object.freeze(roots),
    requestedPlacement: options?.placement,
    requestedBackend: options?.backend,
  });

  return new ColumnHandle(builder, source.id, copiedValues.length, shape);
}

/** @internal Inspect a partially-built column graph. */
export function getColumnDefinition<T extends ColumnValue>(
  value: Column<T>,
): ColumnarDefinition {
  if (!(value instanceof ColumnHandle)) {
    throw new Error("Column was not created by lona column()");
  }
  return value.definition();
}

/** @internal Future compiler hook and structural test helper. */
export function getColumnarDefinition<R extends NumBuildResult>(
  value: ColumnarOutput<R>,
): ColumnarDefinition {
  if (!(value instanceof ColumnarOutputHandle)) {
    throw new Error("ColumnarOutput was not created by lona column.output()");
  }
  return value.definition;
}

export function columnarRoutine<R extends NumBuildResult>(
  build: () => ColumnarOutput<R>,
  opts?: ColumnarRoutineOptions,
): ColumnarRoutine<R> {
  return compileColumnarRoutine<R>(getColumnarDefinition(build()), opts);
}
