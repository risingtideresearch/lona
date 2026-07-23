import type { BackendName, Num, NumStruct, VarMap, VarName } from "lona";

/** A scalar symbolic value or a fixed-width value that can flatten to Nums. */
export type ColumnValue = Num | NumStruct<unknown>;

/** A scalar dependency explicitly brought into a columnar stage. */
export type ExternalNumValue = ColumnValue;

/** Named scalar dependencies preserve their concrete Num/NumStruct types. */
export type StageUsing = Readonly<Record<string, ExternalNumValue>>;

/** Context shape for a stage with no explicit `using` dependencies. */
export type EmptyUsing = Readonly<Record<string, never>>;

export type ExecutionTarget = "cpu" | "gpu";
export type StagePlacement = ExecutionTarget | "auto";
export type PlaceableStageKind =
  "source" | "map" | "reduce" | "then" | "output";

export type CpuBackendName =
  "js-interp" | "js-codegen" | "wasm-interp" | "wasm-codegen";
export type GpuBackendName = "gpu-codegen";
export type ColumnarBackendName = CpuBackendName | GpuBackendName;
export type BackendPreference<T> = T | readonly T[];

export interface PlacementConfig {
  default?: StagePlacement;
  source?: StagePlacement;
  map?: StagePlacement;
  reduce?: StagePlacement;
  then?: StagePlacement;
  output?: StagePlacement;
}

export interface AutoTargetConfig {
  default?: readonly ExecutionTarget[];
  source?: readonly ExecutionTarget[];
  map?: readonly ExecutionTarget[];
  reduce?: readonly ExecutionTarget[];
  then?: readonly ExecutionTarget[];
  output?: readonly ExecutionTarget[];
}

export interface StageOptions {
  placement?: StagePlacement;
  /** Backend required for this stage, overriding routine-level candidates. */
  backend?: ColumnarBackendName;
}

/** Context supplied while tracing a map callback. */
export interface MapContext<TUsing extends StageUsing = EmptyUsing> {
  /** Integer-valued symbolic row index. */
  readonly index: Num;
  /** Explicit scalar dependencies; GPU stages lower these as uniforms. */
  readonly using: TUsing;
}

/** Context supplied to callbacks without a row index. */
export interface UsingContext<TUsing extends StageUsing> {
  readonly using: TUsing;
}

export type ReductionOrder = "left" | "tree";

export interface ReductionOptions extends StageOptions {
  /** Required acknowledgement for a user-defined parallel reduction. */
  associative: boolean;
  order?: ReductionOrder;
}

export interface BuiltInReductionOptions extends StageOptions {
  order?: Exclude<ReductionOrder, "left">;
}

export interface ComponentWiseReductionOptions extends BuiltInReductionOptions {
  /** NumStruct built-ins are explicitly component-wise. */
  componentWise: true;
}

export type NumBuildResult =
  Num | NumStruct<unknown> | readonly (Num | NumStruct<unknown>)[];

/** @internal Runtime brand used by the columnar builder implementation. */
export const COLUMNAR_OUTPUT_RESULT: unique symbol = Symbol(
  "lona.columnarOutputResult",
);

/**
 * A declared routine result whose values depend on columnar stages.
 * It is intentionally not a Num or Column and can only be compiled as part
 * of a columnar routine.
 */
export interface ColumnarOutput<R extends NumBuildResult> {
  readonly [COLUMNAR_OUTPUT_RESULT]: R;
}

/** Flattened numeric result returned by the columnar runtime. */
export type ConcreteResult<R> = R extends Num
  ? number
  : R extends NumStruct<unknown>
    ? number[]
    : R extends readonly (infer T)[]
      ? T extends Num
        ? number[]
        : T extends NumStruct<unknown>
          ? number[][]
          : never
      : never;

export interface ColumnarRoutineOptions {
  /** Placement policy inherited by stages without an explicit placement. */
  placement?: StagePlacement | PlacementConfig;
  /** Ordered target preferences used when a stage resolves to `auto`. */
  auto?: { readonly targets?: AutoTargetConfig };
  /** Ordered backend candidates for each execution target. */
  backends?: {
    readonly cpu?: BackendPreference<CpuBackendName>;
    readonly gpu?: BackendPreference<GpuBackendName>;
  };
  diagnosticCheckpoint?: (phase: string) => void;
}

export interface ColumnarStageInfo {
  readonly id: number;
  readonly kind: "source" | PlaceableStageKind;
  readonly placement?: ExecutionTarget;
  readonly backend?: BackendName;
}

export interface ColumnarStageTiming {
  readonly stageId: number;
  readonly backend?: BackendName;
  readonly milliseconds: number;
}

export interface ColumnarEvaluationStats {
  readonly uploadedBytes: number;
  readonly downloadedBytes: number;
  readonly transferredBytes: number;
  readonly dispatchCount: number;
  readonly readbackCount: number;
  readonly evaluationMilliseconds: number;
  readonly stageTimings: readonly ColumnarStageTiming[];
}

export interface ColumnarRoutine<R extends NumBuildResult> {
  readonly varSlots: readonly VarName[];
  readonly numVars: number;
  readonly stages: readonly ColumnarStageInfo[];
  readonly lastEvaluationStats: ColumnarEvaluationStats | null;

  /** Evaluate on any configured combination of CPU and GPU stages. */
  evalAsync(vars: VarMap): Promise<ConcreteResult<R>>;
  dispose(): void;
}

/** A homogeneous, statically-sized sequence of Num or NumStruct values. */
export interface Column<T extends ColumnValue> {
  readonly length: number;

  map<U extends ColumnValue>(
    build: (value: T, context: MapContext) => U,
    opts?: StageOptions,
  ): Column<U>;

  map<TUsing extends StageUsing, U extends ColumnValue>(spec: {
    using: TUsing;
    build: (value: T, context: MapContext<TUsing>) => U;
    placement?: StagePlacement;
    backend?: ColumnarBackendName;
  }): Column<U>;

  reduce(
    combine: (left: T, right: T) => T,
    initial: T,
    opts: ReductionOptions,
  ): ReducedColumn<T>;

  reduce<TUsing extends StageUsing>(spec: {
    using: TUsing;
    combine: (left: T, right: T, context: UsingContext<TUsing>) => T;
    initial: T;
    associative: boolean;
    order?: ReductionOrder;
    placement?: StagePlacement;
    backend?: ColumnarBackendName;
  }): ReducedColumn<T>;

  sum(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  sum(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  product(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  product(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  min(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  min(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  max(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  max(opts: ComponentWiseReductionOptions): ReducedColumn<T>;

  /** Run a whole-column transformation and continue the graph. */
  then<U extends ColumnValue>(
    build: (values: readonly T[]) => Column<U>,
    opts?: StageOptions,
  ): Column<U>;

  /** Run a whole-column transformation with explicit scalar dependencies. */
  then<TUsing extends StageUsing, U extends ColumnValue>(spec: {
    using: TUsing;
    build: (values: readonly T[], context: UsingContext<TUsing>) => Column<U>;
    placement?: StagePlacement;
    backend?: ColumnarBackendName;
  }): Column<U>;

  /** Return the reduced value without adding an execution stage. */
  output(this: ReducedColumn<T>): ColumnarOutput<T>;

  /**
   * Return the current value(s) without adding an execution stage. A column
   * whose reduced type has been widened to Column may produce either shape.
   */
  output(): ColumnarOutput<T | readonly T[]>;

  /** Produce the final routine output with a whole-column scalar callback. */
  output<R extends NumBuildResult>(
    build: (values: readonly T[]) => R,
    opts?: StageOptions,
  ): ColumnarOutput<R>;

  /** Produce final output with additional explicit scalar dependencies. */
  output<TUsing extends StageUsing, R extends NumBuildResult>(spec: {
    using: TUsing;
    build: (values: readonly T[], context: UsingContext<TUsing>) => R;
    placement?: StagePlacement;
    backend?: ColumnarBackendName;
  }): ColumnarOutput<R>;
}

export interface ReducedColumn<T extends ColumnValue> extends Column<T> {
  readonly length: 1;
}

/** Options for a materialized source column. */
export type SourceOptions = StageOptions;

/** Options for constructing an empty struct column. */
export interface EmptyColumnOptions<
  T extends ColumnValue,
> extends SourceOptions {
  readonly shape: T;
}

/**
 * Type of the future `column` constructor. Keeping this as an interface lets
 * Phase 0 validate inference without publishing a non-functional runtime API.
 */
export interface ColumnFactory {
  (values: readonly Num[], options?: SourceOptions): Column<Num>;

  <T extends NumStruct<T>>(
    values: readonly T[],
    options?: SourceOptions,
  ): Column<T>;

  (values: readonly [], options: EmptyColumnOptions<Num>): Column<Num>;

  <T extends NumStruct<T>>(
    values: readonly [],
    options: EmptyColumnOptions<T>,
  ): Column<T>;
}
