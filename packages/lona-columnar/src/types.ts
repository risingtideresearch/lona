import type { BackendName, Num, NumStruct, VarMap, VarName } from "lona";

/** A scalar symbolic value or a fixed-width value that can flatten to Nums. */
export type ColumnValue = Num | NumStruct<unknown>;

/** A scalar dependency explicitly brought into a structured stage. */
export type ExternalNumValue = ColumnValue;

/** Named scalar dependencies preserve their concrete Num/NumStruct types. */
export type StageUsing = Readonly<Record<string, ExternalNumValue>>;

/** Context shape for a stage with no explicit `using` dependencies. */
export type EmptyUsing = Readonly<Record<string, never>>;

export type ExecutionTarget = "cpu" | "gpu";
export type StagePlacement = ExecutionTarget | "auto";
export type PlaceableStageKind = "map" | "reduce" | "toNums";

export type CpuBackendName =
  "js-interp" | "js-codegen" | "wasm-interp" | "wasm-codegen";
export type GpuBackendName = "gpu-codegen";
export type BackendPreference<T> = T | readonly T[];

export interface PlacementConfig {
  default?: StagePlacement;
  map?: StagePlacement;
  reduce?: StagePlacement;
  toNums?: StagePlacement;
}

export interface AutoTargetConfig {
  default?: readonly ExecutionTarget[];
  map?: readonly ExecutionTarget[];
  reduce?: readonly ExecutionTarget[];
  toNums?: readonly ExecutionTarget[];
}

export interface StageOptions {
  placement?: StagePlacement;
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
  Num | NumStruct<unknown> | readonly Num[] | readonly NumStruct<unknown>[];

/** @internal Runtime brand used by the structured builder implementation. */
export const STRUCTURED_VALUE_RESULT: unique symbol = Symbol(
  "lona.structuredValueResult",
);

/**
 * A scalar Num stage whose inputs may depend on earlier structured stages.
 * It is intentionally not a Num and can only be compiled as part of a
 * structured routine.
 */
export interface StructuredValue<R extends NumBuildResult> {
  readonly [STRUCTURED_VALUE_RESULT]: R;
}

/** Flattened numeric result returned by the structured runtime. */
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

export interface StructuredCompileOptions {
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

export interface StructuredStageInfo {
  readonly id: number;
  readonly kind: "source" | PlaceableStageKind;
  readonly placement?: ExecutionTarget;
  readonly backend?: BackendName;
}

export interface StructuredStageTiming {
  readonly stageId: number;
  readonly backend?: BackendName;
  readonly milliseconds: number;
}

export interface StructuredEvaluationStats {
  readonly uploadedBytes: number;
  readonly downloadedBytes: number;
  readonly transferredBytes: number;
  readonly dispatchCount: number;
  readonly readbackCount: number;
  readonly evaluationMilliseconds: number;
  readonly stageTimings: readonly StructuredStageTiming[];
}

export interface StructuredRoutine<R extends NumBuildResult> {
  readonly varSlots: readonly VarName[];
  readonly numVars: number;
  readonly stages: readonly StructuredStageInfo[];
  readonly lastEvaluationStats: StructuredEvaluationStats | null;

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
  }): ReducedColumn<T>;

  sum(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  sum(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  product(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  product(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  min(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  min(opts: ComponentWiseReductionOptions): ReducedColumn<T>;
  max(this: Column<Num>, opts?: BuiltInReductionOptions): ReducedColumn<T>;
  max(opts: ComponentWiseReductionOptions): ReducedColumn<T>;

  /** Start a scalar Num stage from this column. */
  toNums<R extends NumBuildResult>(
    build: (values: readonly T[]) => R,
    opts?: StageOptions,
  ): StructuredValue<R>;

  /** Start a scalar Num stage with additional explicit scalar dependencies. */
  toNums<TUsing extends StageUsing, R extends NumBuildResult>(spec: {
    using: TUsing;
    build: (values: readonly T[], context: UsingContext<TUsing>) => R;
    placement?: StagePlacement;
  }): StructuredValue<R>;
}

export interface ReducedColumn<T extends ColumnValue> extends Column<T> {
  readonly length: 1;
}

/** Options for constructing an empty struct column. */
export interface EmptyColumnOptions<T extends ColumnValue> {
  readonly shape: T;
}

/**
 * Type of the future `column` constructor. Keeping this as an interface lets
 * Phase 0 validate inference without publishing a non-functional runtime API.
 */
export interface ColumnFactory {
  (values: readonly Num[]): Column<Num>;

  <T extends NumStruct<T>>(values: readonly T[]): Column<T>;

  (values: readonly [], options: EmptyColumnOptions<Num>): Column<Num>;

  <T extends NumStruct<T>>(
    values: readonly [],
    options: EmptyColumnOptions<T>,
  ): Column<T>;
}
