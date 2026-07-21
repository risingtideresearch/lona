import type { NumNode, Variable } from "lona/internal";
import type { StagePlacement } from "./types";
import type { ValueShape } from "./shape";

export type StageId = number;

export interface ScalarBindings {
  readonly names: readonly string[];
  readonly roots: readonly NumNode[];
  readonly params: readonly Variable[];
  readonly shapes: readonly ValueShape[];
  readonly width: number;
}

export type StructuredParamBinding =
  | { readonly kind: "row"; readonly component: number }
  | { readonly kind: "uniform"; readonly component: number }
  | { readonly kind: "index" }
  | { readonly kind: "reduce-left"; readonly component: number }
  | { readonly kind: "reduce-right"; readonly component: number }
  | { readonly kind: "materialized"; readonly component: number };

export interface StructuredParamInput {
  readonly param: Variable;
  readonly binding: StructuredParamBinding;
}

export interface StructuredKernel {
  readonly owner: symbol;
  readonly params: readonly Variable[];
  readonly inputs: readonly StructuredParamInput[];
  readonly inputWidth: number;
  readonly usingWidth: number;
  readonly outputWidth: number;
  readonly indexParam?: Variable;
  readonly roots: readonly NumNode[];
}

export interface SourceStage {
  readonly kind: "source";
  readonly id: StageId;
  readonly count: number;
  readonly shape: ValueShape;
  /** Flattened row-major scalar roots. */
  readonly roots: readonly NumNode[];
}

export interface MapStage {
  readonly kind: "map";
  readonly id: StageId;
  readonly source: StageId;
  readonly count: number;
  readonly inputShape: ValueShape;
  readonly outputShape: ValueShape;
  readonly using: ScalarBindings;
  readonly kernel: StructuredKernel;
  readonly requestedPlacement?: StagePlacement;
}

export type BuiltInReduction = "sum" | "product" | "min" | "max";

export interface ReduceStage {
  readonly kind: "reduce";
  readonly id: StageId;
  readonly source: StageId;
  readonly inputCount: number;
  readonly count: 1;
  readonly shape: ValueShape;
  readonly initial: readonly NumNode[];
  readonly using: ScalarBindings;
  readonly kernel: StructuredKernel;
  readonly associative: boolean;
  readonly order: "left" | "tree";
  readonly builtIn?: BuiltInReduction;
  readonly requestedPlacement?: StagePlacement;
}

export interface ResultShape {
  readonly collection: "single" | "array";
  readonly values: readonly ValueShape[];
}

export interface ToNumsStage {
  readonly kind: "to-nums";
  readonly id: StageId;
  readonly source: StageId;
  readonly sourceCount: number;
  readonly sourceShape: ValueShape;
  readonly params: readonly Variable[];
  readonly inputs: readonly StructuredParamInput[];
  readonly using: ScalarBindings;
  readonly roots: readonly NumNode[];
  readonly resultShape: ResultShape;
  readonly requestedPlacement?: StagePlacement;
}

export type StructuredStage =
  SourceStage | MapStage | ReduceStage | ToNumsStage;

export interface StructuredDefinition {
  readonly stages: readonly StructuredStage[];
  readonly outputStage: StageId;
}

type StageWithoutId = StructuredStage extends infer TStage
  ? TStage extends StructuredStage
    ? Omit<TStage, "id">
    : never
  : never;

export class StructuredBuilder {
  private readonly mutableStages: StructuredStage[] = [];

  add<T extends StageWithoutId>(stage: T): T & { readonly id: StageId } {
    const withId = Object.freeze({
      ...stage,
      id: this.mutableStages.length,
    }) as unknown as T & { readonly id: StageId };
    this.mutableStages.push(withId as StructuredStage);
    return withId;
  }

  definition(outputStage: StageId): StructuredDefinition {
    return Object.freeze({
      stages: Object.freeze([...this.mutableStages]),
      outputStage,
    });
  }
}
