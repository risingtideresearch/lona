import type { NumNode, Variable } from "lona/internal";
import type { ColumnarBackendName, StagePlacement } from "./types";
import type { ValueShape } from "./shape";

export type StageId = number;

export interface ScalarBindings {
  readonly names: readonly string[];
  readonly roots: readonly NumNode[];
  readonly params: readonly Variable[];
  readonly shapes: readonly ValueShape[];
  readonly width: number;
}

export type ColumnarParamBinding =
  | { readonly kind: "row"; readonly component: number }
  | { readonly kind: "uniform"; readonly component: number }
  | { readonly kind: "index" }
  | { readonly kind: "reduce-left"; readonly component: number }
  | { readonly kind: "reduce-right"; readonly component: number }
  | { readonly kind: "materialized"; readonly component: number };

export interface ColumnarParamInput {
  readonly param: Variable;
  readonly binding: ColumnarParamBinding;
}

export interface ColumnarKernel {
  readonly owner: symbol;
  readonly params: readonly Variable[];
  readonly inputs: readonly ColumnarParamInput[];
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
  readonly requestedPlacement?: StagePlacement;
  readonly requestedBackend?: ColumnarBackendName;
}

export interface MapStage {
  readonly kind: "map";
  readonly id: StageId;
  readonly source: StageId;
  readonly count: number;
  readonly inputShape: ValueShape;
  readonly outputShape: ValueShape;
  readonly using: ScalarBindings;
  readonly kernel: ColumnarKernel;
  readonly requestedPlacement?: StagePlacement;
  readonly requestedBackend?: ColumnarBackendName;
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
  readonly kernel: ColumnarKernel;
  readonly associative: boolean;
  readonly order: "left" | "tree";
  readonly builtIn?: BuiltInReduction;
  readonly requestedPlacement?: StagePlacement;
  readonly requestedBackend?: ColumnarBackendName;
}

export interface ResultShape {
  readonly collection: "single" | "array";
  readonly values: readonly ValueShape[];
}

export interface WholeColumnStage {
  readonly kind: "then" | "output";
  readonly id: StageId;
  readonly source: StageId;
  readonly sourceCount: number;
  readonly sourceShape: ValueShape;
  readonly params: readonly Variable[];
  readonly inputs: readonly ColumnarParamInput[];
  readonly using: ScalarBindings;
  readonly roots: readonly NumNode[];
  readonly resultShape: ResultShape;
  readonly requestedPlacement?: StagePlacement;
  readonly requestedBackend?: ColumnarBackendName;
}

export type ColumnarStage =
  SourceStage | MapStage | ReduceStage | WholeColumnStage;

export interface ColumnarDefinition {
  readonly stages: readonly ColumnarStage[];
  /** Current graph endpoint; it is an output only when resultShape is present. */
  readonly outputStage: StageId;
  readonly resultShape?: ResultShape;
}

type StageWithoutId = ColumnarStage extends infer TStage
  ? TStage extends ColumnarStage
    ? Omit<TStage, "id">
    : never
  : never;

export class ColumnarBuilder {
  private readonly mutableStages: ColumnarStage[] = [];

  add<T extends StageWithoutId>(stage: T): T & { readonly id: StageId } {
    const withId = Object.freeze({
      ...stage,
      id: this.mutableStages.length,
    }) as unknown as T & { readonly id: StageId };
    this.mutableStages.push(withId as ColumnarStage);
    return withId;
  }

  /**
   * Append a callback-created column graph, replacing its source with an
   * already-added stage in this graph. Returns the remapped output stage.
   */
  adopt(definition: ColumnarDefinition, sourceReplacement: StageId): StageId {
    const source = definition.stages[0];
    if (!source || source.kind !== "source") {
      throw new Error("callback-created column must start with a source stage");
    }

    const ids = new Map<StageId, StageId>([[source.id, sourceReplacement]]);
    for (const stage of definition.stages.slice(1)) {
      if (stage.kind === "source") {
        throw new Error("cannot adopt a column graph with multiple sources");
      }
      const remappedSource = ids.get(stage.source);
      if (remappedSource === undefined) {
        throw new Error(
          `cannot adopt column stage ${stage.id}: missing source`,
        );
      }
      const { id: _id, ...withoutId } = stage;
      const adopted = this.add({
        ...withoutId,
        source: remappedSource,
      } as StageWithoutId);
      ids.set(stage.id, adopted.id);
    }

    const output = ids.get(definition.outputStage);
    if (output === undefined) {
      throw new Error("cannot adopt callback-created column output");
    }
    return output;
  }

  definition(
    outputStage: StageId,
    resultShape?: ResultShape,
  ): ColumnarDefinition {
    return Object.freeze({
      stages: Object.freeze([...this.mutableStages]),
      outputStage,
      resultShape,
    });
  }
}
