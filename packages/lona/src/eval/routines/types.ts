/**
 * Routine types — compile-once, call-many evaluators of a declared `shape`
 * (Value, MultiValue, Grad, Jacobian), produced by a chosen backend.
 *
 * Every routine exposes the full call surface:
 *   - `eval(vars)`          sync single-point; throws on async-only backends
 *   - `evalAsync(vars)`     async single-point; works on every backend
 *   - `evalBatch(vars)`     async batch with per-variable number arrays
 *   - `evalBatchPacked(..)` async batch with a pre-packed Float32Array
 *
 * Non-vectorized backends implement the batch paths by looping over `eval`;
 * GPU backends drive the batch path natively and synthesize single-point eval
 * via a batch-of-1.
 */
import type { VarName } from "../../core/tree";
import type { GradientResult, JacobianResult } from "../tape";

export type RoutineShape = "value" | "multi-value" | "grad" | "jacobian";

export type VarMap = Map<VarName, number> | Map<string, number>;

/** User-friendly batch input: one array of per-point values per named variable. */
export type VarBatch =
  Record<string, number[]> | Map<VarName, number[]> | Map<string, number[]>;

export type { GradientResult, JacobianResult };

interface RoutineBase {
  readonly shape: RoutineShape;
  readonly varSlots: readonly VarName[];
  readonly numVars: number;
  dispose?(): void;
}

export interface ValueRoutine extends RoutineBase {
  readonly shape: "value";
  /**
   * @param derivatives Optional map of `∂f/∂v` values to seed into derivative
   *   slots. Only meaningful when the DAG contains `Derivative` nodes.
   */
  eval(vars: VarMap, derivatives?: VarMap): number;
  evalAsync(vars: VarMap, derivatives?: VarMap): Promise<number>;
  /** One f32 per point. */
  evalBatch(vars: VarBatch, numPoints?: number): Promise<Float32Array>;
  evalBatchPacked(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array>;
}

export interface MultiValueRoutine extends RoutineBase {
  readonly shape: "multi-value";
  readonly numRoots: number;
  eval(vars: VarMap, derivatives?: VarMap): number[];
  evalAsync(vars: VarMap, derivatives?: VarMap): Promise<number[]>;
  /** `numPoints * numRoots` f32s, interleaved per point. */
  evalBatch(vars: VarBatch, numPoints?: number): Promise<Float32Array>;
  evalBatchPacked(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array>;
}

export interface GradRoutine extends RoutineBase {
  readonly shape: "grad";
  readonly diffVars: readonly VarName[];
  eval(vars: VarMap): GradientResult;
  evalAsync(vars: VarMap): Promise<GradientResult>;
  evalBatch(vars: VarBatch, numPoints?: number): Promise<GradientResult[]>;
  /** `numPoints × (1 + numDiff)` interleaved f32s per point. */
  evalBatchPacked?(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array>;
}

export interface JacobianRoutine extends RoutineBase {
  readonly shape: "jacobian";
  readonly diffVars: readonly VarName[];
  readonly numRoots: number;
  eval(vars: VarMap): JacobianResult;
  evalAsync(vars: VarMap): Promise<JacobianResult>;
  evalBatch(vars: VarBatch, numPoints?: number): Promise<JacobianResult[]>;
  /** `numPoints × numRoots × (1 + numDiff)` interleaved f32s per point. */
  evalBatchPacked?(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array>;
}

export type AnyRoutine =
  ValueRoutine | MultiValueRoutine | GradRoutine | JacobianRoutine;
