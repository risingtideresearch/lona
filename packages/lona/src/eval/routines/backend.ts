/**
 * Backend abstraction — each backend produces minimal "Kernel" objects that
 * implement whichever call path is native for that backend. The routine
 * wrapper in `./compile.ts` turns a Kernel into a full Routine by
 * synthesizing the missing paths (sync↔async, single↔batch).
 */
import type { NumNode, VarName } from "../../core/tree";
import type { CompiledTape } from "../tape";
import type {
  GradRoutine,
  GradientResult,
  JacobianResult,
  JacobianRoutine,
  RoutineShape,
  VarMap,
} from "./types";

export type BaseBackendName =
  | "js-interp"
  | "js-codegen"
  | "wasm-interp"
  | "wasm-codegen"
  | "gpu-interp"
  | "gpu-codegen";

export type SymBackendName =
  "js-codegen-sym" | "wasm-interp-sym" | "wasm-codegen-sym" | "gpu-codegen-sym";

export type BackendName = BaseBackendName | SymBackendName;

// ---------------------------------------------------------------------------
// Kernel shapes — the narrow contract each backend implements
// ---------------------------------------------------------------------------

/**
 * A sync kernel implements single-point `eval`. The routine wrapper will
 * synthesize `evalAsync` (via `Promise.resolve`) and `evalBatch*` (by
 * looping) on top of it.
 */
export interface SyncValueKernel {
  readonly kind: "sync-value";
  readonly numRoots: number;
  eval(vars: VarMap, derivatives?: VarMap): number[];
}
export interface SyncGradKernel {
  readonly kind: "sync-grad";
  readonly diffVars: readonly VarName[];
  eval(vars: VarMap): GradientResult;
}
export interface SyncJacobianKernel {
  readonly kind: "sync-jacobian";
  readonly numRoots: number;
  readonly diffVars: readonly VarName[];
  eval(vars: VarMap): JacobianResult;
}

/** Result of applying arbitrary tangent seeds to a multi-root tape. */
export interface JvpResult {
  readonly vals: number[];
  /** One row per root and one column per tangent direction. */
  readonly tangents: number[][];
}

/**
 * A synchronous, packed multi-root JVP kernel. Values follow the tape's
 * ordinary variable-slot order; seeds use input-major, direction-minor order.
 */
export interface SyncJvpKernel {
  readonly kind: "sync-jvp";
  readonly numRoots: number;
  readonly numDirections: number;
  evalPacked(values: Float64Array, seeds: Float64Array): JvpResult;
}

/**
 * An async-batch kernel implements `evalBatchPacked` natively (GPU). The
 * routine wrapper synthesizes `evalBatch` via `packVarBatch`, synthesizes
 * `evalAsync` via a batch-of-1 call, and makes sync `eval` throw.
 */
export interface AsyncBatchValueKernel {
  readonly kind: "async-batch-value";
  readonly numRoots: number;
  /** Output is `numPoints * numRoots` f32s, interleaved per point. */
  evalBatchPacked(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array>;
}

/**
 * An async-batch grad kernel implements GPU forward-mode autodiff natively.
 * Each GPU thread computes all D partial derivatives in one pass.
 * The routine wrapper synthesizes `eval` (throws), `evalAsync` (batch-of-1),
 * and `evalBatch` (pack + call evalBatchPacked + unpack).
 */
export interface AsyncBatchGradKernel {
  readonly kind: "async-batch-grad";
  readonly diffVars: readonly VarName[];
  /**
   * Evaluate value + all partial derivatives for N points in one dispatch.
   * Output is N × (1 + numDiffVars) f32s: [val, ∂v₀, ∂v₁, …] per point.
   */
  evalBatchPacked(
    varData: Float32Array,
    numPoints: number,
  ): Promise<Float32Array>;
}

export type ValueKernel = SyncValueKernel | AsyncBatchValueKernel;
export type GradKernel = SyncGradKernel | AsyncBatchGradKernel;
export type JacobianKernel = SyncJacobianKernel;
export type JvpKernel = SyncJvpKernel;

export type AnyKernel = ValueKernel | GradKernel | JacobianKernel | JvpKernel;

// ---------------------------------------------------------------------------
// Kernel envelope — adds the tape metadata the routine wrapper needs
// ---------------------------------------------------------------------------

export interface KernelEnvelope<K extends AnyKernel> {
  readonly kernel: K;
  readonly varSlots: readonly VarName[];
  readonly numVars: number;
  readonly backend: BackendName;
  dispose?(): void;
}

// ---------------------------------------------------------------------------
// Backend interface + registry
// ---------------------------------------------------------------------------

export interface Backend {
  readonly name: BackendName;
  readonly supported: ReadonlySet<RoutineShape>;

  compileValue?(tape: CompiledTape): KernelEnvelope<ValueKernel> | null;
  compileGrad?(
    tape: CompiledTape,
    diffVars: VarName[],
  ): KernelEnvelope<GradKernel> | null;
  compileJacobian?(
    tape: CompiledTape,
    diffVars: VarName[],
  ): KernelEnvelope<JacobianKernel> | null;
  compileJvp?(
    tape: CompiledTape,
    numDirections: number,
  ): KernelEnvelope<JvpKernel> | null;

  /** Compile a grad routine from original DAG roots (symbolic differentiation). */
  compileGradFromRoots?(
    roots: NumNode[],
    diffVars: VarName[],
  ): GradRoutine | null;

  /** Compile a jacobian routine from original DAG roots (symbolic differentiation). */
  compileJacobianFromRoots?(
    roots: NumNode[],
    diffVars: VarName[],
  ): JacobianRoutine | null;
}

const registry = new Map<BackendName, Backend>();

export function registerBackend(backend: Backend): void {
  registry.set(backend.name, backend);
}

export function getBackend(name: BackendName): Backend | undefined {
  return registry.get(name);
}

export function syncEvalNotSupported(backend: BackendName): never {
  throw new Error(
    `Sync eval not supported on backend '${backend}' — use evalAsync or evalBatch`,
  );
}
