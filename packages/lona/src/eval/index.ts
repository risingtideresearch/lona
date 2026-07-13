/**
 * `eval/` — focused exports for evaluation and debug transforms.
 *
 * The module is organized in three conceptual groups:
 *
 *   1. **Transformations** — pure `(NumNode[]) → OtherData` functions. Examples:
 *      `compileTape`, `compileTapeFromSerialized` (→
 *      `CompiledTape`); `renderNodeAsDot` (→ graphviz string); the diff and
 *      constant-folding kernels.
 *
 *   2. **Routines** — compile-once, call-many evaluators returned by
 *      `compileValueRoutine` and `compileGradRoutine`. Each routine has a
 *      declared `shape` (value / multi-value / grad / jacobian) and exposes
 *      `eval` (sync, throws on GPU backends), `evalAsync`, `evalBatch`, and
 *      (for value shapes) `evalBatchPacked`.
 *
 *   3. **One-shot helpers** — compile-and-call conveniences: `simpleEval`,
 *      `naiveEval`.
 */

// ---- Tape IR (shared intermediate representation) -----------------------
export {
  type CompiledTape,
  compileTape,
  compileTapeFromSerialized,
  type GradientResult,
  type JacobianResult,
} from "./tape";

// ---- Routines ------------------------------------------------------------
export {
  compileValueRoutine,
  compileGradRoutine,
  type RoutineShape,
  type VarMap,
  type VarBatch,
  type ValueRoutine,
  type MultiValueRoutine,
  type GradRoutine,
  type JacobianRoutine,
  type AnyRoutine,
  type BackendName,
  type CompileOpts,
  initGpu,
  destroyGpu,
} from "./routines";

// ---- One-shot helpers ---------------------------------------------------
export { simpleEval, naiveEval } from "./eval-value";

// `treeEval` is a transform (clones the DAG with `.evalsTo` metadata) — lives
// with the other transforms.
export { treeEval } from "./transforms/tree-eval";

export { renderNodeAsDot } from "./transforms/to-dot";
