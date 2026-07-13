/**
 * Common compiled-function signatures shared by every CPU backend's
 * low-level output. Kept separate from the Kernel / Routine types so each
 * backend can import just these without pulling in the rest of the
 * routine system.
 */
import type { VarName } from "../../../core/tree";

export type CompiledFn = (
  vars: Map<VarName, number>,
  derivatives?: Map<VarName, number>,
) => number;

export type CompiledMultiFn = (
  vars: Map<VarName, number>,
  derivatives?: Map<VarName, number>,
) => number[];
