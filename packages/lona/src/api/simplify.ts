import { genericEval } from "../eval/transforms/generic-eval";
import { ConstantsFoldEval } from "../eval/transforms/fold-constants";
import type { NumNode } from "../core/tree";

// Hash-consing at construction time (see tree-cons.ts) guarantees that
// structurally equal sub-DAGs share a single object, so a separate
// "compress" pass is no longer needed. `simplify` is now just the
// constant-folding / algebraic rewrite pass.
export function simplify(node: NumNode): NumNode {
  return genericEval(node, new ConstantsFoldEval());
}
