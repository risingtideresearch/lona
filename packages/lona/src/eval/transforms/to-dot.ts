import { DotEvalKernel } from "./dot-eval";
import { genericEval } from "./generic-eval";
import { NumNode } from "../../core/tree";

export { DotEvalKernel };

export function renderNodeAsDot(root: NumNode & { evalsTo?: number }): string {
  const kernel = new DotEvalKernel();
  genericEval(root, kernel);
  return kernel.getDot();
}
