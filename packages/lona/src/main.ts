export * from "./core/num";
export type { VarName } from "./core/tree";
export {
  NumNode,
  Derivative,
  DebugNode,
  SelectOp,
  asLiteralValue,
  childrenOfNumNode,
  numNodeLabel,
  isUnaryKind,
  isBinaryKind,
  KIND_SELECT,
  allVariables,
  ZERO_NODE,
  ONE_NODE,
  TWO_NODE,
  NEG_ONE_NODE,
} from "./core/tree";
export {
  varNode,
  litNode,
  binaryNode,
  unaryNode,
  selectNode,
  foreignFnNode,
} from "./core/tree-cons";
export { replaceVariable, partialDerivative } from "./core/tree-walks";
export * from "./core/tree-serialization";
export * from "./api/interfaces";
export * from "./api/fn";
export * from "./api/ops";
export * from "./api/num-struct";
export * from "./api/routine";
export * from "./api/diff";
export * from "./api/expressions";
export * from "./api/value-context";
export * from "./api/simplify";
export * as direct from "./api/direct";
export * from "./types";
export * from "./eval/eval-value";
export {
  compileValueRoutine,
  compileGradRoutine,
  initGpu,
  destroyGpu,
  type RoutineShape,
  type VarMap,
  type VarBatch,
  type GradientResult,
  type JacobianResult,
  type ValueRoutine,
  type MultiValueRoutine,
  type GradRoutine,
  type JacobianRoutine,
  type AnyRoutine,
  type BackendName,
  type CompileOpts,
} from "./eval/routines";
export { compileTape, type CompiledTape } from "./eval/tape";
export { treeEval } from "./eval/transforms/tree-eval";
export { renderNodeAsDot } from "./eval/transforms/to-dot";
export { genericEval } from "./eval/transforms/generic-eval";
export { JSEvalKernel } from "./eval/transforms/js-eval";
export { JSPrecisionEvalKernel } from "./eval/transforms/js-precision-eval";
