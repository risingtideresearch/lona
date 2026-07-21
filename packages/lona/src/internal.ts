/**
 * @internal
 * First-party integration surface. This entry point has no compatibility
 * guarantee and is not intended for application use.
 */
export {
  BinaryOp,
  Derivative,
  ForeignFn,
  LiteralNum,
  NumNode,
  SelectOp,
  UnaryOp,
  Variable,
  childrenOfNumNode,
  isBinaryKind,
  isUnaryKind,
  numNodeLabel,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_LIT,
  KIND_SELECT,
  KIND_VAR,
  type VarName,
} from "./core/tree";
export {
  binaryNode,
  derivativeNode,
  foreignFnNode,
  litNode,
  varNode,
} from "./core/tree-cons";
export { compileTape, type CompiledTape } from "./eval/tape";
export {
  compileValueRoutine,
  compileValueRoutineFromTape,
  type BackendName,
  type MultiValueRoutine,
  type ValueRoutine,
  type VarMap,
} from "./eval/routines";
export {
  destroyGpu,
  initGpu,
  isGpuAvailable,
  mapGpuReadbackBuffer,
  requireGpuDevice,
  type DeviceBufferSlice,
} from "./eval/routines/backends/gpu-util";
export { emitTapeWgsl } from "./eval/routines/backends/gpu-codegen/emit-tape-wgsl";
