export type {
  RoutineShape,
  VarMap,
  VarBatch,
  GradientResult,
  JacobianResult,
  ValueRoutine,
  MultiValueRoutine,
  GradRoutine,
  JacobianRoutine,
  AnyRoutine,
} from "./types";

export type { BackendName } from "./backend";

export {
  compileValueRoutine,
  compileGradRoutine,
  type CompileOpts,
} from "./compile";

// Importing the backend-registry module registers all backends as a
// side-effect. Callers get a fully-loaded registry by importing this module.
import "./backends";

// GPU init + teardown
export { initGpu, destroyGpu } from "./backends/gpu-util";
