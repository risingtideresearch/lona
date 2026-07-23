import type {
  CompiledTape,
  GpuCodegenJvpDeviceKernel,
  GpuCodegenJvpLayout,
} from "lona/internal";
import { compileGpuCodegenJvpDeviceKernel } from "lona/internal";

export type CompiledColumnarGpuJvpTapeKernel = GpuCodegenJvpDeviceKernel;
export type ColumnarGpuJvpTapeLayout = GpuCodegenJvpLayout;

/** Columnar compatibility wrapper around Lona's core device JVP primitive. */
export function compileColumnarGpuJvpTapeKernel(
  tape: CompiledTape,
  layout: ColumnarGpuJvpTapeLayout,
): CompiledColumnarGpuJvpTapeKernel {
  return compileGpuCodegenJvpDeviceKernel(tape, layout);
}
