/**
 * Shared WebGPU utilities: device singleton and readback.
 *
 * Used by gpu-tape-eval.ts and gpu-codegen-eval.ts.
 */

// ---------------------------------------------------------------------------
// GPU device singleton
// ---------------------------------------------------------------------------

let gpuDevice: GPUDevice | null = null;
let gpuInstance: unknown = null;
let initDone = false;

export async function initGpu(): Promise<GPUDevice | null> {
  if (initDone) return gpuDevice;
  initDone = true;
  try {
    let gpu: GPU;

    if (typeof navigator !== "undefined" && navigator.gpu) {
      // Browser: WebGPU is available natively
      gpu = navigator.gpu;
    } else {
      // Node.js: use the webgpu npm package. Hidden behind a runtime variable
      // so browser bundlers don't try to resolve it — the package uses
      // `createRequire` and is Node-only.
      const webgpuModuleName = "webgpu";
      const webgpu = await import(/* @vite-ignore */ webgpuModuleName);
      Object.assign(globalThis, webgpu.globals);
      gpuInstance = webgpu.create([]);
      gpu = gpuInstance as GPU;
    }

    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    gpuDevice = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    return gpuDevice;
  } catch {
    return null;
  }
}

/**
 * Return the cached GPU device or throw. Call `initGpu()` first.
 */
export function requireGpuDevice(): GPUDevice {
  if (!initDone) {
    throw new Error(
      "GPU not initialized — call `await initGpu()` before compiling GPU routines",
    );
  }
  if (!gpuDevice) {
    throw new Error("GPU not available on this platform");
  }
  return gpuDevice;
}

export async function destroyGpu() {
  if (gpuDevice) {
    gpuDevice.destroy();
    gpuDevice = null;
  }
  if (gpuInstance) {
    // Node.js webgpu package needs explicit cleanup
    try {
      const webgpuModuleName = "webgpu";
      const webgpu = (await import(/* @vite-ignore */ webgpuModuleName)) as {
        destroy?: (instance: unknown) => void;
      };
      if (webgpu.destroy) webgpu.destroy(gpuInstance);
    } catch {
      /* ignore */
    }
    gpuInstance = null;
  }
  initDone = false;
}

// ---------------------------------------------------------------------------
// GPU readback helper
// ---------------------------------------------------------------------------

export async function readGpuBuffer(
  device: GPUDevice,
  outputBuf: GPUBuffer,
  stagingBuf: GPUBuffer,
  numBytes: number,
): Promise<Float32Array> {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuf, 0, stagingBuf, 0, numBytes);
  device.queue.submit([encoder.finish()]);

  await stagingBuf.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(stagingBuf.getMappedRange().slice(0));
  stagingBuf.unmap();
  return result;
}
