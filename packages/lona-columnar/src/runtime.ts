import { KIND_LIT, LiteralNum, type NumNode } from "lona/internal";
import {
  compileValueRoutine,
  type BackendName,
  type MultiValueRoutine,
  type ValueRoutine,
  type VarMap,
} from "lona/internal";
import {
  type DeviceBufferSlice,
  isGpuAvailable,
  mapGpuReadbackBuffer,
  requireGpuDevice,
} from "lona/internal";
import type {
  MapStage,
  ReduceStage,
  ResultShape,
  ScalarBindings,
  StructuredDefinition,
  StructuredStage,
  ToNumsStage,
} from "./ir";
import type {
  ConcreteResult,
  CpuBackendName,
  ExecutionTarget,
  GpuBackendName,
  NumBuildResult,
  StructuredCompileOptions,
  StructuredEvaluationStats,
  StructuredRoutine,
  StructuredStageInfo,
  StagePlacement,
} from "./types";
import { compileCpuKernel, type CompiledCpuKernel } from "./cpu/compile-kernel";
import {
  compileGpuMapKernel,
  type CompiledGpuMapKernel,
} from "./gpu/map-kernel";
import {
  compileGpuReduceKernel,
  compileGpuTracedReduceKernel,
  type CompiledGpuReduceKernel,
} from "./gpu/reduce-kernel";

interface CompiledStage {
  readonly stage: StructuredStage;
  readonly placement?: ExecutionTarget;
  readonly backend?: BackendName;
  readonly cpuKernel?: CompiledCpuKernel;
  readonly gpuMapKernel?: CompiledGpuMapKernel;
  readonly gpuReduceKernel?: CompiledGpuReduceKernel;
  readonly uniformGroup?: number;
}

const CPU_BACKENDS: ReadonlySet<string> = new Set([
  "js-interp",
  "js-codegen",
  "wasm-interp",
  "wasm-codegen",
]);
const GPU_BACKENDS: ReadonlySet<string> = new Set(["gpu-codegen"]);

function normalizeRoutineResult(
  routine: ValueRoutine | MultiValueRoutine,
  vars: VarMap,
): number[] {
  const result = routine.eval(vars);
  return typeof result === "number" ? [result] : result;
}

function preference<T>(
  configured: T | readonly T[] | undefined,
  defaults: readonly T[],
  label: string,
): readonly T[] {
  const values =
    configured === undefined
      ? [...defaults]
      : Array.isArray(configured)
        ? [...configured]
        : [configured as T];
  if (values.length === 0) {
    throw new Error(`structured ${label} preference must not be empty`);
  }
  return Object.freeze(values);
}

function configuredCpuBackends(
  opts: StructuredCompileOptions,
): readonly CpuBackendName[] {
  const backends = preference(
    opts.backends?.cpu,
    ["wasm-codegen" as const],
    "CPU backend",
  );
  for (const backend of backends) {
    if (!CPU_BACKENDS.has(backend)) {
      throw new Error(`unsupported structured CPU backend '${backend}'`);
    }
  }
  return backends;
}

function configuredGpuBackends(
  opts: StructuredCompileOptions,
): readonly GpuBackendName[] {
  const backends = preference(
    opts.backends?.gpu,
    ["gpu-codegen" as const],
    "GPU backend",
  );
  for (const backend of backends) {
    if (!GPU_BACKENDS.has(backend)) {
      throw new Error(`unsupported structured GPU backend '${backend}'`);
    }
  }
  return backends;
}

type ExecutableStage = MapStage | ReduceStage | ToNumsStage;
type PlacementKey = "map" | "reduce" | "toNums";

function placementKey(stage: ExecutableStage): PlacementKey {
  return stage.kind === "to-nums" ? "toNums" : stage.kind;
}

function effectiveStagePlacement(
  stage: ExecutableStage,
  opts: StructuredCompileOptions,
): StagePlacement {
  if (stage.requestedPlacement) return stage.requestedPlacement;
  const configured = opts.placement;
  if (typeof configured === "string") return configured;
  return configured?.[placementKey(stage)] ?? configured?.default ?? "auto";
}

function stageTargets(
  stage: ExecutableStage,
  placement: StagePlacement,
  opts: StructuredCompileOptions,
): readonly ExecutionTarget[] {
  if (placement !== "auto") return [placement];
  const configured = opts.auto?.targets;
  const defaults: readonly ExecutionTarget[] =
    stage.kind === "to-nums" ? ["cpu"] : ["gpu", "cpu"];
  const targets = preference(
    configured?.[placementKey(stage)] ?? configured?.default,
    defaults,
    `auto target for ${placementKey(stage)}`,
  );
  for (const target of targets) {
    if (target !== "cpu" && target !== "gpu") {
      throw new Error(`unsupported structured execution target '${target}'`);
    }
  }
  return targets;
}

function gpuUnsupportedReason(stage: ExecutableStage): string | null {
  if (stage.kind === "map") return null;
  if (stage.kind === "to-nums") return "toNums stages execute on CPU";
  if (!stage.associative || stage.order !== "tree") {
    return "GPU reductions require associative: true and tree order";
  }
  return null;
}

function gatherScalarRoots(definition: StructuredDefinition): NumNode[] {
  const roots: NumNode[] = [];
  const seen = new Set<NumNode>();
  const add = (items: readonly NumNode[]): void => {
    for (const root of items) {
      if (seen.has(root) || root.kind === KIND_LIT) continue;
      seen.add(root);
      roots.push(root);
    }
  };

  for (const stage of definition.stages) {
    if (stage.kind === "source") {
      add(stage.roots);
    } else if (stage.kind === "map") {
      add(stage.using.roots);
    } else if (stage.kind === "reduce") {
      add(stage.initial);
      add(stage.using.roots);
    } else if (stage.kind === "to-nums") {
      add(stage.using.roots);
    }
  }
  return roots;
}

function valuesForRoots(
  roots: readonly NumNode[],
  values: ReadonlyMap<NumNode, number>,
): number[] {
  return roots.map((root) => {
    const value = values.get(root);
    if (value === undefined && !values.has(root)) {
      if (root.kind === KIND_LIT) return (root as LiteralNum).value;
      throw new Error("structured scalar root was not evaluated");
    }
    return value!;
  });
}

function usingValues(
  usingBindings: ScalarBindings,
  scalarValues: ReadonlyMap<NumNode, number>,
): number[] {
  return valuesForRoots(usingBindings.roots, scalarValues);
}

function row(
  values: readonly number[],
  width: number,
  index: number,
): number[] {
  return values.slice(index * width, (index + 1) * width);
}

function checkedKernelEval(
  kernel: CompiledCpuKernel,
  inputs: readonly number[],
  expectedWidth: number,
  stage: StructuredStage,
): number[] {
  const result = kernel.eval(inputs);
  if (result.length !== expectedWidth) {
    throw new Error(
      `structured stage ${stage.id} (${stage.kind}) returned ${result.length} values; expected ${expectedWidth}`,
    );
  }
  return result;
}

function evalCpuMap(
  stage: MapStage,
  kernel: CompiledCpuKernel,
  source: readonly number[],
  scalarValues: ReadonlyMap<NumNode, number>,
): number[] {
  const uniforms = usingValues(stage.using, scalarValues);
  const output: number[] = [];
  for (let index = 0; index < stage.count; index++) {
    const inputs = [
      ...row(source, stage.inputShape.width, index),
      ...uniforms,
      index,
    ];
    output.push(
      ...checkedKernelEval(kernel, inputs, stage.outputShape.width, stage),
    );
  }
  return output;
}

interface HostColumn {
  readonly location: "host";
  readonly values: number[];
}

interface DeviceColumn {
  readonly location: "device";
  readonly slice: DeviceBufferSlice;
  readonly count: number;
  readonly width: number;
}

type StageColumn = HostColumn | DeviceColumn;

function combineRows(
  stage: ReduceStage,
  kernel: CompiledCpuKernel,
  left: readonly number[],
  right: readonly number[],
  uniforms: readonly number[],
): number[] {
  return checkedKernelEval(
    kernel,
    [...left, ...right, ...uniforms],
    stage.shape.width,
    stage,
  );
}

function evalReduce(
  stage: ReduceStage,
  kernel: CompiledCpuKernel,
  source: readonly number[],
  scalarValues: ReadonlyMap<NumNode, number>,
): number[] {
  const width = stage.shape.width;
  const initial = valuesForRoots(stage.initial, scalarValues);
  const uniforms = usingValues(stage.using, scalarValues);

  if (stage.builtIn) {
    if (stage.inputCount === 0) return initial;
    let level = Array.from({ length: stage.inputCount }, (_, index) =>
      row(source, width, index),
    );
    while (level.length > 1) {
      const next: number[][] = [];
      for (let index = 0; index < level.length; index += 2) {
        const left = level[index]!;
        const right = level[index + 1];
        next.push(
          right === undefined
            ? left
            : combineRows(stage, kernel, left, right, uniforms),
        );
      }
      level = next;
    }
    return level[0]!;
  }

  if (stage.order === "left") {
    let accumulator = initial;
    for (let index = 0; index < stage.inputCount; index++) {
      accumulator = combineRows(
        stage,
        kernel,
        accumulator,
        row(source, width, index),
        uniforms,
      );
    }
    return accumulator;
  }

  let level: number[][] = [
    initial,
    ...Array.from({ length: stage.inputCount }, (_, index) =>
      row(source, width, index),
    ),
  ];
  while (level.length > 1) {
    const next: number[][] = [];
    for (let index = 0; index < level.length; index += 2) {
      const left = level[index]!;
      const right = level[index + 1];
      next.push(
        right === undefined
          ? left
          : combineRows(stage, kernel, left, right, uniforms),
      );
    }
    level = next;
  }
  return level[0] ?? initial;
}

function evalToNums(
  stage: ToNumsStage,
  kernel: CompiledCpuKernel,
  source: readonly number[],
  scalarValues: ReadonlyMap<NumNode, number>,
): number[] {
  return kernel.eval([...source, ...usingValues(stage.using, scalarValues)]);
}

function decodeResult(shape: ResultShape, flat: readonly number[]): unknown {
  const decoded: Array<number | number[]> = [];
  let offset = 0;
  for (const valueShape of shape.values) {
    const components = flat.slice(offset, offset + valueShape.width);
    if (components.length !== valueShape.width) {
      throw new Error(
        "structured toNums output is shorter than its result shape",
      );
    }
    decoded.push(valueShape.kind === "num" ? components[0]! : components);
    offset += valueShape.width;
  }
  if (offset !== flat.length) {
    throw new Error(
      `structured toNums output has ${flat.length} scalars; expected ${offset}`,
    );
  }
  return shape.collection === "single" ? decoded[0] : decoded;
}

function compileStages(
  definition: StructuredDefinition,
  opts: StructuredCompileOptions,
  cpuBackends: readonly CpuBackendName[],
  gpuBackends: readonly GpuBackendName[],
  checkpoint?: (phase: string) => void,
): CompiledStage[] {
  const compiled: CompiledStage[] = [];
  const uniformSets: NumNode[][] = [];
  const uniformGroup = (roots: readonly NumNode[]): number => {
    const existing = uniformSets.findIndex(
      (candidate) =>
        candidate.length === roots.length &&
        candidate.every((root, index) => root === roots[index]),
    );
    if (existing >= 0) return existing;
    uniformSets.push([...roots]);
    return uniformSets.length - 1;
  };

  const compileCandidate = (
    stage: ExecutableStage,
    target: ExecutionTarget,
    backend: BackendName,
  ): CompiledStage => {
    checkpoint?.(
      `lona:structured:${stage.kind}:${stage.id}:${target}:${backend}:compile`,
    );
    if (target === "gpu") {
      if (stage.kind === "map") {
        return {
          stage,
          placement: "gpu",
          backend,
          gpuMapKernel: compileGpuMapKernel(
            stage.kernel.roots,
            stage.kernel.inputs,
            stage.inputShape.width,
            stage.outputShape.width,
          ),
          uniformGroup: uniformGroup(stage.using.roots),
        };
      }
      if (stage.kind === "reduce") {
        return {
          stage,
          placement: "gpu",
          backend,
          gpuReduceKernel: stage.builtIn
            ? compileGpuReduceKernel(stage.builtIn, stage.shape.width)
            : compileGpuTracedReduceKernel(
                stage.kernel.roots,
                stage.kernel.inputs,
                stage.shape.width,
              ),
          uniformGroup: uniformGroup(stage.using.roots),
        };
      }
      throw new Error("toNums stages execute on CPU");
    }

    const cpuBackend = backend as CpuBackendName;
    if (stage.kind === "map" || stage.kind === "reduce") {
      return {
        stage,
        placement: "cpu",
        backend,
        cpuKernel: compileCpuKernel(
          stage.kernel.roots,
          stage.kernel.inputs,
          cpuBackend,
        ),
      };
    }
    return {
      stage,
      placement: "cpu",
      backend,
      cpuKernel: compileCpuKernel(stage.roots, stage.inputs, cpuBackend),
    };
  };

  try {
    for (const stage of definition.stages) {
      if (stage.kind === "source") {
        compiled.push({ stage });
        continue;
      }

      const placement = effectiveStagePlacement(stage, opts);
      const targets = stageTargets(stage, placement, opts);
      const failures: string[] = [];
      let selected: CompiledStage | null = null;

      for (const target of targets) {
        if (target === "gpu") {
          const unsupported = gpuUnsupportedReason(stage);
          if (unsupported) {
            failures.push(unsupported);
            continue;
          }
          if (!isGpuAvailable()) {
            failures.push("GPU is unavailable");
            continue;
          }
        }

        const backends = target === "gpu" ? gpuBackends : cpuBackends;
        for (const backend of backends) {
          try {
            selected = compileCandidate(stage, target, backend);
            break;
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            failures.push(`${target}/${backend}: ${message}`);
            checkpoint?.(
              `lona:structured:${stage.kind}:${stage.id}:auto:fallback:${target}:${backend}`,
            );
          }
        }
        if (selected) break;
      }

      if (!selected) {
        throw new Error(
          `structured stage ${stage.id} (${stage.kind}) could not satisfy placement '${placement}': ${failures.join("; ")}`,
        );
      }
      if (placement === "auto" && selected.placement !== targets[0]) {
        checkpoint?.(
          `lona:structured:${stage.kind}:${stage.id}:auto:selected:${selected.placement}:${selected.backend}`,
        );
      }
      compiled.push(selected);
    }
    return compiled;
  } catch (error) {
    for (const stage of compiled) {
      stage.cpuKernel?.dispose();
      stage.gpuMapKernel?.dispose();
      stage.gpuReduceKernel?.dispose();
    }
    throw error;
  }
}

export function compileStructuredRoutine<R extends NumBuildResult>(
  definition: StructuredDefinition,
  opts: StructuredCompileOptions = {},
): StructuredRoutine<R> {
  const cpuBackends = configuredCpuBackends(opts);
  const gpuBackends = configuredGpuBackends(opts);
  opts.diagnosticCheckpoint?.("lona:structured:compile:start");

  const scalarRoots = gatherScalarRoots(definition);
  let scalarRoutine: ValueRoutine | MultiValueRoutine | null = null;
  if (scalarRoots.length > 0) {
    const failures: string[] = [];
    for (const backend of cpuBackends) {
      opts.diagnosticCheckpoint?.(
        `lona:structured:source:cpu:${backend}:compile`,
      );
      try {
        scalarRoutine = compileValueRoutine(scalarRoots, { backend });
      } catch (error) {
        failures.push(error instanceof Error ? error.message : String(error));
      }
      if (scalarRoutine) break;
    }
    if (!scalarRoutine) {
      throw new Error(
        `failed to compile structured scalar sources: ${failures.join("; ")}`,
      );
    }
  }

  let compiledStages: CompiledStage[];
  try {
    compiledStages = compileStages(
      definition,
      opts,
      cpuBackends,
      gpuBackends,
      opts.diagnosticCheckpoint,
    );
  } catch (error) {
    scalarRoutine?.dispose?.();
    throw error;
  }
  const output = definition.stages[definition.outputStage];
  if (!output || output.kind !== "to-nums") {
    scalarRoutine?.dispose?.();
    for (const compiled of compiledStages) {
      compiled.cpuKernel?.dispose();
      compiled.gpuMapKernel?.dispose();
      compiled.gpuReduceKernel?.dispose();
    }
    throw new Error("structured routine output must be a toNums stage");
  }

  const stages: readonly StructuredStageInfo[] = Object.freeze(
    compiledStages.map(({ stage, placement, backend }) =>
      Object.freeze({
        id: stage.id,
        kind: stage.kind === "to-nums" ? "toNums" : stage.kind,
        placement,
        backend,
      }),
    ),
  );
  const varSlots = Object.freeze([...(scalarRoutine?.varSlots ?? [])]);
  const numVars = varSlots.length;
  let disposed = false;
  let lastEvaluationStats: StructuredEvaluationStats | null = null;
  const bufferPool = new Map<string, GPUBuffer[]>();

  opts.diagnosticCheckpoint?.("lona:structured:compile:done");
  return {
    varSlots,
    numVars,
    stages,
    get lastEvaluationStats(): StructuredEvaluationStats | null {
      return lastEvaluationStats;
    },
    async evalAsync(vars: VarMap): Promise<ConcreteResult<R>> {
      if (disposed) throw new Error("structured routine has been disposed");
      const evaluationStart = performance.now();
      let uploadedBytes = 0;
      let downloadedBytes = 0;
      let dispatchCount = 0;
      let readbackCount = 0;
      const stageTimings: Array<{
        stageId: number;
        backend?: BackendName;
        milliseconds: number;
      }> = [];

      const scalarNumbers = scalarRoutine
        ? normalizeRoutineResult(scalarRoutine, vars)
        : [];
      const scalarValues = new Map<NumNode, number>();
      for (let index = 0; index < scalarRoots.length; index++) {
        scalarValues.set(scalarRoots[index]!, scalarNumbers[index]!);
      }

      const stageValues = new Map<number, StageColumn>();
      const ownedBuffers = new Map<
        GPUBuffer,
        { readonly key: string; readonly byteLength: number }
      >();
      const uniformBuffers = new Map<number, DeviceBufferSlice>();
      let encoder: GPUCommandEncoder | null = null;

      const createBuffer = (
        byteLength: number,
        usage: GPUBufferUsageFlags,
      ): DeviceBufferSlice => {
        const device = requireGpuDevice();
        if (byteLength > device.limits.maxBufferSize) {
          throw new Error(
            `structured GPU buffer requires ${byteLength} bytes; device limit is ${device.limits.maxBufferSize}`,
          );
        }
        const allocatedBytes = Math.max(byteLength, 4);
        const key = `${allocatedBytes}:${usage}`;
        const available = bufferPool.get(key);
        const reused = available?.pop();
        if (reused) {
          opts.diagnosticCheckpoint?.(
            `lona:structured:buffer-reuse:bytes:${allocatedBytes}`,
          );
        }
        const buffer =
          reused ??
          device.createBuffer({
            size: allocatedBytes,
            usage,
          });
        ownedBuffers.set(buffer, { key, byteLength: allocatedBytes });
        return { buffer, offset: 0, byteLength };
      };

      const upload = (
        values: readonly number[],
        stageId: number,
      ): DeviceBufferSlice => {
        const data = Float32Array.from(values);
        const slice = createBuffer(
          data.byteLength,
          // COPY_SRC: an uploaded column can itself be the source of a later
          // device-side copy (e.g. the non-builtin GPU reduce's
          // initial-value prepend), not just a compute-shader input.
          GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        );
        if (data.byteLength > 0) {
          requireGpuDevice().queue.writeBuffer(
            slice.buffer,
            0,
            data.buffer,
            data.byteOffset,
            data.byteLength,
          );
        }
        uploadedBytes += data.byteLength;
        opts.diagnosticCheckpoint?.(
          `lona:structured:transfer:${stageId}:host-to-device`,
        );
        opts.diagnosticCheckpoint?.(
          `lona:structured:cost:upload-bytes:${data.byteLength}`,
        );
        return slice;
      };

      const ensureDevice = (
        column: StageColumn,
        count: number,
        width: number,
        stageId: number,
      ): DeviceColumn => {
        if (column.location === "device") {
          if (column.count !== count || column.width !== width) {
            throw new Error(
              `structured device column shape mismatch at stage ${stageId}`,
            );
          }
          return column;
        }
        return {
          location: "device",
          slice: upload(column.values, stageId),
          count,
          width,
        };
      };

      const ensureHost = async (
        column: StageColumn,
        stageId: number,
      ): Promise<HostColumn> => {
        if (column.location === "host") return column;
        if (column.slice.byteLength === 0) {
          return { location: "host", values: [] };
        }
        const staging = requireGpuDevice().createBuffer({
          size: column.slice.byteLength,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const pending = encoder ?? requireGpuDevice().createCommandEncoder();
        pending.copyBufferToBuffer(
          column.slice.buffer,
          column.slice.offset,
          staging,
          0,
          column.slice.byteLength,
        );
        requireGpuDevice().queue.submit([pending.finish()]);
        opts.diagnosticCheckpoint?.("lona:structured:gpu:submit");
        encoder = null;
        try {
          const values = await mapGpuReadbackBuffer(
            staging,
            column.slice.byteLength,
          );
          downloadedBytes += column.slice.byteLength;
          readbackCount++;
          opts.diagnosticCheckpoint?.(
            `lona:structured:transfer:${stageId}:device-to-host`,
          );
          opts.diagnosticCheckpoint?.(
            `lona:structured:cost:download-bytes:${column.slice.byteLength}`,
          );
          return { location: "host", values: Array.from(values) };
        } finally {
          staging.destroy();
        }
      };

      try {
        for (const compiled of compiledStages) {
          const { stage } = compiled;
          const stageStart = performance.now();
          try {
            if (stage.kind === "source") {
              stageValues.set(stage.id, {
                location: "host",
                values: valuesForRoots(stage.roots, scalarValues),
              });
              continue;
            }

            const source = stageValues.get(stage.source);
            if (!source) {
              throw new Error(
                `structured stage ${stage.id} is missing source ${stage.source}`,
              );
            }

            if (stage.kind === "map" && compiled.gpuMapKernel) {
              const input = ensureDevice(
                source,
                stage.count,
                stage.inputShape.width,
                stage.id,
              );
              const outputSlice = createBuffer(
                stage.count * stage.outputShape.width * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
              );
              let uniforms: DeviceBufferSlice | null = null;
              if (compiled.gpuMapKernel.uniformWidth > 0) {
                const group = compiled.uniformGroup!;
                uniforms = uniformBuffers.get(group) ?? null;
                if (!uniforms) {
                  uniforms = upload(
                    usingValues(stage.using, scalarValues),
                    stage.id,
                  );
                  uniformBuffers.set(group, uniforms);
                }
              }
              encoder ??= requireGpuDevice().createCommandEncoder();
              compiled.gpuMapKernel.encode(
                encoder,
                input.slice,
                uniforms,
                outputSlice,
                stage.count,
              );
              dispatchCount += compiled.gpuMapKernel.dispatchCount;
              stageValues.set(stage.id, {
                location: "device",
                slice: outputSlice,
                count: stage.count,
                width: stage.outputShape.width,
              });
            } else if (stage.kind === "reduce" && compiled.gpuReduceKernel) {
              if (stage.inputCount === 0) {
                stageValues.set(stage.id, {
                  location: "host",
                  values: valuesForRoots(stage.initial, scalarValues),
                });
                continue;
              }
              const sourceInput = ensureDevice(
                source,
                stage.inputCount,
                stage.shape.width,
                stage.id,
              );
              let input = sourceInput;
              let reductionCount = stage.inputCount;
              if (!stage.builtIn) {
                const initial = Float32Array.from(
                  valuesForRoots(stage.initial, scalarValues),
                );
                const combinedSlice = createBuffer(
                  (stage.inputCount + 1) * stage.shape.width * 4,
                  GPUBufferUsage.STORAGE |
                    GPUBufferUsage.COPY_DST |
                    GPUBufferUsage.COPY_SRC,
                );
                requireGpuDevice().queue.writeBuffer(
                  combinedSlice.buffer,
                  0,
                  initial.buffer,
                  initial.byteOffset,
                  initial.byteLength,
                );
                uploadedBytes += initial.byteLength;
                opts.diagnosticCheckpoint?.(
                  `lona:structured:cost:upload-bytes:${initial.byteLength}`,
                );
                encoder ??= requireGpuDevice().createCommandEncoder();
                encoder.copyBufferToBuffer(
                  sourceInput.slice.buffer,
                  sourceInput.slice.offset,
                  combinedSlice.buffer,
                  initial.byteLength,
                  sourceInput.slice.byteLength,
                );
                reductionCount++;
                input = {
                  location: "device",
                  slice: combinedSlice,
                  count: reductionCount,
                  width: stage.shape.width,
                };
              }
              if (reductionCount === 1) {
                stageValues.set(stage.id, { ...input, count: 1 });
                continue;
              }
              const outputSlice = createBuffer(
                stage.shape.width * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
              );
              let uniforms: DeviceBufferSlice | null = null;
              if (compiled.gpuReduceKernel.uniformWidth > 0) {
                const group = compiled.uniformGroup!;
                uniforms = uniformBuffers.get(group) ?? null;
                if (!uniforms) {
                  uniforms = upload(
                    usingValues(stage.using, scalarValues),
                    stage.id,
                  );
                  uniformBuffers.set(group, uniforms);
                }
              }
              encoder ??= requireGpuDevice().createCommandEncoder();
              compiled.gpuReduceKernel.encode(
                encoder,
                input.slice,
                uniforms,
                outputSlice,
                reductionCount,
              );
              dispatchCount +=
                compiled.gpuReduceKernel.dispatchCount(reductionCount);
              stageValues.set(stage.id, {
                location: "device",
                slice: outputSlice,
                count: 1,
                width: stage.shape.width,
              });
            } else {
              const hostSource = await ensureHost(source, stage.id);
              const values =
                stage.kind === "map"
                  ? evalCpuMap(
                      stage,
                      compiled.cpuKernel!,
                      hostSource.values,
                      scalarValues,
                    )
                  : stage.kind === "reduce"
                    ? evalReduce(
                        stage,
                        compiled.cpuKernel!,
                        hostSource.values,
                        scalarValues,
                      )
                    : evalToNums(
                        stage,
                        compiled.cpuKernel!,
                        hostSource.values,
                        scalarValues,
                      );
              stageValues.set(stage.id, { location: "host", values });
            }
          } finally {
            stageTimings.push({
              stageId: stage.id,
              backend: compiled.backend,
              milliseconds: performance.now() - stageStart,
            });
          }
        }

        const result = stageValues.get(output.id);
        if (!result) throw new Error("structured routine produced no output");
        const flatOutput = (await ensureHost(result, output.id)).values;
        return decodeResult(
          output.resultShape,
          flatOutput,
        ) as ConcreteResult<R>;
      } finally {
        for (const [buffer, allocation] of ownedBuffers) {
          const available = bufferPool.get(allocation.key) ?? [];
          available.push(buffer);
          bufferPool.set(allocation.key, available);
        }
        const evaluationMilliseconds = performance.now() - evaluationStart;
        lastEvaluationStats = Object.freeze({
          uploadedBytes,
          downloadedBytes,
          transferredBytes: uploadedBytes + downloadedBytes,
          dispatchCount,
          readbackCount,
          evaluationMilliseconds,
          stageTimings: Object.freeze([...stageTimings]),
        });
        opts.diagnosticCheckpoint?.(
          `lona:structured:cost:transferred-bytes:${uploadedBytes + downloadedBytes}`,
        );
        opts.diagnosticCheckpoint?.(
          `lona:structured:cost:dispatches:${dispatchCount}`,
        );
        opts.diagnosticCheckpoint?.(
          `lona:structured:cost:readbacks:${readbackCount}`,
        );
        opts.diagnosticCheckpoint?.(
          `lona:structured:cost:evaluation-ms:${evaluationMilliseconds}`,
        );
      }
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      scalarRoutine?.dispose?.();
      for (const compiled of compiledStages) {
        compiled.cpuKernel?.dispose();
        compiled.gpuMapKernel?.dispose();
        compiled.gpuReduceKernel?.dispose();
      }
      for (const buffers of bufferPool.values()) {
        for (const buffer of buffers) buffer.destroy();
      }
      bufferPool.clear();
    },
  };
}
