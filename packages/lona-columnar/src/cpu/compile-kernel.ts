import type { NumNode, VarName } from "lona/internal";
import {
  compileJvpRoutineFromTape,
  compileValueRoutineFromTape,
  type MultiValueRoutine,
  type ValueRoutine,
} from "lona/internal";
import { compileColumnarTape } from "../compile-tape";
import type { ColumnarParamInput } from "../ir";
import type { CpuBackendName } from "../types";

export interface CompiledCpuKernel {
  readonly arity: number;
  eval(values: readonly number[]): number[];
  dispose(): void;
}

export interface CompiledCpuJvpKernel {
  readonly arity: number;
  readonly outputWidth: number;
  readonly numDirections: number;
  eval(
    values: readonly number[],
    tangents: readonly number[],
  ): { values: number[]; tangents: number[] };
  dispose(): void;
}

function resultArray(
  routine: ValueRoutine | MultiValueRoutine,
  vars: Map<VarName, number>,
): number[] {
  const result = routine.eval(vars);
  return typeof result === "number" ? [result] : result;
}

export function compileCpuJvpKernel(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
  backend: CpuBackendName,
  numDirections: number,
): CompiledCpuJvpKernel {
  const compiled = compileColumnarTape(roots, inputs);
  const routine = compileJvpRoutineFromTape(compiled.tape, numDirections, {
    backend,
  });
  const inputIndex = new Map(
    inputs.map((input, index) => [input.param.name, index] as const),
  );
  const usedInputs = compiled.tapeInputs.map((input) => {
    const index = inputIndex.get(input.param.name);
    if (index === undefined) {
      throw new Error("columnar JVP tape contains an undeclared input");
    }
    return index;
  });

  return {
    arity: inputs.length,
    outputWidth: roots.length,
    numDirections,
    eval(values, tangents) {
      if (values.length !== inputs.length) {
        throw new Error(
          `columnar CPU JVP expected ${inputs.length} inputs, got ${values.length}`,
        );
      }
      if (tangents.length !== inputs.length * numDirections) {
        throw new Error(
          `columnar CPU JVP expected ${inputs.length * numDirections} tangents, got ${tangents.length}`,
        );
      }
      const packedValues = Float64Array.from(
        usedInputs,
        (index) => values[index]!,
      );
      const packedTangents = new Float64Array(
        usedInputs.length * numDirections,
      );
      for (let used = 0; used < usedInputs.length; used++) {
        const declared = usedInputs[used]!;
        for (let direction = 0; direction < numDirections; direction++) {
          packedTangents[used * numDirections + direction] =
            tangents[declared * numDirections + direction]!;
        }
      }
      const result = routine.evalPacked(packedValues, packedTangents);
      return { values: result.vals, tangents: result.tangents.flat() };
    },
    dispose(): void {
      routine.dispose?.();
    },
  };
}

/** Compile a private-variable kernel directly to tape, preserving positional metadata. */
export function compileCpuKernel(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
  backend: CpuBackendName,
): CompiledCpuKernel {
  const compiled = compileColumnarTape(roots, inputs);
  const routine = compileValueRoutineFromTape(compiled.tape, { backend });

  return {
    arity: inputs.length,
    eval(values: readonly number[]): number[] {
      if (values.length !== inputs.length) {
        throw new Error(
          `columnar CPU kernel expected ${inputs.length} inputs, got ${values.length}`,
        );
      }
      const vars = new Map<VarName, number>();
      for (let index = 0; index < compiled.inputNames.length; index++) {
        vars.set(compiled.inputNames[index]!, values[index]!);
      }
      return resultArray(routine, vars);
    },
    dispose(): void {
      routine.dispose?.();
    },
  };
}
