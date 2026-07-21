import type { NumNode, VarName } from "lona/internal";
import {
  compileValueRoutineFromTape,
  type MultiValueRoutine,
  type ValueRoutine,
} from "lona/internal";
import { compileStructuredTape } from "../compile-tape";
import type { StructuredParamInput } from "../ir";
import type { CpuBackendName } from "../types";

export interface CompiledCpuKernel {
  readonly arity: number;
  eval(values: readonly number[]): number[];
  dispose(): void;
}

function resultArray(
  routine: ValueRoutine | MultiValueRoutine,
  vars: Map<VarName, number>,
): number[] {
  const result = routine.eval(vars);
  return typeof result === "number" ? [result] : result;
}

/** Compile a private-variable kernel directly to tape, preserving positional metadata. */
export function compileCpuKernel(
  roots: readonly NumNode[],
  inputs: readonly StructuredParamInput[],
  backend: CpuBackendName,
): CompiledCpuKernel {
  const compiled = compileStructuredTape(roots, inputs);
  const routine = compileValueRoutineFromTape(compiled.tape, { backend });

  return {
    arity: inputs.length,
    eval(values: readonly number[]): number[] {
      if (values.length !== inputs.length) {
        throw new Error(
          `structured CPU kernel expected ${inputs.length} inputs, got ${values.length}`,
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
