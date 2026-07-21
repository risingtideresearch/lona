import type { NumNode, VarName } from "lona/internal";
import { compileTape, type CompiledTape } from "lona/internal";
import type { StructuredParamInput } from "./ir";

export interface CompiledStructuredTape {
  readonly tape: CompiledTape;
  /** Inputs in the stage-declared positional order. */
  readonly inputs: readonly StructuredParamInput[];
  /** Private variable name corresponding to each positional input. */
  readonly inputNames: readonly VarName[];
  /** Used inputs in the actual CompiledTape variable-slot order. */
  readonly tapeInputs: readonly StructuredParamInput[];
}

/** Compile a structured scalar kernel whose inputs are private Variables. */
export function compileStructuredTape(
  roots: readonly NumNode[],
  inputs: readonly StructuredParamInput[],
): CompiledStructuredTape {
  const inputByName = new Map<VarName, StructuredParamInput>();
  const inputNames: VarName[] = [];

  for (let index = 0; index < inputs.length; index++) {
    const input = inputs[index]!;
    const name = input.param.name;
    if (inputByName.has(name)) {
      throw new Error(
        `structured input ${index} binds the same variable more than once`,
      );
    }
    inputByName.set(name, input);
    inputNames.push(name);
  }

  const tape = compileTape([...roots]);
  if (!tape) throw new Error("failed to compile structured variable tape");

  const tapeInputs = tape.varSlots.slice(0, tape.numVars).map((name) => {
    const input = inputByName.get(name);
    if (!input) {
      throw new Error("structured tape contains an undeclared input variable");
    }
    return input;
  });

  return Object.freeze({
    tape,
    inputs: Object.freeze([...inputs]),
    inputNames: Object.freeze(inputNames),
    tapeInputs: Object.freeze(tapeInputs),
  });
}
