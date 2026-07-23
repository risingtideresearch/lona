import type { NumNode, VarName } from "lona/internal";
import { compileTape, type CompiledTape } from "lona/internal";
import type { ColumnarParamInput } from "./ir";

export interface CompiledColumnarTape {
  readonly tape: CompiledTape;
  /** Inputs in the stage-declared positional order. */
  readonly inputs: readonly ColumnarParamInput[];
  /** Private variable name corresponding to each positional input. */
  readonly inputNames: readonly VarName[];
  /** Used inputs in the actual CompiledTape variable-slot order. */
  readonly tapeInputs: readonly ColumnarParamInput[];
}

/** Compile a columnar scalar kernel whose inputs are private Variables. */
export function compileColumnarTape(
  roots: readonly NumNode[],
  inputs: readonly ColumnarParamInput[],
): CompiledColumnarTape {
  const inputByName = new Map<VarName, ColumnarParamInput>();
  const inputNames: VarName[] = [];

  for (let index = 0; index < inputs.length; index++) {
    const input = inputs[index]!;
    const name = input.param.name;
    if (inputByName.has(name)) {
      throw new Error(
        `columnar input ${index} binds the same variable more than once`,
      );
    }
    inputByName.set(name, input);
    inputNames.push(name);
  }

  const tape = compileTape([...roots]);
  if (!tape) throw new Error("failed to compile columnar variable tape");

  const tapeInputs = tape.varSlots.slice(0, tape.numVars).map((name) => {
    const input = inputByName.get(name);
    if (!input) {
      throw new Error("columnar tape contains an undeclared input variable");
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
