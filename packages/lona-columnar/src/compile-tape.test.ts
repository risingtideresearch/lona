import { describe, expect, test } from "vitest";
import { Num } from "lona";
import { Variable } from "lona/internal";
import { varNode } from "lona/internal";
import { compileValueRoutineFromTape } from "lona/internal";
import { compileColumnarTape } from "./compile-tape";

function param(label: string, index: number): Variable {
  return varNode(Symbol(`${label}.param.${index}`));
}

describe("compileColumnarTape", () => {
  test("uses the parameter Variables directly without rebuilding the Num DAG", () => {
    const a = param("direct-binding", 0);
    const b = param("direct-binding", 1);
    const root = new Num(a).mul(2).add(new Num(b));
    const originalRoot = root.n;

    const compiled = compileColumnarTape(
      [root.n],
      [
        { param: a, binding: { kind: "row", component: 0 } },
        { param: b, binding: { kind: "uniform", component: 0 } },
      ],
    );

    expect(root.n).toBe(originalRoot);
    expect(compiled.inputs.map((input) => input.binding.kind)).toEqual([
      "row",
      "uniform",
    ]);
    expect(compiled.inputNames).toEqual([a.name, b.name]);
    expect(compiled.inputNames.every((name) => typeof name === "symbol")).toBe(
      true,
    );
    expect(new Set(compiled.tape.varSlots)).toEqual(
      new Set(compiled.inputNames),
    );
    expect(compiled.tapeInputs.map((input) => input.binding.kind)).toEqual(
      compiled.tape.varSlots.map(
        (name) =>
          compiled.inputs[compiled.inputNames.indexOf(name)]!.binding.kind,
      ),
    );

    const routine = compileValueRoutineFromTape(compiled.tape, {
      backend: "js-interp",
    });
    expect(
      routine.eval(
        new Map([
          [compiled.inputNames[0]!, 3],
          [compiled.inputNames[1]!, 4],
        ]),
      ),
    ).toBe(10);
  });

  test("rejects missing and duplicate variable bindings", () => {
    const a = param("invalid-binding", 0);
    const b = param("invalid-binding", 1);
    const root = new Num(a).add(new Num(b));

    expect(() =>
      compileColumnarTape(
        [root.n],
        [{ param: a, binding: { kind: "row", component: 0 } }],
      ),
    ).toThrow(/undeclared input variable/);

    expect(() =>
      compileColumnarTape(
        [root.n],
        [
          { param: a, binding: { kind: "row", component: 0 } },
          { param: a, binding: { kind: "uniform", component: 0 } },
        ],
      ),
    ).toThrow(/same variable more than once/);
  });

  test("preserves unused positional inputs in metadata", () => {
    const used = param("unused-binding", 0);
    const unused = param("unused-binding", 1);
    const compiled = compileColumnarTape(
      [new Num(used).square().n],
      [
        { param: used, binding: { kind: "row", component: 0 } },
        { param: unused, binding: { kind: "index" } },
      ],
    );

    expect(compiled.inputs).toHaveLength(2);
    expect(compiled.inputNames).toHaveLength(2);
    expect(compiled.tape.numVars).toBe(1);
    expect(compiled.tape.varSlots).toEqual([compiled.inputNames[0]]);
    expect(compiled.tapeInputs).toEqual([compiled.inputs[0]]);
    expect(used).toBeInstanceOf(Variable);
  });
});
