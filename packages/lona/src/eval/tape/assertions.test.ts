import { describe, expect, test } from "vitest";
import type { CompiledTape } from "./compiled-tape";
import { TapeAssertionError } from "./assertions";
import { compileTape } from "./compile";
import {
  OP_ADD,
  OP_ASSERT_NONZERO,
  OP_ASSERT_ZERO,
  OP_LIT,
  OP_VAR,
} from "./opcodes";
import { binaryNode, varNode } from "../../core/tree-cons";
import {
  evalTape,
  compileForwardAutodiff,
} from "../routines/backends/js-interp/tape-eval";
import { compileFunctionFromTape } from "../routines/backends/js-codegen/codegen";
import { compileWasmTapeFromTape } from "../routines/backends/wasm-interp/tape-eval";
import { compileWasmFromTape } from "../routines/backends/wasm-codegen/codegen";

function passingAssertionTape(): CompiledTape {
  return {
    opcodes: new Uint8Array([
      OP_VAR,
      OP_ASSERT_ZERO,
      OP_LIT,
      OP_ASSERT_NONZERO,
    ]),
    argA: new Int32Array([0, 0, 0, 2]),
    argB: new Int32Array([0, 10, 0, 11]),
    literals: new Float64Array([42]),
    varSlots: ["x"],
    numVars: 1,
    rootIndices: [1, 3],
    assertions: [
      { id: 10, tapeIndex: 1, kind: "zero" },
      { id: 11, tapeIndex: 3, kind: "nonzero" },
    ],
  };
}

function failingAssertionTape(kind: "zero" | "nonzero"): CompiledTape {
  return {
    opcodes: new Uint8Array([
      OP_VAR,
      kind === "zero" ? OP_ASSERT_ZERO : OP_ASSERT_NONZERO,
    ]),
    argA: new Int32Array([0, 0]),
    argB: new Int32Array([0, 7]),
    literals: new Float64Array([]),
    varSlots: ["x"],
    numVars: 1,
    rootIndices: [1],
    assertions: [{ id: 7, tapeIndex: 1, kind }],
  };
}

describe("tape assertion opcodes", () => {
  test("JS interpreter treats passing assertions as value-producing ops", () => {
    expect(evalTape(passingAssertionTape(), new Map([["x", 0]]))).toEqual([
      0, 42,
    ]);
  });

  test("JS interpreter throws when an assertion fails", () => {
    expect(() =>
      evalTape(failingAssertionTape("zero"), new Map([["x", 7]])),
    ).toThrow(TapeAssertionError);
    expect(() =>
      evalTape(failingAssertionTape("nonzero"), new Map([["x", 0]])),
    ).toThrow(TapeAssertionError);
  });

  test("JS codegen supports passing assertions and throws on failures", () => {
    expect(
      compileFunctionFromTape(passingAssertionTape())(new Map([["x", 0]])),
    ).toEqual([0, 42]);
    expect(() =>
      compileFunctionFromTape(failingAssertionTape("zero"))(
        new Map([["x", 7]]),
      ),
    ).toThrow(TapeAssertionError);
    expect(() =>
      compileFunctionFromTape(failingAssertionTape("nonzero"))(
        new Map([["x", 0]]),
      ),
    ).toThrow(TapeAssertionError);
  });

  test("WASM backends support passing assertions and throw on failures", () => {
    expect(
      compileWasmTapeFromTape(passingAssertionTape())(new Map([["x", 0]])),
    ).toEqual([0, 42]);
    expect(
      compileWasmFromTape(passingAssertionTape())(new Map([["x", 0]])),
    ).toEqual([0, 42]);

    expect(() =>
      compileWasmTapeFromTape(failingAssertionTape("zero"))(
        new Map([["x", 7]]),
      ),
    ).toThrow(TapeAssertionError);
    expect(() =>
      compileWasmFromTape(failingAssertionTape("nonzero"))(new Map([["x", 0]])),
    ).toThrow(TapeAssertionError);
  });

  test("compileTapeWithGuardPrelude emits assertions before value roots", () => {
    const x = varNode("x");
    const y = varNode("y");
    const value = binaryNode("ADD", x, y);
    const tape = compileTape([value], {
      guardPrelude: [
        { node: x, kind: "nonzero", id: 123, source: "x must be nonzero" },
      ],
    })!;

    expect(tape.assertions).toEqual([
      {
        id: 123,
        tapeIndex: 1,
        kind: "nonzero",
        source: "x must be nonzero",
      },
    ]);
    expect(tape.opcodes[1]).toBe(OP_ASSERT_NONZERO);
    expect(tape.opcodes[tape.rootIndices[0]]).toBe(OP_ADD);
    expect(tape.assertions![0]!.tapeIndex).toBeLessThan(tape.rootIndices[0]);
    expect(tape.rootIndices).toHaveLength(1);
    expect(
      evalTape(
        tape,
        new Map([
          ["x", 2],
          ["y", 3],
        ]),
      ),
    ).toEqual([5]);
    expect(() =>
      evalTape(
        tape,
        new Map([
          ["x", 0],
          ["y", 3],
        ]),
      ),
    ).toThrow(TapeAssertionError);
  });

  test("JS forward autodiff propagates through passing assertions", () => {
    const tape: CompiledTape = {
      opcodes: new Uint8Array([OP_VAR, OP_ASSERT_NONZERO]),
      argA: new Int32Array([0, 0]),
      argB: new Int32Array([0, 0]),
      literals: new Float64Array([]),
      varSlots: ["x"],
      numVars: 1,
      rootIndices: [1],
      assertions: [{ id: 0, tapeIndex: 1, kind: "nonzero" }],
    };

    const grad = compileForwardAutodiff(tape, ["x"])(new Map([["x", 3]]));

    expect(grad.val).toBe(3);
    expect(grad.gradient).toEqual([1]);
  });
});
