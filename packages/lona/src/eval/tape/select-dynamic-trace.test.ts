import { describe, expect, test } from "vitest";
import fs from "node:fs";
import path from "node:path";
import {
  BinaryOp,
  LiteralNum,
  SelectOp,
  Variable,
  type NumNode,
  type VarName,
} from "../../core/tree";
import {
  deserializeNumDAG,
  type SerializedNumDAG,
} from "../../core/tree-serialization";
import { compileTape } from "./compile";
import { OP_VAR } from "./opcodes";
import { TapeAssertionError } from "./assertions";
import { compileDynamicSelectTraceTape } from "./select-dynamic-trace";
import { evalTape } from "../routines/backends/js-interp/tape-eval";

function lit(value: number): LiteralNum {
  return new LiteralNum(value);
}

function v(name: VarName): Variable {
  return new Variable(name);
}

function add(left: NumNode, right: NumNode): BinaryOp {
  return new BinaryOp("ADD", left, right);
}

function fixturePath(filename: string): string {
  return path.resolve(import.meta.dirname, "../../../bench/fixtures", filename);
}

describe("dynamic select trace tape", () => {
  test("emits guards and only the selected branch", () => {
    const x = v("x");
    const y = v("y");
    const root = new SelectOp(x, add(x, lit(1)), add(y, lit(1000)));
    const vars = new Map<VarName, number>([
      ["x", 2],
      ["y", 10],
    ]);

    const dynamic = compileDynamicSelectTraceTape([root], vars)!;
    const full = compileTape([root])!;

    expect(evalTape(dynamic.tape, vars)[0]).toBe(evalTape(full, vars)[0]);
    expect(dynamic.selectDecisions).toBe(1);
    expect(dynamic.guardAssertions).toBe(1);
    expect(dynamic.tape.opcodes.length).toBeLessThan(full.opcodes.length);

    const ySlot = dynamic.tape.varSlots.indexOf("y");
    const emittedYVars = Array.from(dynamic.tape.opcodes).filter(
      (op, i) => op === OP_VAR && dynamic.tape.argA[i] === ySlot,
    );
    expect(emittedYVars.length).toBe(0);
  });

  test("guarded tape throws when the traced select decision no longer holds", () => {
    const x = v("x");
    const root = new SelectOp(x, add(x, lit(1)), lit(0));
    const dynamic = compileDynamicSelectTraceTape(
      [root],
      new Map<VarName, number>([["x", 2]]),
    )!;

    expect(() =>
      evalTape(dynamic.tape, new Map<VarName, number>([["x", 0]])),
    ).toThrow(TapeAssertionError);
  });

  test("gridfinity fixture matches full eval at representative points", () => {
    const raw = fs.readFileSync(
      fixturePath("gridfinity-3d-select-numtree.json"),
      "utf-8",
    );
    const dag = JSON.parse(raw).select.dag as SerializedNumDAG;
    const root = deserializeNumDAG(dag);
    const full = compileTape([root])!;

    for (const [x, y, z] of [
      [0, 0, 10],
      [-45, -25, -5],
      [45, 25, 25],
      [0, 0, -5],
    ] as const) {
      const vars = new Map<VarName, number>([
        ["x", x],
        ["y", y],
        ["z", z],
      ]);
      const dynamic = compileDynamicSelectTraceTape([root], vars)!;
      expect(evalTape(dynamic.tape, vars)[0]).toBe(evalTape(full, vars)[0]);
      expect(dynamic.tape.opcodes.length).toBeLessThan(full.opcodes.length);
      expect(dynamic.guardAssertions).toBeGreaterThan(0);
    }
  });
});
