import { describe, expect, test } from "vitest";
import { Num } from "lona";
import {
  KIND_VAR,
  Variable,
  childrenOfNumNode,
  numNodeLabel,
} from "lona/internal";
import { binaryNode, varNode } from "lona/internal";
import { compileTape } from "lona/internal";

describe("columnar parameter variables", () => {
  test("uses unique labelled symbols as variable names", () => {
    const name = Symbol("column.map.param.0");
    const a = varNode(name);
    const sameA = varNode(name);
    const anotherA = varNode(Symbol("column.map.param.0"));

    expect(a).toBeInstanceOf(Variable);
    expect(a.kind).toBe(KIND_VAR);
    expect(a).toBe(sameA);
    expect(a).not.toBe(anotherA);
    expect(a.name).toBe(name);
    expect(a.name).not.toBe(anotherA.name);
    expect(childrenOfNumNode(a)).toEqual([]);
    expect(numNodeLabel(a)).toBe("VAR:Symbol(column.map.param.0)");
  });

  test("participates in commutative canonical ordering", () => {
    const a = varNode(Symbol("column.map.param.0"));
    const b = varNode(Symbol("column.map.param.1"));

    expect(binaryNode("ADD", a, b)).toBe(binaryNode("ADD", b, a));
    expect(new Num(a).add(new Num(b)).n).toBe(new Num(b).add(new Num(a)).n);
  });

  test("compiles through the ordinary tape path", () => {
    const param = varNode(Symbol("column.map.param.0"));
    const tape = compileTape([param]);

    expect(tape?.varSlots).toEqual([param.name]);
  });
});
