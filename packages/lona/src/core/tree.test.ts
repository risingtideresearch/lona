import { describe, test, expect } from "vitest";
import { asNum, variableNum } from "./num";
import { replaceVariable } from "./tree-walks";
import {
  allVariables,
  KIND_LIT,
  KIND_VAR,
  KIND_SQRT,
  KIND_CBRT,
  KIND_COS,
  KIND_ACOS,
  KIND_ASIN,
  KIND_TAN,
  KIND_ATAN,
  KIND_LOG,
  KIND_EXP,
  KIND_ABS,
  KIND_NEG,
  KIND_SIN,
  KIND_SIGN,
  KIND_NOT,
  KIND_TANH,
  KIND_LOG1P,
  KIND_DEBUG,
  KIND_ADD,
  KIND_SUB,
  KIND_MUL,
  KIND_DIV,
  KIND_MOD,
  KIND_ATAN2,
  KIND_MIN,
  KIND_MAX,
  KIND_COMPARE,
  KIND_AND,
  KIND_OR,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_SELECT,
  isBinaryKind,
  isUnaryKind,
} from "./tree";
import {
  OP_LIT,
  OP_VAR,
  OP_SQRT,
  OP_CBRT,
  OP_COS,
  OP_ACOS,
  OP_ASIN,
  OP_TAN,
  OP_ATAN,
  OP_LOG,
  OP_EXP,
  OP_ABS,
  OP_NEG,
  OP_SIN,
  OP_SIGN,
  OP_NOT,
  OP_TANH,
  OP_LOG1P,
  OP_DEBUG,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_MOD,
  OP_ATAN2,
  OP_MIN,
  OP_MAX,
  OP_COMPARE,
  OP_AND,
  OP_OR,
} from "../eval/tape";
import {
  litNode,
  varNode,
  binaryNode,
  unaryNode,
  selectNode,
} from "./tree-cons";

describe("all variables", () => {
  test("finds no variables in a Num without vars", () => {
    const num = asNum(1).sub(asNum(2)).add(asNum(3));
    expect(allVariables(num.n)).toEqual(new Set([]));
  });

  test("finds a single variable", () => {
    const num = asNum(1)
      .sub(asNum(2))
      .add(asNum(3))
      .add(variableNum("x").mul(asNum(5)));
    expect(allVariables(num.n)).toEqual(new Set(["x"]));
  });
});

describe("Replace variable", () => {
  test("replaces a single variable", () => {
    const num = asNum(1)
      .sub(asNum(2))
      .add(asNum(3))
      .add(variableNum("x").mul(asNum(5)));

    const replaced = replaceVariable(num.n, new Map([["x", 10]]));
    expect(allVariables(replaced)).toEqual(new Set([]));
  });

  test("replaces keep unspecified variables", () => {
    const num = asNum(1)
      .sub(asNum(2))
      .add(asNum(3))
      .add(variableNum("x").mul(variableNum("y")));

    const replaced = replaceVariable(num.n, new Map([["x", 10]]));
    expect(allVariables(replaced)).toEqual(new Set(["y"]));
  });
});

describe("kind dispatch tags", () => {
  // If these ever drift apart from the OP_* constants in tape-opcodes.ts,
  // `compileTape` breaks in subtle ways because it pushes `node.kind`
  // directly as an opcode byte. Lock the invariant with a test so the
  // next reader can change them together.
  test("leaf kinds match the tape opcodes", () => {
    expect(KIND_LIT).toBe(OP_LIT);
    expect(KIND_VAR).toBe(OP_VAR);
  });

  test("unary kinds match the tape opcodes", () => {
    expect(KIND_SQRT).toBe(OP_SQRT);
    expect(KIND_CBRT).toBe(OP_CBRT);
    expect(KIND_COS).toBe(OP_COS);
    expect(KIND_ACOS).toBe(OP_ACOS);
    expect(KIND_ASIN).toBe(OP_ASIN);
    expect(KIND_TAN).toBe(OP_TAN);
    expect(KIND_ATAN).toBe(OP_ATAN);
    expect(KIND_LOG).toBe(OP_LOG);
    expect(KIND_EXP).toBe(OP_EXP);
    expect(KIND_ABS).toBe(OP_ABS);
    expect(KIND_NEG).toBe(OP_NEG);
    expect(KIND_SIN).toBe(OP_SIN);
    expect(KIND_SIGN).toBe(OP_SIGN);
    expect(KIND_NOT).toBe(OP_NOT);
    expect(KIND_TANH).toBe(OP_TANH);
    expect(KIND_LOG1P).toBe(OP_LOG1P);
    expect(KIND_DEBUG).toBe(OP_DEBUG);
  });

  test("binary kinds match the tape opcodes", () => {
    expect(KIND_ADD).toBe(OP_ADD);
    expect(KIND_SUB).toBe(OP_SUB);
    expect(KIND_MUL).toBe(OP_MUL);
    expect(KIND_DIV).toBe(OP_DIV);
    expect(KIND_MOD).toBe(OP_MOD);
    expect(KIND_ATAN2).toBe(OP_ATAN2);
    expect(KIND_MIN).toBe(OP_MIN);
    expect(KIND_MAX).toBe(OP_MAX);
    expect(KIND_COMPARE).toBe(OP_COMPARE);
    expect(KIND_AND).toBe(OP_AND);
    expect(KIND_OR).toBe(OP_OR);
  });

  test("non-tape kinds are outside the opcode range", () => {
    // Derivative and ForeignFn aren't valid opcodes — they live above
    // the tape range so range checks like `isUnaryKind` and
    // `isBinaryKind` correctly exclude them.
    expect(KIND_DERIVATIVE).toBeGreaterThanOrEqual(60);
    expect(KIND_FOREIGN).toBeGreaterThanOrEqual(60);
    expect(KIND_SELECT).toBeGreaterThanOrEqual(60);
    expect(isUnaryKind(KIND_DERIVATIVE)).toBe(false);
    expect(isBinaryKind(KIND_DERIVATIVE)).toBe(false);
    expect(isUnaryKind(KIND_FOREIGN)).toBe(false);
    expect(isBinaryKind(KIND_FOREIGN)).toBe(false);
    expect(isUnaryKind(KIND_SELECT)).toBe(false);
    expect(isBinaryKind(KIND_SELECT)).toBe(false);
  });

  test("constructed nodes expose the right kind", () => {
    expect(litNode(42).kind).toBe(KIND_LIT);
    expect(varNode("x").kind).toBe(KIND_VAR);
    expect(unaryNode("NEG", varNode("x")).kind).toBe(KIND_NEG);
    expect(unaryNode("ABS", varNode("x")).kind).toBe(KIND_ABS);
    expect(binaryNode("ADD", varNode("x"), varNode("y")).kind).toBe(KIND_ADD);
    expect(binaryNode("MUL", varNode("x"), varNode("y")).kind).toBe(KIND_MUL);
    expect(selectNode(varNode("c"), varNode("x"), varNode("y")).kind).toBe(
      KIND_SELECT,
    );
  });

  test("isUnaryKind and isBinaryKind partition the tape space", () => {
    // Every tape-reachable node is either LIT, VAR, unary, or binary.
    expect(isUnaryKind(KIND_LIT)).toBe(false);
    expect(isUnaryKind(KIND_VAR)).toBe(false);
    expect(isUnaryKind(KIND_ABS)).toBe(true);
    expect(isUnaryKind(KIND_DEBUG)).toBe(true);
    expect(isBinaryKind(KIND_ADD)).toBe(true);
    expect(isBinaryKind(KIND_OR)).toBe(true);
    expect(isBinaryKind(KIND_VAR)).toBe(false);
  });
});
