import { describe, test, expect } from "vitest";
import { binaryNode, litNode, varNode } from "./tree-cons";

/**
 * Canonical-operand-order tests for hash-cons `binaryNode`.
 *
 * Lives in its own file so the additional `test(...)` registrations
 * don't perturb the test-collection phase of `tree-cons.test.ts`
 * (Vitest discovers both, but they load independently).
 */
describe("binaryNode canonical operand ordering", () => {
  test("canonicalises commutative operand order", () => {
    // ADD / MUL / MIN / MAX are commutative: `op(a, b)` and `op(b, a)`
    // must dedupe to the same node so downstream hash-cons and
    // foldConstants rules don't need mirrored variants.
    const x = varNode("cx");
    const y = varNode("cy");
    const z = varNode("cz");
    expect(binaryNode("ADD", x, y)).toBe(binaryNode("ADD", y, x));
    expect(binaryNode("MUL", x, y)).toBe(binaryNode("MUL", y, x));
    expect(binaryNode("MIN", x, y)).toBe(binaryNode("MIN", y, x));
    expect(binaryNode("MAX", x, y)).toBe(binaryNode("MAX", y, x));

    // Nested: ADD(z, ADD(x, y)) === ADD(ADD(y, x), z)
    const inner1 = binaryNode("ADD", x, y);
    const inner2 = binaryNode("ADD", y, x);
    expect(inner1).toBe(inner2);
    expect(binaryNode("ADD", z, inner1)).toBe(binaryNode("ADD", inner2, z));
  });

  test("leaves non-commutative ops untouched", () => {
    const x = varNode("ncx");
    const y = varNode("ncy");
    expect(binaryNode("SUB", x, y)).not.toBe(binaryNode("SUB", y, x));
    expect(binaryNode("DIV", x, y)).not.toBe(binaryNode("DIV", y, x));
    expect(binaryNode("MOD", x, y)).not.toBe(binaryNode("MOD", y, x));
    expect(binaryNode("ATAN2", x, y)).not.toBe(binaryNode("ATAN2", y, x));
    expect(binaryNode("COMPARE", x, y)).not.toBe(binaryNode("COMPARE", y, x));
    // AND/OR have non-commutative numeric semantics in this codebase.
    expect(binaryNode("AND", x, y)).not.toBe(binaryNode("AND", y, x));
    expect(binaryNode("OR", x, y)).not.toBe(binaryNode("OR", y, x));
  });

  test("biases literals to the left of commutative ops", () => {
    // foldConstants pattern-matches on `left.kind === KIND_LIT`, so the
    // canonical form must always put a literal on the left when one side
    // is a literal.
    const x = varNode("litx");
    const three = litNode(3);
    const left = binaryNode("ADD", x, three);
    const right = binaryNode("ADD", three, x);
    expect(left).toBe(right);
    expect(left.left).toBe(three);
    expect(left.right).toBe(x);
  });

  test("handles NaN literal operands", () => {
    const x = varNode("nanx");
    const nan = litNode(NaN);
    // NaN is still a literal so it sorts before non-literals.
    const n1 = binaryNode("ADD", x, nan);
    const n2 = binaryNode("ADD", nan, x);
    expect(n1).toBe(n2);
    expect(n1.left).toBe(nan);
  });
});
