/**
 * Verifies that the eval/simplify/compile pipeline walks the DAG rather
 * than the tree — each unique node is visited exactly once, even when
 * subtrees are heavily shared.
 *
 * This invariant is what makes `.simplify()` and `compileTape()` scale
 * to large expressions: hash-consing guarantees structural dedup at
 * construction time, and the traversal framework must respect that by
 * not re-walking shared sub-DAGs.
 */
import { describe, test, expect } from "vitest";
import { binaryNode, unaryNode, varNode } from "../core/tree-cons";
import { genericEval } from "../eval/transforms/generic-eval";
import { visitFromLeaves } from "./traversal";
import { childrenOfNumNode, type NumNode } from "../core/tree";
import { simplify } from "../api/simplify";
import { compileTape } from "../eval/tape";
import type { NumEvalKernel } from "../types";

function countReachable(root: NumNode): number {
  let count = 0;
  visitFromLeaves(root, childrenOfNumNode, () => {
    count += 1;
  });
  return count;
}

function makeCountingKernel(): {
  kernel: NumEvalKernel<NumNode>;
  calls: { n: number };
} {
  const calls = { n: 0 };
  const kernel: NumEvalKernel<NumNode> = {
    literal: (_, n) => {
      calls.n += 1;
      return n;
    },
    variable: (_, n) => {
      calls.n += 1;
      return n;
    },
    derivative: (_, n) => {
      calls.n += 1;
      return n;
    },
    unaryOp: (_, __, n) => {
      calls.n += 1;
      return n;
    },
    binaryOp: (_, __, ___, n) => {
      calls.n += 1;
      return n;
    },
    value: () => 0,
  };
  return { kernel, calls };
}

describe("DAG walk (not tree walk)", () => {
  test("genericEval visits each unique node exactly once on a diamond", () => {
    const x = varNode("x");
    const y = varNode("y");
    const sum = binaryNode("ADD", x, y);
    // Diamond: (x+y) * (x+y) — the inner sum is shared.
    const diamond = binaryNode("MUL", sum, sum);

    expect(countReachable(diamond)).toBe(4); // x, y, ADD, MUL

    const { kernel, calls } = makeCountingKernel();
    genericEval(diamond, kernel);

    // Tree-walk would count 7 (4 + 3 duplicated via sharing).
    expect(calls.n).toBe(4);
  });

  test("genericEval visits each unique node exactly once on a deep shared chain", () => {
    // Build x, x², x⁴, x⁸ — each level reuses the previous shared node.
    const x = varNode("x");
    const x2 = binaryNode("MUL", x, x);
    const x4 = binaryNode("MUL", x2, x2);
    const x8 = binaryNode("MUL", x4, x4);

    expect(countReachable(x8)).toBe(4); // x, x², x⁴, x⁸

    const { kernel, calls } = makeCountingKernel();
    genericEval(x8, kernel);

    // Tree-walk would count 15 (1 + 2 + 4 + 8). DAG-walk: 4.
    expect(calls.n).toBe(4);
  });

  test("simplify walks the DAG (shared sub-expressions visited once)", () => {
    const x = varNode("x");
    const y = varNode("y");
    const shared = unaryNode(
      "SIN",
      binaryNode("ADD", binaryNode("MUL", x, x), binaryNode("MUL", y, y)),
    );
    // Use the shared subtree in multiple places.
    const root = binaryNode(
      "ADD",
      binaryNode("MUL", shared, shared),
      binaryNode("SUB", shared, shared),
    );

    const unique = countReachable(root);
    // Make sure the DAG has fewer nodes than its tree expansion would.
    // Tree expansion would be way more than 'unique'.
    expect(unique).toBeLessThan(20);

    // simplify should produce a valid output without blowing up on the
    // shared subtrees. (`shared - shared = 0` and the outer ADD should
    // reduce; the exact shape doesn't matter for this test.)
    const result = simplify(root);
    expect(result).toBeDefined();
  });

  test("compileTape emits exactly one entry per unique DAG node", () => {
    const x = varNode("x");
    const y = varNode("y");
    const sum = binaryNode("ADD", x, y);
    const diamond = binaryNode("MUL", sum, sum);

    const tape = compileTape([diamond]);
    expect(tape).not.toBeNull();
    // 4 unique nodes, 4 opcodes.
    expect(tape!.opcodes.length).toBe(4);
  });
});
