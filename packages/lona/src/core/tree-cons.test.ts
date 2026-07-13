import { describe, test, expect } from "vitest";
import {
  NumTreeContext,
  binaryNode,
  debugNode,
  derivativeNode,
  foreignFnNode,
  litNode,
  unaryNode,
  varNode,
} from "./tree-cons";
import { asNum, variableNum } from "./num";
import { NumValueContext, getContext, withContext } from "../api/value-context";
import {
  ZERO_NODE,
  ONE_NODE,
  TWO_NODE,
  NEG_ONE_NODE,
  LiteralNum,
} from "./tree";
import { serializeNumDAG, deserializeNumDAG } from "./tree-serialization";

describe("hash-cons factories", () => {
  test("litNode returns the same object for equal values", () => {
    expect(litNode(3.14)).toBe(litNode(3.14));
    expect(litNode(42)).toBe(litNode(42));
  });

  test("litNode returns the canonical sentinels", () => {
    expect(litNode(0)).toBe(ZERO_NODE);
    expect(litNode(1)).toBe(ONE_NODE);
    expect(litNode(2)).toBe(TWO_NODE);
    expect(litNode(-1)).toBe(NEG_ONE_NODE);
  });

  test("litNode collapses -0 onto the canonical ZERO_NODE", () => {
    // Hash-consing treats `-0` and `+0` as the same mathematical constant
    // and returns the same sentinel. `DIV_BY_ZERO_FALLBACK` in num.ts
    // already erases the only place the sign-of-zero distinction would
    // leak back into evaluation (division by zero), so there is nothing
    // to preserve.
    expect(litNode(-0)).toBe(ZERO_NODE);
    expect(litNode(0)).toBe(ZERO_NODE);
  });

  test("litNode handles NaN", () => {
    const nan1 = litNode(NaN);
    const nan2 = litNode(NaN);
    expect(nan1).toBe(nan2);
    expect(Number.isNaN(nan1.value)).toBe(true);
  });

  test("varNode returns the same object for equal names", () => {
    expect(varNode("x")).toBe(varNode("x"));
    expect(varNode("x")).not.toBe(varNode("y"));
  });

  test("varNode distinguishes symbol names", () => {
    const s1 = Symbol("v");
    const s2 = Symbol("v"); // different symbol, same description
    expect(varNode(s1)).toBe(varNode(s1));
    expect(varNode(s1)).not.toBe(varNode(s2));
  });

  test("unaryNode dedups by operation and operand", () => {
    const x = varNode("x");
    expect(unaryNode("NEG", x)).toBe(unaryNode("NEG", x));
    expect(unaryNode("NEG", x)).not.toBe(unaryNode("SIN", x));
  });

  test("binaryNode dedups by operation and both operands", () => {
    const x = varNode("x");
    const y = varNode("y");
    expect(binaryNode("ADD", x, y)).toBe(binaryNode("ADD", x, y));
    expect(binaryNode("ADD", x, y)).not.toBe(binaryNode("SUB", x, y));
  });

  test("debugNode dedups by original and debug string", () => {
    const x = varNode("x");
    expect(debugNode(x, "tag")).toBe(debugNode(x, "tag"));
    expect(debugNode(x, "tag")).not.toBe(debugNode(x, "other"));
  });

  test("derivativeNode dedups by variable", () => {
    const x = varNode("x");
    const y = varNode("y");
    expect(derivativeNode(x)).toBe(derivativeNode(x));
    expect(derivativeNode(x)).not.toBe(derivativeNode(y));
  });

  test("foreignFnNode never dedups (closures are opaque)", () => {
    const lit = litNode(1);
    const f = (v: readonly number[]) => v[0]!;
    const d = () => lit;
    const a = foreignFnNode([lit], f, d);
    const b = foreignFnNode([lit], f, d);
    expect(a).not.toBe(b);
  });
});

// These tests depend on num.ts and should arguably be moved to num.test.ts, but
// are kept here as they specifically test hash-consing of the underlying nodes.

describe("Num factory integration", () => {
  test("x + y === x + y (structural sharing through Num)", () => {
    const x = variableNum("x");
    const y = variableNum("y");
    expect(x.add(y).n).toBe(x.add(y).n);
  });

  test("x * x === x * x", () => {
    const x = variableNum("x");
    expect(x.mul(x).n).toBe(x.mul(x).n);
  });

  test("sqrt(x² + y²) is shared across calls", () => {
    const x1 = variableNum("x");
    const y1 = variableNum("y");
    const e1 = x1.mul(x1).add(y1.mul(y1)).sqrt();

    const x2 = variableNum("x");
    const y2 = variableNum("y");
    const e2 = x2.mul(x2).add(y2.mul(y2)).sqrt();

    expect(e1.n).toBe(e2.n);
  });

  test("a subterm reused twice produces identical sub-DAG edges", () => {
    // (x+y) appears twice; hash-consing must give one node for it.
    const x = variableNum("x");
    const y = variableNum("y");
    const s1 = x.add(y);
    const s2 = x.add(y);
    const combined = s1.mul(s2);
    // Walking the binary node: both children should be identical.
    const bin = combined.n as unknown as { left: unknown; right: unknown };
    expect(bin.left).toBe(bin.right);
  });

  test("literal operations hit the cache (3 === 3)", () => {
    const a = asNum(3);
    const b = asNum(3);
    expect(a.n).toBe(b.n);
  });

  test("deserialize produces cons'd trees that compose with new builds", () => {
    const x = variableNum("x");
    const expr = x.add(1);
    const round = deserializeNumDAG(serializeNumDAG(expr.n));
    // Rebuilding via the Num factory after deserialize should hit the cache.
    expect(x.add(1).n).toBe(round);
  });

  test("raw `new LiteralNum` does not collide with cons'd identity", () => {
    // Nodes built outside the factory are intentionally *not* in the cons
    // table. This documents the invariant: production code must go through
    // factories.
    const rawLit = new LiteralNum(999);
    const consLit = litNode(999);
    expect(rawLit).not.toBe(consLit);
    // But cons'd reads of the same value still dedup with each other.
    expect(litNode(999)).toBe(consLit);
  });
});

describe("NumTreeContext isolation", () => {
  test("two fresh contexts produce independent nodes", () => {
    const ctxA = new NumTreeContext();
    const ctxB = new NumTreeContext();
    // Same variable name, different contexts → different objects.
    expect(ctxA.varNode("p")).not.toBe(ctxB.varNode("p"));
    // Inside a single context, identity still holds.
    expect(ctxA.varNode("p")).toBe(ctxA.varNode("p"));
  });

  test("contexts share the literal sentinels but not other literals", () => {
    const ctxA = new NumTreeContext();
    const ctxB = new NumTreeContext();
    // Sentinel literals (0, 1, 2, -1) are shared by design — existing
    // reference checks like `node === ZERO_NODE` must keep working.
    expect(ctxA.litNode(0)).toBe(ctxB.litNode(0));
    expect(ctxA.litNode(1)).toBe(ctxB.litNode(1));
    // Non-sentinel literals are context-private.
    expect(ctxA.litNode(7)).not.toBe(ctxB.litNode(7));
  });

  test("binaryNode on independent contexts does not cross-cache", () => {
    const ctxA = new NumTreeContext();
    const ctxB = new NumTreeContext();
    const xA = ctxA.varNode("x");
    const yA = ctxA.varNode("y");
    const xB = ctxB.varNode("x");
    const yB = ctxB.varNode("y");
    // Different contexts, different operand identities, different results.
    expect(ctxA.binaryNode("ADD", xA, yA)).not.toBe(
      ctxB.binaryNode("ADD", xB, yB),
    );
  });

  // These tests depend on value-context and should be moved to
  // value-context.test.ts. Keeping them here for now for easier to inspect git
  // diff.

  test("withContext restores and has no effect on NumNode's identity", () => {
    const outerCtx = new NumValueContext();
    const innerCtx = new NumValueContext();
    withContext(outerCtx, () => {
      // Create a NumNode while innerCtx is current.
      const inside = withContext(innerCtx, () => varNode("q"));

      // After withContext(innerCtx), outerCtx is restored.
      expect(getContext()).toBe(outerCtx);

      // The NumNode is the same regardless of which NumValueContext was active,
      // since nodes live in a shared NumTreeContext.
      const outside = varNode("q");
      expect(outside).toBe(inside);
    });
  });

  test("withContext restores previous context on exception", () => {
    const outerCtx = new NumValueContext();
    const innerCtx = new NumValueContext();
    withContext(outerCtx, () => {
      expect(() =>
        withContext(innerCtx, () => {
          throw new Error("boom");
        }),
      ).toThrow("boom");
      expect(getContext()).toBe(outerCtx);
    });
  });

  test("nested withContext restores in LIFO order", () => {
    const initialCtx = new NumValueContext();
    const ctx1 = new NumValueContext();
    const ctx2 = new NumValueContext();
    withContext(initialCtx, () => {
      expect(getContext()).toBe(initialCtx);
      withContext(ctx1, () => {
        expect(getContext()).toBe(ctx1);
        withContext(ctx2, () => {
          expect(getContext()).toBe(ctx2);
        });
        expect(getContext()).toBe(ctx1);
      });
      expect(getContext()).toBe(initialCtx);
    });
  });
});
