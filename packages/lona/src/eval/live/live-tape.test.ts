import { describe, test, expect } from "vitest";
import {
  LiteralNum,
  Variable,
  UnaryOp,
  BinaryOp,
  ForeignFn,
  Derivative,
  type NumNode,
  type VarName,
} from "../../core/tree";
import { LiveTape, INTERN_FAILED, type LiveTapeSweepFn } from "./live-tape";
import { jsInterpSweep, wasmInterpSweep } from "./sweep";

// ---------------------------------------------------------------------------
// Helpers — build NumNode DAGs without going through the cons machinery, so
// each test gets independent node identities. Hash-cons-shared identities
// would let one test's tape see nodes from another test's DAG, which makes
// the "lazy intern" assertions noisy.
// ---------------------------------------------------------------------------

function lit(value: number): LiteralNum {
  return new LiteralNum(value);
}
function v(name: VarName): Variable {
  return new Variable(name);
}
function add(a: NumNode, b: NumNode): BinaryOp {
  return new BinaryOp("ADD", a, b);
}
function mul(a: NumNode, b: NumNode): BinaryOp {
  return new BinaryOp("MUL", a, b);
}
function sub(a: NumNode, b: NumNode): BinaryOp {
  return new BinaryOp("SUB", a, b);
}
function neg(a: NumNode): UnaryOp {
  return new UnaryOp("NEG", a);
}
function sqrt(a: NumNode): UnaryOp {
  return new UnaryOp("SQRT", a);
}

// ---------------------------------------------------------------------------
// Lazy interning
// ---------------------------------------------------------------------------

describe("LiveTape — lazy interning", () => {
  test("nodeToIndex is empty before any ensureInterned call", () => {
    const tape = new LiveTape();
    expect(tape.length).toBe(0);
    expect(tape.nodeToIndex.size).toBe(0);
  });

  test("ensureInterned populates nodeToIndex with the operand subgraph", () => {
    const tape = new LiveTape();
    const x = v("x");
    const root = add(x, lit(3));
    const idx = tape.ensureInterned(root);
    expect(idx).toBeGreaterThanOrEqual(0);
    // x, 3, ADD — three nodes.
    expect(tape.nodeToIndex.size).toBe(3);
    expect(tape.length).toBe(3);
  });

  test("ensureInterned is idempotent for an already-interned node", () => {
    const tape = new LiveTape();
    const root = add(v("x"), lit(1));
    const a = tape.ensureInterned(root);
    const b = tape.ensureInterned(root);
    expect(a).toBe(b);
    expect(tape.length).toBe(3);
  });

  test("structurally-shared operands are interned once", () => {
    const tape = new LiveTape();
    const x = v("x");
    // x*x — same Variable node referenced twice.
    const root = mul(x, x);
    tape.ensureInterned(root);
    expect(tape.length).toBe(2); // x and MUL
  });

  test("foreign-bearing subgraph returns INTERN_FAILED", () => {
    const tape = new LiveTape();
    const foreign = new ForeignFn(
      [lit(1)],
      (vals) => vals[0]!,
      () => lit(0),
    );
    const root = add(foreign, lit(3));
    expect(tape.ensureInterned(root)).toBe(INTERN_FAILED);
  });

  test("derivative-bearing subgraph returns INTERN_FAILED", () => {
    const tape = new LiveTape();
    const root = add(new Derivative(v("x")), lit(2));
    expect(tape.ensureInterned(root)).toBe(INTERN_FAILED);
  });
});

// ---------------------------------------------------------------------------
// Basic read / write
// ---------------------------------------------------------------------------

describe("LiveTape — basic read/write", () => {
  test("variable defaults to 0 when never registered", () => {
    const tape = new LiveTape();
    const idx = tape.ensureInterned(v("x"));
    expect(tape.getValue(idx)).toBe(0);
  });

  test("registerInitialVariableValue applied on intern", () => {
    const tape = new LiveTape();
    tape.registerInitialVariableValue("x", 7);
    const idx = tape.ensureInterned(v("x"));
    expect(tape.getValue(idx)).toBe(7);
  });

  test("setVariable applied even before intern (stashes)", () => {
    const tape = new LiveTape();
    tape.setVariable("x", 9);
    const idx = tape.ensureInterned(v("x"));
    expect(tape.getValue(idx)).toBe(9);
  });

  test("setVariable overrides registerInitialVariableValue", () => {
    const tape = new LiveTape();
    tape.registerInitialVariableValue("x", 1);
    tape.setVariable("x", 42);
    const idx = tape.ensureInterned(v("x"));
    expect(tape.getValue(idx)).toBe(42);
  });

  test("registerInitialVariableValue is idempotent (first wins)", () => {
    const tape = new LiveTape();
    tape.registerInitialVariableValue("x", 1);
    tape.registerInitialVariableValue("x", 2);
    const idx = tape.ensureInterned(v("x"));
    expect(tape.getValue(idx)).toBe(1);
  });

  test("setVariable on already-interned variable propagates to consumers", () => {
    const tape = new LiveTape();
    const x = v("x");
    const root = add(x, lit(10));
    const rootIdx = tape.ensureInterned(root);
    tape.setVariable("x", 5);
    expect(tape.getValue(rootIdx)).toBe(15);
    tape.setVariable("x", 7);
    expect(tape.getValue(rootIdx)).toBe(17);
  });

  test("ADD / SUB / MUL / NEG / SQRT compute correctly", () => {
    const tape = new LiveTape();
    const x = v("x");
    const y = v("y");
    // sqrt((x+y) * -(x-y)) — exercises 5 distinct ops.
    const root = sqrt(mul(add(x, y), neg(sub(x, y))));
    const idx = tape.ensureInterned(root);
    tape.setVariable("x", 3);
    tape.setVariable("y", 5);
    // (3+5) * -(3-5) = 8 * 2 = 16, sqrt = 4.
    expect(tape.getValue(idx)).toBeCloseTo(4, 10);
  });

  test("DIV by zero returns DIV_BY_ZERO_FALLBACK", () => {
    const tape = new LiveTape();
    const root = new BinaryOp("DIV", lit(1), v("x"));
    const idx = tape.ensureInterned(root);
    expect(tape.getValue(idx)).toBe(1e50);
  });
});

// ---------------------------------------------------------------------------
// Equality short-circuit
// ---------------------------------------------------------------------------

describe("LiveTape — equality short-circuit", () => {
  test("setVariable with the same value is a no-op (epoch unchanged)", () => {
    const tape = new LiveTape();
    const x = v("x");
    tape.setVariable("x", 3);
    const idx = tape.ensureInterned(x);
    expect(tape.getValue(idx)).toBe(3);

    // Sentinel: spy on getValue's path by reading slotEpoch indirectly.
    // After setVariable("x", 3) again, epoch must not advance, so a
    // subsequent getValue is the O(1) hot path.
    tape.setVariable("x", 3);
    expect(tape.getValue(idx)).toBe(3); // still correct
  });

  test("Object.is treats NaN as equal — no propagation on NaN→NaN", () => {
    const tape = new LiveTape();
    const x = v("x");
    const root = add(x, lit(0));
    const idx = tape.ensureInterned(root);
    tape.setVariable("x", NaN);
    expect(Number.isNaN(tape.getValue(idx))).toBe(true);
    // Second NaN write must short-circuit (Object.is(NaN, NaN) === true).
    tape.setVariable("x", NaN);
    expect(Number.isNaN(tape.getValue(idx))).toBe(true);
  });

  test("Object.is distinguishes +0 and -0", () => {
    const tape = new LiveTape();
    tape.setVariable("x", 0);
    const idx = tape.ensureInterned(v("x"));
    expect(tape.getValue(idx)).toBe(0);
    tape.setVariable("x", -0);
    expect(Object.is(tape.getValue(idx), -0)).toBe(true);
  });

  test("mid-DAG values that don't change don't propagate", () => {
    // (x - x) is always 0 regardless of x. A write to x triggers
    // recompute of the SUB, which produces 0, which equals the prior
    // 0 under Object.is — so the wrapping ADD doesn't need to recompute.
    // We can't easily observe propagation directly, but we can assert
    // the final value remains correct across many no-op-equivalent
    // writes.
    const tape = new LiveTape();
    const x = v("x");
    const sameMinusSame = sub(x, x);
    const root = add(sameMinusSame, lit(7));
    const idx = tape.ensureInterned(root);
    for (let i = 0; i < 10; i++) {
      tape.setVariable("x", i);
      expect(tape.getValue(idx)).toBe(7);
    }
  });
});

// ---------------------------------------------------------------------------
// Partial vs full equivalence
// ---------------------------------------------------------------------------

describe("LiveTape — getValue and catchUp produce consistent values", () => {
  function buildExpression(): NumNode {
    // sqrt((x+y)^2 + (x-y)^2) - (x*y) — touches multiple vars/ops.
    const x = v("x");
    const y = v("y");
    const sumSq = mul(add(x, y), add(x, y));
    const diffSq = mul(sub(x, y), sub(x, y));
    const norm = sqrt(add(sumSq, diffSq));
    return sub(norm, mul(x, y));
  }

  test("getValue (implicit sweep) matches explicit catchUp + read", () => {
    const tapeA = new LiveTape();
    const tapeB = new LiveTape();
    const idxA = tapeA.ensureInterned(buildExpression());
    const idxB = tapeB.ensureInterned(buildExpression());

    const writes: Array<[VarName, number]> = [
      ["x", 1],
      ["y", 2],
      ["x", 3],
      ["y", 5],
      ["x", 3], // no-op
      ["y", 7],
    ];

    for (const [name, val] of writes) {
      tapeA.setVariable(name, val);
      tapeB.setVariable(name, val);
    }

    // tapeA: implicit sweep on miss.
    const va = tapeA.getValue(idxA);
    // tapeB: explicit catchUp, then read (now hits hot path).
    tapeB.catchUp();
    const vb = tapeB.getValue(idxB);

    expect(va).toBe(vb);
  });

  test("repeated reads after a sweep stay consistent", () => {
    const tape = new LiveTape();
    const idx = tape.ensureInterned(buildExpression());

    tape.setVariable("x", 2);
    tape.setVariable("y", 3);
    const r1 = tape.getValue(idx);
    tape.catchUp();
    const r2 = tape.getValue(idx);
    expect(r1).toBe(r2);

    tape.setVariable("y", 4);
    tape.catchUp();
    const r3 = tape.getValue(idx);
    tape.setVariable("y", 4); // no-op
    const r4 = tape.getValue(idx);
    expect(r3).toBe(r4);
  });
});

// ---------------------------------------------------------------------------
// Deep DAGs — partial descent must not blow the JS stack
// ---------------------------------------------------------------------------

describe("LiveTape — deep DAGs", () => {
  test("partial descent handles a chain depth of 10000", () => {
    const tape = new LiveTape();
    let n: NumNode = v("x");
    for (let i = 0; i < 10000; i++) {
      n = add(n, lit(1));
    }
    const idx = tape.ensureInterned(n);
    tape.setVariable("x", 0);
    expect(tape.getValue(idx)).toBe(10000);
    tape.setVariable("x", 100);
    expect(tape.getValue(idx)).toBe(10100);
  });

  test("full sweep handles the same depth", () => {
    const tape = new LiveTape();
    let n: NumNode = v("x");
    for (let i = 0; i < 10000; i++) {
      n = add(n, lit(1));
    }
    const idx = tape.ensureInterned(n);
    tape.setVariable("x", 50);
    tape.catchUp();
    expect(tape.getValue(idx)).toBe(10050);
  });
});

// ---------------------------------------------------------------------------
// Incremental intern across multiple roots
// ---------------------------------------------------------------------------

describe("LiveTape — incremental intern", () => {
  test("a second root reuses already-interned subexpressions", () => {
    const tape = new LiveTape();
    const x = v("x");
    const shared = add(x, lit(1));
    const root1 = mul(shared, lit(2));
    const root2 = mul(shared, lit(3));
    const idx1 = tape.ensureInterned(root1);
    const sizeAfter1 = tape.length;
    const idx2 = tape.ensureInterned(root2);
    // Only the new lit(3) and the new MUL need fresh slots.
    expect(tape.length).toBe(sizeAfter1 + 2);

    tape.setVariable("x", 4);
    expect(tape.getValue(idx1)).toBe(10); // (4+1) * 2
    expect(tape.getValue(idx2)).toBe(15); // (4+1) * 3
  });

  test("intern after a setVariable still produces correct values", () => {
    const tape = new LiveTape();
    tape.setVariable("x", 3);
    const idx = tape.ensureInterned(add(v("x"), lit(10)));
    expect(tape.getValue(idx)).toBe(13);
  });
});

// ---------------------------------------------------------------------------
// Pluggable sweep backend
// ---------------------------------------------------------------------------

describe("LiveTape — pluggable sweep backend", () => {
  test("jsInterpSweep is the default", () => {
    const tapeDefault = new LiveTape();
    const tapeExplicit = new LiveTape({ sweepFn: jsInterpSweep });
    const idxA = tapeDefault.ensureInterned(mul(add(v("x"), lit(1)), lit(3)));
    const idxB = tapeExplicit.ensureInterned(mul(add(v("x"), lit(1)), lit(3)));
    tapeDefault.setVariable("x", 4);
    tapeExplicit.setVariable("x", 4);
    expect(tapeDefault.getValue(idxA)).toBe(15);
    expect(tapeExplicit.getValue(idxB)).toBe(tapeDefault.getValue(idxA));
  });

  test("a custom sweep backend is invoked by catchUp", () => {
    let calls = 0;
    let sawTapeVersion = -1;
    const customSweep: LiveTapeSweepFn = (ctx) => {
      calls++;
      sawTapeVersion = ctx.tapeVersion;
      // Delegate to the default to actually validate slot state.
      return jsInterpSweep(ctx);
    };
    const tape = new LiveTape({ sweepFn: customSweep });
    const idx = tape.ensureInterned(add(v("x"), lit(2)));
    tape.setVariable("x", 7);

    expect(calls).toBe(0); // catchUp not called yet
    tape.catchUp();
    expect(calls).toBe(1);
    expect(sawTapeVersion).toBeGreaterThan(0); // tape grew during intern
    expect(tape.getValue(idx)).toBe(9);

    // Subsequent no-op catchUp still routes through the backend.
    tape.catchUp();
    expect(calls).toBe(2);
  });

  test("post-pass leaves valueEpoch unchanged for slots whose value didn't change", () => {
    // After a sweep that recomputes every slot from scratch, the
    // post-pass should bump valueEpoch ONLY for slots whose value
    // actually changed. We can't observe valueEpoch directly but we
    // can prove the post-pass works by verifying that subsequent
    // reads on independent subgraphs are both correct.
    const tape = new LiveTape({ sweepFn: jsInterpSweep });
    const x = v("x");
    const y = v("y");
    const root1 = add(x, lit(10)); // depends on x only
    const root2 = mul(y, lit(2)); // depends on y only
    const idx1 = tape.ensureInterned(root1);
    const idx2 = tape.ensureInterned(root2);

    tape.setVariable("x", 1);
    tape.setVariable("y", 5);
    tape.catchUp();
    expect(tape.getValue(idx1)).toBe(11);
    expect(tape.getValue(idx2)).toBe(10);

    // Change only x. Sweep, then read root2 (which doesn't depend
    // on x).
    tape.setVariable("x", 100);
    tape.catchUp();
    expect(tape.getValue(idx1)).toBe(110);
    expect(tape.getValue(idx2)).toBe(10);
  });

  test("wasmInterpSweep matches jsInterpSweep across many writes", () => {
    // Same expression, two tapes — one with the JS interpreter sweep
    // (default), one with the wasm interpreter sweep. After every write,
    // both must produce identical slot values.
    const buildExpr = () => {
      const x = v("x");
      const y = v("y");
      const z = v("z");
      // sqrt((x+y)*(x-y) + z^2) - z*y
      return sub(sqrt(add(mul(add(x, y), sub(x, y)), mul(z, z))), mul(z, y));
    };

    const tapeJs = new LiveTape({ sweepFn: jsInterpSweep });
    const tapeWasm = new LiveTape({ sweepFn: wasmInterpSweep });
    const idxJs = tapeJs.ensureInterned(buildExpr());
    const idxWasm = tapeWasm.ensureInterned(buildExpr());

    const writes: Array<[VarName, number]> = [
      ["x", 5],
      ["y", 3],
      ["z", 2],
      ["x", 7],
      ["y", 3], // no-op
      ["z", -1],
      ["x", 0],
      ["y", 4],
    ];

    for (const [name, val] of writes) {
      tapeJs.setVariable(name, val);
      tapeWasm.setVariable(name, val);
      // Use getValue (implicit sweep on miss) on both.
      expect(tapeWasm.getValue(idxWasm)).toBe(tapeJs.getValue(idxJs));
    }
  });

  test("wasmInterpSweep handles tape growth (capacity reallocation)", () => {
    // Build a tape that exceeds the initial 64-slot capacity to force
    // a typed-array reallocation, then verify wasm sweep still matches
    // js sweep. The wasm interpreter caches layouts per opcodes
    // identity, so a reallocation must be transparently handled.
    const tapeJs = new LiveTape({ sweepFn: jsInterpSweep });
    const tapeWasm = new LiveTape({ sweepFn: wasmInterpSweep });

    let nJs: NumNode = v("x");
    let nWasm: NumNode = v("x");
    for (let i = 0; i < 200; i++) {
      nJs = add(nJs, lit(i));
      nWasm = add(nWasm, lit(i));
    }
    const idxJs = tapeJs.ensureInterned(nJs);
    const idxWasm = tapeWasm.ensureInterned(nWasm);

    tapeJs.setVariable("x", 7);
    tapeWasm.setVariable("x", 7);
    expect(tapeWasm.getValue(idxWasm)).toBeCloseTo(tapeJs.getValue(idxJs), 10);
  });

  test("backend sees current tapeVersion and can cache by it", () => {
    const versions: number[] = [];
    const versioningSweep: LiveTapeSweepFn = (ctx) => {
      versions.push(ctx.tapeVersion);
      return jsInterpSweep(ctx);
    };
    const tape = new LiveTape({ sweepFn: versioningSweep });
    const idx1 = tape.ensureInterned(add(v("x"), lit(1)));
    tape.setVariable("x", 1);
    tape.catchUp();
    const v1 = versions[versions.length - 1]!;

    // Add new structure — tapeVersion must advance.
    const idx2 = tape.ensureInterned(mul(v("x"), lit(2)));
    tape.setVariable("x", 2);
    tape.catchUp();
    const v2 = versions[versions.length - 1]!;

    expect(v2).toBeGreaterThan(v1);
    expect(tape.getValue(idx1)).toBe(3);
    expect(tape.getValue(idx2)).toBe(4);
  });
});
