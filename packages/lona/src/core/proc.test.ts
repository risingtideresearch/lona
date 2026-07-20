import { describe, expect, test } from "vitest";
import { defineProc, callProc, mapReduce } from "./proc";
import { variableNum, asNum, Num } from "./num";
import { KIND_LIT, Project, childrenOfNumNode, type NumNode } from "./tree";
import { projectNode } from "./tree-cons";
import {
  inlineProcs,
  countReachableIncludingProcBodies,
  partialDerivative,
} from "./tree-walks";
import { serializeNumDAG, deserializeNumDAG } from "./tree-serialization";
import { compileTape } from "../eval/tape";
import {
  compileValueRoutine,
  type ValueRoutine,
  type MultiValueRoutine,
} from "../eval/routines";

const evalNodeAt = (n: NumNode, vars: Record<string, number>): number => {
  const r = compileValueRoutine([n]) as ValueRoutine;
  return r.eval(new Map(Object.entries(vars))) as number;
};
const evalAt = (n: Num, vars: Record<string, number>): number =>
  evalNodeAt(n.n, vars);

describe("proc — value equivalence with the inlined graph", () => {
  test("single call matches the hand-inlined expression", () => {
    const proc = defineProc(2, ([a, b]) => a!.mul(b!).add(a!));
    const x = variableNum("x");
    const y = variableNum("y");

    const [called] = callProc(proc, [x, y]);
    const inline = x.mul(y).add(x);

    for (const [xv, yv] of [
      [2, 3],
      [-1.5, 4],
      [0, 7],
      [10, -2.25],
    ]) {
      const vars = { x: xv!, y: yv! };
      expect(evalAt(called!, vars)).toBeCloseTo(evalAt(inline, vars), 10);
    }
  });

  test("mapReduce fold matches a plain JS loop", () => {
    const items = [1, 2, 3, 4, 5].map((k) => k * 0.5);
    const t = variableNum("t");

    // proc(a, c) = (a * t + c)^2 ; fold by sum. `t` enters as an arg (uniform),
    // `c` as a per-item constant — nothing captured.
    const rolled = mapReduce(
      items,
      2,
      (c) => [t, c],
      ([a, c]) => a!.mul(a!).add(c!).square(),
      (u, v) => u.add(v),
      asNum(0),
    );

    let looped = asNum(0);
    for (const c of items) looped = looped.add(t.mul(t).add(c).square());

    for (const tv of [0.3, -1, 2.5]) {
      expect(evalAt(rolled, { t: tv })).toBeCloseTo(
        evalAt(looped, { t: tv }),
        9,
      );
    }
  });
});

describe("proc — emission sharing", () => {
  test("multi-output proc emits its shared body once", () => {
    // g(a) = { s=a*a; [s, s+1, s+2] } — `s` must be emitted once, not 3×.
    const proc = defineProc(1, ([a]) => {
      const s = a!.mul(a!);
      return [s, s.add(1), s.add(2)];
    });
    const x = variableNum("x");
    const outs = callProc(proc, [x]);
    const tape = compileTape(outs.map((o) => o.n));
    expect(tape).not.toBeNull();
    // VAR(x), MUL(s), LIT(1), ADD, LIT(2), ADD = 6 ops; projections are aliases.
    expect(tape!.opcodes.length).toBe(6);

    const r = compileValueRoutine(outs.map((o) => o.n)) as MultiValueRoutine;
    const v = r.eval(new Map([["x", 3]])) as number[];
    expect(v).toEqual([9, 10, 11]);
  });

  test("a body literal is emitted once across many calls", () => {
    const proc = defineProc(1, ([a]) => a!.add(3.5));
    const roots = [0, 1, 2, 3, 4].map(
      (i) => callProc(proc, [variableNum(`x${i}`)])[0]!.n,
    );
    const tape = compileTape(roots);
    expect(tape).not.toBeNull();
    const litCount = [...tape!.opcodes].filter((op) => op === KIND_LIT).length;
    expect(litCount).toBe(1); // the shared 3.5, independent of call count
  });
});

describe("proc — validation", () => {
  test("rejects a body that captures a variable", () => {
    expect(() => defineProc(1, ([a]) => a!.add(variableNum("oops")))).toThrow(
      /captures variable 'oops'/,
    );
  });

  test("callProc rejects an arity mismatch", () => {
    const proc = defineProc(2, ([a, b]) => a!.add(b!));
    expect(() => callProc(proc, [asNum(1)])).toThrow(/arity 2/);
  });

  test("rejects nested procs (the review's repro)", () => {
    const inner = defineProc(1, ([x]) => x!.add(1));
    expect(() =>
      defineProc(1, ([x]) => callProc(inner, [x!])[0]!.mul(2)),
    ).toThrow(/nested procs are not supported/);
  });

  test("projectNode rejects an out-of-range output", () => {
    const proc = defineProc(1, ([a]) => a!.mul(a!)); // one output
    const [only] = callProc(proc, [variableNum("pr_x")]);
    const call = (only!.n as Project).call;
    expect(() => projectNode(call, 1)).toThrow(/out of range/);
    expect(() => projectNode(call, -1)).toThrow(/out of range/);
  });
});

describe("proc — projections are sort-stamped", () => {
  test("commutative ops over two projections canonicalize consistently", () => {
    const proc = defineProc(1, ([a]) => [a!.mul(a!), a!.add(1)]);
    const [p, q] = callProc(proc, [variableNum("cx")]);
    // ADD is commutative; without stamped sort-ids both projections would get
    // id 0 and `p+q` / `q+p` would fail to dedup.
    expect(p!.add(q!).n).toBe(q!.add(p!).n);
  });
});

describe("proc — inlineProcs / transforms", () => {
  test("inlineProcs desugars to an equivalent, serializable plain DAG", () => {
    const proc = defineProc(2, ([a, b]) => a!.mul(b!).add(a!));
    const [called] = callProc(proc, [variableNum("ix"), variableNum("iy")]);
    const [inlined] = inlineProcs([called!.n]);

    const vars = { ix: 2, iy: 5 };
    expect(evalNodeAt(inlined!, vars)).toBeCloseTo(evalAt(called!, vars), 10);
    // proc nodes threw before; the inlined DAG serializes cleanly.
    expect(() => serializeNumDAG(inlined!)).not.toThrow();
  });

  test("inlineProcs shares one memo across roots (repeated calls collapse)", () => {
    const proc = defineProc(1, ([a]) => a!.mul(a!));
    const x = variableNum("sx");
    const r0 = callProc(proc, [x])[0]!.n;
    const r1 = callProc(proc, [x])[0]!.n;
    const [i0, i1] = inlineProcs([r0, r1]);
    expect(i0).toBe(i1); // identical (proc, args) collapse to one node
  });

  test("countReachableIncludingProcBodies includes the shared body", () => {
    const proc = defineProc(1, ([a]) => a!.mul(a!).add(a!));
    const roots = [0, 1, 2].map(
      (i) => callProc(proc, [variableNum(`cc${i}`)])[0]!.n,
    );
    const withBodies = countReachableIncludingProcBodies(roots);
    // plain reachability (childrenOfNumNode) excludes proc bodies…
    const seen = new Set<NumNode>();
    const stack = [...roots];
    while (stack.length) {
      const n = stack.pop()!;
      if (seen.has(n)) continue;
      seen.add(n);
      for (const c of childrenOfNumNode(n)) stack.push(c);
    }
    expect(withBodies).toBeGreaterThan(seen.size);
  });

  test("partialDerivative throws on proc nodes rather than silently mishandling", () => {
    const proc = defineProc(1, ([a]) => a!.mul(a!));
    const [p] = callProc(proc, [variableNum("dx")]);
    expect(() => partialDerivative(p!.n, "dx")).toThrow(/inline/i);
  });
});

describe("proc — serialization round-trips", () => {
  test("a multi-output proc graph serializes and restores equivalently", () => {
    const proc = defineProc(2, ([a, b]) => {
      const s = a!.mul(b!); // shared across both outputs
      return [s.add(a!), s.mul(2)];
    });
    const x = variableNum("szx");
    const y = variableNum("szy");
    const [o0, o1] = callProc(proc, [x, y]);
    const root = o0!.add(o1!); // combines both projections of one call

    const data = serializeNumDAG(root.n);
    expect(data.procs?.length).toBe(1);
    expect(data.procs![0]!.arity).toBe(2);

    const restored = deserializeNumDAG(data);
    const vars = { szx: 3, szy: 4 };
    expect(evalNodeAt(restored, vars)).toBeCloseTo(
      evalNodeAt(root.n, vars),
      10,
    );
  });

  test("round-trip survives JSON.stringify/parse", () => {
    const proc = defineProc(1, ([a]) => a!.mul(a!).add(1));
    const [p] = callProc(proc, [variableNum("jx")]);
    const data = JSON.parse(JSON.stringify(serializeNumDAG(p!.n)));
    expect(evalNodeAt(deserializeNumDAG(data), { jx: 5 })).toBe(26);
  });
});
