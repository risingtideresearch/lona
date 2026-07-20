/**
 * Procs — reusable, closed subgraphs that let a repeated builder loop share one
 * body instead of inlining N structurally-distinct copies into the `Num` DAG.
 *
 * `defineProc` traces a body once against symbolic `Param`s and validates it;
 * `callProc` applies it, returning one `Num` per output (all sharing a single
 * `Call`, so a multi-output proc's body is emitted once); `mapReduce` is the
 * ergonomic form for a single-output fold over a collection.
 *
 * The whole abstraction disappears at compile time: `compileTape` inlines each
 * call into the flat value tape (see eval/tape/emit.ts). Backends never see it.
 * The memory win is that the *retained object DAG* stays O(body + calls) rather
 * than O(calls · body).
 */
import { Num, asNum } from "./num";
import { type Proc } from "./tree";
import { callNode, newProcTag, paramNode, projectNode } from "./tree-cons";
import { validateProcBody } from "./proc-validate";

/**
 * Trace and validate a proc body once. `build` receives `arity` symbolic params
 * and returns one output (`Num`) or several (`Num[]`). Throws if the body is not
 * closed over params + literal constants, or is not tape-legal — see
 * {@link validateProcBody}.
 */
export function defineProc(
  arity: number,
  build: (params: Num[]) => Num | Num[],
): Proc {
  if (!Number.isInteger(arity) || arity < 0)
    throw new Error(
      `defineProc: arity must be a non-negative integer, got ${arity}`,
    );
  const tag = newProcTag();
  const params = Array.from({ length: arity }, (_, i) => paramNode(tag, i));
  const outs = build(params.map((p) => new Num(p)));
  const outArray = Array.isArray(outs) ? outs : [outs];
  if (outArray.length === 0)
    throw new Error("defineProc: build returned no outputs");
  const body = Object.freeze(outArray.map((o) => o.n));
  validateProcBody(params, body);
  return { tag, arity, params: Object.freeze(params), body };
}

/**
 * Apply a proc to an argument tuple. Returns one `Num` per proc output, all
 * projecting the *same* `Call` — so downstream use of several outputs emits the
 * shared body only once. `args` length must equal the proc's arity.
 */
export function callProc(proc: Proc, args: ReadonlyArray<Num | number>): Num[] {
  if (args.length !== proc.arity)
    throw new Error(
      `callProc: proc arity ${proc.arity}, got ${args.length} args`,
    );
  const argNodes = Object.freeze(args.map((a) => asNum(a).n));
  const call = callNode(proc, argNodes);
  return proc.body.map((_, out) => new Num(projectNode(call, out)));
}

/**
 * Reroll a single-output JS loop into `defineProc` + one `Call` per item, folded
 * by an associative `reduce`. The body is traced once; each item contributes one
 * call and its argument fan-in, so the DAG grows O(body + items) rather than
 * O(items · body). Captures (e.g. a fixed cut plane) are passed as explicit
 * params via `argsOf`, never closed over.
 */
export function mapReduce<T>(
  items: readonly T[],
  arity: number,
  argsOf: (item: T) => Array<Num | number>,
  build: (params: Num[]) => Num,
  reduce: (a: Num, b: Num) => Num,
  init: Num,
): Num {
  const proc = defineProc(arity, build);
  let acc = init;
  for (const item of items) {
    const [out] = callProc(proc, argsOf(item));
    acc = reduce(acc, out!);
  }
  return acc;
}
