/**
 * Validation for proc bodies. A proc body must be **closed** and **tape-legal**,
 * and — for v1 — **flat** (no nested procs):
 *
 *   - closed: every reachable leaf is a `Param` of *this* proc (checked by object
 *     identity, so a publicly-constructed impostor `Param` with a matching tag is
 *     rejected) or a literal constant. No captured `Variable`.
 *   - tape-legal: no `ForeignFn` / `Derivative`.
 *   - flat: no `Call` / `Project`. A nested call is part of the shared outer body
 *     but would be instantiated under different outer argument bindings, which the
 *     identity-keyed emission cache cannot distinguish. Nested procs are not
 *     needed for the current workloads; forbid them rather than mis-emit.
 *
 * The walk accepts only the known tape-legal operator kinds and throws on any
 * other kind, so unsupported nodes fail loudly instead of slipping through as
 * childless leaves.
 */
import {
  childrenOfNumNode,
  isBinaryKind,
  isUnaryKind,
  KIND_CALL,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_LIT,
  KIND_PARAM,
  KIND_PROJECT,
  KIND_SELECT,
  KIND_VAR,
  Param,
  Variable,
  type NumNode,
} from "./tree";

export function validateProcBody(
  params: readonly Param[],
  body: readonly NumNode[],
): void {
  const ownParams = new Set<Param>(params);
  const seen = new Set<NumNode>();
  const stack: NumNode[] = [...body];

  while (stack.length > 0) {
    const n = stack.pop()!;
    if (seen.has(n)) continue;
    seen.add(n);

    const kind = n.kind;

    if (kind === KIND_LIT) continue;

    if (kind === KIND_PARAM) {
      if (!ownParams.has(n as Param)) {
        const p = n as Param;
        throw new Error(
          `defineProc: body references a foreign parameter (tag ${p.procTag}, ` +
            `index ${p.index}) that is not one of this proc's own params.`,
        );
      }
      continue;
    }

    if (kind === KIND_VAR) {
      throw new Error(
        `defineProc: body captures variable '${String(
          (n as Variable).name,
        )}'. Pass it as an explicit parameter — proc bodies must be closed ` +
          `over params and literal constants only.`,
      );
    }
    if (kind === KIND_FOREIGN) {
      throw new Error(
        "defineProc: body contains a ForeignFn, which cannot be lowered into " +
          "the value tape.",
      );
    }
    if (kind === KIND_DERIVATIVE) {
      throw new Error(
        "defineProc: body contains a Derivative, which cannot be lowered into " +
          "the value tape.",
      );
    }
    if (kind === KIND_CALL || kind === KIND_PROJECT) {
      throw new Error(
        "defineProc: nested procs are not supported — a proc body may not " +
          "contain a Call or Project. Inline the inner computation instead.",
      );
    }

    if (isUnaryKind(kind) || isBinaryKind(kind) || kind === KIND_SELECT) {
      for (const child of childrenOfNumNode(n)) stack.push(child);
      continue;
    }

    throw new Error(`defineProc: unsupported node kind ${kind} in proc body.`);
  }
}
