import { Num, variableNum } from "../core/num";
import type { VarName } from "../core/tree";

import { simpleEval } from "../eval/eval-value";
import { INTERN_FAILED } from "../eval/tape";
import { LiveTape } from "../eval/live/live-tape";

/**
 * A stateful object that owns a LiveTape and exposes convenient methods to set
 * variable values and read node values.
 *
 * Consumers can create multiple contexts to manage separate sets of variable
 * values. For convenience (avoid passing a context explicitly to many
 * functions), it is also possible to temporarily install a context as the
 * "current" one using `withContext`.
 */
export class NumValueContext {
  private readonly liveTape = new LiveTape();

  /**
   * Evaluates the given `Num` with variable values provided by this context.
   *
   * If `n` is a number, it is returned as is.
   */
  asNumber(n: number | Num): number {
    // Lazily interns into the live tape
    // on first call. Subgraphs containing foreign or derivative nodes
    // cannot be tape-resident; for those, fall back to `simpleEval`.
    if (typeof n === "number") {
      return n;
    }
    const node = n.n;
    const idx = this.liveTape.ensureInterned(node);
    if (idx === INTERN_FAILED) return simpleEval(node);
    return this.liveTape.getValue(idx);
  }

  /**
   * Creates a Num variable with the given name and register `val` as its
   * initial value in this context.
   */
  variable(name: VarName, val = 0): Num {
    this.registerInitialVariableValue(name, val);
    return variableNum(name);
  }

  /**
   * Sets the current value of variable `name`. If no node referencing the
   * variable has been interned yet, the value is stashed and applied on
   * first intern.
   */
  setVariable(name: VarName, value: number): void {
    this.liveTape.setVariable(name, value);
  }

  /**
   * Registers a default value for a variable that has not yet been
   * interned. Idempotent — the first registration per name wins.
   */
  registerInitialVariableValue(name: VarName, value: number): void {
    this.liveTape.registerInitialVariableValue(name, value);
  }

  /**
   * Monotonic counter that advances on every value-changing `setVariable` call.
   *
   * Consumers can compare a stored snapshot of this number to detect whether
   * anything a tape-resident `Num` depends on may have changed.
   */
  get epoch(): number {
    return this.liveTape.epoch;
  }
}

let defaultContext: NumValueContext | undefined = undefined;
let currentContext: NumValueContext | undefined = undefined;

/**
 * Returns the given `ctx` or, if not given, the current context.
 */
export function getContext(ctx?: NumValueContext): NumValueContext {
  const context = ctx ?? currentContext ?? defaultContext;
  if (!context) {
    throw Error(
      "No NumValueContext provided. Please explicitly use a context, e.g., wrap your code with withContext().",
    );
  }
  return context;
}

/**
 * Run `fn` with `ctx` installed as the current context, then restore the
 * previous one. Nested calls restore in LIFO order. Safe under exceptions.
 *
 * Note: not safe across `await` suspensions unless the awaited work stays
 * in the same microtask. Build the nodes you need synchronously.
 */
export function withContext<T>(ctx: NumValueContext, fn: () => T): T {
  const prev = currentContext;
  currentContext = ctx;
  try {
    return fn();
  } finally {
    currentContext = prev;
  }
}

/**
 * Creates a new `NumValueContext` and sets it as "default" context.
 *
 * This context will be used by global functions like `getContext`, `asNumber`
 * and `variable` whenever there is no explicit context given via `withContext`.
 *
 * Throws if called more than once without an intervening
 * `clearDefaultContext()`, or called within a `withContext()` block.
 *
 * This function is useful for quick scripts and tests that don't need multiple
 * contexts. For production code, prefer using `withContext` which makes context
 * management explicit and less bug-prone.
 */
export function initDefaultContext(): NumValueContext {
  if (defaultContext) {
    throw Error("There is already a default context initialized.");
  }
  if (currentContext) {
    throw Error("Calling initDefaultContext() within a `withContext()` block.");
  }
  defaultContext = new NumValueContext();
  return defaultContext;

  // Note: we prefer using a separate `defaultContext` variable instead of just
  // checking and setting `currentContext`, as it makes it safer in case someone
  // calls `clearDefaultContext()` within a `withContext()` block. Basically, it
  // allows us to know whether the current context is the default one or one set
  // by `withContext()`.
}

/**
 * Removes the default context created by `initDefaultContext`.
 *
 * Throws if there is no default context, or if called within a `withContext()`
 * block.
 */
export function clearDefaultContext() {
  if (!defaultContext) {
    throw Error("There is no default context to clear.");
  }
  if (currentContext) {
    throw Error(
      "Calling clearDefaultContext() within a `withContext()` block.",
    );
  }
  defaultContext = undefined;
}

// Convenient global functions working with the current context. These are just
// thin wrappers around the corresponding `NumValueContext` methods, but they
// save the caller from having to call `getContext()` all the time.

export function asNumber(n: number | Num, ctx?: NumValueContext): number {
  return getContext(ctx).asNumber(n);
}

export function variable(name: VarName, val = 0, ctx?: NumValueContext): Num {
  return getContext(ctx).variable(name, val);
}

export function setVariable(
  name: VarName,
  value: number,
  ctx?: NumValueContext,
): void {
  getContext(ctx).setVariable(name, value);
}

export function registerInitialVariableValue(
  name: VarName,
  value: number,
  ctx?: NumValueContext,
): void {
  getContext(ctx).registerInitialVariableValue(name, value);
}

// XXX: rename to the simpler `epoch` to exactly match NumValueContext naming?
export function currentValueEpoch(ctx?: NumValueContext): number {
  return getContext(ctx).epoch;
}
