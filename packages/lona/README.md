# Lona

**L**ittle **O**dd **N**umeric **A**utograd.

Lona is a small symbolic-numeric library for TypeScript. You build expressions
out of `Num` values; under the hood Lona records the computation as a hash-consed
DAG, so the same expression can be:

- evaluated immediately via `asNumber` and Lona's live variable values,
- differentiated symbolically (full Jacobian, automatic),
- compiled to a fast routine and called in a tight loop or batched on GPU,
- serialized to JSON and replayed elsewhere.

The library is intentionally small. There is one object type to know — `Num` —
and one optional template-literal sugar — `nexpr`. Everything else is either a
plain function on `Num` or a way to compile a `Num` into something faster.

---

## Quick start

```ts
import { variable, nexpr, asNumber, setVariable } from "lona";

const x = variable("x", 0.5);
const y = variable("y", 1.0);

// Method chaining:
const f = x.square().add(y.square()).sqrt();

// Or as a template literal:
const g = nexpr`sqrt(${x} ** 2 + ${y} ** 2)`;

asNumber(f); // 1.118033988749895, using x=0.5 and y=1.0

setVariable("x", 3);
setVariable("y", 4);
asNumber(f); // 5

f.eval({ x: 3, y: 4 }); // 5, using an explicit per-call variable map
```

Both `f` and `g` are `Num` instances wrapping the same kind of DAG. They behave
identically; pick whichever reads better at the call site.

---

## The `Num` API

A `Num` is a symbolic expression wrapper around a hash-consed `NumNode`
(`num.n`). It does **not** store a concrete value field; there is no `num.v`.
When you need the current numeric value of a `Num`, call `asNumber(num)`; this
routes through Lona's live evaluator and the current variable values.

### Constructing values

```ts
import { variable, asNum, ONE, TWO, ZERO } from "lona";

variable("x"); // a free variable named "x", initial live value 0
variable("x", 3.14); // free variable with initial live value 3.14
asNum(7); // wrap a JS number as a literal Num
ONE;
TWO;
ZERO; // shared literal singletons
```

`asNum(n)` is a no-op when `n` is already a `Num`, so you can sprinkle it
defensively anywhere you accept `Num | number`.

### Live concrete values with `asNumber`

`Num` values are symbolic. The supported path for extracting a concrete numeric
value is `asNumber(value)`:

```ts
import { variable, asNumber, setVariable, currentValueEpoch } from "lona";

const x = variable("x", 2); // registers x's initial live value
const y = variable("y", 5);
const f = x.square().add(y);

asNumber(f); // 9

setVariable("x", 4);
asNumber(f); // 21

setVariable("y", -1);
asNumber(f); // 15
```

`asNumber` also accepts raw JavaScript numbers, which pass through unchanged:

```ts
asNumber(42); // 42
```

Variable values live in Lona's live-value context and are addressed by variable
name:

- `variable(name, initialValue)` registers an initial live value. The first
  registration for a name wins; constructing another `variable` with the same
  name does not reset it. A pending `setVariable` for that name takes
  precedence over this initial value.
- `setVariable(name, value)` changes the current live value. Expressions that
  depend on that variable reflect the new value the next time you call
  `asNumber`.
- `setVariable` can be called before any `Num` using that variable has been
  evaluated; the value is stored and applied when the variable is first used.
- Writing the same value again is a no-op. `currentValueEpoch()` advances only
  when a variable value actually changes, which is useful if you keep your own
  caches around live `Num` values.

This live path is separate from `num.eval(...)` and compiled routines. Use
`asNumber` when you want the current value managed by `setVariable`; use
`eval`/`compile` when you want to pass an explicit variable map for one call or
a tight loop.

### Arithmetic and elementary functions

Every `Num` method returns a new `Num`. Operands can be `Num` or `number`:

```ts
x.add(y) | x.sub(y) | x.mul(y) | x.div(y) | x.mod(y);
x.neg() | x.inv() | x.abs() | x.sign();
x.square() | x.powi(k); // integer-power fast path
x.sqrt() | x.cbrt() | x.safeSqrt(); // safeSqrt clamps the input to ≥ 0
x.exp() | x.log() | x.log1p();
x.sin() | x.cos() | x.tan();
x.asin() | x.acos() | x.atan();
x.tanh() | x.smoothabs() | x.softplus() | x.softminus();
```

Free functions for things that don't fit a single-receiver shape:

```ts
import { atan2, hypot, max, min, clamp, sigmoid } from "lona";

atan2(y, x);
hypot(x, y);
max(a, b, c);
clamp(x, 0, 1);
sigmoid(x);
```

### Comparisons and branches

Comparisons return a `Num` that is `1` (true) or `0` (false), so they compose
with arithmetic:

```ts
x.lessThan(y);
x.lessThanOrEqual(y);
x.greaterThan(y);
x.greaterThanOrEqual(y);
x.equals(y);
x.compare(y); // -1 | 0 | 1
x.and(y);
x.or(y);
x.not();
```

Lona is symbolic, so JavaScript `if` only runs while building the expression.
Use `ifTruthyElse` (or its alias `select`) to express a branch as a value:

```ts
import { ifTruthyElse } from "lona";

const safeRecip = ifTruthyElse(x.equals(0), ZERO, ONE.div(x));
```

For longer conditionals, use the fluent `when(...).then(...).else(...)` API.
`elseIf` chains are supported, and conditions may be passed directly or as
callbacks:

```ts
import { when } from "lona";

const adjusted = when(x.lessThan(0))
  .then(() => x.neg())
  .elseIf(() => x.lessThan(3))
  .then(() => x.add(10))
  .else(() => x.add(20));
```

This builds the same kind of symbolic select expression as nested
`ifTruthyElse` calls, but reads like an `if / else if / else` chain. Branches
are callbacks so the same fluent shape also works with numeric/direct contexts.
When using the symbolic `Num` API, all branches must still be valid to build.

For switch-like logic, use `cases(selector)`:

```ts
import { cases } from "lona";

const chosen = cases(mode)
  .case(0, () => x)
  .case(1, () => x.square())
  .case(2, () => x.sqrt())
  .default(() => ZERO);
```

Each `.case(value, branch)` compares `selector.equals(value)` in order and
falls through to `.default(...)` if no case matches.

### One-shot evaluation

```ts
num.eval({ x: 3, y: 4 }); // pass variables for this call
num.eval(new Map([["x", 3]])); // Map also accepted
num.eval(); // ok for expressions with no variables, or when omissions should be 0
```

`num.eval` lazily compiles a value routine on first call and caches it on the
`Num` instance, so subsequent calls in a tight loop are fast. It does not read
or mutate the live values used by `asNumber`/`setVariable`; pass the values you
want in the call. If you want to control compilation explicitly — pick a
backend, or share the same routine across many `Num` wrappers — use
`num.compile()` directly (next section).

### Other methods

```ts
num.simplify(); // run constant folding / algebraic rewrites; returns a new Num
num.debug("tag"); // attach a label that surfaces during graph rendering
num.asDot(); // Graphviz DOT representation, useful for inspection
num.toJSON(); // { dag }
Num.fromJSON(j); // round-trip
```

---

## `nexpr` — template literal syntax

`nexpr` parses a JS template string into the same kind of DAG as the method API:

```ts
import { nexpr, variable } from "lona";

const x = variable("x");
const y = variable("y");

nexpr`(${x} + ${y}) ** 2 - 4 * ${x} * ${y}`;
nexpr`sqrt(1 - ${x} ** 2)`;
```

**Operators**: `+ - * / %`, plus `**` and `^` (both mean exponentiation,
right-associative). Unary minus is supported with the same precedence as JS:
`-2 ** 2` parses as `-(2 ** 2)`.

**Functions**: `sqrt`, `cbrt`, `abs`, `log`, `log1p`, `exp`, `sin`, `cos`,
`tan`, `asin`, `acos`, `atan`. Identifiers must be followed by `(...)`.

**Placeholders**: `${value}` accepts either a `Num` or a `number`. Numbers
become literals; `Num`s splice their underlying DAG in directly, so there is no
re-parse cost for sub-expressions you've already built.

The parser specializes integer powers: `${x} ** 3` compiles to repeated
multiplication (`x * x * x`), which is both faster and exactly differentiable.
Non-integer or symbolic exponents fall back to `exp(log(base) * exponent)`.

`nexpr` is purely a builder — it does not introduce a runtime parser at
evaluation time. The string is parsed once when the template literal is first
evaluated, producing a `Num` you reuse like any other.

---

## `compile` and routines

`num.eval(...)` is fine for occasional calls. When you need to evaluate the
same expression many times — once per pixel, per frame, per gradient-descent
step — compile it into a **routine**:

```ts
const f = nexpr`sqrt(${x} ** 2 + ${y} ** 2)`;
const routine = f.compile();

routine.eval({ x: 3, y: 4 }); // 5
routine.eval({ x: 0, y: 1 }); // 1
```

A routine bakes the DAG into native code (or a tape, or a shader — see
[Backends](#backends) below) once at compile time, then evaluates it
without re-walking the symbolic tree.

### What is a routine?

A `Routine` is an object with a fixed **shape** describing what it computes,
and a uniform **call surface** describing how you invoke it.

#### Shapes

| Shape         | Returned by                              | One call gives you                    |
| ------------- | ---------------------------------------- | ------------------------------------- |
| `value`       | `compileValueRoutine([f])` (single root) | a single `number`                     |
| `multi-value` | `compileValueRoutine([f, g, ...])`       | `number[]`, one per root              |
| `grad`        | `compileGradRoutine([f], vars)`          | `{ value, gradient }` for one root    |
| `jacobian`    | `compileGradRoutine([f, g, ...], vars)`  | `{ values, jacobian }` for many roots |

`num.compile()` is a thin wrapper around `compileValueRoutine([num.n])` that
unwraps the result and asserts the shape is `value`.

#### Call surface

Every routine implements four entry points. Each backend implements at least
one natively; the others are synthesized.

```ts
routine.eval(vars)                     // sync, single point
routine.evalAsync(vars)                // async, single point
routine.evalBatch(vars, numPoints?)    // async, many points
routine.evalBatchPacked(buf, numPoints) // async, pre-packed Float32Array
```

- **Sync vs async**: sync `eval` throws on backends that are intrinsically
  async (GPU). If you don't know what backend you're on, use `evalAsync`.
- **Single vs batch**: `evalBatch` takes one array of values per variable
  (`{ x: [...], y: [...] }`) and returns a `Float32Array`. GPU backends
  evaluate the whole batch in parallel; CPU backends loop. The `*Packed`
  variant skips the per-call repacking when you already have your inputs in
  the right layout — useful for tight inner loops over millions of points.

#### Variables

Routines bind variables by name. The `varSlots` field on a routine lists the
names it reads, in compile-time-fixed order; `numVars` is the count. Pass
inputs as a `VarMap` (`Record<string, number>` or `Map`).

If your DAG contains explicit `Derivative` nodes (rare; produced by
`fullDerivative` and friends), `eval` accepts an optional second `derivatives`
map for seeding partial-derivative slots. You can ignore this argument
otherwise.

### Differentiation via routines

```ts
import { compileGradRoutine } from "lona";

const r = compileGradRoutine([f.n], ["x", "y"]);
const { value, gradient } = r.eval({ x: 3, y: 4 });
// value:    5
// gradient: { x: 0.6, y: 0.8 }
```

`compileGradRoutine` takes the underlying `NumNode[]`, not `Num[]`, because
it's the lower-level routine API. Pass `f.n` to get the node out of a `Num`.

If you want the symbolic gradient as another DAG (e.g. to inspect, simplify,
or compose with more arithmetic), use `fullDerivative`, `gradient`, or
`jacobian` from the `lona` top level — those return `NumNode`s rather than a
runtime routine.

### Backends

A backend is a strategy for turning a compiled tape into a callable. Lona
ships with six:

| Backend        | Native call    | Best for                                         |
| -------------- | -------------- | ------------------------------------------------ |
| `js-interp`    | sync, scalar   | tiny expressions, debugging                      |
| `js-codegen`   | sync, scalar   | small/medium expressions, no toolchain           |
| `wasm-interp`  | sync, scalar   | medium expressions, more shapes                  |
| `wasm-codegen` | sync, scalar   | **default**; large expressions, fastest CPU path |
| `gpu-interp`   | async, batched | very large expressions on huge batches           |
| `gpu-codegen`  | async, batched | huge batches with simple expressions             |

The default is `wasm-codegen` for value and grad routines (and `wasm-interp`
for jacobians of multi-root DAGs, which the codegen path doesn't specialize).
Override with the `backend` option:

```ts
const r = f.compile({ backend: "gpu-codegen" });
const r = compileGradRoutine([f.n], ["x"], { backend: "js-interp" });
```

GPU backends require a one-time async init:

```ts
import { initGpu, destroyGpu } from "lona";

await initGpu(); // before compiling any GPU routine
// ... use routines ...
destroyGpu(); // tear down the device
```

Compiling a GPU routine before `initGpu` resolves throws synchronously.

**Under Node.js**, `destroyGpu()` is **mandatory** at the end of any script
that called `initGpu()`. The `webgpu` package holds a native handle that keeps
the event loop alive; without `destroyGpu()` the process never exits and CLI
runs / test workers / benchmarks hang. In the browser you can skip it — the
device is reclaimed when the page unloads.

### When does compilation fail?

`compileValueRoutine` and `compileGradRoutine` return `null` if the DAG
contains constructs that can't be compiled to a tape — most notably
`ForeignFn` nodes (user-supplied opaque numerical functions). In that case
`num.eval()` falls back to a tree walk transparently; if you called
`num.compile()` directly it throws, since you asked for a routine
specifically.

---

## Symbolic differentiation

For pure-DAG manipulation (no compilation), the `api/diff` family operates on
`NumNode`s:

```ts
import { fullDerivative, gradient, jacobian } from "lona";

fullDerivative(f.n); // a NumNode whose Derivative slots are populated
gradient(f.n, ["x", "y"]); // NumNode[] — ∂f/∂x, ∂f/∂y
jacobian([f.n, g.n], ["x", "y"]); // NumNode[][]
```

These are useful when you want the gradient to participate in further symbolic
work (simplification, composition, fingerprinting). For "I just want the
numbers", prefer `compileGradRoutine` — it's faster and avoids materializing
the gradient DAG.

---

## Serialization

```ts
const json = num.toJSON(); // { dag: SerializedNumDAG }
const back = Num.fromJSON(json);
```

The serialized form preserves DAG sharing — a sub-expression referenced ten
times serializes once and rehydrates as the same shared node. Round-tripping
through JSON does not lose hash-consing.

---

## Project layout

```
src/
├── core/      Num, NumNode tree, hash-consing, serialization
├── api/       fn, ops, diff, expressions (nexpr), simplify
├── dag/       generic DAG traversal helpers
├── eval/      tape IR, transforms (kernels), routines, backends
└── utils/     hashing
```

The public API lives in `main.ts`. Anything not re-exported there is internal
and may move between releases.
