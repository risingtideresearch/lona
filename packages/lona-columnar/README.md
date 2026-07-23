# lona-columnar

Experimental columnar map/reduce execution for `lona`.

## Columnar columns (experimental)

Columnar columns are available through the optional `lona-columnar` package. They preserve
a map/reduce axis instead of expanding it into one scalar DAG. CPU stages support all
four CPU backends. Placement and backend selection are configured
separately: stages choose `"auto"`, `"cpu"`, or `"gpu"`, while `columnarRoutine` supplies
ordered backend candidates for each target. By default, auto source stages try CPU then GPU,
map and reduce stages try GPU then CPU, and callback-based `then` and `output` stages run on CPU.
A GPU source is evaluated directly into device-resident column storage. Adjacent GPU stages
remain device-resident, and transfers are inferred from resolved producer and consumer
placements.

```ts
import { asNum, variable } from "lona";
import { columnarRoutine, column } from "lona-columnar";

const x = variable("x");
const y = variable("y");
const offset = variable("offset");

const routine = columnarRoutine(
  () =>
    column([x, y], { placement: "cpu" })
      .map({
        using: { offset },
        build: (value, { index, using }) => value.add(using.offset).add(index),
        placement: "cpu",
      })
      .reduce((a, b) => a.add(b), asNum(0), {
        associative: true,
        order: "tree",
        placement: "cpu",
      })
      .output(([total]) => total!),
  {
    placement: "cpu",
    backends: { cpu: "wasm-codegen" },
  },
);

await routine.evalAsync(
  new Map([
    ["x", 2],
    ["y", 5],
    ["offset", 10],
  ]),
); // 28

routine.dispose();
```

A column may contain either `Num` values or one homogeneous `NumStruct` shape. Its flattened
source DAG can be placed directly on the GPU without an intermediate readback or column upload:

```ts
const source = column([x.add(1), y.mul(2)], { placement: "gpu" });
```

Map and reduce callbacks are traced once over formal parameters. External scalar
`Num`/`NumStruct` dependencies must be declared under `using`; they are evaluated once per
routine invocation.
`then` runs a whole-column callback whose callback-created `column(...)` is incorporated into
the same graph and can be followed by more column stages:

```ts
const expanded = column(values)
  .then(([a, b]) => column([a!.add(b!), b!.mul(2)]))
  .map((value) => value.max(0))
  .sum();
```

The returned column must be non-empty. `output()` closes the graph. With no callback it returns
the current column directly and adds no execution stage; a reduced scalar therefore remains on
the GPU until the final readback:

```ts
const routine = columnarRoutine(() =>
  column(values).sum({ placement: "gpu" }).output(),
);
const total: number = await routine.evalAsync(vars);
```

`output(build)` adds a final CPU whole-column scalar callback when post-processing is needed.
Columnar routines use an async evaluation surface because GPU output requires a readback.

To require a map to run on GPU, set a hard stage placement. Backend selection remains a
routine-level concern:

```ts
await initGpu();

const mapped = column(values).map({
  using: { scale },
  build: (value, { index, using }) => value.mul(using.scale).add(index),
  placement: "gpu",
});

const routine = columnarRoutine(() => mapped.output(([value]) => value!), {
  backends: {
    cpu: ["wasm-codegen", "js-codegen"],
    gpu: ["gpu-codegen"],
  },
});
```

GPU reductions may use explicit built-ins:

```ts
const totals = mapped.sum({
  componentWise: true, // required for NumStruct columns
  order: "tree",
  placement: "gpu",
});
```

`sum`, `product`, `min`, and `max` support scalar columns and explicitly component-wise
`NumStruct` columns. Their empty identities are `0`, `1`, `Infinity`, and `-Infinity`. GPU tree
reductions propagate NaN and define signed-zero min/max behavior.

Arbitrary reduction callbacks also support GPU tree execution when `associative: true` is
acknowledged. Their explicit initial value participates as the first tree element, and `using`
dependencies are uploaded as broadcast uniforms. Ordered `left` reductions remain CPU-only.
A callback-based `then` or `output` stage causes an inferred device readback. Direct `output()`
only reads the final result. GPU values use f32 while CPU stages normally use f64.

Placement may be configured by stage kind. Explicit stage placement wins over the kind policy,
which wins over the default:

```ts
const routine = columnarRoutine(() => result, {
  placement: {
    default: "auto",
    source: "auto",
    map: "auto",
    reduce: "auto",
    then: "cpu",
    output: "cpu",
  },
  auto: {
    targets: {
      source: ["cpu", "gpu"],
      map: ["gpu", "cpu"],
      reduce: ["gpu", "cpu"],
      then: ["cpu"],
      output: ["cpu"],
    },
  },
  backends: {
    cpu: ["wasm-codegen", "js-codegen"],
    gpu: ["gpu-codegen"],
  },
});
```

For `"auto"`, targets and their backends are tried in configured order during compilation. A
hard `"cpu"` or `"gpu"` placement may try another backend on that same target, but never changes
target. Transfers are inferred only after placements resolve. After evaluation,
`routine.lastEvaluationStats` reports upload/download bytes, dispatch and readback counts, total
wall time, and per-stage timings. Device buffers are reused by size and usage between completed
evaluations.

---
