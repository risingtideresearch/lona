# lona-columnar

Experimental columnar map/reduce execution for `lona`.

## Structured columns (experimental)

Structured columns are available through the optional `lona-columnar` package. They preserve
a map/reduce axis instead of expanding it into one scalar DAG. CPU stages support all
four CPU backends. Placement and backend selection are configured
separately: stages choose `"auto"`, `"cpu"`, or `"gpu"`, while `buildStructuredRoutine` supplies
ordered backend candidates for each target. By default, auto map and reduce stages try GPU then
CPU, while `toNums` runs on CPU. Adjacent GPU stages remain device-resident; uploads and
readbacks are inferred from the resolved producer and consumer placements.

```ts
import { asNum, variable } from "lona";
import { buildStructuredRoutine, column } from "lona-columnar";

const x = variable("x");
const y = variable("y");
const offset = variable("offset");

const routine = buildStructuredRoutine(
  () =>
    column([x, y])
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
      .toNums(([total]) => total!),
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

A column may contain either `Num` values or one homogeneous `NumStruct` shape. Map and reduce
callbacks are traced once over formal parameters. External scalar `Num`/`NumStruct`
dependencies must be declared under `using`; they are evaluated once per routine invocation.
`toNums` crosses from a column back into an ordinary scalar Num stage. Structured routines use
an async `eval` surface so GPU stages can be added without API changes.

To require a map to run on GPU, set a hard stage placement. Backend selection remains a
routine-level concern:

```ts
await initGpu();

const mapped = column(values).map({
  using: { scale },
  build: (value, { index, using }) => value.mul(using.scale).add(index),
  placement: "gpu",
});

const routine = buildStructuredRoutine(
  () => mapped.toNums(([value]) => value!),
  {
    backends: {
      cpu: ["wasm-codegen", "js-codegen"],
      gpu: ["gpu-codegen"],
    },
  },
);
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
A CPU consumer such as `toNums` causes an inferred device readback. GPU values use f32 while CPU
stages normally use f64.

Placement may be configured by stage kind. Explicit stage placement wins over the kind policy,
which wins over the default:

```ts
const routine = buildStructuredRoutine(() => result, {
  placement: {
    default: "auto",
    map: "auto",
    reduce: "auto",
    toNums: "cpu",
  },
  auto: {
    targets: {
      map: ["gpu", "cpu"],
      reduce: ["gpu", "cpu"],
      toNums: ["cpu"],
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
