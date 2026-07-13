# Lona

**L**ittle **O**dd **N**umeric **A**utograd.

Lona is a small symbolic-numeric library for TypeScript. You build expressions
out of `Num` values; under the hood Lona records the computation as a
hash-consed DAG, so the same expression can be evaluated immediately,
differentiated symbolically, compiled to a fast routine (CPU or GPU), or
serialized and replayed elsewhere.

This is a monorepo (npm workspaces). The library itself lives in
[`packages/lona`](packages/lona) — see that package's
[README](packages/lona/README.md) for the full API (quick start, `nexpr`,
compilation/backends, differentiation, serialization).

## Packages

| Package                               | What it is                                                                              |
| ------------------------------------- | --------------------------------------------------------------------------------------- |
| [`lona`](packages/lona)               | The core library — `Num`, autograd, compilation backends. Start here.                   |
| [`lona-geom`](packages/lona-geom)     | Symbolic 2D/3D geometry primitives (`Point`, `Vec2`/`Vec3`, `Angle`, …) built on `Num`. |
| [`lona-curves`](packages/lona-curves) | Curve builders (Catmull-Rom, Hobby, natural spline, PCHIP, …) built on `lona-geom`.     |

Dependency direction: `lona` → `lona-geom` → `lona-curves`. `lona-curves` has
an interactive demo (`npm run demo:show-curves -w packages/lona-curves`); see
its [README](packages/lona-curves/README.md).

## Working in this repo

```bash
npm install          # installs all workspaces
npm run format        # prettier --write, whole repo
npm run formatcheck    # prettier --check, whole repo
```

Each package also has its own `test`, `lint`, `typecheck`, `build`, and
`check` scripts — run them with `-w packages/<name>`, e.g.:

```bash
npm run test -w packages/lona
```
