# Lona

**L**ittle **O**dd **N**umeric **A**utograd.

Lona is a small symbolic-numeric library for TypeScript. You build expressions
out of `Num` values; under the hood Lona records the computation as a
hash-consed DAG, so the same expression can be evaluated immediately,
differentiated symbolically, compiled to a fast routine (CPU or GPU), or
serialized and replayed elsewhere.

This is an npm-workspaces monorepo. The library itself lives in
[`packages/lona`](packages/lona) — see that package's
[README](packages/lona/README.md) for the full API.

## Packages

| Package                               | What it is                                                                              |
| ------------------------------------- | --------------------------------------------------------------------------------------- |
| [`lona`](packages/lona)               | The core library — `Num`, autograd, and compilation backends.                           |
| [`lona-geom`](packages/lona-geom)     | Symbolic 2D/3D geometry primitives (`Point`, `Vec2`/`Vec3`, `Angle`, …) built on `Num`. |
| [`lona-curves`](packages/lona-curves) | Curve builders (Catmull-Rom, Hobby, natural spline, PCHIP, …) built on `lona-geom`.     |

Dependency direction: `lona` → `lona-geom` → `lona-curves`.

Interactive applications live under [`demos`](demos). Benchmarks and their
fixtures remain with the core implementation under [`packages/lona/bench`](packages/lona/bench).

## Working in this repo

Use Node 22 (see [`.nvmrc`](.nvmrc)).

```bash
npm install
npm run build       # libraries, then demos
npm run test        # all workspace tests
npm run check       # format, lint, types, tests, and build
npm run watch       # watch all three libraries
```

Tool versions and shared ESLint, TypeScript, Prettier, and Vite-library
configuration are managed from the repository root. Package scripts remain
available through npm workspaces, for example:

```bash
npm run test -w packages/lona
npm run test:integration -w packages/lona # large real-world fixtures
npm run typecheck -w packages/lona-curves
npm run dev -w demos/show-curves
```

See [`demos/README.md`](demos/README.md) for the runnable demos.
