# Demos

Each demo lives in its own sub-folder with its own `vite.config.js`.

## Profile surface optimisation

Variable profile fitting demo using Levenberg–Marquardt on the residual jacobian,
with configurable optimisation backend (`js-interp` or `wasm-interp`) and optional
select specialisation (`trace` / `full-trace`). Profile x values are driven by
squared variables (`x = u²`), the minimum y is fixed at 0, and extra profile
points can be added in the UI.

```bash
npx vite --config packages/lona/demos/profile-surface-optimisation/vite.config.js
```

or build it with:

```bash
npx vite build --config packages/lona/demos/profile-surface-optimisation/vite.config.js
```
