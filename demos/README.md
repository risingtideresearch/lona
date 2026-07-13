# Demos

The demos are private npm workspaces. They import the local package sources, so
they follow library changes without requiring a package build first.

## Spline playground

Interactive curve builder and sampler using `lona-curves`:

```bash
npm run dev -w demos/show-curves
```

## Profile surface optimisation

Variable profile fitting using Levenberg–Marquardt, with configurable numeric
backend and select specialization:

```bash
npm run dev -w demos/profile-surface-optimisation
```

Build both demos as part of the full repository build:

```bash
npm run build
```
