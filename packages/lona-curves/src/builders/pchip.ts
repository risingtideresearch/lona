import { asNum, ifTruthyElse, type Num } from "lona";
import { Point, Vec2 } from "lona-geom";
import type { Spline } from "../spline";
import { splineFromHermite } from "./hermite";

// ---------------------------------------------------------------------------
// Monotone cubic Hermite (PCHIP) — interpolates without overshoot.
//
// Functional curve `y = f(x)`: both the station positions `x` (the spline
// knots) and the `y` values may be symbolic `Num`s — knots stay symbolic all
// the way through, so a curve whose stations are themselves blended from
// several sources re-solves as the blend moves. The monotonicity sign-tests
// that branch in the classic numeric algorithm become `ifTruthyElse` selects
// here, so the whole curve is one differentiable graph. (`select` doesn't
// short-circuit, so both arms are always built; lona's `div` has a finite
// divide-by-zero fallback, so the guarded harmonic-mean arm is safe to
// evaluate even where it would divide by zero.)
// ---------------------------------------------------------------------------

/** Fritsch–Carlson end-point slope with the standard sign/limit guards. */
function pchipEnd(h0: Num, h1: Num, d0: Num, d1: Num): Num {
  const m = d0.mul(h0.mul(2).add(h1)).sub(d1.mul(h0)).div(h0.add(h1));
  // sign(m) !== sign(d0) → 0
  const wrongSign = m.sign().equals(d0.sign()).not();
  // sign(d0) !== sign(d1) && |m| > 3|d0| → clamp to 3·d0
  const overshoot = d0
    .sign()
    .equals(d1.sign())
    .not()
    .and(m.abs().greaterThan(d0.abs().mul(3)));
  return ifTruthyElse(
    wrongSign,
    asNum(0),
    ifTruthyElse(overshoot, d0.mul(3), m),
  );
}

/** Monotone PCHIP slopes for symbolic `ys` over ascending (possibly symbolic) `xs`. */
export function pchipSlopes(xs: (Num | number)[], ys: Num[]): Num[] {
  const n = xs.length;
  const x = xs.map(asNum);
  const h: Num[] = new Array(n - 1);
  const d: Num[] = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = x[i + 1].sub(x[i]);
    d[i] = ys[i + 1].sub(ys[i]).div(h[i]);
  }
  if (n === 1) return [asNum(0)];
  if (n === 2) return [d[0], d[0]];

  const m: Num[] = new Array(n);
  for (let i = 1; i < n - 1; i++) {
    const w1 = h[i].mul(2).add(h[i - 1]);
    const w2 = h[i].add(h[i - 1].mul(2));
    const harmonic = w1.add(w2).div(w1.div(d[i - 1]).add(w2.div(d[i])));
    // local extremum or sign change between secants → flat
    m[i] = ifTruthyElse(
      d[i - 1].mul(d[i]).lessThanOrEqual(0),
      asNum(0),
      harmonic,
    );
  }
  m[0] = pchipEnd(h[0], h[1], d[0], d[1]);
  m[n - 1] = pchipEnd(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
  return m;
}

/**
 * Per-point left/right slopes for a "knuckled" monotone curve. `ks[i] ∈ [0,1]`
 * blends the smooth PCHIP slope toward the one-sided secants: `k=0` → C¹ smooth
 * (plain PCHIP); `k=1` → each side takes its own secant, so an isolated `k=1` is
 * a knuckle and two adjacent `k=1` points bound a straight segment. End points
 * have a single side. Every blend stays sign-consistent and within the
 * Fritsch–Carlson monotonicity box, so every `k` stays monotone-safe.
 */
export function knuckleSlopes(
  xs: (Num | number)[],
  ys: Num[],
  ks: Num[],
): { L: Num[]; R: Num[] } {
  const n = xs.length;
  const x = xs.map(asNum);
  const m = pchipSlopes(x, ys);
  const L = m.slice();
  const R = m.slice();
  for (let i = 0; i < n; i++) {
    const k = (ks[i] ?? asNum(0)).max(0).min(1);
    if (i > 0) {
      const leftSecant = ys[i].sub(ys[i - 1]).div(x[i].sub(x[i - 1]));
      L[i] = m[i].add(leftSecant.sub(m[i]).mul(k));
    }
    if (i < n - 1) {
      const rightSecant = ys[i + 1].sub(ys[i]).div(x[i + 1].sub(x[i]));
      R[i] = m[i].add(rightSecant.sub(m[i]).mul(k));
    }
  }
  return { L, R };
}

/** A PCHIP data point: `x` and `y` may be symbolic; optional knuckle `k ∈ [0,1]`. */
export interface PchipStation {
  x: Num | number;
  y: Num;
  k?: Num;
}

/**
 * Build a monotone PCHIP curve through `stations` (must be sorted ascending in
 * `x` — with symbolic knots this is the caller's contract, it cannot be
 * checked at build time). The result is a 2D parametric `Spline` whose
 * parameter is `x`: querying `curve.at(x)` returns the point `(x, y(x))`.
 * Optional per-station `k` adds knuckles (see `knuckleSlopes`).
 */
export function buildPchip(stations: PchipStation[]): Spline<Point, Vec2> {
  const n = stations.length;
  if (n < 2) {
    throw new Error(`buildPchip needs at least 2 stations (got ${n})`);
  }
  const xs = stations.map((s) => asNum(s.x));
  const ys = stations.map((s) => s.y);
  const ks = stations.map((s) => s.k ?? asNum(0));
  const { L, R } = knuckleSlopes(xs, ys, ks);

  const points = stations.map((s, i) => new Point(xs[i], s.y));
  // The 2D tangent of P(x) = (x, y(x)) is (1, slope). Knuckles give a point
  // different leaving (R) and arriving (L) slopes.
  const leaving = R.map((slope) => new Vec2(asNum(1), slope));
  const arriving = L.map((slope) => new Vec2(asNum(1), slope));

  return splineFromHermite(points, leaving, arriving, xs);
}
