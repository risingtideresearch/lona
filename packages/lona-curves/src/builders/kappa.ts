import { asNum, ifTruthyElse, type Num } from "lona";
import { Point, Vec2 } from "lona-geom";
import { QuadraticSeg, Spline } from "../spline";
import { solveTridiagonal } from "../utils/tridiagonal";

// ---------------------------------------------------------------------------
// κ-curves (open, 2D) — Yan, Schiller, Wilensky, Carr & Schaefer, "κ-Curves:
// Interpolation at Local Maximum Curvature" (SIGGRAPH 2017; the curve behind
// Illustrator's curvature tool). The output is a chain of **quadratic** Béziers
// (hence `QuadraticSeg`, not the cubics every other builder emits), and each
// input point is interpolated **at the local maximum-curvature parameter** of
// its segment — so the input points are not the Bézier control points, and the
// joints between segments are solved for.
//
// The reference algorithm (the author's public C++/Mathematica, GPL — only the
// math is reproduced here) alternates three sub-steps to a fixed point:
//
//   1. λ  — joint placement for G²: each joint rides the segment between two
//           consecutive middle control points, blended by λ. λ has a closed
//           form in two triangle areas (`lambda`).
//   2. t  — the max-curvature parameter of each segment. Requiring the data
//           point to sit at the curvature maximum of the quadratic through the
//           current endpoints yields a cubic in t (`maxParamT`).
//   3. c₁ — the middle control points. With the joints written as λ-blends of
//           the middles, the interpolation constraints `cᵢ(tᵢ) = pᵢ` form one
//           tridiagonal system, solved per coordinate by a fixed Thomas sweep.
//
// Adaptation to the `Num` model: the library has no data-dependent control flow,
// so the iteration is **unrolled a fixed number of times** (`iterations`), just
// like the tridiagonal / Gaussian solves elsewhere are fixed sweeps. The cubic's
// discriminant branch (Cardano vs. trigonometric roots) and the λ degeneracy
// guard become branch-free `ifTruthyElse` selects with `safeSqrt`-guarded
// radicals, so the whole curve stays one differentiable graph of the inputs.
//
// 2D by nature: a quadratic Bézier is planar, so interpolating an arbitrary
// out-of-plane point at its curvature maximum is not generally solvable — the
// construction has no direct 3D analogue.
// ---------------------------------------------------------------------------

export interface KappaOptions {
  /**
   * Fixed number of local/global refinement passes (the unrolled fixed-point
   * iteration). Interpolation of the input points is exact at any count; more
   * passes converge the *max-curvature* placement (typically settled within a
   * handful — the reference uses 50, but the placement is essentially fixed by
   * ~10). Default 10.
   */
  iterations?: number;
}

const TWO_PI = Math.PI * 2;

/** Unsigned triangle area `½|(B−A)×(C−B)|`. */
function triArea(a: Point, b: Point, c: Point): Num {
  return a.vecTo(b).cross(b.vecTo(c)).abs().div(2);
}

/**
 * The G²-continuity blend for the joint between two consecutive segments,
 * `λ = √A₁ / (√A₁ + √A₂)` with `A₁ = area(c0,c1,c3)`, `A₂ = area(c1,c3,c4)`
 * (`c0` segment start, `c1`/`c3` the two middles, `c4` the next segment end).
 * Degenerate (collinear) neighbourhood ⇒ both areas vanish ⇒ default ½.
 */
function lambda(c0: Point, c1: Point, c3: Point, c4: Point): Num {
  const s1 = triArea(c0, c1, c3).safeSqrt();
  const s2 = triArea(c1, c3, c4).safeSqrt();
  const denom = s1.add(s2);
  return ifTruthyElse(denom.greaterThan(0), s1.div(denom), asNum(0.5));
}

/**
 * The parameter `t ∈ [0,1]` at which the quadratic Bézier through endpoints
 * `c0`, `c2` whose middle control point is fixed by interpolating `cp` would
 * place `cp` at its maximum curvature. Reduces to the cubic
 * `a·t³ + b·t² + c·t + d = 0`; solved branch-free via the depressed form
 * `u³ + p·u + q = 0`, selecting Cardano's single real root when the
 * discriminant `4p³+27q² ≥ 0` and the trigonometric three-root formula
 * (picking the root in `[0,1]`) otherwise. Radicals are `safeSqrt`-guarded so
 * the unselected branch never produces `NaN`.
 */
function maxParamT(c0: Point, cp: Point, c2: Point): Num {
  const v02 = c0.vecTo(c2); // c2 − c0
  const vp0 = cp.vecTo(c0); // c0 − cp
  const vp2 = cp.vecTo(c2); // c2 − cp

  const a = v02.dot(v02);
  const b = v02.dot(vp0).mul(3);
  // 3c0 − 2cp − c2 = 3(c0−cp) − (c2−cp)
  const c = vp0.scale(3).sub(vp2).dot(vp0);
  const d = vp0.dot(vp0).neg();

  const a2 = a.mul(a);
  const a3 = a2.mul(a);
  const p = a.mul(c).mul(3).sub(b.mul(b)).div(a2.mul(3));
  const q = b
    .mul(b)
    .mul(b)
    .mul(2)
    .sub(a.mul(b).mul(c).mul(9))
    .add(a2.mul(d).mul(27))
    .div(a3.mul(27));
  const shift = b.div(a.mul(3)); // u = t + b/(3a)
  const p3 = p.mul(p).mul(p);
  const disc = p3.mul(4).add(q.mul(q).mul(27));

  // Cardano (one real root): u = ∛(−q/2 + S) + ∛(−q/2 − S), S = √(q²/4 + p³/27).
  const halfNegQ = q.neg().div(2);
  const s = q.square().div(4).add(p3.div(27)).safeSqrt();
  const cardano = halfNegQ.add(s).cbrt().add(halfNegQ.sub(s).cbrt()).sub(shift);

  // Trigonometric (three real roots; here p < 0): pick the one in [0,1].
  const amp = p.neg().div(3).safeSqrt().mul(2); // 2√(−p/3)
  const acosArg = q
    .mul(3)
    .div(p.mul(2))
    .mul(asNum(-3).div(p).safeSqrt())
    .max(-1)
    .min(1);
  const phi = acosArg.acos().div(3);
  const root = (k: number): Num =>
    amp.mul(phi.sub((TWO_PI * k) / 3).cos()).sub(shift);
  const inRange = (t: Num): Num =>
    t.greaterThanOrEqual(0).and(t.lessThanOrEqual(1));
  const t0 = root(0);
  const t1 = root(1);
  const t2 = root(2);
  const trig = ifTruthyElse(inRange(t0), t0, ifTruthyElse(inRange(t1), t1, t2));

  return ifTruthyElse(disc.greaterThanOrEqual(0), cardano, trig).max(0).min(1);
}

/** Affine blend `(1−λ)·a + λ·b`. */
function blend(a: Point, b: Point, l: Num): Point {
  return a.add(a.vecTo(b).scale(l));
}

/**
 * Build an open κ-curve through `points` (≥ 2). The result is a 2D `Spline` of
 * quadratic Béziers; the end points are the curve's endpoints and every interior
 * point is interpolated at a local maximum of curvature. Query it with
 * `curve.at(u)` over the uniform knots `0 … n` (`n` = number of segments).
 */
export function buildKappa(
  points: Point[],
  options: KappaOptions = {},
): Spline<Point, Vec2> {
  const N = points.length;
  if (N < 2) {
    throw new Error(`buildKappa needs at least 2 points (got ${N})`);
  }

  // Two points: a single straight quadratic (collinear control points).
  if (N === 2) {
    const mid = points[0].midPoint(points[1]);
    return new Spline([0, 1], [new QuadraticSeg(points[0], mid, points[1])]);
  }

  // Three points: one segment with both endpoints pinned. t is fixed (it uses
  // only the data point and the fixed endpoints), so the middle is one shot:
  // 2(1−t)t·c₁ = p₁ − (1−t)²p₀ − t²p₂.
  if (N === 3) {
    const t = maxParamT(points[0], points[1], points[2]);
    const omt = asNum(1).sub(t);
    const inv = omt.mul(t).mul(2).inv();
    const coord = (k: "x" | "y"): Num =>
      points[1][k]
        .sub(omt.square().mul(points[0][k]))
        .sub(t.square().mul(points[2][k]))
        .mul(inv);
    const mid = new Point(coord("x"), coord("y"));
    return new Spline([0, 1], [new QuadraticSeg(points[0], mid, points[2])]);
  }

  // General case: n = N − 2 segments, segment i interpolating points[i+1].
  const n = N - 2;
  const iterations = options.iterations ?? 10;

  // Per-segment control points: start (c₀), middle (c₁), end (c₂). Seed the
  // middles at the data points and the joints at the chord midpoints; the two
  // outer ends are pinned to the first/last input point.
  const start: Point[] = [];
  const mid: Point[] = [];
  const end: Point[] = [];
  for (let i = 1; i <= n; i++) {
    start.push(points[i - 1].midPoint(points[i]));
    mid.push(points[i]);
    end.push(points[i].midPoint(points[i + 1]));
  }
  start[0] = points[0];
  end[n - 1] = points[N - 1];

  for (let iter = 0; iter < iterations; iter++) {
    const t = mid.map((_, i) => maxParamT(start[i], points[i + 1], end[i]));
    const ld: Num[] = [];
    for (let i = 0; i < n - 1; i++) {
      ld.push(lambda(start[i], mid[i], mid[i + 1], end[i + 1]));
    }

    // Tridiagonal interpolation system in the middle control points. Each row
    // sums to 1 (a convex combination of the three middles), with the two
    // pinned endpoints moved to the right-hand side. The unknowns are the
    // control points themselves: `Point` is a `NumStruct`, so the shared solver
    // eliminates over both coordinates at once.
    const sub: Num[] = new Array(n);
    const diag: Num[] = new Array(n);
    const sup: Num[] = new Array(n);
    const rhs: Point[] = new Array(n);

    const t0 = t[0];
    const o0 = asNum(1).sub(t0);
    diag[0] = o0.mul(t0).mul(2).add(asNum(1).sub(ld[0]).mul(t0.square()));
    sup[0] = ld[0].mul(t0.square());
    rhs[0] = new Point(
      points[1].x.sub(o0.square().mul(points[0].x)),
      points[1].y.sub(o0.square().mul(points[0].y)),
    );

    for (let i = 1; i < n - 1; i++) {
      const ti = t[i];
      const oi = asNum(1).sub(ti);
      const o2 = oi.square();
      const t2 = ti.square();
      sub[i] = asNum(1)
        .sub(ld[i - 1])
        .mul(o2);
      diag[i] = ld[i - 1]
        .mul(o2)
        .add(oi.mul(ti).mul(2))
        .add(asNum(1).sub(ld[i]).mul(t2));
      sup[i] = ld[i].mul(t2);
      rhs[i] = points[i + 1];
    }

    const tl = t[n - 1];
    const ol = asNum(1).sub(tl);
    const ol2 = ol.square();
    sub[n - 1] = asNum(1)
      .sub(ld[n - 2])
      .mul(ol2);
    diag[n - 1] = ld[n - 2].mul(ol2).add(ol.mul(tl).mul(2));
    rhs[n - 1] = new Point(
      points[n].x.sub(tl.square().mul(points[N - 1].x)),
      points[n].y.sub(tl.square().mul(points[N - 1].y)),
    );

    const corners = solveTridiagonal<Point>(sub, diag, sup, rhs);
    for (let i = 0; i < n; i++) mid[i] = corners[i];

    // Recompute the joints from the new middles and this pass's λ.
    for (let i = 0; i < n - 1; i++) {
      const joint = blend(mid[i], mid[i + 1], ld[i]);
      end[i] = joint;
      start[i + 1] = joint;
    }
  }

  const segments = mid.map((m, i) => new QuadraticSeg(start[i], m, end[i]));
  const knots = Array.from({ length: n + 1 }, (_, i) => i);
  return new Spline(knots, segments);
}
