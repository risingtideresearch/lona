import { asNum, type Num, type NumStruct } from "lona";
import {
  QuadraticSeg,
  Spline,
  type CurvePoint,
  type CurveVec,
  type Segment,
  type VecOf,
} from "../spline";
import { buildKnots, type CurveParameterization } from "./parameterization";

// ---------------------------------------------------------------------------
// Yuksel interpolating splines — Cem Yuksel, "A Class of C² Interpolating
// Splines" (ACM TOG 2020). Unlike the Bézier builders, a segment here is *not*
// a polynomial Bézier: each is a trigonometric blend of two overlapping local
// interpolants, so it gets its own `Segment` type (`YukselSeg`) rather than
// reusing `CubicSeg`/`QuadraticSeg`.
//
// For every interior point pᵢ there is a local function Fᵢ that interpolates the
// triple (pᵢ₋₁, pᵢ, pᵢ₊₁) — here the parabola through them (the paper also
// admits circular/elliptic interpolants). The curve over the interval
// [pⱼ, pⱼ₊₁] blends the two interpolants centred at its endpoints:
//
//   C(t) = cos²(πt/2)·Fⱼ(u) + sin²(πt/2)·Fⱼ₊₁(u),   t ∈ [0,1]
//
// At t=0 the blend is pure Fⱼ (which interpolates pⱼ); at t=1 pure Fⱼ₊₁ (which
// interpolates pⱼ₊₁) — so every point is interpolated. The trig weights have
// vanishing 1st/2nd derivatives at the ends, and at each junction the "other"
// interpolant also passes through the shared point (Fⱼ(uⱼ₊₁) = pⱼ₊₁), which is
// what makes the join C². The two boundary segments have only one neighbouring
// interpolant, so they use it on both sides (a blend of an interpolant with
// itself is that interpolant).
//
// Affine throughout (point ± vector, scale, scalar trig weights), so it runs in
// 2D and 3D unchanged. Everything stays a `Num`, so the curve is one
// differentiable graph of the input coordinates.
// ---------------------------------------------------------------------------

/**
 * The parabola through three points, parametrised by a global knot value `u`:
 * Lagrange interpolation on nodes `(a, b, c)` (the three knots) returning the
 * point at `b` (the centre) plus weighted offsets to its neighbours. Affine, so
 * it is generic over dimension.
 */
class LocalQuadratic<
  P extends CurvePoint<P, V>,
  V extends CurveVec<V> = VecOf<P>,
> {
  private readonly toPrev: V;
  private readonly toNext: V;

  constructor(
    private readonly center: P,
    private readonly a: Num, // knot of prev
    private readonly b: Num, // knot of center
    private readonly c: Num, // knot of next
    prev: P,
    next: P,
  ) {
    this.toPrev = center.vecTo(prev);
    this.toNext = center.vecTo(next);
  }

  /** `F(u) = center + L_prev(u)·(prev−center) + L_next(u)·(next−center)`. */
  at(u: Num): P {
    const { a, b, c } = this;
    const lPrev = u
      .sub(b)
      .mul(u.sub(c))
      .div(a.sub(b).mul(a.sub(c)));
    const lNext = u
      .sub(a)
      .mul(u.sub(b))
      .div(c.sub(a).mul(c.sub(b)));
    return this.center
      .add(this.toPrev.scale(lPrev))
      .add(this.toNext.scale(lNext));
  }

  /** `dF/du` — the Lagrange basis derivatives applied to the same offsets. */
  tangent(u: Num): V {
    const { a, b, c } = this;
    const dPrev = u
      .mul(2)
      .sub(b)
      .sub(c)
      .div(a.sub(b).mul(a.sub(c)));
    const dNext = u
      .mul(2)
      .sub(a)
      .sub(b)
      .div(c.sub(a).mul(c.sub(b)));
    return this.toPrev.scale(dPrev).add(this.toNext.scale(dNext));
  }
}

/** Weight on the right interpolant: `sin²(πt/2)` (so the left gets `cos²`). */
function blendWeight(t: Num): Num {
  const s = t.mul(Math.PI / 2).sin();
  return s.mul(s);
}

/**
 * One Yuksel segment: the trigonometric blend of a `left` and `right` local
 * interpolant over the knot interval `[u0, u0+h]`. The local parameter `t ∈
 * [0,1]` maps to the global knot `u = u0 + h·t`; both interpolants are evaluated
 * there and blended by `cos²/sin²(πt/2)`.
 */
export class YukselSeg<
  P extends CurvePoint<P, V>,
  V extends CurveVec<V> = VecOf<P>,
> implements Segment<P, V> {
  constructor(
    private readonly u0: Num,
    private readonly h: Num,
    private readonly left: LocalQuadratic<P, V>,
    private readonly right: LocalQuadratic<P, V>,
  ) {}

  private knot(t: Num): Num {
    return this.u0.add(this.h.mul(t));
  }

  /** Point at local parameter `t ∈ [0,1]` (`t` may be a symbolic `Num`). */
  at(t: Num | number): P {
    const tt = asNum(t);
    const u = this.knot(tt);
    const a = this.left.at(u);
    const b = this.right.at(u);
    // (1−w)·a + w·b, affine.
    return a.add(a.vecTo(b).scale(blendWeight(tt)));
  }

  /**
   * Tangent w.r.t. the *local* parameter `t` (like the Bézier segments; scale by
   * `1/Δknot` for the global derivative). With `A=left(u)`, `B=right(u)`,
   * `w=sin²(πt/2)`: `C(t) = A + w·(B−A)`, so
   * `C'(t) = A' + w'·(B−A) + w·(B'−A')`, where `A'/B'` carry the `du/dt = h`
   * chain factor and `w' = (π/2)·sin(πt)`.
   */
  tangentAt(t: Num | number): V {
    const tt = asNum(t);
    const u = this.knot(tt);
    const a = this.left.at(u);
    const b = this.right.at(u);
    const aPrime = this.left.tangent(u).scale(this.h);
    const bPrime = this.right.tangent(u).scale(this.h);
    const w = blendWeight(tt);
    const wPrime = tt
      .mul(Math.PI)
      .sin()
      .mul(Math.PI / 2);
    return aPrime
      .add(a.vecTo(b).scale(wPrime))
      .add(bPrime.sub(aPrime).scale(w));
  }
}

export interface YukselOptions {
  /**
   * Knot spacing — see {@link CurveParameterization}. `"uniform"` (default) uses
   * the point index; `"centripetal"`/`"chordal"` space the knots by chord
   * length, which tames overshoot when the points are unevenly spaced.
   */
  parameterization?: CurveParameterization;
}

/**
 * Build a Yuksel C² interpolating spline through `points` (≥ 2). The result is a
 * `Spline` of {@link YukselSeg} pieces interpolating every point; query it with
 * `curve.at(u)` over the knots (`0 … n−1` for uniform spacing). Generic over 2D
 * and 3D.
 */
export function buildYuksel<
  P extends CurvePoint<P, V> & NumStruct<P>,
  V extends CurveVec<V> & NumStruct<V> = VecOf<P>,
>(points: P[], options: YukselOptions = {}): Spline<P, V> {
  const n = points.length;
  if (n < 2) {
    throw new Error(`buildYuksel needs at least 2 points (got ${n})`);
  }

  const knots = buildKnots(points, options.parameterization ?? "uniform");

  // Two points: a single straight segment (no triple to interpolate).
  if (n === 2) {
    const mid = points[0].add(points[0].vecTo(points[1]).scale(0.5));
    return new Spline<P, V>(knots, [
      new QuadraticSeg<P, V>(points[0], mid, points[1]),
    ]);
  }

  // Local interpolant centred at each interior point pᵢ (i = 1 … n−2).
  const interp: LocalQuadratic<P, V>[] = new Array(n);
  for (let i = 1; i <= n - 2; i++) {
    interp[i] = new LocalQuadratic<P, V>(
      points[i],
      knots[i - 1],
      knots[i],
      knots[i + 1],
      points[i - 1],
      points[i + 1],
    );
  }

  // Segment j spans [pⱼ, pⱼ₊₁], blending the interpolants centred at its ends.
  // The boundary segments fall back to their single available interpolant on
  // both sides (blending it with itself reproduces it exactly).
  const segments: Segment<P, V>[] = [];
  for (let j = 0; j < n - 1; j++) {
    const left = interp[j] ?? interp[j + 1];
    const right = interp[j + 1] ?? interp[j];
    segments.push(
      new YukselSeg<P, V>(knots[j], knots[j + 1].sub(knots[j]), left, right),
    );
  }

  return new Spline<P, V>(knots, segments);
}
