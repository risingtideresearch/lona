import { asNum, type Num, type NumStruct } from "lona";
import { Spline, type CurvePoint, type CurveVec } from "../spline";
import { solveTridiagonal } from "../utils/tridiagonal";
import { segmentSecants, splineFromHermite } from "./hermite";
import { buildKnots, type CurveParameterization } from "./parameterization";

// ---------------------------------------------------------------------------
// Natural cubic spline — the C² interpolant with zero second derivative at the
// ends ("natural" boundary). Unlike Catmull-Rom (a local C¹ rule), the tangents
// here are coupled: requiring the second derivative to match across every
// interior knot yields one tridiagonal system, solved once for the per-point
// tangents. With those tangents the curve goes out through the shared
// Hermite→Bézier conversion like every other tangent-based builder.
//
// The system is diagonally dominant, so the Thomas (tridiagonal) elimination
// below is stable and — being a fixed forward/back sweep with no pivoting —
// stays entirely within `Num` arithmetic: coefficients are `Num`s built from
// the (possibly symbolic) knot spacing, unknowns are vectors `V`. Affine, so it
// runs unchanged in 2D and 3D.
// ---------------------------------------------------------------------------

export interface NaturalOptions {
  /**
   * Knot spacing — see {@link CurveParameterization}. `"uniform"` (default) uses
   * the segment index; `"centripetal"`/`"chordal"` space the parameter by chord
   * length, which reduces overshoot when the points are unevenly spaced.
   */
  parameterization?: CurveParameterization;
}

/**
 * Build a natural cubic spline through `points` (C², zero end curvature).
 *
 * Tangents `mᵢ` (derivatives w.r.t. the global parameter) solve, per the C²
 * condition on a knot interval of width `hᵢ = tᵢ₊₁ − tᵢ` with secant
 * `δᵢ = (Pᵢ₊₁ − Pᵢ)/hᵢ`:
 *
 *   interior i: `mᵢ₋₁/hᵢ₋₁ + 2(1/hᵢ₋₁ + 1/hᵢ)·mᵢ + mᵢ₊₁/hᵢ = 3(δᵢ₋₁/hᵢ₋₁ + δᵢ/hᵢ)`
 *   ends (natural): `2m₀ + m₁ = 3δ₀` and `mₙ₋₂ + 2mₙ₋₁ = 3δₙ₋₂`
 *
 * Generic over 2D/3D points.
 */
export function buildNatural<
  P extends CurvePoint<P, V> & NumStruct<P>,
  V extends CurveVec<V> & NumStruct<V>,
>(points: P[], options: NaturalOptions = {}): Spline<P, V> {
  const n = points.length;
  if (n < 2) {
    throw new Error(`buildNatural needs at least 2 points (got ${n})`);
  }
  const knots = buildKnots(points, options.parameterization ?? "uniform");
  // Interval widths h[i] = knots[i+1] − knots[i] and secants δ[i] = ΔP / h[i].
  const { h, delta } = segmentSecants<P, V>(points, knots);

  // Tridiagonal rows (a·mᵢ₋₁ + b·mᵢ + c·mᵢ₊₁ = r): scalar a/b/c, vector r.
  const a: Num[] = new Array(n);
  const b: Num[] = new Array(n);
  const c: Num[] = new Array(n);
  const r: V[] = new Array(n);

  // First row — natural end: 2·m₀ + m₁ = 3·δ₀.
  b[0] = asNum(2);
  c[0] = asNum(1);
  r[0] = delta[0].scale(3);
  // Interior rows.
  for (let i = 1; i < n - 1; i++) {
    const invPrev = h[i - 1].inv();
    const invNext = h[i].inv();
    a[i] = invPrev;
    b[i] = invPrev.add(invNext).mul(2);
    c[i] = invNext;
    r[i] = delta[i - 1].scale(invPrev).add(delta[i].scale(invNext)).scale(3);
  }
  // Last row — natural end: mₙ₋₂ + 2·mₙ₋₁ = 3·δₙ₋₂.
  a[n - 1] = asNum(1);
  b[n - 1] = asNum(2);
  r[n - 1] = delta[n - 2].scale(3);

  // One tridiagonal solve for the per-point tangents (scalar coefficients,
  // vector unknowns).
  const tangents = solveTridiagonal<V>(a, b, c, r);

  // C²: the same tangent leaves and arrives at each point.
  return splineFromHermite(points, tangents, tangents, knots);
}
