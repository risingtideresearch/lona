import { type NumStruct } from "lona";
import { Spline, type CurvePoint, type CurveVec } from "../spline";
import { segmentSecants, splineFromHermite } from "./hermite";
import { buildKnots, type CurveParameterization } from "./parameterization";

/** @deprecated use {@link CurveParameterization}. */
export type CatmullRomParameterization = CurveParameterization;

export interface CatmullRomOptions {
  /**
   * 0 = standard Catmull-Rom, 1 = zero tangents (straight chords). Scales every
   * tangent by `1 - tension`.
   */
  tension?: number;
  /**
   * Knot spacing. `"uniform"` (default) uses segment index; `"centripetal"`
   * (`|ΔP|^½`) and `"chordal"` (`|ΔP|`) space knots by chord length, which tames
   * the overshoot and self-intersections uniform Catmull-Rom shows near sharp
   * turns. See {@link CurveParameterization}.
   */
  parameterization?: CurveParameterization;
}

/**
 * Catmull-Rom: a C¹ interpolating spline through `points`. Like the natural
 * spline, it decomposes the curve into knot intervals `h` and secants `δ`
 * (`segmentSecants`), then sets each per-point tangent — but the rule here is
 * *local*: each interior tangent is the two adjacent secants averaged with the
 * *opposite* interval as weight,
 * `mᵢ = (hᵢ₋₁·δᵢ + hᵢ·δᵢ₋₁) / (hᵢ₋₁ + hᵢ)` — the closed form of the non-uniform
 * Barry–Goldman tangent, reducing to the centred secant `(δᵢ + δᵢ₋₁)/2` for
 * uniform knots. The ends take the one-sided secant. (The natural spline, by
 * contrast, couples all tangents through one tridiagonal solve.)
 */
export function buildCatmullRom<
  P extends CurvePoint<P, V> & NumStruct<P>,
  V extends CurveVec<V> & NumStruct<V>,
>(points: P[], options: CatmullRomOptions = {}): Spline<P, V> {
  const n = points.length;
  if (n < 2) {
    throw new Error(`buildCatmullRom needs at least 2 points (got ${n})`);
  }
  const scale = 1 - (options.tension ?? 0);
  const knots = buildKnots(points, options.parameterization ?? "uniform");
  const { h, delta } = segmentSecants<P, V>(points, knots);

  const tangents: V[] = points.map((_, i) => {
    let m: V;
    if (i === 0) {
      m = delta[0]; // one-sided secant at the start
    } else if (i === n - 1) {
      m = delta[n - 2]; // one-sided secant at the end
    } else {
      const w = h[i - 1].add(h[i]).inv();
      m = delta[i].scale(h[i - 1].mul(w)).add(delta[i - 1].scale(h[i].mul(w)));
    }
    return m.scale(scale);
  });

  // C¹: the same tangent leaves and arrives at each point.
  return splineFromHermite(points, tangents, tangents, knots);
}
