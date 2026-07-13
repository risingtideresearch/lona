import { asNum, type Num, type NumStruct } from "lona";
import { CubicSeg, Spline, type CurvePoint, type CurveVec } from "../spline";

// ---------------------------------------------------------------------------
// Hermite → Bézier — the shared foundation every tangent-based builder emits
// through. A cubic Hermite piece on a knot interval of width `h`, with endpoints
// `p0`, `p1` and parameter-derivative tangents `m0` (leaving p0) and `m1`
// (arriving at p1), becomes the cubic Bézier:
//
//   c0 = p0,  c1 = p0 + (h/3)·m0,  c2 = p1 − (h/3)·m1,  c3 = p1
//
// Catmull-Rom, natural cubic, and pchip all differ only in how they compute the
// per-point tangents; the conversion to control points is this one place.
// ---------------------------------------------------------------------------

/**
 * Knot interval widths and secant velocities — the shared input prelude for
 * tangent-based builders. For each segment `i`: width `h[i] = knots[i+1] −
 * knots[i]` and secant `δ[i] = (P[i+1] − P[i]) / h[i]` (the average velocity
 * across the segment). Builders then derive per-point tangents from these —
 * locally (Catmull-Rom's weighted secant average) or by a global solve
 * (natural spline's tridiagonal system) — before emitting through the
 * Hermite→Bézier conversion below.
 */
export function segmentSecants<
  P extends CurvePoint<P, V>,
  V extends CurveVec<V>,
>(points: P[], knots: Num[]): { h: Num[]; delta: V[] } {
  const h: Num[] = [];
  const delta: V[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const width = knots[i + 1].sub(knots[i]);
    h.push(width);
    delta.push(points[i].vecTo(points[i + 1]).scale(width.inv()));
  }
  return { h, delta };
}

/** One cubic Bézier segment from Hermite data (endpoints + leaving/arriving tangents). */
export function bezierFromHermite<
  P extends CurvePoint<P, V>,
  V extends CurveVec<V>,
>(p0: P, p1: P, m0: V, m1: V, h: Num | number): CubicSeg<P, V> {
  const third = asNum(h).div(3);
  return new CubicSeg<P, V>(
    p0,
    p0.add(m0.scale(third)),
    p1.add(m1.scale(third.neg())),
    p1,
  );
}

/**
 * Assemble a `Spline` from points, per-point tangents, and knots.
 *
 * Tangents are given as two arrays so a builder can request a tangent *jump* at
 * a point (a knuckle): segment `i` leaves `points[i]` along `leaving[i]` and
 * arrives at `points[i+1]` along `arriving[i+1]`. For a smooth (C¹) curve pass
 * the same array for both.
 */
export function splineFromHermite<
  P extends CurvePoint<P, V> & NumStruct<P>,
  V extends CurveVec<V> & NumStruct<V>,
>(
  points: P[],
  leaving: V[],
  arriving: V[],
  knots: (Num | number)[],
): Spline<P, V> {
  const k = knots.map(asNum);
  const segments: CubicSeg<P, V>[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const h = k[i + 1].sub(k[i]);
    segments.push(
      bezierFromHermite<P, V>(
        points[i],
        points[i + 1],
        leaving[i],
        arriving[i + 1],
        h,
      ),
    );
  }
  return new Spline<P, V>(k, segments);
}
