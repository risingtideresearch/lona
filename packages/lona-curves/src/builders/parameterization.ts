import { asNum, type Num, type NumStruct } from "lona";

// ---------------------------------------------------------------------------
// Knot parameterization — how a builder spaces the parameter values at the
// control points. Shared by every interpolating builder whose knots are not
// dictated by the data (Catmull-Rom, natural spline, …). `uniform` uses the
// segment index; `centripetal` (`|ΔP|^½`) and `chordal` (`|ΔP|`) space knots by
// chord length, which tames the overshoot uniform spacing shows near sharp
// turns. Chord lengths stay symbolic `Num`s, so the parameterization is itself
// a `Num` and tracks the (possibly symbolic) point positions.
// ---------------------------------------------------------------------------

/** Knot spacing (the exponent α on chord length: 0, ½, or 1). */
export type CurveParameterization = "uniform" | "centripetal" | "chordal";

/** Euclidean chord length between two points, generic via their components. */
export function chordLength<P extends NumStruct<P>>(a: P, b: P): Num {
  const an = a.toNums();
  const bn = b.toNums();
  let sumSq: Num = asNum(0);
  for (let i = 0; i < an.length; i++) {
    const d = an[i].sub(bn[i]);
    sumSq = sumSq.add(d.mul(d));
  }
  return sumSq.sqrt();
}

/** Cumulative (symbolic) knot vector for the chosen parameterization. */
export function buildKnots<P extends NumStruct<P>>(
  points: P[],
  parameterization: CurveParameterization,
): Num[] {
  if (parameterization === "uniform") return points.map((_, i) => asNum(i));
  const knots: Num[] = [asNum(0)];
  for (let i = 0; i < points.length - 1; i++) {
    const len = chordLength(points[i], points[i + 1]);
    // chordal: |ΔP|;  centripetal: |ΔP|^½
    const dt = parameterization === "chordal" ? len : len.sqrt();
    knots.push(knots[i].add(dt));
  }
  return knots;
}
