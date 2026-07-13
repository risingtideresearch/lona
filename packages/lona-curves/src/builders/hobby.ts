import { asNum, atan2, type Num } from "lona";
import { Point, Vec2, angleRromRad } from "lona-geom";
import { CubicSeg, Spline } from "../spline";

// ---------------------------------------------------------------------------
// Hobby spline (open, 2D) — John Hobby's interpolating curve from MetaPost.
//
// The shape is chosen by giving each knot a tangent *direction* so that a
// linearised ("mock") curvature is continuous across knots; the Bézier handle
// *lengths* then come from Hobby's velocity function. Unlike the affine
// builders this is genuinely 2D: it works with signed turning angles (`ψ`),
// departure/arrival angles (`θ`/`φ`) and rotations, none of which have a
// canonical signed analogue in 3D.
//
// This is the default-tension (τ=1), default-curl, non-cyclic case — the
// well-behaved core. The turning angles and the θ-system are solved in radians
// (`atan2`/`sin`/`cos` are differentiable `Num` ops), then converted to the
// cos/sin `Angle` algebra to place the handles. Everything stays a `Num`, so
// the whole curve is one differentiable graph of the input coordinates.
//
// The math follows MetaPost's `mp_solve_choices` (public domain), specialised
// to τ=1: aa = bb = ½, the chord factor (3 − 1/τ) = 2, and the curl ratio = 1.
// ---------------------------------------------------------------------------

const SQRT2 = Math.SQRT2;
const SQRT5 = Math.sqrt(5);

/**
 * MetaPost's velocity: the handle length (as a fraction of the chord) for a
 * Bézier leaving at angle `θ` and arriving at angle `φ`, given the partner
 * angle's sin/cos. Capped at 4 to avoid runaway handles on near-degenerate
 * turns.
 */
function velocity(st: Num, ct: Num, sf: Num, cf: Num): Num {
  const num = st
    .sub(sf.div(16))
    .mul(sf.sub(st.div(16)))
    .mul(ct.sub(cf))
    .mul(SQRT2)
    .add(2);
  const den = ct
    .mul(SQRT5 - 1)
    .add(cf.mul(3 - SQRT5))
    .add(2)
    .mul(1.5);
  return num.div(den).min(4);
}

/**
 * Build an open Hobby spline through `points` (≥ 2). The result is a 2D
 * `Spline` of cubic Béziers interpolating every point with a smooth,
 * "mock-curvature continuous" shape; query it with `curve.at(u)` over the
 * uniform knots `0 … n−1`.
 */
export function buildHobby(points: Point[]): Spline<Point, Vec2> {
  const n = points.length;
  if (n < 2) {
    throw new Error(`buildHobby needs at least 2 points (got ${n})`);
  }

  // Chord vectors and lengths between consecutive points.
  const chord: Vec2[] = [];
  const d: Num[] = [];
  for (let i = 0; i < n - 1; i++) {
    const v = points[i].vecTo(points[i + 1]);
    chord.push(v);
    d.push(v.norm());
  }

  const knots = points.map((_, i) => i);

  // Two points: a single straight-chord Bézier (the θ-system is degenerate).
  if (n === 2) {
    const handle = chord[0].scale(asNum(1).div(3));
    const seg = new CubicSeg(
      points[0],
      points[0].add(handle),
      points[1].sub(handle),
      points[1],
    );
    return new Spline(knots, [seg]);
  }

  // Turning angle ψ at each interior knot (signed angle from chord i−1 to i);
  // the ends have no turn.
  const psi: Num[] = points.map(() => asNum(0));
  for (let i = 1; i <= n - 2; i++) {
    const prev = chord[i - 1];
    const next = chord[i];
    psi[i] = atan2(prev.cross(next), prev.dot(next));
  }

  // Solve the tridiagonal θ-system by Gaussian elimination (MetaPost's sweep).
  // uu/vv are the eliminated coefficient and right-hand side at each knot.
  const uu: Num[] = new Array(n);
  const vv: Num[] = new Array(n);
  const theta: Num[] = new Array(n);

  // First knot — default curl ⇒ uu₀ = 1, vv₀ = −ψ₁.
  uu[0] = asNum(1);
  vv[0] = psi[1].neg();

  for (let i = 1; i < n - 1; i++) {
    // τ=1 constants: aa = bb = ½, chord factor = 2.
    const dd = d[i].mul(2);
    const ee = d[i - 1].mul(2);
    const cc = asNum(1).sub(uu[i - 1].mul(0.5));
    const ff = ee.div(ee.add(dd.mul(cc)));
    uu[i] = ff.mul(0.5);
    const ffPrev = asNum(1).sub(ff).div(cc);
    const acc = psi[i + 1].mul(uu[i]).neg().sub(psi[i].mul(ffPrev));
    vv[i] = acc.sub(vv[i - 1].mul(ffPrev.mul(0.5)));
  }

  // Last knot — default curl ⇒ θₙ₋₁ = −vvₙ₋₂ / (1 − uuₙ₋₂).
  theta[n - 1] = vv[n - 2].div(asNum(1).sub(uu[n - 2])).neg();
  for (let i = n - 2; i >= 0; i--) {
    theta[i] = vv[i].sub(theta[i + 1].mul(uu[i]));
  }

  // φ at each knot follows from θ + φ + ψ = 0.
  const phi = psi.map((p, i) => p.neg().sub(theta[i]));

  // Place the two handles of each segment from its departure/arrival angles.
  const segments: CubicSeg<Point, Vec2>[] = [];
  for (let i = 0; i < n - 1; i++) {
    const dep = angleRromRad(theta[i]);
    const arr = angleRromRad(phi[i + 1]);
    const velOut = velocity(dep.sin(), dep.cos(), arr.sin(), arr.cos());
    const velIn = velocity(arr.sin(), arr.cos(), dep.sin(), dep.cos());
    const c1 = points[i].add(chord[i].rotate(dep).scale(velOut));
    const c2 = points[i + 1].sub(chord[i].rotate(arr.neg()).scale(velIn));
    segments.push(new CubicSeg(points[i], c1, c2, points[i + 1]));
  }

  return new Spline(knots, segments);
}
