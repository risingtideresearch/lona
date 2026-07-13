import { asNum, ifTruthyElse, type Num } from "lona";
import { Point, Vec2 } from "lona-geom";
import { QuadraticSeg, Spline, type Segment } from "../spline";

// ---------------------------------------------------------------------------
// Jiang & Chen, "G² Interpolating Spline with Local Maximum Curvature"
// (ACM TOG 2025).
//
// This implementation follows the paper's construction much more closely than
// the original demo prototype:
//
//   • each data point pᵢ owns a quadratic interpolation function Fᵢ whose
//     endpoints (the paper's F-points) are placed along the
//     curvature-parameterized parabola from §4.2;
//   • open curves are handled by reflected virtual end points (§4.4);
//   • a segment Cᵢ blends Fᵢ and Fᵢ₊₁ as in Eq. (1), using the reference
//     implementation's parametric quartic blend (x(t), y(t)):
//       Cᵢ(t) = y(t) Fᵢ(Bᵢ + (1−Bᵢ)x(t))
//             + (1−y(t)) Fᵢ₊₁(Bᵢ₊₁ x(t));
//   • the blend endpoint parameters are the MATLAB reference implementation's
//     closed-form `getab` solve, i.e. Eq. (13)'s quadratic LMC constraints plus
//     the G² curvature-matching ratio.
//
// Deliberate adaptation: the public MATLAB code uses data-dependent flat-point
// indexing and mutable vector assignments. This TypeScript port keeps the same
// non-flat formulas but avoids data-dependent topology changes so symbolic Num
// graphs remain reusable while dragging points in the demo.
// ---------------------------------------------------------------------------

const EPS = 1e-6;
const BLEND_EPS = 1e-3;
const F_POINT_SCALE = 0.83;
const TWO_PI = Math.PI * 2;

export interface JiangChenOptions {
  /**
   * Per-point curvature magnitude αᵢ from §4.2. A single number applies to all
   * points; an array gives one per point. Omit for αᵢ = 1. Larger values sharpen
   * the local feature around pᵢ; αᵢ = 0 gives the base κ-parabola tangent frame.
   */
  curvature?: number | number[];
}

/** A quadratic Bézier interpolation function Fᵢ. */
class QuadraticFunction {
  constructor(
    readonly c0: Point,
    readonly c1: Point,
    readonly c2: Point,
    readonly b: Num,
  ) {}

  at(t: Num | number): Point {
    const tt = asNum(t);
    const omt = asNum(1).sub(tt);
    return pointLin(
      this.c0,
      this.c1,
      this.c2,
      omt.square(),
      omt.mul(tt).mul(2),
      tt.square(),
    );
  }

  tangentAt(t: Num | number): Vec2 {
    const tt = asNum(t);
    const omt = asNum(1).sub(tt);
    return this.c0
      .vecTo(this.c1)
      .scale(omt.mul(2))
      .add(this.c1.vecTo(this.c2).scale(tt.mul(2)));
  }

  second(): Vec2 {
    return this.c1.vecTo(this.c2).scale(2).sub(this.c0.vecTo(this.c1).scale(2));
  }
}

/** One Jiang-Chen blended segment over local t ∈ [0, 1]. */
export class JiangChenSeg implements Segment<Point, Vec2> {
  constructor(
    private readonly left: QuadraticFunction,
    private readonly right: QuadraticFunction,
    private readonly leftBlend: Num,
    private readonly rightBlend: Num,
  ) {}

  at(t: Num | number): Point {
    const tt = asNum(t);
    const x = blendX(tt, this.leftBlend, this.rightBlend);
    const y = blendY(tt);
    const a = this.left.at(this.left.b.add(asNum(1).sub(this.left.b).mul(x)));
    const b = this.right.at(this.right.b.mul(x));
    return a.add(a.vecTo(b).scale(asNum(1).sub(y)));
  }

  tangentAt(t: Num | number): Vec2 {
    const tt = asNum(t);
    const x = blendX(tt, this.leftBlend, this.rightBlend);
    const xPrime = blendXDerivative(tt, this.leftBlend, this.rightBlend);
    const y = blendY(tt);
    const yPrime = blendYDerivative(tt);
    const uL = this.left.b.add(asNum(1).sub(this.left.b).mul(x));
    const uR = this.right.b.mul(x);
    const a = this.left.at(uL);
    const b = this.right.at(uR);
    const aPrime = this.left
      .tangentAt(uL)
      .scale(asNum(1).sub(this.left.b).mul(xPrime));
    const bPrime = this.right.tangentAt(uR).scale(this.right.b.mul(xPrime));

    // C = y A + (1−y) B = A + (1−y)(B−A)
    return aPrime
      .add(bPrime.sub(aPrime).scale(asNum(1).sub(y)))
      .add(a.vecTo(b).scale(yPrime.neg()));
  }
}

interface LocalFrame {
  point: Point;
  tangent: Vec2;
  second: Vec2;
}

function pointLin(
  a: Point,
  b: Point,
  c: Point,
  wa: Num,
  wb: Num,
  wc: Num,
): Point {
  return new Point(
    a.x.mul(wa).add(b.x.mul(wb)).add(c.x.mul(wc)),
    a.y.mul(wa).add(b.y.mul(wb)).add(c.y.mul(wc)),
  );
}

/** κ-curve maximum-curvature parameter for a quadratic through c0/cp/c2. */
function maxParamT(c0: Point, cp: Point, c2: Point): Num {
  const v02 = c0.vecTo(c2);
  const vp0 = cp.vecTo(c0);
  const vp2 = cp.vecTo(c2);

  const a = v02.dot(v02);
  const b = v02.dot(vp0).mul(3);
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
  const shift = b.div(a.mul(3));
  const p3 = p.mul(p).mul(p);
  const disc = p3.mul(4).add(q.mul(q).mul(27));

  const halfNegQ = q.neg().div(2);
  const s = q.square().div(4).add(p3.div(27)).safeSqrt();
  const cardano = halfNegQ.add(s).cbrt().add(halfNegQ.sub(s).cbrt()).sub(shift);

  const amp = p.neg().div(3).safeSqrt().mul(2);
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

function localFrame(prev: Point, point: Point, next: Point): LocalFrame {
  const b = maxParamT(prev, point, next);
  const omt = asNum(1).sub(b);
  const inv = omt.mul(b).mul(2).inv();
  const mid = new Point(
    point.x.sub(omt.square().mul(prev.x)).sub(b.square().mul(next.x)).mul(inv),
    point.y.sub(omt.square().mul(prev.y)).sub(b.square().mul(next.y)).mul(inv),
  );
  return {
    point,
    tangent: new QuadraticFunction(prev, mid, next, b).tangentAt(b),
    second: mid.vecTo(next).scale(2).sub(prev.vecTo(mid).scale(2)),
  };
}

function tildeAt(frame: LocalFrame, alpha: Num, d: Num): Point {
  return frame.point
    .add(frame.tangent.scale(d))
    .add(frame.second.scale(alpha.mul(0.5).mul(d.square())));
}

function signedBoundaryRoot(
  frame: LocalFrame,
  alpha: Num,
  target: Point,
  boundaryDirection: Vec2,
  wantPositive: boolean,
): Num {
  // Solve <p + dT + α/2 d²A − target, boundaryDirection> = 0.
  const a = frame.second.dot(boundaryDirection).mul(alpha).mul(0.5);
  const b = frame.tangent.dot(boundaryDirection);
  const c = target.vecTo(frame.point).dot(boundaryDirection);
  const disc = b.square().sub(a.mul(c).mul(4)).safeSqrt();
  const denom = a.mul(2);
  const r1 = b.neg().add(disc).div(denom);
  const r2 = b.neg().sub(disc).div(denom);
  const linear = c.neg().div(b);
  const useQuadratic = a.abs().greaterThan(EPS);
  const qr1 = ifTruthyElse(useQuadratic, r1, linear);
  const qr2 = ifTruthyElse(useQuadratic, r2, linear);

  if (wantPositive) {
    const big = asNum(1e6);
    const c1 = ifTruthyElse(qr1.greaterThan(EPS), qr1, big);
    const c2 = ifTruthyElse(qr2.greaterThan(EPS), qr2, big);
    const chosen = c1.min(c2);
    return ifTruthyElse(chosen.lessThan(1e5), chosen, asNum(1));
  }

  const negBig = asNum(-1e6);
  const c1 = ifTruthyElse(qr1.lessThan(-EPS), qr1, negBig);
  const c2 = ifTruthyElse(qr2.lessThan(-EPS), qr2, negBig);
  const chosen = c1.max(c2);
  return ifTruthyElse(chosen.greaterThan(-1e5), chosen, asNum(-1));
}

function buildFunction(
  frames: LocalFrame[],
  points: Point[],
  index: number,
  alpha: Num,
): QuadraticFunction {
  const frame = frames[index];
  const prevFrame = frames[Math.max(0, index - 1)];
  const nextFrame = frames[Math.min(frames.length - 1, index + 1)];
  const prev = points[Math.max(0, index - 1)];
  const next = points[Math.min(points.length - 1, index + 1)];

  const d0a = signedBoundaryRoot(frame, alpha, prev, prevFrame.tangent, false);
  const d0b = signedBoundaryRoot(frame, alpha, prev, prevFrame.second, false);
  const d2a = signedBoundaryRoot(frame, alpha, next, nextFrame.tangent, true);
  const d2b = signedBoundaryRoot(frame, alpha, next, nextFrame.second, true);
  const d0 = d0a.max(d0b).mul(F_POINT_SCALE);
  const d2 = d2a.min(d2b).mul(F_POINT_SCALE);

  const b = d0
    .neg()
    .div(d2.sub(d0))
    .max(EPS)
    .min(1 - EPS);
  const c0 = tildeAt(frame, alpha, d0);
  const c2 = tildeAt(frame, alpha, d2);
  const omt = asNum(1).sub(b);
  const inv = omt.mul(b).mul(2).inv();
  const c1 = new Point(
    frame.point.x
      .sub(omt.square().mul(c0.x))
      .sub(b.square().mul(c2.x))
      .mul(inv),
    frame.point.y
      .sub(omt.square().mul(c0.y))
      .sub(b.square().mul(c2.y))
      .mul(inv),
  );
  return new QuadraticFunction(c0, c1, c2, b);
}

function projection(v: Vec2, basis: Vec2): Num {
  return v.dot(basis).div(basis.dot(basis).max(EPS));
}

function solveBlendQuadratic(a: Num, b: Num, c: Num): Num {
  // Reference solveQuadpoly: (-b + sqrt(b² - 4ac)) / (2a), with the
  // infeasible/degenerate case falling back to 2, so 1/root becomes 0.5.
  const disc = b.square().sub(a.mul(c).mul(4));
  const root = b.neg().add(disc.safeSqrt()).div(a.mul(2));
  const valid = a.abs().greaterThan(EPS).and(disc.greaterThanOrEqual(0));
  return ifTruthyElse(valid.and(root.greaterThan(EPS)), root, asNum(2));
}

function getBlendParams(functions: QuadraticFunction[]): [Num, Num][] {
  const k = functions.length;
  const params: [Num, Num][] = [];
  for (let i = 0; i < k; i++) {
    const f = functions[i];
    const next = functions[(i + 1) % k];
    const prev = functions[(i + k - 1) % k];
    const p = f.at(f.b);
    const n = f.second();
    const t = f.tangentAt(f.b);
    const s = f.b;
    const sNext = next.b;
    const sPrev = prev.b;
    const oneMinusS = asNum(1).sub(s);
    const oneMinusSPrev = asNum(1).sub(sPrev);

    const b0 = projection(p.vecTo(next.c1), n);
    const b2 = projection(p.vecTo(prev.c1), n);
    const d0 = projection(p.vecTo(next.c0), n);
    const d2 = projection(p.vecTo(prev.c2), n);
    const a0 = projection(p.vecTo(next.c0), t);
    const a2 = projection(p.vecTo(prev.c2), t);

    const rootA = solveBlendQuadratic(
      d0
        .mul(3 / 8)
        .div(sNext)
        .mul(asNum(1).add(a0.div(oneMinusS))),
      d0.mul(-4).div(3).div(sNext),
      oneMinusS
        .div(sNext)
        .mul(a0)
        .add(d0.sub(b0).mul(2))
        .sub(oneMinusS.div(sNext)),
    );
    const rootB = solveBlendQuadratic(
      d2
        .mul(3 / 8)
        .div(oneMinusSPrev)
        .mul(asNum(1).sub(a2.div(s))),
      d2.mul(-4).div(3).div(oneMinusSPrev),
      s
        .neg()
        .div(oneMinusSPrev)
        .mul(a2)
        .add(d2.sub(b2).mul(2))
        .sub(s.div(oneMinusSPrev)),
    );

    let a = rootA.inv().min(0.5).max(BLEND_EPS);
    let b = rootB.inv().min(0.5).max(BLEND_EPS);

    const leftScore = d0.div(oneMinusS.square()).div(a.square());
    const rightScore = d2.div(s.square()).div(b.square());
    const adjustedA = s
      .div(oneMinusS)
      .mul(d0.div(d2.max(EPS)).safeSqrt())
      .mul(b)
      .min(0.5)
      .max(BLEND_EPS);
    const adjustedB = oneMinusS
      .div(s)
      .mul(d2.div(d0.max(EPS)).safeSqrt())
      .mul(a)
      .min(0.5)
      .max(BLEND_EPS);
    const adjustB = leftScore.greaterThan(rightScore);
    a = ifTruthyElse(adjustB, a, adjustedA);
    b = ifTruthyElse(adjustB, adjustedB, b);

    params.push([a, b]);
  }
  return params;
}

function quarticBezier(
  t: Num,
  p0: Num | number,
  p1: Num | number,
  p2: Num | number,
  p3: Num | number,
  p4: Num | number,
): Num {
  const omt = asNum(1).sub(t);
  return asNum(p0)
    .mul(omt.powi(4))
    .add(asNum(p1).mul(omt.powi(3)).mul(t).mul(4))
    .add(asNum(p2).mul(omt.square()).mul(t.square()).mul(6))
    .add(asNum(p3).mul(omt).mul(t.powi(3)).mul(4))
    .add(asNum(p4).mul(t.powi(4)));
}

function cubicBezierDerivative(
  t: Num,
  p0: Num | number,
  p1: Num | number,
  p2: Num | number,
  p3: Num | number,
  p4: Num | number,
): Num {
  const omt = asNum(1).sub(t);
  return asNum(p1)
    .sub(p0)
    .mul(omt.powi(3))
    .add(asNum(p2).sub(p1).mul(omt.square()).mul(t).mul(3))
    .add(asNum(p3).sub(p2).mul(omt).mul(t.square()).mul(3))
    .add(asNum(p4).sub(p3).mul(t.powi(3)))
    .mul(4);
}

function blendX(t: Num, left: Num, right: Num): Num {
  return quarticBezier(t, 0, left, 0.5, asNum(1).sub(right), 1);
}

function blendXDerivative(t: Num, left: Num, right: Num): Num {
  return cubicBezierDerivative(t, 0, left, 0.5, asNum(1).sub(right), 1);
}

function blendY(t: Num): Num {
  return quarticBezier(t, 1, 1, 0.5, 0, 0);
}

function blendYDerivative(t: Num): Num {
  return cubicBezierDerivative(t, 1, 1, 0.5, 0, 0);
}

function reflectedVirtualStart(points: Point[]): Point {
  return points[0].sub(points[0].vecTo(points[1]));
}

function reflectedVirtualEnd(points: Point[]): Point {
  const n = points.length;
  return points[n - 1].add(points[n - 2].vecTo(points[n - 1]));
}

/**
 * Build an open Jiang-Chen style spline through `points` (≥ 2). The result is a
 * 2D `Spline` of blended quadratic interpolation functions over knots 0…n−1.
 */
export function buildJiangChen(
  points: Point[],
  options: JiangChenOptions = {},
): Spline<Point, Vec2> {
  const n = points.length;
  if (n < 2) {
    throw new Error(`buildJiangChen needs at least 2 points (got ${n})`);
  }

  if (n === 2) {
    const mid = points[0].midPoint(points[1]);
    return new Spline([0, 1], [new QuadraticSeg(points[0], mid, points[1])]);
  }

  const extended = [
    reflectedVirtualStart(points),
    ...points,
    reflectedVirtualEnd(points),
  ];
  const frames = extended.map((point, i) => {
    if (i === 0) {
      return localFrame(
        point.sub(point.vecTo(extended[1])),
        point,
        extended[1],
      );
    }
    if (i === extended.length - 1) {
      return localFrame(
        extended[i - 1],
        point,
        point.add(extended[i - 1].vecTo(point)),
      );
    }
    return localFrame(extended[i - 1], point, extended[i + 1]);
  });

  const alphaAt = (i: number): Num => {
    const c = options.curvature;
    if (Array.isArray(c)) return asNum(c[i - 1] ?? 1).max(0);
    if (c !== undefined) return asNum(c).max(0);
    return asNum(1);
  };

  const functions = extended.map((_, i) =>
    buildFunction(frames, extended, i, alphaAt(i)),
  );
  const params = getBlendParams(functions);

  const segments: Segment<Point, Vec2>[] = [];
  for (let i = 0; i < n - 1; i++) {
    const leftIndex = i + 1;
    const rightIndex = i + 2;
    segments.push(
      new JiangChenSeg(
        functions[leftIndex],
        functions[rightIndex],
        params[leftIndex][0],
        params[rightIndex][1],
      ),
    );
  }

  return new Spline(
    Array.from({ length: n }, (_, i) => i),
    segments,
  );
}
