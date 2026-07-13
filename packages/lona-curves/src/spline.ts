import { asNum, selectStruct, type Num, type NumStruct } from "lona";

// ---------------------------------------------------------------------------
// Geometry capability boundary
//
// The spline machinery is affine: it only ever does `point ± vector`, scales
// vectors, and takes vectors between points. Those operations are identical in
// 2D and 3D, so the IR is generic over a `(Point, Vec)` pair. Both geom's
// `(Point, Vec2)` and `(Point3D, Vec3)` structurally satisfy the minimal
// interfaces below — nothing here imports a concrete geom class.
// ---------------------------------------------------------------------------

/** The minimal linear-vector operations the curve math needs. */
export interface CurveVec<V> {
  add(other: V): V;
  sub(other: V): V;
  scale(k: Num | number): V;
  neg(): V;
}

/** The minimal affine-point operations the curve math needs. */
export interface CurvePoint<P, V extends CurveVec<V>> {
  add(v: V): P;
  vecTo(other: P): V;
}

/**
 * Recover the vector type a point type produces (e.g. `Vec2` from `Point`). The
 * `CurveVec & NumStruct` guard lets it serve as the `V` default on the classes:
 * it provably satisfies their bound for abstract `P`, and resolves to the
 * concrete vector (`Vec2`/`Vec3`) at a `new` site.
 */
export type VecOf<P> = P extends { vecTo(other: P): infer V }
  ? V extends CurveVec<V> & NumStruct<V>
    ? V
    : never
  : never;

// ---------------------------------------------------------------------------
// The built curve (interchange representation)
//
// Every builder produces a `Spline` of Bézier segments. Most emit cubics
// (`CubicSeg`); κ-curves emit quadratics (`QuadraticSeg`). A `Spline` only ever
// queries its segments through `at`/`tangentAt`, so it holds the shared
// `Segment` interface and is agnostic to the Bézier degree. Querying never
// re-fits.
//
// The segment classes carry the `(P, V)` constraint on the type so it is
// declared once rather than on every query function, `V` is a genuine (used)
// parameter so no phantom is needed, and call sites read fluently
// (`seg.at(t)`, `spline.sample(n)`).
// ---------------------------------------------------------------------------

/**
 * The query surface every Bézier piece shares, regardless of degree: a point
 * and a tangent at a local parameter `t ∈ [0,1]`. `Spline` holds these, so a
 * curve can mix degrees (cubic and quadratic) transparently.
 */
export interface Segment<P, V> {
  /** Point at local parameter `t ∈ [0,1]` (`t` may be a symbolic `Num`). */
  at(t: Num | number): P;
  /** Tangent (first derivative) at local parameter `t`. */
  tangentAt(t: Num | number): V;
}

/** A single cubic Bézier piece, with control points in the geometry type `P`. */
export class CubicSeg<
  P extends CurvePoint<P, V>,
  V extends CurveVec<V> = VecOf<P>,
> implements Segment<P, V> {
  constructor(
    readonly c0: P,
    readonly c1: P,
    readonly c2: P,
    readonly c3: P,
  ) {}

  /** Point at local parameter `t ∈ [0,1]` (de Casteljau). `t` may be a symbolic `Num`. */
  at(t: Num | number): P {
    const lerp = (a: P, b: P): P => a.add(a.vecTo(b).scale(t)); // a + (b-a)·t
    const ab = lerp(this.c0, this.c1);
    const bc = lerp(this.c1, this.c2);
    const cd = lerp(this.c2, this.c3);
    return lerp(lerp(ab, bc), lerp(bc, cd));
  }

  /**
   * Tangent (first derivative) at local parameter `t`:
   * `B'(t) = 3[(c1-c0)(1-t)² + 2(c2-c1)(1-t)t + (c3-c2)t²]`. This is the
   * derivative w.r.t. the *local* parameter, so its magnitude depends on the
   * segment's parameter width — fine for directions; scale by `1/Δknot` for the
   * derivative w.r.t. a global parameter.
   */
  tangentAt(t: Num | number): V {
    const tt = asNum(t);
    const omt = asNum(1).sub(tt); // 1 - t
    const v01 = this.c0.vecTo(this.c1);
    const v12 = this.c1.vecTo(this.c2);
    const v23 = this.c2.vecTo(this.c3);
    return v01
      .scale(omt.mul(omt).mul(3))
      .add(v12.scale(omt.mul(tt).mul(6)))
      .add(v23.scale(tt.mul(tt).mul(3)));
  }
}

/** A single quadratic Bézier piece, with control points in the geometry type `P`. */
export class QuadraticSeg<
  P extends CurvePoint<P, V>,
  V extends CurveVec<V> = VecOf<P>,
> implements Segment<P, V> {
  constructor(
    readonly c0: P,
    readonly c1: P,
    readonly c2: P,
  ) {}

  /** Point at local parameter `t ∈ [0,1]` (de Casteljau). `t` may be a symbolic `Num`. */
  at(t: Num | number): P {
    const lerp = (a: P, b: P): P => a.add(a.vecTo(b).scale(t)); // a + (b-a)·t
    return lerp(lerp(this.c0, this.c1), lerp(this.c1, this.c2));
  }

  /**
   * Tangent (first derivative) at local parameter `t`:
   * `B'(t) = 2[(c1-c0)(1-t) + (c2-c1)t]`. Like `CubicSeg.tangentAt` this is the
   * derivative w.r.t. the *local* parameter; scale by `1/Δknot` for the global one.
   */
  tangentAt(t: Num | number): V {
    const tt = asNum(t);
    const omt = asNum(1).sub(tt); // 1 - t
    const v01 = this.c0.vecTo(this.c1);
    const v12 = this.c1.vecTo(this.c2);
    return v01.scale(omt.mul(2)).add(v12.scale(tt.mul(2)));
  }
}

/**
 * A piecewise-Bézier curve (cubic and/or quadratic segments).
 *
 * `knots` are the parameter values at the segment boundaries (ascending,
 * `length === segments.length + 1`). For a parametric curve they are typically
 * `0..n` or chord-length; for a functional curve (pchip) they are the
 * independent-coordinate breakpoints. They are `Num`s, so a parameterization
 * may itself be symbolic (e.g. chord-length knots from symbolic points); the
 * constructor accepts plain `number`s for convenience and wraps them.
 */
export class Spline<
  P extends CurvePoint<P, V> & NumStruct<P>,
  V extends CurveVec<V> & NumStruct<V> = VecOf<P>,
> {
  readonly knots: Num[];

  constructor(
    knots: (Num | number)[],
    readonly segments: Segment<P, V>[],
  ) {
    this.knots = knots.map(asNum);
    if (this.knots.length !== segments.length + 1) {
      throw new Error(
        `Spline expects knots.length === segments.length + 1 ` +
          `(got ${this.knots.length} knots, ${segments.length} segments)`,
      );
    }
  }

  /** The `[start, end]` parameter range spanned by the knots. */
  get domain(): [Num, Num] {
    return [this.knots[0], this.knots[this.knots.length - 1]];
  }

  /**
   * Evaluate a per-segment quantity at a global parameter `u`, choosing the
   * active segment with a branch-free `selectStruct` cascade over the knots
   * (the whole point of the library: everything stays a `Num`, so location
   * can't branch on `u`). `u` below the first knot / at-or-above the last
   * extrapolates the first / last segment. Works in any dimension because the
   * results self-describe (`NumStruct`).
   */
  private cascade<S extends NumStruct<S>>(
    u: Num,
    evalSeg: (seg: Segment<P, V>, t: Num) => S,
  ): S {
    const { knots, segments } = this;
    const last = segments.length - 1;
    const localParam = (i: number): Num =>
      u.sub(knots[i]).div(knots[i + 1].sub(knots[i]));

    let result = evalSeg(segments[last], localParam(last));
    for (let i = last - 1; i >= 0; i--) {
      result = selectStruct(
        u.lessThan(knots[i + 1]),
        evalSeg(segments[i], localParam(i)),
        result,
      );
    }
    return result;
  }

  /** Point at global parameter `u`. */
  at(u: Num | number): P {
    return this.cascade(asNum(u), (seg, t) => seg.at(t));
  }

  /** Tangent at global parameter `u` (see `CubicSeg.tangentAt` on local scaling). */
  tangentAt(u: Num | number): V {
    return this.cascade(asNum(u), (seg, t) => seg.tangentAt(t));
  }

  /** Sample `count` points evenly across the domain (both ends inclusive). */
  sample(count: number): P[] {
    if (count < 1) return [];
    const [start, end] = this.domain;
    const span = end.sub(start);
    const out: P[] = [];
    for (let k = 0; k < count; k++) {
      const frac = count === 1 ? 0 : k / (count - 1);
      out.push(this.at(start.add(span.mul(frac))));
    }
    return out;
  }
}
