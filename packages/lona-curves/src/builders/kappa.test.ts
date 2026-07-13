import { describe, expect, test } from "vitest";
import { asNum, naiveEval, variableNum } from "lona";
import { Point } from "lona-geom";
import { ex, val } from "../test-utils";
import { buildKappa } from "../main";

const p = (x: number, y: number) => new Point(asNum(x), asNum(y));

interface XY {
  x: number;
  y: number;
}

/** Evaluate a dense sampling of the curve to plain numbers (done once, reused). */
function sampleXY(curve: { sample(n: number): Point[] }, n: number): XY[] {
  return curve.sample(n).map((s) => ({ x: val(s.x), y: val(s.y) }));
}

/** Smallest distance from `target` to a pre-evaluated sampling. */
function minDist(xy: XY[], target: Point): number {
  const tx = val(target.x);
  const ty = val(target.y);
  let best = Infinity;
  for (const s of xy) {
    const d = Math.hypot(s.x - tx, s.y - ty);
    if (d < best) best = d;
  }
  return best;
}

describe("buildKappa", () => {
  const pts = [p(0, 0), p(1, 2), p(2, -1), p(3, 1.5), p(4, 0)];
  const curve = buildKappa(pts);
  const dense = sampleXY(curve, 1200);

  test("pins the first and last points as the curve endpoints", () => {
    ex(curve.at(0).x).toBeCloseTo(0);
    ex(curve.at(0).y).toBeCloseTo(0);
    ex(curve.at(curve.domain[1]).x).toBeCloseTo(4);
    ex(curve.at(curve.domain[1]).y).toBeCloseTo(0);
  });

  test("interpolates every input point", () => {
    for (const pt of pts) {
      expect(minDist(dense, pt)).toBeLessThan(0.02);
    }
  });

  test("the domain runs over the segment indices (n = pts − 2)", () => {
    ex(asNum(curve.domain[0])).toBeCloseTo(0);
    ex(asNum(curve.domain[1])).toBeCloseTo(pts.length - 2);
  });

  test("each interior point is a local maximum of curvature", () => {
    // Curvature magnitude (numeric) along the dense sample should peak right at
    // each input point — the defining κ-curve property.
    const xy = dense;
    const curvature = (i: number): number => {
      const a = xy[i - 1];
      const b = xy[i];
      const c = xy[i + 1];
      const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
      const denom = Math.hypot(b.x - a.x, b.y - a.y) ** 3 || Infinity;
      return Math.abs(cross) / denom;
    };
    // For each interior input point, the nearest sample index should sit at a
    // local curvature maximum (>= its neighbours a short stride away).
    for (const pt of pts.slice(1, -1)) {
      const tx = val(pt.x);
      const ty = val(pt.y);
      let peak = 1;
      let bestD = Infinity;
      for (let i = 1; i < xy.length - 1; i++) {
        const d = Math.hypot(xy[i].x - tx, xy[i].y - ty);
        if (d < bestD) {
          bestD = d;
          peak = i;
        }
      }
      const stride = 8;
      const lo = Math.max(1, peak - stride);
      const hi = Math.min(xy.length - 2, peak + stride);
      expect(curvature(peak)).toBeGreaterThanOrEqual(curvature(lo) - 1e-9);
      expect(curvature(peak)).toBeGreaterThanOrEqual(curvature(hi) - 1e-9);
    }
  });

  test("two points reduce to the straight chord", () => {
    const seg = buildKappa([p(0, 0), p(2, 1)]);
    ex(seg.at(0.5).x).toBeCloseTo(1);
    ex(seg.at(0.5).y).toBeCloseTo(0.5);
  });

  test("a collinear run stays on the line", () => {
    const line = buildKappa([p(0, 0), p(1, 1), p(2, 2), p(3, 3), p(4, 4)]);
    for (const s of line.sample(50)) {
      ex(s.y.sub(s.x)).toBeCloseTo(0);
    }
  });

  test("interpolates a 3-point curve", () => {
    const c3 = buildKappa([p(0, 0), p(1, 1), p(2, 0)]);
    const xy = sampleXY(c3, 600);
    for (const pt of [p(0, 0), p(1, 1), p(2, 0)]) {
      expect(minDist(xy, pt)).toBeLessThan(0.02);
    }
  });

  test("a mirror-symmetric point set yields a symmetric curve", () => {
    const sym = buildKappa([p(-2, 0), p(-1, 1), p(0, 2), p(1, 1), p(2, 0)]);
    const mid = sym.domain[1].div(2);
    // The apex sample at the parametric midpoint sits on the axis of symmetry.
    ex(sym.at(mid).x).toBeCloseTo(0);
    // Mirrored parameters give mirrored points.
    const left = sym.at(mid.sub(0.6));
    const right = sym.at(mid.add(0.6));
    ex(left.x.add(right.x)).toBeCloseTo(0);
    ex(left.y.sub(right.y)).toBeCloseTo(0);
  });

  test("stays a pure Num graph for symbolic inputs", () => {
    const names = ["a", "b", "c", "d", "e"];
    const coords: Record<string, number> = {
      ax: 0,
      ay: 0,
      bx: 1,
      by: 2,
      cx: 2,
      cy: -1,
      dx: 3,
      dy: 1.5,
      ex: 4,
      ey: 0,
    };
    const symbolic = buildKappa(
      names.map(
        (nm) => new Point(variableNum(`${nm}x`), variableNum(`${nm}y`)),
      ),
    );
    const bind = new Map(Object.entries(coords));
    const at = symbolic.at(1.3);
    // Binding the variables reproduces the equivalent numeric build.
    const numeric = curve.at(1.3);
    expect(naiveEval(at.x.n, bind)).toBeCloseTo(val(numeric.x));
    expect(naiveEval(at.y.n, bind)).toBeCloseTo(val(numeric.y));
  });
});
