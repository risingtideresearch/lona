import { describe, test } from "vitest";
import { asNum } from "lona";
import { Point, Point3D } from "lona-geom";
import { ex, val } from "../test-utils";
import { buildNatural } from "../main";

const p = (x: number, y: number) => new Point(asNum(x), asNum(y));
const p3 = (x: number, y: number, z: number) =>
  new Point3D(asNum(x), asNum(y), asNum(z));

describe("buildNatural (2D)", () => {
  const pts = [p(0, 0), p(1, 1), p(2, 0)];
  const curve = buildNatural(pts);

  test("interpolates every control point at its knot", () => {
    pts.forEach((pt, i) => {
      ex(curve.at(i).x).toBeCloseTo(val(pt.x));
      ex(curve.at(i).y).toBeCloseTo(val(pt.y));
    });
  });

  test("matches the analytic natural-spline midpoint", () => {
    // Tangents solve to m = [(1,1.5), (1,0), (1,-1.5)]; the first segment's
    // de Casteljau midpoint is (c0 + 3c1 + 3c2 + c3)/8 = (0.5, 0.6875).
    ex(curve.at(0.5).x).toBeCloseTo(0.5);
    ex(curve.at(0.5).y).toBeCloseTo(0.6875);
  });

  test("a collinear run stays exactly on the line", () => {
    const line = buildNatural([p(0, 0), p(1, 2), p(2, 4), p(3, 6)]);
    ex(line.at(1.5).y).toBeCloseTo(3);
    ex(line.at(2.5).y).toBeCloseTo(5);
  });

  test("two points reduce to the straight chord", () => {
    const seg = buildNatural([p(0, 0), p(2, 1)]);
    ex(seg.at(0.5).x).toBeCloseTo(1);
    ex(seg.at(0.5).y).toBeCloseTo(0.5);
  });

  test("domain runs over the segment indices", () => {
    ex(asNum(curve.domain[0])).toBeCloseTo(0);
    ex(asNum(curve.domain[1])).toBeCloseTo(2);
  });
});

describe("buildNatural parameterization", () => {
  // Unevenly spaced points: a short hop then a long one.
  const pts = [p(0, 0), p(1, 1), p(5, 0)];

  test("centripetal knots are sqrt-chord spaced", () => {
    const c = buildNatural(pts, { parameterization: "centripetal" });
    ex(asNum(c.knots[1])).toBeCloseTo(Math.sqrt(Math.SQRT2));
  });

  test("parameterization changes the interior shape", () => {
    const uni = buildNatural(pts, { parameterization: "uniform" });
    const chordal = buildNatural(pts, { parameterization: "chordal" });
    const yUni = val(uni.at(uni.knots[1].div(2)).y);
    const yChordal = val(chordal.at(chordal.knots[1].div(2)).y);
    ex(asNum(Math.abs(yUni - yChordal))).toBeGreaterThan(1e-3);
  });

  test("still interpolates the control points under chordal spacing", () => {
    const c = buildNatural(pts, { parameterization: "chordal" });
    pts.forEach((pt, i) => {
      ex(c.at(c.knots[i]).x).toBeCloseTo(val(pt.x));
      ex(c.at(c.knots[i]).y).toBeCloseTo(val(pt.y));
    });
  });
});

describe("buildNatural runs in 3D", () => {
  const curve = buildNatural([
    p3(0, 0, 0),
    p3(1, 1, 1),
    p3(2, 0, 2),
    p3(3, 1, 3),
  ]);

  test("interpolates control points incl. z", () => {
    ex(curve.at(2).x).toBeCloseTo(2);
    ex(curve.at(2).y).toBeCloseTo(0);
    ex(curve.at(2).z).toBeCloseTo(2);
  });
});
