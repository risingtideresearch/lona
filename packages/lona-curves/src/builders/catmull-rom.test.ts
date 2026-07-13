import { describe, test } from "vitest";
import { asNum } from "lona";
import { Point, Point3D } from "lona-geom";
import { ex, val } from "../test-utils";
import { buildCatmullRom } from "../main";

const p = (x: number, y: number) => new Point(asNum(x), asNum(y));
const p3 = (x: number, y: number, z: number) =>
  new Point3D(asNum(x), asNum(y), asNum(z));

describe("buildCatmullRom (2D)", () => {
  const pts = [p(0, 0), p(1, 1), p(2, 0), p(3, 1)];
  const curve = buildCatmullRom(pts);

  test("interpolates every control point at its knot", () => {
    for (let i = 0; i < pts.length; i++) {
      ex(curve.at(i).x).toBeCloseTo(i);
      ex(curve.at(i).y).toBeCloseTo(i % 2 === 0 ? 0 : 1);
    }
  });

  test("domain runs over the segment indices", () => {
    ex(asNum(curve.domain[0])).toBeCloseTo(0);
    ex(asNum(curve.domain[1])).toBeCloseTo(3);
  });

  test("a collinear run stays on the line", () => {
    // Points on y = 2x → the interpolant is exactly that line.
    const line = buildCatmullRom([p(0, 0), p(1, 2), p(2, 4), p(3, 6)]);
    ex(line.at(1.5).y).toBeCloseTo(3);
    ex(line.at(2.5).y).toBeCloseTo(5);
  });

  test("tension = 1 makes segments straight chords", () => {
    const slack = buildCatmullRom(pts, { tension: 1 });
    // zero tangents → the segment from (1,1) to (2,0) is its straight chord
    ex(slack.at(1.5).x).toBeCloseTo(1.5);
    ex(slack.at(1.5).y).toBeCloseTo(0.5);
  });
});

describe("non-uniform parameterization", () => {
  // Deliberately uneven spacing: chords of length 3 then 1.
  const pts = [p(0, 0), p(3, 0), p(4, 1)];

  test("chordal knots are the cumulative chord lengths", () => {
    const c = buildCatmullRom(pts, { parameterization: "chordal" });
    ex(asNum(c.knots[0])).toBeCloseTo(0);
    ex(asNum(c.knots[1])).toBeCloseTo(3); // |(3,0)-(0,0)|
    ex(asNum(c.knots[2])).toBeCloseTo(3 + Math.SQRT2); // + |(4,1)-(3,0)|
  });

  test("centripetal knots use the sqrt of chord length", () => {
    const c = buildCatmullRom(pts, { parameterization: "centripetal" });
    ex(asNum(c.knots[1])).toBeCloseTo(Math.sqrt(3));
    ex(asNum(c.knots[2])).toBeCloseTo(Math.sqrt(3) + Math.SQRT2 ** 0.5);
  });

  test("still interpolates every control point at its knot", () => {
    for (const param of ["centripetal", "chordal"] as const) {
      const c = buildCatmullRom(pts, { parameterization: param });
      pts.forEach((pt, i) => {
        ex(c.at(c.knots[i]).x).toBeCloseTo(val(pt.x));
        ex(c.at(c.knots[i]).y).toBeCloseTo(val(pt.y));
      });
    }
  });

  test("parameterization changes the interior shape", () => {
    const uni = buildCatmullRom(pts, { parameterization: "uniform" });
    const cen = buildCatmullRom(pts, { parameterization: "centripetal" });
    // Same control points, but the midpoint of the first segment differs.
    const yUni = val(uni.at(uni.knots[1].div(2)).y);
    const yCen = val(cen.at(cen.knots[1].div(2)).y);
    ex(asNum(Math.abs(yUni - yCen))).toBeGreaterThan(1e-3);
  });
});

describe("buildCatmullRom runs in 3D", () => {
  const curve = buildCatmullRom([
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
