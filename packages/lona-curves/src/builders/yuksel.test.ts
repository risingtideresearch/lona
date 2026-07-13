import { describe, expect, test } from "vitest";
import { asNum, naiveEval, variableNum } from "lona";
import { Point, Point3D } from "lona-geom";
import { ex, val } from "../test-utils";
import { buildYuksel } from "../main";

const p = (x: number, y: number) => new Point(asNum(x), asNum(y));
const p3 = (x: number, y: number, z: number) =>
  new Point3D(asNum(x), asNum(y), asNum(z));

describe("buildYuksel (2D)", () => {
  const pts = [p(0, 0), p(1, 2), p(2, -1), p(3, 1.5), p(4, 0)];
  const curve = buildYuksel(pts);

  test("interpolates every control point at its knot (exactly)", () => {
    pts.forEach((pt, i) => {
      ex(curve.at(i).x).toBeCloseTo(val(pt.x));
      ex(curve.at(i).y).toBeCloseTo(val(pt.y));
    });
  });

  test("pins the endpoints", () => {
    ex(curve.at(0).x).toBeCloseTo(0);
    ex(curve.at(0).y).toBeCloseTo(0);
    ex(curve.at(curve.domain[1]).x).toBeCloseTo(4);
    ex(curve.at(curve.domain[1]).y).toBeCloseTo(0);
  });

  test("domain runs over the point indices", () => {
    ex(asNum(curve.domain[0])).toBeCloseTo(0);
    ex(asNum(curve.domain[1])).toBeCloseTo(pts.length - 1);
  });

  test("is C¹ across interior knots (tangent continuous)", () => {
    // Sample the tangent just below and just above each interior knot; for a C¹
    // join the two one-sided tangents agree.
    for (let i = 1; i < pts.length - 1; i++) {
      const before = curve.tangentAt(i - 1e-4);
      const after = curve.tangentAt(i + 1e-4);
      ex(before.x.sub(after.x)).toBeCloseTo(0);
      ex(before.y.sub(after.y)).toBeCloseTo(0);
    }
  });

  test("analytic tangent matches a finite difference", () => {
    const h = 1e-5;
    for (const u of [0.37, 1.5, 2.6, 3.2]) {
      const fd = (k: "x" | "y") =>
        (val(curve.at(u + h)[k]) - val(curve.at(u - h)[k])) / (2 * h);
      ex(curve.tangentAt(u).x).toBeCloseTo(fd("x"), 3);
      ex(curve.tangentAt(u).y).toBeCloseTo(fd("y"), 3);
    }
  });

  test("three points reproduce the parabola through them", () => {
    // With 3 points the whole curve is the single interpolating parabola.
    const par = buildYuksel([p(0, 0), p(1, 1), p(2, 0)]);
    // Parabola through (0,0),(1,1),(2,0) is y = x(2−x); check an off-knot point.
    const at = par.at(0.5);
    ex(at.x).toBeCloseTo(0.5);
    ex(at.y).toBeCloseTo(0.5 * (2 - 0.5)); // 0.75
  });

  test("two points reduce to the straight chord", () => {
    const seg = buildYuksel([p(0, 0), p(2, 1)]);
    ex(seg.at(0.5).x).toBeCloseTo(1);
    ex(seg.at(0.5).y).toBeCloseTo(0.5);
  });

  test("a collinear run stays on the line", () => {
    const line = buildYuksel([p(0, 0), p(1, 1), p(2, 2), p(3, 3), p(4, 4)]);
    for (const s of line.sample(40)) {
      ex(s.y.sub(s.x)).toBeCloseTo(0);
    }
  });

  test("stays a pure Num graph for symbolic inputs", () => {
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
    const symbolic = buildYuksel(
      ["a", "b", "c", "d", "e"].map(
        (nm) => new Point(variableNum(`${nm}x`), variableNum(`${nm}y`)),
      ),
    );
    const bind = new Map(Object.entries(coords));
    const at = symbolic.at(1.3);
    const numeric = curve.at(1.3);
    expect(naiveEval(at.x.n, bind)).toBeCloseTo(val(numeric.x));
    expect(naiveEval(at.y.n, bind)).toBeCloseTo(val(numeric.y));
  });
});

describe("buildYuksel parameterization", () => {
  const pts = [p(0, 0), p(1, 1), p(5, 0)];

  test("still interpolates under chordal spacing", () => {
    const c = buildYuksel(pts, { parameterization: "chordal" });
    pts.forEach((pt, i) => {
      ex(c.at(c.knots[i]).x).toBeCloseTo(val(pt.x));
      ex(c.at(c.knots[i]).y).toBeCloseTo(val(pt.y));
    });
  });
});

describe("buildYuksel runs in 3D", () => {
  const curve = buildYuksel([
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
