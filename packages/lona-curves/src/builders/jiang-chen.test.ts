import { describe, expect, test } from "vitest";
import { asNum, naiveEval, variableNum } from "lona";
import { Point } from "lona-geom";
import { ex, val } from "../test-utils";
import { buildJiangChen } from "../main";

const p = (x: number, y: number) => new Point(asNum(x), asNum(y));

describe("buildJiangChen (formulation)", () => {
  const pts = [p(0, 0), p(1, 2), p(2, -1), p(3, 1.5), p(4, 0)];
  const curve = buildJiangChen(pts);

  test("interpolates every control point at its knot", () => {
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

  test("blend endpoints: B(0)=1, B(1)=0 ⇒ segment ends hit the points", () => {
    // first segment local ends land on p0 and p1
    ex(curve.at(0).x).toBeCloseTo(0);
    ex(curve.at(1).x).toBeCloseTo(1);
    ex(curve.at(1).y).toBeCloseTo(2);
  });

  test("is G¹ across interior knots (tangent direction continuous)", () => {
    for (let i = 1; i < pts.length - 1; i++) {
      const before = curve.tangentAt(i - 1e-4);
      const after = curve.tangentAt(i + 1e-4);
      // Same direction: cross product ≈ 0 and dot > 0.
      const cross = before.x.mul(after.y).sub(before.y.mul(after.x));
      ex(cross).toBeCloseTo(0);
      expect(
        val(before.x.mul(after.x).add(before.y.mul(after.y))),
      ).toBeGreaterThan(0);
    }
  });

  test("analytic tangent matches a finite difference", () => {
    const h = 1e-5;
    for (const u of [0.3, 1.5, 2.6, 3.4]) {
      const fd = (k: "x" | "y") =>
        (val(curve.at(u + h)[k]) - val(curve.at(u - h)[k])) / (2 * h);
      ex(curve.tangentAt(u).x).toBeCloseTo(fd("x"), 3);
      ex(curve.tangentAt(u).y).toBeCloseTo(fd("y"), 3);
    }
  });

  test("higher curvature parameter increases the curvature near a point", () => {
    const three = [p(0, 0), p(1, 1), p(2, 0)];
    // Discrete (Menger) curvature 1/R = 4·area / (|ab|·|bc|·|ca|) near the apex.
    const curvatureNear = (a: number): number => {
      const c = buildJiangChen(three, { curvature: a });
      const xy = [0.9, 0.95, 1.0].map((u) => {
        const at = c.at(u);
        return { x: val(at.x), y: val(at.y) };
      });
      const [A, B, C] = xy;
      const area =
        Math.abs((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)) / 2;
      const d = (u: typeof A, v: typeof A) => Math.hypot(u.x - v.x, u.y - v.y);
      return (4 * area) / (d(A, B) * d(B, C) * d(C, A));
    };
    expect(curvatureNear(3)).toBeGreaterThan(curvatureNear(0.3));
  });

  test("collinear points stay on the line (zero oriented curvature)", () => {
    const line = buildJiangChen([p(0, 0), p(1, 1), p(2, 2), p(3, 3), p(4, 4)]);
    for (const s of line.sample(40)) {
      ex(s.y.sub(s.x)).toBeCloseTo(0);
    }
  });

  test("two points reduce to the straight chord", () => {
    const seg = buildJiangChen([p(0, 0), p(2, 1)]);
    ex(seg.at(0.5).x).toBeCloseTo(1);
    ex(seg.at(0.5).y).toBeCloseTo(0.5);
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
    const symbolic = buildJiangChen(
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
