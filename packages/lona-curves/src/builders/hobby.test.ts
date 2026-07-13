import { describe, test } from "vitest";
import { asNum } from "lona";
import { Point } from "lona-geom";
import { ex, val } from "../test-utils";
import { buildHobby } from "../main";

const p = (x: number, y: number) => new Point(asNum(x), asNum(y));

describe("buildHobby (open, 2D)", () => {
  const pts = [p(0, 0), p(1, 1), p(2, 0), p(3, 1)];
  const curve = buildHobby(pts);

  test("interpolates every control point at its knot", () => {
    pts.forEach((pt, i) => {
      ex(curve.at(i).x).toBeCloseTo(val(pt.x));
      ex(curve.at(i).y).toBeCloseTo(val(pt.y));
    });
  });

  test("domain runs over the segment indices", () => {
    ex(asNum(curve.domain[0])).toBeCloseTo(0);
    ex(asNum(curve.domain[1])).toBeCloseTo(3);
  });

  test("two points reduce to the straight chord", () => {
    const seg = buildHobby([p(0, 0), p(2, 1)]);
    ex(seg.at(0.5).x).toBeCloseTo(1);
    ex(seg.at(0.5).y).toBeCloseTo(0.5);
  });

  test("a collinear run stays on the line", () => {
    const line = buildHobby([p(0, 0), p(1, 0), p(2, 0), p(3, 0)]);
    ex(line.at(1.5).x).toBeCloseTo(1.5);
    ex(line.at(1.5).y).toBeCloseTo(0);
    ex(line.at(2.5).y).toBeCloseTo(0);
  });
});

describe("buildHobby symmetry", () => {
  // A symmetric "hat": the curve must be its own mirror image about x = 1.
  const curve = buildHobby([p(0, 0), p(1, 1), p(2, 0)]);

  test("mirror-symmetric points map to mirror-symmetric outputs", () => {
    const left = curve.at(0.5);
    const right = curve.at(1.5);
    ex(left.x).toBeCloseTo(2 - val(right.x));
    ex(left.y).toBeCloseTo(val(right.y));
  });

  test("the apex bulges above the straight chords", () => {
    // Hobby rounds the corner, so the first half peaks above the chord y = x.
    ex(curve.at(0.5).y).toBeGreaterThan(0.5);
  });
});
