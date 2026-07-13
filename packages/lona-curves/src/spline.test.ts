import { describe, test, expect } from "vitest";
import { asNum } from "lona";
import { Point, Point3D } from "lona-geom";
import { ex } from "./test-utils";
import { CubicSeg, Spline } from "./main";

const p = (x: number, y: number) => new Point(asNum(x), asNum(y));
const p3 = (x: number, y: number, z: number) =>
  new Point3D(asNum(x), asNum(y), asNum(z));

describe("CubicSeg (2D)", () => {
  // A known cubic: B(½) = (c0 + 3c1 + 3c2 + c3) / 8.
  const seg = new CubicSeg(p(0, 0), p(0, 1), p(1, 1), p(1, 0));

  test("endpoints are interpolated", () => {
    ex(seg.at(0).x).toBeCloseTo(0);
    ex(seg.at(0).y).toBeCloseTo(0);
    ex(seg.at(1).x).toBeCloseTo(1);
    ex(seg.at(1).y).toBeCloseTo(0);
  });

  test("midpoint matches the Bézier average", () => {
    ex(seg.at(0.5).x).toBeCloseTo(0.5);
    ex(seg.at(0.5).y).toBeCloseTo(0.75);
  });

  test("tangent at midpoint is horizontal", () => {
    ex(seg.tangentAt(0.5).x).toBeCloseTo(1.5);
    ex(seg.tangentAt(0.5).y).toBeCloseTo(0);
  });

  test("local parameter t may be symbolic (Num)", () => {
    ex(seg.at(asNum(0.5)).y).toBeCloseTo(0.75);
  });
});

describe("Spline query (2D)", () => {
  // Two segments over knots [0, 1, 2]; second is a flat line (1,0)→(2,0).
  const spline = new Spline(
    [0, 1, 2],
    [
      new CubicSeg(p(0, 0), p(0, 1), p(1, 1), p(1, 0)),
      new CubicSeg(p(1, 0), p(4 / 3, 0), p(5 / 3, 0), p(2, 0)),
    ],
  );

  test("u in the second interval locates the second segment", () => {
    ex(spline.at(1.5).x).toBeCloseTo(1.5);
    ex(spline.at(1.5).y).toBeCloseTo(0);
  });

  test("a symbolic Num parameter resolves the same way", () => {
    ex(spline.at(asNum(0.5)).y).toBeCloseTo(0.75); // first segment midpoint
    ex(spline.at(asNum(1.5)).x).toBeCloseTo(1.5); // second segment midpoint
  });

  test("tangentAt selects the active segment's tangent", () => {
    ex(spline.tangentAt(0.5).x).toBeCloseTo(1.5); // first segment is the S-curve
    ex(spline.tangentAt(1.5).y).toBeCloseTo(0); // second segment is flat
  });

  test("domain spans the knots", () => {
    ex(spline.domain[0]).toBeCloseTo(0);
    ex(spline.domain[1]).toBeCloseTo(2);
  });

  test("sample covers the full domain inclusively", () => {
    const pts = spline.sample(3);
    expect(pts).toHaveLength(3);
    ex(pts[0].x).toBeCloseTo(0);
    ex(pts[2].x).toBeCloseTo(2);
  });

  test("constructor enforces knots.length === segments.length + 1", () => {
    expect(() => new Spline([0, 1], spline.segments)).toThrow();
  });
});

describe("the same fluent class runs in 3D", () => {
  const seg = new CubicSeg(p3(0, 0, 0), p3(0, 1, 0), p3(1, 1, 1), p3(1, 0, 1));

  test("midpoint carries the z coordinate", () => {
    const mid = seg.at(0.5);
    ex(mid.x).toBeCloseTo(0.5);
    ex(mid.y).toBeCloseTo(0.75);
    ex(mid.z).toBeCloseTo(0.5);
  });
});

describe("the cascade works in 3D too", () => {
  const s3 = new Spline(
    [0, 1],
    [new CubicSeg(p3(0, 0, 0), p3(0, 1, 0), p3(1, 1, 1), p3(1, 0, 1))],
  );

  test("at() carries the z coordinate", () => {
    const mid = s3.at(asNum(0.5));
    ex(mid.x).toBeCloseTo(0.5);
    ex(mid.y).toBeCloseTo(0.75);
    ex(mid.z).toBeCloseTo(0.5);
  });
});
