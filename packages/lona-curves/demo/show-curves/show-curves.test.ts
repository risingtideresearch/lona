import { describe, expect, test } from "vitest";
import {
  cloneInitialPoints,
  createCurveSampleRoutine,
  pointsForSpline,
  sampleCount,
  sampleSpline,
  samplerCacheKey,
  type DemoPoint,
} from "./src/curve-model";

const points: DemoPoint[] = [
  { id: 1, x: 20, y: 80 },
  { id: 2, x: 100, y: 20 },
  { id: 3, x: 180, y: 90 },
];

describe("show-curves demo model", () => {
  test("samples a Catmull-Rom curve through the first and last points", () => {
    const samples = sampleSpline(points, "catmull-centripetal");

    expect(samples).toHaveLength(sampleCount(points.length));
    expect(samples[0].x).toBeCloseTo(points[0].x);
    expect(samples[0].y).toBeCloseTo(points[0].y);
    expect(samples[samples.length - 1].x).toBeCloseTo(points[2].x);
    expect(samples[samples.length - 1].y).toBeCloseTo(points[2].y);
  });

  test("samples the Natural, Hobby, κ-curve, Yuksel and Jiang-Chen builders through endpoints", () => {
    for (const kind of [
      "natural-uniform",
      "hobby",
      "kappa",
      "yuksel",
      "jiang-chen",
    ] as const) {
      const samples = sampleSpline(points, kind);

      expect(samples).toHaveLength(sampleCount(points.length));
      expect(samples[0].x).toBeCloseTo(points[0].x);
      expect(samples[0].y).toBeCloseTo(points[0].y);
      expect(samples[samples.length - 1].x).toBeCloseTo(points[2].x);
      expect(samples[samples.length - 1].y).toBeCloseTo(points[2].y);
    }
  });

  test("Catmull-Rom samples are visibly curved for non-collinear points", () => {
    const samples = sampleSpline(
      [
        { id: 1, x: 0, y: 0 },
        { id: 2, x: 100, y: 0 },
        { id: 3, x: 100, y: 100 },
      ],
      "catmull-uniform",
    );

    const firstSegmentSamples = samples.filter(
      (sample) => sample.x >= 0 && sample.x <= 100,
    );
    expect(
      Math.min(...firstSegmentSamples.map((sample) => sample.y)),
    ).toBeLessThan(-1);
  });

  test("does not change the sampler cache key when points move", () => {
    const moved = points.map((point) =>
      point.id === 2 ? { ...point, x: point.x + 40, y: point.y + 100 } : point,
    );

    expect(samplerCacheKey(moved, "catmull-centripetal", "js-codegen")).toBe(
      samplerCacheKey(points, "catmull-centripetal", "js-codegen"),
    );
    expect(samplerCacheKey(moved, "pchip", "js-codegen")).toBe(
      samplerCacheKey(points, "pchip", "js-codegen"),
    );
    expect(samplerCacheKey(points, "pchip", "js-codegen", "trace")).not.toBe(
      samplerCacheKey(points, "pchip", "js-codegen", "off"),
    );
    expect(
      samplerCacheKey(
        points.map((point) =>
          point.id === 2 ? { ...point, knuckle: true } : point,
        ),
        "pchip",
        "js-codegen",
      ),
    ).not.toBe(samplerCacheKey(points, "pchip", "js-codegen"));
  });

  test("compiles point coordinates as variables in one reusable routine", () => {
    const routine = createCurveSampleRoutine(points, "catmull-uniform");

    expect(routine?.variableNames).toEqual([
      "p1.x",
      "p1.y",
      "p2.x",
      "p2.y",
      "p3.x",
      "p3.y",
    ]);

    const moved = points.map((point) =>
      point.id === 2 ? { ...point, y: point.y + 100 } : point,
    );
    const originalSamples = routine?.sample(points).points ?? [];
    const movedSamples = routine?.sample(moved).points ?? [];

    expect(movedSamples[Math.floor(movedSamples.length / 2)].y).not.toBeCloseTo(
      originalSamples[Math.floor(originalSamples.length / 2)].y,
    );
  });

  test("keeps PCHIP in user order", () => {
    const userOrdered = [points[2], points[0], points[1]];
    const ordered = pointsForSpline(userOrdered, "pchip");

    expect(ordered.map((point) => point.x)).toEqual([180, 20, 100]);
    expect(ordered).toBe(userOrdered);
  });

  test("samples PCHIP from the first numbered point to the last", () => {
    const userOrdered = [points[2], points[0], points[1]];
    const samples = sampleSpline(userOrdered, "pchip");

    expect(samples[0]).toEqual({ x: 180, y: 90 });
    expect(samples[samples.length - 1]).toEqual({ x: 100, y: 20 });
  });

  test("clones initial points for reset state", () => {
    const a = cloneInitialPoints();
    const b = cloneInitialPoints();

    a[0].x = -1;
    expect(b[0].x).not.toBe(-1);
  });
});
