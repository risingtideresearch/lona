import { describe, expect, test } from "vitest";
import { asNum, naiveEval, variableNum } from "lona";
import { ex, val } from "../test-utils";
import { buildPchip } from "../main";

const station = (x: number, y: number) => ({ x, y: asNum(y) });

describe("buildPchip", () => {
  const stations = [station(0, 0), station(1, 1), station(2, 4), station(3, 9)];
  const curve = buildPchip(stations);

  test("interpolates every station: at(x_i) = (x_i, y_i)", () => {
    for (const s of stations) {
      ex(curve.at(s.x).x).toBeCloseTo(s.x);
      ex(curve.at(s.x).y).toBeCloseTo(val(s.y));
    }
  });

  test("monotone data produces no overshoot", () => {
    // A step-like monotone non-decreasing profile: PCHIP must stay in [0, 1].
    const c = buildPchip([
      station(0, 0),
      station(1, 0),
      station(2, 1),
      station(3, 1),
    ]);
    for (let x = 0; x <= 3; x += 0.1) {
      const y = val(c.at(x).y);
      ex(asNum(y)).toBeGreaterThanOrEqual(-1e-6);
      ex(asNum(y)).toBeLessThanOrEqual(1 + 1e-6);
    }
  });

  test("k = 1 everywhere makes each segment a straight chord", () => {
    const one = asNum(1);
    const c = buildPchip([
      { x: 0, y: asNum(0), k: one },
      { x: 1, y: asNum(1), k: one },
      { x: 2, y: asNum(4), k: one },
      { x: 3, y: asNum(9), k: one },
    ]);
    // Mid of each segment equals the linear average of its endpoints.
    ex(c.at(0.5).y).toBeCloseTo(0.5); // between (0,0) and (1,1)
    ex(c.at(1.5).y).toBeCloseTo(2.5); // between (1,1) and (2,4)
    ex(c.at(2.5).y).toBeCloseTo(6.5); // between (2,4) and (3,9)
  });

  test("the y-values are symbolic — the curve is one Num graph", () => {
    // Build from a fresh Num and confirm the queried y tracks it.
    const h = asNum(5);
    const c = buildPchip([station(0, 0), { x: 1, y: h }, station(2, 0)]);
    ex(c.at(1).y).toBeCloseTo(5); // peak interpolates the symbolic height
    // monotone-to-the-peak: no overshoot above the peak height
    ex(c.at(0.5).y).toBeLessThanOrEqual(5 + 1e-6);
  });

  test("the knots are symbolic too — a station slides with a variable", () => {
    // The peak station sits at x = 1 + t: one graph, re-solved per binding.
    const t = variableNum("t");
    const c = buildPchip([
      station(0, 0),
      { x: t.add(1), y: asNum(2) },
      station(4, 0),
    ]);
    const yAt = (x: number, tv: number): number =>
      naiveEval(c.at(x).y.n, new Map([["t", tv]]));
    // the interpolated peak follows the knot as t moves
    expect(yAt(1, 0)).toBeCloseTo(2);
    expect(yAt(2, 1)).toBeCloseTo(2);
    expect(yAt(3, 2)).toBeCloseTo(2);
    // the fixed end stations still interpolate exactly at any binding
    expect(yAt(0, 0)).toBeCloseTo(0);
    expect(yAt(4, 1)).toBeCloseTo(0);
    // shape-preserving on both sides of the moved peak: y stays in [0, 2]
    for (let x = 0; x <= 4; x += 0.25) {
      const y = yAt(x, 1);
      expect(y).toBeGreaterThanOrEqual(-1e-6);
      expect(y).toBeLessThanOrEqual(2 + 1e-6);
    }
  });
});
