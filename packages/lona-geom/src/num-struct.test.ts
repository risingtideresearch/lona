import { describe, test } from "vitest";
import { asNum, casesStruct, mapNums, selectStruct, whenStruct } from "lona";
import { ex } from "./test-utils";
import { Point, angleFromDeg } from "./geom";
import { Point3D } from "./geom-3d";

describe("self-describing geom values (no layout passed)", () => {
  const a = new Point(asNum(1), asNum(2));
  const b = new Point(asNum(3), asNum(4));

  test("selectStruct picks one whole point by the condition", () => {
    ex(selectStruct(asNum(1), a, b).x).toBeCloseTo(1);
    ex(selectStruct(asNum(1), a, b).y).toBeCloseTo(2);
    ex(selectStruct(asNum(0), a, b).x).toBeCloseTo(3);
    ex(selectStruct(asNum(0), a, b).y).toBeCloseTo(4);
  });

  test("the same generic select works in 3D", () => {
    const p = new Point3D(asNum(1), asNum(2), asNum(3));
    const q = new Point3D(asNum(4), asNum(5), asNum(6));
    const picked = selectStruct(asNum(0), p, q);
    ex(picked.x).toBeCloseTo(4);
    ex(picked.z).toBeCloseTo(6);
  });

  test("select keeps a constrained Angle valid (picks one whole input)", () => {
    const picked = selectStruct(asNum(1), angleFromDeg(30), angleFromDeg(200));
    ex(picked.asDeg()).toBeCloseTo(30);
    ex(picked.cos().square().add(picked.sin().square())).toBeCloseTo(1);
  });

  test("mapNums applies a scalar op to every component", () => {
    const p = mapNums(new Point(asNum(2), asNum(5)), (n) => n.mul(10));
    ex(p.x).toBeCloseTo(20);
    ex(p.y).toBeCloseTo(50);
  });

  test("whenStruct / casesStruct infer the type from the branch values", () => {
    const w = whenStruct(asNum(0))
      .then(a)
      .elseIf(asNum(1))
      .then(b)
      .else(new Point(asNum(9), asNum(9)));
    ex(w.x).toBeCloseTo(3); // first false, second true → b

    const pick = (sel: number) =>
      casesStruct(asNum(sel))
        .case(0, a)
        .case(1, b)
        .default(new Point(asNum(9), asNum(9)));
    ex(pick(1).x).toBeCloseTo(3);
    ex(pick(7).x).toBeCloseTo(9);
  });
});
