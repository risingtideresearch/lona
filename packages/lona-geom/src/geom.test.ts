import { test, describe } from "vitest";

import { ex } from "./test-utils";
import { angleFromDeg, angleRromRad, asVec, vecFromPolarCoords } from "./geom";

describe("Angle", () => {
  test("angle from rad", () => {
    ex(angleRromRad(0).cos()).toBeCloseTo(1);
    ex(angleRromRad(Math.PI).cos()).toBeCloseTo(-1);
    ex(angleRromRad(Math.PI / 2).cos()).toBeCloseTo(0);
    ex(angleRromRad(0).sin()).toBeCloseTo(0);
    ex(angleRromRad(Math.PI).sin()).toBeCloseTo(0);
    ex(angleRromRad(Math.PI / 2).sin()).toBeCloseTo(1);
  });

  test("angle from deg", () => {
    ex(angleFromDeg(0).cos()).toBeCloseTo(1);
    ex(angleFromDeg(180).cos()).toBeCloseTo(-1);
    ex(angleFromDeg(90).cos()).toBeCloseTo(0);
    ex(angleFromDeg(45).cos()).toBeCloseTo(Math.sqrt(2) / 2);
    ex(angleFromDeg(45).sin()).toBeCloseTo(Math.sqrt(2) / 2);
  });

  test("as_deg", () => {
    ex(angleRromRad(0).asDeg()).toBeCloseTo(0);
    ex(angleRromRad(Math.PI).asDeg()).toBeCloseTo(180);
    ex(angleRromRad(Math.PI / 2).asDeg()).toBeCloseTo(90);
    ex(angleRromRad(Math.PI / 4).asDeg()).toBeCloseTo(45);
    ex(angleFromDeg(123).asDeg()).toBeCloseTo(123);
    ex(angleFromDeg(-123).asDeg()).toBeCloseTo(-123);
  });

  test("as_rad", () => {
    ex(angleRromRad(0).asRad()).toBeCloseTo(0);
    ex(angleRromRad(Math.PI).asRad()).toBeCloseTo(Math.PI);
    ex(angleRromRad(1.1).asRad()).toBeCloseTo(1.1);
    ex(angleRromRad(-1.1).asRad()).toBeCloseTo(-1.1);
  });

  function _d(x: number) {
    return angleFromDeg(x);
  }

  test("add", () => {
    ex(_d(0).add(_d(0)).asDeg()).toBeCloseTo(0);
    ex(_d(3).add(_d(120)).asDeg()).toBeCloseTo(123);
    ex(_d(90).add(_d(90)).asDeg()).toBeCloseTo(180);
    ex(_d(90).add(_d(-90)).asDeg()).toBeCloseTo(0);
    ex(_d(5).add(_d(50)).asDeg()).toBeCloseTo(55);
    ex(_d(250).add(_d(100)).asDeg()).toBeCloseTo(-10);
    ex(_d(250).add(_d(200)).asDeg()).toBeCloseTo(90);
  });

  test("sub", () => {
    ex(_d(0).sub(_d(0)).asDeg()).toBeCloseTo(0);
    ex(_d(120).sub(_d(3)).asDeg()).toBeCloseTo(117);
    ex(_d(180).sub(_d(90)).asDeg()).toBeCloseTo(90);
    ex(_d(0).sub(_d(90)).asDeg()).toBeCloseTo(-90);
    ex(_d(50).sub(_d(5)).asDeg()).toBeCloseTo(45);
    ex(_d(100).sub(_d(250)).asDeg()).toBeCloseTo(-150);
    ex(_d(-160).sub(_d(250)).asDeg()).toBeCloseTo(-50);
    ex(_d(-160).sub(_d(30)).asDeg()).toBeCloseTo(170);
  });

  test("double", () => {
    ex(_d(0).double().asDeg()).toBeCloseTo(0);
    ex(_d(-0).double().asDeg()).toBeCloseTo(0);
    ex(_d(90).double().asDeg()).toBeCloseTo(180);
    ex(_d(180).double().asDeg()).toBeCloseTo(0);
    ex(_d(270).double().asDeg()).toBeCloseTo(180);
    ex(_d(360).double().asDeg()).toBeCloseTo(0);

    ex(_d(40).double().asDeg()).toBeCloseTo(80);
    ex(_d(120).double().asDeg()).toBeCloseTo(-120);
    ex(_d(200).double().asDeg()).toBeCloseTo(40);
    ex(_d(300).double().asDeg()).toBeCloseTo(-120);
    ex(_d(-30).double().asDeg()).toBeCloseTo(-60);
    ex(_d(-100).double().asDeg()).toBeCloseTo(160);
  });

  test("half", () => {
    ex(_d(0).half().asDeg()).toBeCloseTo(0);
    ex(_d(-0).half().asDeg()).toBeCloseTo(0);
    ex(_d(90).half().asDeg()).toBeCloseTo(45);
    ex(_d(180).half().asDeg()).toBeCloseTo(90);
    ex(_d(120).half().asDeg()).toBeCloseTo(60);
    ex(_d(200).half().asDeg()).toBeCloseTo(100);
    ex(_d(300).half().asDeg()).toBeCloseTo(150);
    ex(_d(-30).half().asDeg()).toBeCloseTo(165);
    ex(_d(-100).half().asDeg()).toBeCloseTo(130);
  });

  test("perp", () => {
    ex(_d(0).perp().asDeg()).toBeCloseTo(90);
    ex(_d(-0).perp().asDeg()).toBeCloseTo(90);
    ex(_d(90).perp().asDeg()).toBeCloseTo(180);
    ex(_d(180).perp().asDeg()).toBeCloseTo(-90);
    ex(_d(120).perp().asDeg()).toBeCloseTo(-150);
    ex(_d(200).perp().asDeg()).toBeCloseTo(-70);
    ex(_d(300).perp().asDeg()).toBeCloseTo(30);
    ex(_d(-30).perp().asDeg()).toBeCloseTo(60);
    ex(_d(-100).perp().asDeg()).toBeCloseTo(-10);
  });

  test("opposite", () => {
    ex(_d(0).opposite().asDeg()).toBeCloseTo(-180);
    // -0 and 0 produce the same cos/sin, so opposite is the same (-180 ≡ 180)
    ex(_d(-0).opposite().asDeg()).toBeCloseTo(-180);
    ex(_d(90).opposite().asDeg()).toBeCloseTo(-90);
    ex(_d(180).opposite().asDeg()).toBeCloseTo(0);
    ex(_d(270).opposite().asDeg()).toBeCloseTo(90);
    ex(_d(360).opposite().asDeg()).toBeCloseTo(180);

    ex(_d(50).opposite().asDeg()).toBeCloseTo(-130);
    ex(_d(150).opposite().asDeg()).toBeCloseTo(-30);
    ex(_d(200).opposite().asDeg()).toBeCloseTo(20);
    ex(_d(300).opposite().asDeg()).toBeCloseTo(120);

    ex(_d(-30).opposite().asDeg()).toBeCloseTo(150);
    ex(_d(-160).opposite().asDeg()).toBeCloseTo(20);
  });

  test("neg", () => {
    ex(_d(0).neg().asDeg()).toBeCloseTo(0);
    ex(_d(-0).neg().asDeg()).toBeCloseTo(0);
    ex(_d(90).neg().asDeg()).toBeCloseTo(-90);
    ex(_d(180).neg().asDeg()).toBeCloseTo(-180);
    ex(_d(270).neg().asDeg()).toBeCloseTo(90);
    ex(_d(360).neg().asDeg()).toBeCloseTo(0);
    ex(_d(50).neg().asDeg()).toBeCloseTo(-50);
    ex(_d(150).neg().asDeg()).toBeCloseTo(-150);
    ex(_d(200).neg().asDeg()).toBeCloseTo(160);
    ex(_d(300).neg().asDeg()).toBeCloseTo(60);
    ex(_d(-30).neg().asDeg()).toBeCloseTo(30);
    ex(_d(-160).neg().asDeg()).toBeCloseTo(160);
  });

  test("as_sort_value", () => {
    const angles = [
      0, 90, 180, 270, 360, 45, 135, 225, 315, 30, 120, 210, 300, 60, 150, 240,
      330, 15, 105, 195, 285, 75, 165, 255, 345,
    ];

    const cartesianProduct = angles.flatMap((a) => angles.map((b) => [a, b]));

    cartesianProduct.forEach(([a, b]) => {
      ex(_d(a).asSortValue().sub(_d(b).asSortValue()).sign()).toBeCloseTo(
        Math.sign(a - b),
      );
    });
  });
});

describe("Vec2", () => {
  test("add", () => {
    const v = asVec(1, 2).add(asVec(3, 4));
    ex(v.x).toBeCloseTo(4);
    ex(v.y).toBeCloseTo(6);
  });

  test("sub", () => {
    const v = asVec(1, 2).sub(asVec(3, 4));
    ex(v.x).toBeCloseTo(-2);
    ex(v.y).toBeCloseTo(-2);
  });

  test("scale", () => {
    const v = asVec(1, 2).scale(3);
    ex(v.x).toBeCloseTo(3);
    ex(v.y).toBeCloseTo(6);
  });

  test("dot", () => {
    const v = asVec(1, 2).dot(asVec(3, 4));
    ex(v).toBeCloseTo(11);
  });

  test("cross", () => {
    const v = asVec(1, 2).cross(asVec(3, 4));
    ex(v).toBeCloseTo(-2);
  });

  test("norm", () => {
    const v = asVec(3, 4).norm();
    ex(v).toBeCloseTo(5);
  });

  test("normalize", () => {
    ex(asVec(3, 4).normalize().norm()).toBeCloseTo(1);
  });

  test("as_angle", () => {
    ex(asVec(3, 3).asAngle().asDeg()).toBeCloseTo(45);
    ex(asVec(3, -3).asAngle().asDeg()).toBeCloseTo(-45);
    ex(asVec(-3, 3).asAngle().asDeg()).toBeCloseTo(135);
    ex(asVec(-3, -3).asAngle().asDeg()).toBeCloseTo(-135);
  });

  test("vec_from_polar_coords", () => {
    const v = vecFromPolarCoords(3, angleFromDeg(33));
    ex(v.asAngle().asDeg()).toBeCloseTo(33);
    ex(v.norm()).toBeCloseTo(3);
  });

  test("rotate", () => {
    const v = vecFromPolarCoords(3, angleFromDeg(33)).rotate(angleFromDeg(22));
    ex(v.asAngle().asDeg()).toBeCloseTo(55);

    const v2 = vecFromPolarCoords(3, angleFromDeg(33)).rotate(
      angleFromDeg(-55),
    );
    ex(v2.asAngle().asDeg()).toBeCloseTo(-22);
  });
});
