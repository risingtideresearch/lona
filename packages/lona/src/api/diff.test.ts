import { test } from "vitest";
import { Num, variableNum } from "../core/num";
import { diff } from "./ops";

import { exVar } from "../test-utils";

test("just a square", () => {
  const x = variableNum("x");
  const f = x.mul(x);

  const df = diff(f);
  //writeFileSync("debug.dot", df.asDot());
  exVar(df, { x: 2, d_x: 1 }).toBeCloseTo(4);
  exVar(df, { x: 6, d_x: 1 }).toBeCloseTo(12);
  exVar(df, { x: 0, d_x: 1 }).toBeCloseTo(0);
});

test("a simple polynomial", () => {
  const x = variableNum("x");
  const f = x.mul(x).add(x).add(1);

  const df = diff(f);
  //writeFileSync("debug.dot", df.asDot());
  exVar(df, { x: 2, d_x: 1 }).toBeCloseTo(5);
  exVar(df, { x: 6, d_x: 1 }).toBeCloseTo(13);
  exVar(df, { x: 0, d_x: 1 }).toBeCloseTo(1);
});

test("roots", () => {
  const x = variableNum("x");
  const f = x.sqrt();
  const f2 = x.cbrt();

  const df = diff(f);
  const df2 = diff(f2);

  const values = [0.2, 8, 9, 27, 33];

  values.forEach((v) => {
    exVar(df, { x: v, d_x: 1 }).toBeCloseTo(1 / (2 * Math.sqrt(v)));
    exVar(df2, { x: v, d_x: 1 }).toBeCloseTo(1 / (3 * Math.cbrt(v ** 2)));
  });
});

test("trig functions", () => {
  const x = variableNum("x");
  const sin = x.sin();
  const cos = x.cos();
  const tan = x.tan();
  const asin = x.asin();
  const acos = x.acos();
  const atan = x.atan();

  const funcs: [Num, (a: number) => number, string][] = [
    [sin, Math.cos, "sin"],
    [cos, (v) => -Math.sin(v), "cos"],
    [tan, (v) => 1 / Math.cos(v) ** 2, "tan"],
    [asin, (v) => 1 / Math.sqrt(1 - v ** 2), "asin"],
    [acos, (v) => -1 / Math.sqrt(1 - v ** 2), "acos"],
    [atan, (v) => 1 / (1 + v ** 2), "atan"],
  ];

  const values = [0, 0.2, 0.4, 0.8, 0.99];

  funcs.forEach(([f, mathF]) => {
    const df = diff(f);
    values.forEach((v) => {
      const expected = mathF(v);
      exVar(df, { x: v, d_x: 1 }).toBeCloseTo(expected);
    });
  });
});

test("exp and log", () => {
  const x = variableNum("x");
  const exp = x.exp();
  const log = x.log();

  const dfExp = diff(exp);
  const dfLog = diff(log);

  const values = [0.2, 0.4, 0.8, 0.99];

  values.forEach((v) => {
    exVar(dfExp, { x: v, d_x: 1 }).toBeCloseTo(Math.exp(v));
    exVar(dfLog, { x: v, d_x: 1 }).toBeCloseTo(1 / v);
  });

  exVar(dfExp, { x: 0, d_x: 1 }).toBeCloseTo(1);
  // This should be inifinity, but we are doing some fishy stuff to avoid
  // NaNs
  exVar(dfLog, { x: 0, d_x: 1 }).toBeCloseTo(1e50);
});

test("abs", () => {
  const x = variableNum("x");
  const abs = x.abs();

  const dfAbs = diff(abs);

  const values = [-1, -0.2, 0, 0.2, 1];

  values.forEach((v) => {
    exVar(dfAbs, { x: v, d_x: 1 }).toBeCloseTo(Math.sign(v));
  });
});

test("sign", () => {
  const x = variableNum("x");
  const sign = x.sign();

  const dfSign = diff(sign);

  const values = [-1, -0.2, 0, 0.2, 1];

  values.forEach((v) => {
    exVar(dfSign, { x: v, d_x: 1 }).toBeCloseTo(0);
  });
});

test("tanh", () => {
  const x = variableNum("x");
  const tanh = x.tanh();

  const dfTanh = diff(tanh);

  const values = [-1, -0.2, 0, 0.2, 1];

  values.forEach((v) => {
    exVar(dfTanh, { x: v, d_x: 1 }).toBeCloseTo(1 - Math.tanh(v) ** 2);
  });
});

test("log1p", () => {
  const x = variableNum("x");
  const log1p = x.log1p();

  const dfLog1p = diff(log1p);

  const values = [-0.2, 0, 0.2, 1];

  values.forEach((v) => {
    exVar(dfLog1p, { x: v, d_x: 1 }).toBeCloseTo(1 / (1 + v));
  });

  // This should be inifinity, but we are doing some fishy stuff to avoid NaNs
  exVar(dfLog1p, { x: -1, d_x: 1 }).toBeCloseTo(1e50);
});

test("and", () => {
  const x = variableNum("x");
  const and = x.and(x.mul(2).sub(2));

  //const dfFun = !x ? 1: 2

  const df = diff(and);

  exVar(df, { x: 1, d_x: 1 }).toBeCloseTo(2);
  exVar(df, { x: 0, d_x: 1 }).toBeCloseTo(1);
});

test("or", () => {
  const x = variableNum("x");
  const or = x.or(x.mul(2).sub(2));

  const df = diff(or);

  exVar(df, { x: 1, d_x: 1 }).toBeCloseTo(1);
  exVar(df, { x: 0, d_x: 1 }).toBeCloseTo(2);
});

test("min", () => {
  const x = variableNum("x");
  const y = variableNum("y");
  const min = x.min(x.mul(y));

  const df = diff(min);

  exVar(df, { x: 2, y: 1.2, d_x: 1, d_y: 0 }).toBeCloseTo(1);
  exVar(df, { x: 2, y: 0.9, d_x: 1, d_y: 0 }).toBeCloseTo(0.9);

  // It always chooses the rhs value when they are equal
  exVar(df, { x: 0, y: 0.9, d_x: 1, d_y: 0 }).toBeCloseTo(0.9);
  exVar(df, { x: 0, y: 1.2, d_x: 1, d_y: 0 }).toBeCloseTo(1.2);
});

test("max", () => {
  const x = variableNum("x");
  const y = variableNum("y");
  const max = x.max(x.mul(y));

  const df = diff(max);

  exVar(df, { x: 2, y: 1.2, d_x: 1, d_y: 0 }).toBeCloseTo(1.2);
  exVar(df, { x: 2, y: 0.9, d_x: 1, d_y: 0 }).toBeCloseTo(1);

  // It always chooses the rhs value when they are equal
  exVar(df, { x: 0, y: 1.2, d_x: 1, d_y: 0 }).toBeCloseTo(1.2);
  exVar(df, { x: 0, y: 0.9, d_x: 1, d_y: 0 }).toBeCloseTo(0.9);
});
