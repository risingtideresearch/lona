import { expect, test } from "vitest";

import { ex, exVar } from "../test-utils";
import { variableNum } from "../core/num";

import {
  max,
  min,
  atan2,
  select,
  ifTruthyElse,
  when,
  cases,
  hypot,
  clamp,
  sigmoid,
} from "./ops";
import { buildRoutine } from "./routine";

test("max", () => {
  ex(max(1, 3)).toBeCloseTo(3);
  ex(max(1, 0)).toBeCloseTo(1);
  ex(max(1, -1)).toBeCloseTo(1);
  ex(max(0.3, 0.1)).toBeCloseTo(0.3);
  ex(max(0.3, 0.1, -2, 12, -123.1)).toBeCloseTo(12);
});

test("min", () => {
  ex(min(1, 3)).toBeCloseTo(1);
  ex(min(1, 0)).toBeCloseTo(0);
  ex(min(1, -1)).toBeCloseTo(-1);
  ex(min(0.3, 0.1)).toBeCloseTo(0.1);
  ex(min(0.3, 0.1, -2, -123.1)).toBeCloseTo(-123.1);
});

test("atan2", () => {
  ex(atan2(1, 1)).toBeCloseTo(Math.PI / 4);
  ex(atan2(1, 0)).toBeCloseTo(Math.PI / 2);
  ex(atan2(1, -1)).toBeCloseTo((3 * Math.PI) / 4);
  ex(atan2(0, 1)).toBeCloseTo(0);
  ex(atan2(0, -1)).toBeCloseTo(Math.PI);
  ex(atan2(-1, 1)).toBeCloseTo(-Math.PI / 4);
  ex(atan2(-1, 0)).toBeCloseTo(-Math.PI / 2);
  ex(atan2(-1, -1)).toBeCloseTo((-3 * Math.PI) / 4);
});

test("select", () => {
  ex(select(1, 3, 4)).toBeCloseTo(3);
  ex(select(0, 3, 4)).toBeCloseTo(4);
  ex(select(-1, 3, 4)).toBeCloseTo(3);
  exVar(select(variableNum("c"), 3, 4), { c: 0 }).toBeCloseTo(4);
  exVar(select(variableNum("c"), 3, 4), { c: 2 }).toBeCloseTo(3);

  const c = variableNum("c");
  const compiled = select(c, c.add(1), c.add(2));
  expect(compiled.eval({ c: 0 })).toBeCloseTo(2);
  expect(compiled.eval({ c: 2 })).toBeCloseTo(3);
});

test("if_non_zero_else", () => {
  ex(ifTruthyElse(1, 3, 4)).toBeCloseTo(3);
  ex(ifTruthyElse(0, 3, 4)).toBeCloseTo(4);
  ex(ifTruthyElse(-1, 3, 4)).toBeCloseTo(3);
  ex(ifTruthyElse(0.3, 0.1, 0.2)).toBeCloseTo(0.1);
  ex(ifTruthyElse(0.1, 0.1, 0.2)).toBeCloseTo(0.1);
  ex(ifTruthyElse(0.1, 0.2, 0.1)).toBeCloseTo(0.2);
  ex(ifTruthyElse(-0.1, 0.2, 0.1)).toBeCloseTo(0.2);
  ex(ifTruthyElse(-0, 0.1, 0.2)).toBeCloseTo(0.2);
  ex(ifTruthyElse(0, 10000, 0.2)).toBeCloseTo(0.2);
});

test("when", () => {
  ex(
    when(1)
      .then(() => 3)
      .else(() => 4),
  ).toBeCloseTo(3);
  ex(
    when(0)
      .then(() => 3)
      .else(() => 4),
  ).toBeCloseTo(4);
  ex(
    when(0)
      .then(() => 1)
      .elseIf(1)
      .then(() => 2)
      .else(() => 3),
  ).toBeCloseTo(2);
  ex(
    when(0)
      .then(() => 1)
      .elseIf(0)
      .then(() => 2)
      .else(() => 3),
  ).toBeCloseTo(3);

  const x = variableNum("x");
  const select = when(x)
    .then(() => x.add(1))
    .else(() => x.sub(1));
  exVar(select, { x: 2 }).toBeCloseTo(3);
  exVar(select, { x: 0 }).toBeCloseTo(-1);

  const chainedSelect = when(x.lessThan(0))
    .then(() => x.neg())
    .elseIf(() => x.lessThan(3))
    .then(() => x.add(10))
    .else(() => x.add(20));
  exVar(chainedSelect, { x: -2 }).toBeCloseTo(2);
  exVar(chainedSelect, { x: 2 }).toBeCloseTo(12);
  exVar(chainedSelect, { x: 5 }).toBeCloseTo(25);
});

test("cases", () => {
  const x = variableNum("x");
  const select = cases(x)
    .case(-2, () => x.neg())
    .case(2, () => x.add(10))
    .default(() => x.add(20));

  exVar(select, { x: -2 }).toBeCloseTo(2);
  exVar(select, { x: 2 }).toBeCloseTo(12);
  exVar(select, { x: 5 }).toBeCloseTo(25);

  ex(
    cases(1)
      .case(0, () => 0)
      .case(1, () => 42)
      .default(() => -1),
  ).toBeCloseTo(42);
});

test("hypot", () => {
  ex(hypot(3, 4)).toBeCloseTo(5);
  ex(hypot(1, 1)).toBeCloseTo(Math.sqrt(2));
  ex(hypot(1, 0)).toBeCloseTo(1);
  ex(hypot(0, 1)).toBeCloseTo(1);
  ex(hypot(0, 0)).toBeCloseTo(0);
});

test("clamp", () => {
  ex(clamp(1, 3, 4)).toBeCloseTo(3);
  ex(clamp(5, 3, 4)).toBeCloseTo(4);
  ex(clamp(3.5, 3, 4)).toBeCloseTo(3.5);
  ex(clamp(3, 3, 4)).toBeCloseTo(3);
  ex(clamp(4, 3, 4)).toBeCloseTo(4);
  ex(clamp(1, 3, 3)).toBeCloseTo(3);
  ex(clamp(5, 3, 3)).toBeCloseTo(3);
  ex(clamp(3.5, 3, 3)).toBeCloseTo(3);
  ex(clamp(3, 3, 3)).toBeCloseTo(3);
  ex(clamp(4, 3, 3)).toBeCloseTo(3);
});

test("sigmoid", () => {
  ex(sigmoid(0)).toBeCloseTo(0.5);
  ex(sigmoid(1)).toBeCloseTo(1 / (1 + Math.exp(-1)));
  ex(sigmoid(-1)).toBeCloseTo(1 / (1 + Math.exp(1)));
  ex(sigmoid(3)).toBeCloseTo(1 / (1 + Math.exp(-3)));
  ex(sigmoid(-3)).toBeCloseTo(1 / (1 + Math.exp(3)));
  ex(sigmoid(10)).toBeCloseTo(1);
  ex(sigmoid(-10)).toBeCloseTo(0);
});

test("buildRoutine builds symbolic value and grad routines", () => {
  const checkpoints: string[] = [];
  const valueRoutine = buildRoutine(
    ({ variable, asNum, when }) => {
      const x = variable("x");
      const y = variable("y");
      return asNum(2)
        .mul(x)
        .add(
          when(y.greaterThan(0))
            .then(() => y.square())
            .else(() => y.neg()),
        );
    },
    {
      backend: "js-interp",
      diagnosticCheckpoint: (phase) => checkpoints.push(phase),
    },
  );

  expect(checkpoints).toContain("lona:tape:start");
  expect(valueRoutine.shape).toBe("value");
  expect(
    valueRoutine.eval(
      new Map([
        ["x", 3],
        ["y", 4],
      ]),
    ),
  ).toBeCloseTo(22, 10);

  const gradRoutine = buildRoutine(
    ({ variable, asNum, when }) => {
      const x = variable("x");
      const y = variable("y");
      return asNum(2)
        .mul(x)
        .add(
          when(y.greaterThan(0))
            .then(() => y.square())
            .else(() => y.neg()),
        );
    },
    { backend: "js-interp", diffVars: "all" },
  );

  expect(gradRoutine.shape).toBe("grad");
  const grad = gradRoutine.eval(
    new Map([
      ["x", 3],
      ["y", 4],
    ]),
  );
  expect(grad.val).toBeCloseTo(22, 10);
  expect(grad.gradient[0]).toBeCloseTo(2, 10);
  expect(grad.gradient[1]).toBeCloseTo(8, 10);
});

test("buildRoutine builds symbolic jacobian routines", () => {
  const routine = buildRoutine(
    ({ variable }) => {
      const x = variable("x");
      const y = variable("y");
      return [x.mul(y), x.add(y.square())];
    },
    { backend: "js-interp", diffVars: ["x", "y"] },
  );

  expect(routine.shape).toBe("jacobian");
  const result = routine.eval(
    new Map([
      ["x", 2],
      ["y", 3],
    ]),
  );
  expect(result.vals[0]).toBeCloseTo(6, 10);
  expect(result.vals[1]).toBeCloseTo(11, 10);
  expect(result.jacobian[0]![0]).toBeCloseTo(3, 10);
  expect(result.jacobian[0]![1]).toBeCloseTo(2, 10);
  expect(result.jacobian[1]![0]).toBeCloseTo(1, 10);
  expect(result.jacobian[1]![1]).toBeCloseTo(6, 10);
});
