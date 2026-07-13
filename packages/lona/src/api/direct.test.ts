import { describe, expect, test } from "vitest";
import { variableNum as symbolicVariableNum } from "../core/num";
import type { VarName } from "../core/tree";
import { compileGradRoutine } from "../eval/routines";
import { atan2 as symbolicAtan2 } from "./ops";
import * as direct from "./direct";

const xName = "x" as VarName;
const yName = "y" as VarName;

function directVars(
  xv: number,
  yv: number,
): [direct.DirectNum, direct.DirectNum] {
  return [direct.variableNum(xName, xv), direct.variableNum(yName, yv)];
}

function varMap(xv: number, yv: number): Map<VarName, number> {
  return new Map<VarName, number>([
    [xName, xv],
    [yName, yv],
  ]);
}

describe("direct DirectNum", () => {
  test("computes value and derivatives while building expressions", () => {
    const [x, y] = directVars(2, 3);
    const f = x.square().add(x.mul(y).mul(3)).add(y.sin());

    expect(f.value).toBeCloseTo(22 + Math.sin(3), 10);
    expect(f.derivative(xName)).toBeCloseTo(13, 10);
    expect(f.derivative(yName)).toBeCloseTo(6 + Math.cos(3), 10);
    const gradient = f.gradient([xName, yName]);
    expect(gradient[0]).toBeCloseTo(13, 10);
    expect(gradient[1]).toBeCloseTo(6 + Math.cos(3), 10);
  });

  test("matches symbolic gradient routine for common operations", () => {
    const sx = symbolicVariableNum(xName);
    const sy = symbolicVariableNum(yName);
    const symbolicExprs = [
      sx.div(sy.add(4)),
      sx.mul(sx).add(sy.mul(sy)).add(1).sqrt(),
      sx.sub(sy).abs(),
      sx.mul(0.1).exp(),
      sx.mul(sx).add(1).log(),
      sx.sin().mul(sy.cos()),
      sx.atan().add(sy.tanh()),
      symbolicAtan2(sx, sy),
    ];

    const directExprs = (xv: number, yv: number) => {
      const [x, y] = directVars(xv, yv);
      return [
        x.div(y.add(4)),
        x.mul(x).add(y.mul(y)).add(1).sqrt(),
        x.sub(y).abs(),
        x.mul(0.1).exp(),
        x.mul(x).add(1).log(),
        x.sin().mul(y.cos()),
        x.atan().add(y.tanh()),
        direct.atan2(x, y),
      ];
    };

    const routines = symbolicExprs.map((expr) => {
      const routine = compileGradRoutine([expr.n], [xName, yName], {
        backend: "js-interp",
      });
      if (!routine || routine.shape !== "grad") {
        throw new Error("Expected gradient routine");
      }
      return routine;
    });

    for (const [xv, yv] of [
      [1, 2],
      [-3, 5],
      [0.5, -1.5],
      [10, -7],
    ] as const) {
      const exprs = directExprs(xv, yv);
      for (let i = 0; i < exprs.length; i++) {
        const got = exprs[i]!;
        const expected = routines[i]!.eval(varMap(xv, yv));
        expect(got.value).toBeCloseTo(expected.val, 8);
        expect(got.derivative(xName)).toBeCloseTo(expected.gradient[0]!, 8);
        expect(got.derivative(yName)).toBeCloseTo(expected.gradient[1]!, 8);
      }
    }
  });

  test("branches min/max using the selected value", () => {
    const [x, y] = directVars(5, 3);
    expect(x.max(y).value).toBe(5);
    expect(x.max(y).derivative(xName)).toBe(1);
    expect(x.max(y).derivative(yName)).toBe(0);

    expect(x.min(y).value).toBe(3);
    expect(x.min(y).derivative(xName)).toBe(0);
    expect(x.min(y).derivative(yName)).toBe(1);
  });

  test("wrapNumFn exposes eval and derivative helpers", () => {
    const f = direct.wrapNumFn<[number, number]>((x, y) =>
      x.square().add(x.mul(y).mul(3)).add(y.sin()),
    );

    expect(f.eval(2, 3)).toBeCloseTo(22 + Math.sin(3), 10);
    expect(f.derivative(0).eval(2, 3)).toBeCloseTo(13, 10);
    expect(f.derivative("arg1").eval(2, 3)).toBeCloseTo(6 + Math.cos(3), 10);
    expect(f.vars()).toEqual(["arg0", "arg1"]);
  });

  test("when evaluates only the selected branch", () => {
    const x = direct.variableNum(xName, 2);

    const fromThenElse = direct
      .when(x.greaterThan(0))
      .then(() => x.square())
      .else(() => {
        throw new Error("unselected branch should not run");
      });

    expect(fromThenElse.value).toBe(4);
    expect(fromThenElse.derivative(xName)).toBe(4);

    const fromElse = direct
      .when(x.lessThan(0))
      .then(() => {
        throw new Error("unselected branch should not run");
      })
      .else(() => x.neg());

    expect(fromElse.value).toBe(-2);
    expect(fromElse.derivative(xName)).toBe(-1);
  });

  test("when supports elseIf chains", () => {
    const x = direct.variableNum(xName, 2);

    const result = direct
      .when(x.lessThan(0))
      .then(() => {
        throw new Error("unselected branch should not run");
      })
      .elseIf(x.lessThan(3))
      .then(() => x.add(10))
      .elseIf(() => {
        throw new Error("condition after selected branch should not run");
      })
      .then(() => {
        throw new Error("unselected branch should not run");
      })
      .else(() => x.neg());

    expect(result.value).toBe(12);
    expect(result.derivative(xName)).toBe(1);
  });

  test("cases supports switch-like case clauses", () => {
    const x = direct.variableNum(xName, 2);

    const result = direct
      .cases(x)
      .case(-1, () => {
        throw new Error("unselected branch should not run");
      })
      .case(2, () => x.add(10))
      .case(3, () => {
        throw new Error("unselected branch should not run");
      })
      .default(() => x.neg());

    expect(result.value).toBe(12);
    expect(result.derivative(xName)).toBe(1);

    const fallback = direct
      .cases(x)
      .case(0, () => {
        throw new Error("unselected branch should not run");
      })
      .default(() => x.neg());

    expect(fallback.value).toBe(-2);
    expect(fallback.derivative(xName)).toBe(-1);
  });

  test("routineFromDirectFn reevaluates direct expressions from routine inputs", async () => {
    const f = (x: direct.DirectNum, y: direct.DirectNum) =>
      x.square().add(x.mul(y).mul(3)).add(y.sin());
    const routine = direct.routineFromDirectFn(
      f,
      [xName, yName],
      [xName, yName],
    );

    const at23 = routine.eval(
      new Map<VarName, number>([
        [xName, 2],
        [yName, 3],
      ]),
    );
    expect(at23.val).toBeCloseTo(22 + Math.sin(3), 10);
    expect(at23.gradient[0]).toBeCloseTo(13, 10);
    expect(at23.gradient[1]).toBeCloseTo(6 + Math.cos(3), 10);

    const at11 = routine.eval(
      new Map<VarName, number>([
        [xName, 1],
        [yName, 1],
      ]),
    );
    expect(at11.val).toBeCloseTo(4 + Math.sin(1), 10);
    expect(at11.gradient[0]).toBeCloseTo(5, 10);
    expect(at11.gradient[1]).toBeCloseTo(3 + Math.cos(1), 10);

    const batch = await routine.evalBatch(
      new Map<VarName, number[]>([
        [xName, [2, 1]],
        [yName, [3, 1]],
      ]),
    );
    expect(batch[0]!.val).toBeCloseTo(at23.val, 10);
    expect(batch[1]!.val).toBeCloseTo(at11.val, 10);

    const packed = await routine.evalBatchPacked!(
      new Float32Array([2, 3, 1, 1]),
      2,
    );
    expect(packed[0]).toBeCloseTo(at23.val, 5);
    expect(packed[3]).toBeCloseTo(at11.val, 5);
  });

  test("buildRoutine discovers direct variables and reevaluates inputs", async () => {
    const routine = direct.buildRoutine(
      ({ variable, asNum, when }) => {
        const x = variable(xName);
        const y = variable(yName);
        return asNum(2)
          .mul(x)
          .add(
            when(y.greaterThan(0))
              .then(() => y.square())
              .else(() => y.neg()),
          );
      },
      { diffVars: "all" },
    );

    expect(routine.shape).toBe("grad");
    expect(routine.varSlots).toEqual([xName, yName]);

    const at34 = routine.eval(varMap(3, 4));
    expect(at34.val).toBeCloseTo(22, 10);
    expect(at34.gradient[0]).toBeCloseTo(2, 10);
    expect(at34.gradient[1]).toBeCloseTo(8, 10);

    const at25 = routine.eval(varMap(2, 5));
    expect(at25.val).toBeCloseTo(29, 10);
    expect(at25.gradient[0]).toBeCloseTo(2, 10);
    expect(at25.gradient[1]).toBeCloseTo(10, 10);

    const packed = await routine.evalBatchPacked!(
      new Float32Array([3, 4, 2, 5]),
      2,
    );
    expect(packed[0]).toBeCloseTo(22, 5);
    expect(packed[1]).toBeCloseTo(2, 5);
    expect(packed[2]).toBeCloseTo(8, 5);
    expect(packed[3]).toBeCloseTo(29, 5);
    expect(packed[4]).toBeCloseTo(2, 5);
    expect(packed[5]).toBeCloseTo(10, 5);
  });

  test("buildRoutine builds direct jacobian routines", () => {
    const routine = direct.buildRoutine(
      ({ variable }) => {
        const x = variable(xName);
        const y = variable(yName);
        return [x.mul(y), x.add(y.square())];
      },
      { diffVars: [xName, yName] },
    );

    expect(routine.shape).toBe("jacobian");
    const result = routine.eval(varMap(2, 3));
    expect(result.vals[0]).toBeCloseTo(6, 10);
    expect(result.vals[1]).toBeCloseTo(11, 10);
    expect(result.jacobian[0]![0]).toBeCloseTo(3, 10);
    expect(result.jacobian[0]![1]).toBeCloseTo(2, 10);
    expect(result.jacobian[1]![0]).toBeCloseTo(1, 10);
    expect(result.jacobian[1]![1]).toBeCloseTo(6, 10);
  });
});
