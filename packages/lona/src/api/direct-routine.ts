import type { VarName } from "../core/tree";
import type {
  GradRoutine,
  GradientResult,
  JacobianResult,
  JacobianRoutine,
  MultiValueRoutine,
  ValueRoutine,
  VarBatch,
  VarMap,
} from "../eval/routines";
import type {
  Branch,
  Condition,
  NumericInput,
  RoutineBuilder,
  RoutineDiffVars,
  WhenApi,
  WhenChainApi,
} from "./interfaces";
import { DirectNum, asNum, variableNum, when as directWhen } from "./direct";

export type DirectRoutine =
  ValueRoutine | MultiValueRoutine | GradRoutine | JacobianRoutine;

export type BuildRoutineOptions = {
  diffVars?: RoutineDiffVars;
};

type SingleBuildRoutineOptions = BuildRoutineOptions & {
  diffVars?: undefined;
};

type DiffBuildRoutineOptions = BuildRoutineOptions & {
  diffVars: RoutineDiffVars;
};

function recordVar(vars: VarName[], seen: Set<VarName>, name: VarName): void {
  if (!seen.has(name)) {
    seen.add(name);
    vars.push(name);
  }
}

function discoveryWhen(condition: Condition<DirectNum>): WhenApi<DirectNum> {
  if (typeof condition === "function") condition();

  const chain = (): WhenChainApi<DirectNum> => ({
    elseIf: (nextCondition: Condition<DirectNum>): WhenApi<DirectNum> =>
      discoveryWhen(nextCondition),
    else: (fallback: Branch<DirectNum>): DirectNum => {
      fallback();
      return asNum(0);
    },
  });

  return {
    then: (ifNonZero: Branch<DirectNum>): WhenChainApi<DirectNum> => {
      ifNonZero();
      return chain();
    },
  };
}

function inferBatchSize(vars: VarBatch, explicit?: number): number {
  if (explicit !== undefined) return explicit;
  if (vars instanceof Map) {
    const first = vars.values().next().value as number[] | undefined;
    return first?.length ?? 1;
  }
  const first = Object.values(vars)[0];
  return first?.length ?? 1;
}

function varMapValue(vars: VarMap, name: VarName): number {
  return (vars as Map<VarName, number>).get(name) ?? 0;
}

function varBatchValue(vars: VarBatch, name: VarName, point: number): number {
  if (vars instanceof Map) {
    return (vars as Map<VarName, number[]>).get(name)?.[point] ?? 0;
  }
  return (vars as Record<string, number[]>)[String(name)]?.[point] ?? 0;
}

function rootsFromResult(
  result: DirectNum | readonly DirectNum[],
): DirectNum[] {
  return Array.isArray(result) ? result.slice() : [result];
}

function rootsFromBuildResult(
  result: NumericInput<DirectNum> | readonly NumericInput<DirectNum>[],
): DirectNum[] {
  if (Array.isArray(result)) {
    return (result as readonly NumericInput<DirectNum>[]).map(asNum);
  }
  return [asNum(result as NumericInput<DirectNum>)];
}

function resultFromBuildResult(
  result: NumericInput<DirectNum> | readonly NumericInput<DirectNum>[],
): DirectNum | readonly DirectNum[] {
  const roots = rootsFromBuildResult(result);
  return Array.isArray(result) ? roots : roots[0]!;
}

function gradientResult(
  root: DirectNum,
  diffVars: readonly VarName[],
): GradientResult {
  return {
    val: root.value,
    gradient: diffVars.map((name) => root.derivative(name)),
  };
}

function jacobianResult(
  roots: readonly DirectNum[],
  diffVars: readonly VarName[],
): JacobianResult {
  return {
    vals: roots.map((root) => root.value),
    jacobian: roots.map((root) =>
      diffVars.map((name) => root.derivative(name)),
    ),
  };
}

export type DirectRoutineFnResult = DirectNum | readonly DirectNum[];
export type DirectRoutineFn = (...args: DirectNum[]) => DirectRoutineFnResult;

function directArgsFromVarMap(
  varNames: readonly VarName[],
  vars: VarMap,
): DirectNum[] {
  return varNames.map((name) => variableNum(name, varMapValue(vars, name)));
}

function directArgsFromVarBatch(
  varNames: readonly VarName[],
  vars: VarBatch,
  point: number,
): DirectNum[] {
  return varNames.map((name) =>
    variableNum(name, varBatchValue(vars, name, point)),
  );
}

function directArgsFromPacked(
  varNames: readonly VarName[],
  varData: Float32Array,
  point: number,
): DirectNum[] {
  return varNames.map((name, index) =>
    variableNum(name, varData[point * varNames.length + index] ?? 0),
  );
}

function evalDirectRoutineFn(
  fn: DirectRoutineFn,
  varNames: readonly VarName[],
  vars: VarMap,
): DirectNum[] {
  return rootsFromResult(fn(...directArgsFromVarMap(varNames, vars)));
}

function evalDirectRoutineFnBatchPoint(
  fn: DirectRoutineFn,
  varNames: readonly VarName[],
  vars: VarBatch,
  point: number,
): DirectNum[] {
  return rootsFromResult(fn(...directArgsFromVarBatch(varNames, vars, point)));
}

function evalDirectRoutineFnPackedPoint(
  fn: DirectRoutineFn,
  varNames: readonly VarName[],
  varData: Float32Array,
  point: number,
): DirectNum[] {
  return rootsFromResult(fn(...directArgsFromPacked(varNames, varData, point)));
}

export function buildRoutine(
  builder: RoutineBuilder<DirectNum, NumericInput<DirectNum>>,
  opts?: SingleBuildRoutineOptions,
): ValueRoutine;
export function buildRoutine(
  builder: RoutineBuilder<DirectNum, readonly NumericInput<DirectNum>[]>,
  opts?: SingleBuildRoutineOptions,
): MultiValueRoutine;
export function buildRoutine(
  builder: RoutineBuilder<DirectNum, NumericInput<DirectNum>>,
  opts: DiffBuildRoutineOptions,
): GradRoutine;
export function buildRoutine(
  builder: RoutineBuilder<DirectNum, readonly NumericInput<DirectNum>[]>,
  opts: DiffBuildRoutineOptions,
): JacobianRoutine;
export function buildRoutine(
  builder: RoutineBuilder<
    DirectNum,
    NumericInput<DirectNum> | readonly NumericInput<DirectNum>[]
  >,
  opts: BuildRoutineOptions = {},
): DirectRoutine {
  const varNames: VarName[] = [];
  const seen = new Set<VarName>();
  const discoveryResult = builder({
    variable: (name: VarName): DirectNum => {
      recordVar(varNames, seen, name);
      return variableNum(name, 0);
    },
    asNum,
    when: discoveryWhen,
  });
  if (rootsFromBuildResult(discoveryResult).length === 0) {
    throw new Error("buildRoutine requires a root");
  }

  const diffVars = opts.diffVars === "all" ? varNames : (opts.diffVars ?? []);

  const fn: DirectRoutineFn = (...args: DirectNum[]) => {
    const byName = new Map<VarName, DirectNum>();
    for (let i = 0; i < varNames.length; i++) {
      byName.set(varNames[i]!, args[i]!);
    }
    return resultFromBuildResult(
      builder({
        variable: (name: VarName): DirectNum => {
          const value = byName.get(name);
          if (!value) {
            throw new Error(
              `Variable '${String(name)}' was not discovered while building the routine`,
            );
          }
          return value;
        },
        asNum,
        when: directWhen,
      }),
    );
  };

  return (
    routineFromDirectFn as (
      fn: DirectRoutineFn,
      varNames: readonly VarName[],
      diffVars?: readonly VarName[],
    ) => DirectRoutine
  )(fn, varNames, diffVars);
}

export function routineFromDirectFn(
  fn: (...args: DirectNum[]) => DirectNum,
  varNames: readonly VarName[],
): ValueRoutine;
export function routineFromDirectFn(
  fn: (...args: DirectNum[]) => readonly DirectNum[],
  varNames: readonly VarName[],
): MultiValueRoutine;
export function routineFromDirectFn(
  fn: (...args: DirectNum[]) => DirectNum,
  varNames: readonly VarName[],
  diffVars: readonly VarName[],
): GradRoutine;
export function routineFromDirectFn(
  fn: (...args: DirectNum[]) => readonly DirectNum[],
  varNames: readonly VarName[],
  diffVars: readonly VarName[],
): JacobianRoutine;
export function routineFromDirectFn(
  fn: DirectRoutineFn,
  varNames: readonly VarName[],
  diffVars: readonly VarName[] = [],
): DirectRoutine {
  const base = {
    varSlots: varNames,
    numVars: varNames.length,
  };
  const rootsAtZero = rootsFromResult(
    fn(...varNames.map((name) => variableNum(name, 0))),
  );

  if (diffVars.length > 0) {
    if (rootsAtZero.length === 1) {
      return {
        ...base,
        shape: "grad",
        diffVars,
        eval: (vars: VarMap) =>
          gradientResult(evalDirectRoutineFn(fn, varNames, vars)[0]!, diffVars),
        evalAsync: (vars: VarMap) =>
          Promise.resolve(
            gradientResult(
              evalDirectRoutineFn(fn, varNames, vars)[0]!,
              diffVars,
            ),
          ),
        evalBatch: (vars: VarBatch, numPoints?: number) => {
          const n = inferBatchSize(vars, numPoints);
          return Promise.resolve(
            Array.from({ length: n }, (_, point) =>
              gradientResult(
                evalDirectRoutineFnBatchPoint(fn, varNames, vars, point)[0]!,
                diffVars,
              ),
            ),
          );
        },
        evalBatchPacked: (varData: Float32Array, numPoints: number) => {
          const rowWidth = 1 + diffVars.length;
          const out = new Float32Array(numPoints * rowWidth);
          for (let point = 0; point < numPoints; point++) {
            const result = gradientResult(
              evalDirectRoutineFnPackedPoint(fn, varNames, varData, point)[0]!,
              diffVars,
            );
            out[point * rowWidth] = result.val;
            for (let d = 0; d < diffVars.length; d++) {
              out[point * rowWidth + 1 + d] = result.gradient[d]!;
            }
          }
          return Promise.resolve(out);
        },
      } satisfies GradRoutine;
    }

    return {
      ...base,
      shape: "jacobian",
      diffVars,
      numRoots: rootsAtZero.length,
      eval: (vars: VarMap) =>
        jacobianResult(evalDirectRoutineFn(fn, varNames, vars), diffVars),
      evalAsync: (vars: VarMap) =>
        Promise.resolve(
          jacobianResult(evalDirectRoutineFn(fn, varNames, vars), diffVars),
        ),
      evalBatch: (vars: VarBatch, numPoints?: number) => {
        const n = inferBatchSize(vars, numPoints);
        return Promise.resolve(
          Array.from({ length: n }, (_, point) =>
            jacobianResult(
              evalDirectRoutineFnBatchPoint(fn, varNames, vars, point),
              diffVars,
            ),
          ),
        );
      },
      evalBatchPacked: (varData: Float32Array, numPoints: number) => {
        const rowWidth = 1 + diffVars.length;
        const out = new Float32Array(numPoints * rootsAtZero.length * rowWidth);
        for (let point = 0; point < numPoints; point++) {
          const result = jacobianResult(
            evalDirectRoutineFnPackedPoint(fn, varNames, varData, point),
            diffVars,
          );
          for (let root = 0; root < result.vals.length; root++) {
            const offset =
              point * rootsAtZero.length * rowWidth + root * rowWidth;
            out[offset] = result.vals[root]!;
            for (let d = 0; d < diffVars.length; d++) {
              out[offset + 1 + d] = result.jacobian[root]![d]!;
            }
          }
        }
        return Promise.resolve(out);
      },
    } satisfies JacobianRoutine;
  }

  if (rootsAtZero.length === 1) {
    return {
      ...base,
      shape: "value",
      eval: (vars: VarMap) => evalDirectRoutineFn(fn, varNames, vars)[0]!.value,
      evalAsync: (vars: VarMap) =>
        Promise.resolve(evalDirectRoutineFn(fn, varNames, vars)[0]!.value),
      evalBatch: (vars: VarBatch, numPoints?: number) => {
        const n = inferBatchSize(vars, numPoints);
        const out = new Float32Array(n);
        for (let point = 0; point < n; point++) {
          out[point] = evalDirectRoutineFnBatchPoint(
            fn,
            varNames,
            vars,
            point,
          )[0]!.value;
        }
        return Promise.resolve(out);
      },
      evalBatchPacked: (varData: Float32Array, numPoints: number) => {
        const out = new Float32Array(numPoints);
        for (let point = 0; point < numPoints; point++) {
          out[point] = evalDirectRoutineFnPackedPoint(
            fn,
            varNames,
            varData,
            point,
          )[0]!.value;
        }
        return Promise.resolve(out);
      },
    } satisfies ValueRoutine;
  }

  return {
    ...base,
    shape: "multi-value",
    numRoots: rootsAtZero.length,
    eval: (vars: VarMap) =>
      evalDirectRoutineFn(fn, varNames, vars).map((root) => root.value),
    evalAsync: (vars: VarMap) =>
      Promise.resolve(
        evalDirectRoutineFn(fn, varNames, vars).map((root) => root.value),
      ),
    evalBatch: (vars: VarBatch, numPoints?: number) => {
      const n = inferBatchSize(vars, numPoints);
      const out = new Float32Array(n * rootsAtZero.length);
      for (let point = 0; point < n; point++) {
        const roots = evalDirectRoutineFnBatchPoint(fn, varNames, vars, point);
        for (let root = 0; root < roots.length; root++) {
          out[point * rootsAtZero.length + root] = roots[root]!.value;
        }
      }
      return Promise.resolve(out);
    },
    evalBatchPacked: (varData: Float32Array, numPoints: number) => {
      const out = new Float32Array(numPoints * rootsAtZero.length);
      for (let point = 0; point < numPoints; point++) {
        const roots = evalDirectRoutineFnPackedPoint(
          fn,
          varNames,
          varData,
          point,
        );
        for (let root = 0; root < roots.length; root++) {
          out[point * rootsAtZero.length + root] = roots[root]!.value;
        }
      }
      return Promise.resolve(out);
    },
  } satisfies MultiValueRoutine;
}
