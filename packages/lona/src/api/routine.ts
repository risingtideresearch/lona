import { Num, asNum, variableNum } from "../core/num";
import type { VarName } from "../core/tree";
import {
  compileGradRoutine,
  compileValueRoutine,
  type AnyRoutine,
  type CompileOpts,
  type GradRoutine,
  type JacobianRoutine,
  type MultiValueRoutine,
  type ValueRoutine,
} from "../eval/routines";
import type {
  NumericInput,
  RoutineBuilder,
  RoutineDiffVars,
} from "./interfaces";
import { when } from "./ops";

export type BuildRoutineOptions = CompileOpts & {
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

function rootsFromResult(
  result: NumericInput<Num> | readonly NumericInput<Num>[],
): Num[] {
  if (Array.isArray(result)) {
    return (result as readonly NumericInput<Num>[]).map(asNum);
  }
  return [asNum(result as NumericInput<Num>)];
}

function compileOpts(opts: BuildRoutineOptions): CompileOpts {
  return {
    backend: opts.backend,
    selectSpecialization: opts.selectSpecialization,
    diagnosticCheckpoint: opts.diagnosticCheckpoint,
  };
}

function requireRoutine<TRoutine extends AnyRoutine>(
  routine: TRoutine | null,
): TRoutine {
  if (!routine) throw new Error("Failed to build routine");
  return routine;
}

export function buildRoutine(
  builder: RoutineBuilder<Num, NumericInput<Num>>,
  opts?: SingleBuildRoutineOptions,
): ValueRoutine;
export function buildRoutine(
  builder: RoutineBuilder<Num, readonly NumericInput<Num>[]>,
  opts?: SingleBuildRoutineOptions,
): MultiValueRoutine;
export function buildRoutine(
  builder: RoutineBuilder<Num, NumericInput<Num>>,
  opts: DiffBuildRoutineOptions,
): GradRoutine;
export function buildRoutine(
  builder: RoutineBuilder<Num, readonly NumericInput<Num>[]>,
  opts: DiffBuildRoutineOptions,
): JacobianRoutine;
export function buildRoutine(
  builder: RoutineBuilder<
    Num,
    NumericInput<Num> | readonly NumericInput<Num>[]
  >,
  opts: BuildRoutineOptions = {},
): AnyRoutine {
  const varNames: VarName[] = [];
  const seen = new Set<VarName>();
  const result = builder({
    variable: (name: VarName): Num => {
      recordVar(varNames, seen, name);
      return variableNum(name);
    },
    asNum,
    when,
  });
  const roots = rootsFromResult(result);
  if (roots.length === 0) throw new Error("buildRoutine requires a root");

  const diffVars =
    opts.diffVars === "all"
      ? varNames
      : (opts.diffVars as VarName[] | undefined);

  if (diffVars) {
    return requireRoutine(
      compileGradRoutine(
        roots.map((root) => root.n),
        [...diffVars],
        compileOpts(opts),
      ),
    );
  }

  return requireRoutine(
    compileValueRoutine(
      roots.map((root) => root.n),
      compileOpts(opts),
    ),
  );
}
