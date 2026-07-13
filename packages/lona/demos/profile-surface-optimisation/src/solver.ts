import type { NumericApi, NumericInput, RoutineBuildContext } from "lona";
import { direct } from "lona";
import {
  optimiseResiduals,
  type BackendChoice,
  type LevenbergMarquardtStats,
  type OptimisationSettings,
  type ResidualBuilder,
  type SelectSpecializationChoice,
} from "./optimisation";

export type {
  BackendChoice,
  OptimisationSettings,
  SelectSpecializationChoice,
} from "./optimisation";

/**
 * A profile point whose x coordinate is expressed in some numeric API (the
 * symbolic `Num`, the `direct` forward-autodiff num, …). The y coordinate is
 * always a plain number — only the x coordinates are optimisation variables.
 */
type GeometryPoint<TNum> = {
  x: NumericInput<TNum>;
  y: number;
};

export type NumericProfilePoint = {
  x: number;
  y: number;
};

export type ProfileVariable = {
  /** Variable name used in the Lona expression. */
  name: string;
  /** Which point's x coordinate this variable drives. */
  pointIndex: number;
  /** Original/profile-space x. The solver variable is initialised to sqrt(initialX). */
  initialX: number;
};

export type LimitTarget = {
  name: string;
  limitY: number;
  targetSurface: number;
};

export type OptimisationStats = LevenbergMarquardtStats & {
  expressionBuildMs: number;
  totalMs: number;
  backend: BackendChoice;
  selectSpecialization: SelectSpecializationChoice;
};

export type SurfaceRow = LimitTarget & {
  surface: number;
};

export type OptimisationResult = {
  originalVariables: Record<string, number>;
  finalVariables: Record<string, number>;
  originalProfile: NumericProfilePoint[];
  finalProfile: NumericProfilePoint[];
  originalSurfaces: SurfaceRow[];
  finalSurfaces: SurfaceRow[];
  stats: OptimisationStats;
};

export function profileVariablesFor(
  profile: NumericProfilePoint[],
): ProfileVariable[] {
  return profile.slice(1).map((point, index) => ({
    name: `x_${index + 1}`,
    pointIndex: index + 1,
    initialX: point.x,
  }));
}

function lerpProfileX<TNum extends NumericApi<TNum>>(
  ctx: RoutineBuildContext<TNum>,
  y: TNum,
  a: GeometryPoint<TNum>,
  b: GeometryPoint<TNum>,
): TNum {
  const t = y.sub(a.y).div(b.y - a.y);
  return ctx.asNum(a.x).add(t.mul(ctx.asNum(b.x).sub(a.x)));
}

function fullSliceSurface<TNum extends NumericApi<TNum>>(
  ctx: RoutineBuildContext<TNum>,
  a: GeometryPoint<TNum>,
  b: GeometryPoint<TNum>,
): TNum {
  // Full mirrored trapezoid area:
  //   height * (2*a.x + 2*b.x) / 2 = height * (a.x + b.x)
  return ctx.asNum(b.y - a.y).mul(ctx.asNum(a.x).add(b.x));
}

function partialSliceSurface<TNum extends NumericApi<TNum>>(
  ctx: RoutineBuildContext<TNum>,
  limitY: TNum,
  a: GeometryPoint<TNum>,
  b: GeometryPoint<TNum>,
): TNum {
  const xAtLimit = lerpProfileX(ctx, limitY, a, b);
  return limitY.sub(a.y).mul(ctx.asNum(a.x).add(xAtLimit));
}

function surfaceBelowY<TNum extends NumericApi<TNum>>(
  ctx: RoutineBuildContext<TNum>,
  limitY: TNum,
  profile: GeometryPoint<TNum>[],
): TNum {
  if (profile.length < 2) throw new Error("profile needs at least 2 points");
  if (profile[0]!.y > profile[profile.length - 1]!.y) {
    throw new Error("profile points must be sorted from bottom to top");
  }

  let surface = ctx.asNum(0);
  for (let i = 0; i < profile.length - 1; i++) {
    const a = profile[i]!;
    const b = profile[i + 1]!;

    const sliceContribution = ctx
      .when(limitY.greaterThanOrEqual(b.y))
      .then(() => fullSliceSurface(ctx, a, b))
      .elseIf(limitY.greaterThan(a.y))
      .then(() => partialSliceSurface(ctx, limitY, a, b))
      .else(() => 0);

    surface = surface.add(sliceContribution);
  }

  return surface;
}

function buildVariableProfile<TNum extends NumericApi<TNum>>(
  ctx: RoutineBuildContext<TNum>,
  baseProfile: NumericProfilePoint[],
  profileVariables: ProfileVariable[],
): GeometryPoint<TNum>[] {
  const variableByPoint = new Map<number, TNum>(
    profileVariables.map((variable) => [
      variable.pointIndex,
      ctx.variable(variable.name).square(),
    ]),
  );

  return baseProfile.map((point, pointIndex) => ({
    x: variableByPoint.get(pointIndex) ?? point.x,
    y: point.y,
  }));
}

function initialVariableValues(
  profileVariables: ProfileVariable[],
): Record<string, number> {
  return Object.fromEntries(
    profileVariables.map((variable) => [
      variable.name,
      Math.sqrt(Math.max(0, variable.initialX)),
    ]),
  );
}

/**
 * Build the residual vector with whatever numeric API `ctx` exposes. The same
 * builder drives every backend: the compiled symbolic backends discover the
 * expression once and lower it to a routine, while the `direct` backend simply
 * re-runs it on every evaluation with forward-mode derivatives.
 */
function buildResiduals<TNum extends NumericApi<TNum>>(
  ctx: RoutineBuildContext<TNum>,
  baseProfile: NumericProfilePoint[],
  targets: LimitTarget[],
  profileVariables: ProfileVariable[],
  settings: OptimisationSettings,
): TNum[] {
  const profile = buildVariableProfile(ctx, baseProfile, profileVariables);
  const residuals: TNum[] = [];

  for (const target of targets) {
    const surface = surfaceBelowY(ctx, ctx.asNum(target.limitY), profile);
    const normaliser = Math.max(Math.abs(target.targetSurface), 1);
    residuals.push(surface.sub(target.targetSurface).div(normaliser));
  }

  const regularisationScale = Math.sqrt(settings.regularisationWeight);
  if (regularisationScale > 0) {
    for (const variable of profileVariables) {
      const normaliser = Math.max(Math.abs(variable.initialX), 1);
      residuals.push(
        ctx
          .variable(variable.name)
          .square()
          .sub(variable.initialX)
          .div(normaliser)
          .mul(regularisationScale),
      );
    }
  }

  return residuals;
}

/**
 * A `RoutineBuildContext` backed by the `direct` num, bound to concrete
 * variable values. Reusing the generic geometry above, this lets us evaluate
 * the profile and surfaces for display without any compilation step.
 */
function directEvalContext(
  values: Record<string, number>,
): RoutineBuildContext<direct.DirectNum> {
  return {
    variable: (name) => direct.variableNum(name, values[String(name)] ?? 0),
    asNum: direct.asNum,
    when: direct.when,
  };
}

function evaluateNumericProfile(
  baseProfile: NumericProfilePoint[],
  profileVariables: ProfileVariable[],
  values: Record<string, number>,
): NumericProfilePoint[] {
  const ctx = directEvalContext(values);
  const profile = buildVariableProfile(ctx, baseProfile, profileVariables);
  return profile.map((point) => ({
    x: ctx.asNum(point.x).value,
    y: point.y,
  }));
}

function evaluatedSurfaces(
  baseProfile: NumericProfilePoint[],
  profileVariables: ProfileVariable[],
  targets: LimitTarget[],
  values: Record<string, number>,
): SurfaceRow[] {
  const ctx = directEvalContext(values);
  const profile = buildVariableProfile(ctx, baseProfile, profileVariables);
  return targets.map((target) => ({
    ...target,
    surface: surfaceBelowY(ctx, ctx.asNum(target.limitY), profile).value,
  }));
}

export function runOptimisation(
  baseProfile: NumericProfilePoint[],
  targets: LimitTarget[],
  profileVariables: ProfileVariable[],
  settings: OptimisationSettings,
): OptimisationResult {
  const totalStart = performance.now();
  const buildStart = performance.now();
  const builder: ResidualBuilder = (ctx) =>
    buildResiduals(ctx, baseProfile, targets, profileVariables, settings);
  const variableNames = profileVariables.map((variable) => variable.name);
  const originalVariables = initialVariableValues(profileVariables);
  const expressionBuildMs = performance.now() - buildStart;

  const optimisationResult = optimiseResiduals(
    builder,
    variableNames,
    originalVariables,
    settings,
  );
  const finalVariables = optimisationResult.values;
  const totalMs = performance.now() - totalStart;

  return {
    originalVariables,
    finalVariables,
    originalProfile: evaluateNumericProfile(
      baseProfile,
      profileVariables,
      originalVariables,
    ),
    finalProfile: evaluateNumericProfile(
      baseProfile,
      profileVariables,
      finalVariables,
    ),
    originalSurfaces: evaluatedSurfaces(
      baseProfile,
      profileVariables,
      targets,
      originalVariables,
    ),
    finalSurfaces: evaluatedSurfaces(
      baseProfile,
      profileVariables,
      targets,
      finalVariables,
    ),
    stats: {
      ...optimisationResult.stats,
      expressionBuildMs,
      totalMs,
      backend: settings.backend,
      selectSpecialization: settings.selectSpecialization,
    },
  };
}
