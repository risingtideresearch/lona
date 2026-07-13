import {
  asNum,
  compileValueRoutine,
  variableNum,
  type BackendName,
  type MultiValueRoutine,
  type NumNode,
  type ValueRoutine,
} from "lona";
import { Point, Vec2 } from "lona-geom";
import {
  buildHobby,
  buildJiangChen,
  buildKappa,
  buildKnots,
  buildNatural,
  buildYuksel,
  knuckleSlopes,
  splineFromHermite,
  type Segment,
} from "lona-curves";

export type SplineKind =
  | "catmull-uniform"
  | "catmull-centripetal"
  | "catmull-chordal"
  | "catmull-straight"
  | "natural-uniform"
  | "natural-centripetal"
  | "natural-chordal"
  | "hobby"
  | "kappa"
  | "kappa-3"
  | "yuksel"
  | "jiang-chen"
  | "pchip";

export interface DemoPoint {
  id: number;
  x: number;
  y: number;
  knuckle?: boolean;
  curvature?: number;
  contourId?: number;
}

export interface XY {
  x: number;
  y: number;
}

export type DemoBackend = Extract<
  BackendName,
  "js-codegen" | "js-interp" | "wasm-codegen" | "wasm-interp"
>;

export type BranchPruning = "off" | "trace" | "full-trace";

export interface CurveSampleResult {
  points: XY[];
  evalMs: number;
}

interface VariablePoint {
  point: Point;
  xName: string;
  yName: string;
}

interface SplineInputGroup {
  numericPoints: DemoPoint[];
  variablePoints: VariablePoint[];
}

export interface CurveSampleRoutine {
  readonly backend: DemoBackend;
  readonly branchPruning: BranchPruning;
  readonly compileMs: number;
  readonly variableNames: string[];
  readonly sampleCount: number;
  sample(points: DemoPoint[]): CurveSampleResult;
}

export const initialPoints: DemoPoint[] = [
  { id: 1, x: 100, y: 480 },
  { id: 2, x: 190, y: 110 },
  { id: 3, x: 340, y: 455 },
  { id: 4, x: 520, y: 95 },
  { id: 5, x: 720, y: 430 },
];

export function cloneInitialPoints(): DemoPoint[] {
  return initialPoints.map((point) => ({ ...point }));
}

export function pointsForSpline(
  points: DemoPoint[],
  _kind: SplineKind,
): DemoPoint[] {
  // Keep the user's order for every builder. PCHIP is built parametrically, not
  // as an x-sorted y=f(x) graph, so dragging x positions only re-evaluates
  // variables and does not recompile.
  return points;
}

export function sampleCount(pointCount: number): number {
  return Math.max(96, pointCount * 32);
}

function groupSplineInputs(
  numericPoints: DemoPoint[],
  variablePoints: VariablePoint[],
): SplineInputGroup[] {
  const groups = new Map<number, SplineInputGroup>();
  numericPoints.forEach((point, index) => {
    const contourId = point.contourId ?? 0;
    let group = groups.get(contourId);
    if (!group) {
      group = { numericPoints: [], variablePoints: [] };
      groups.set(contourId, group);
    }
    group.numericPoints.push(point);
    group.variablePoints.push(variablePoints[index]);
  });
  return [...groups.values()];
}

export function samplerCacheKey(
  points: DemoPoint[],
  kind: SplineKind,
  backend: DemoBackend,
  branchPruning: BranchPruning = "off",
): string {
  // Only compile when the expression topology changes: backend, spline builder,
  // number/order of user points, or sample count. Dragging point coordinates must
  // only update variable values and re-evaluate the existing routine.
  const topology = points
    .map(
      (point) =>
        `${point.id}@${point.contourId ?? 0}${point.knuckle ? "k" : "s"}:${point.curvature ?? 1}`,
    )
    .join(",");
  return `${backend}:${branchPruning}:${kind}:${topology}:${sampleCount(points.length)}`;
}

export function createCurveSampleRoutine(
  points: DemoPoint[],
  kind: SplineKind,
  backend: DemoBackend = "js-codegen",
  branchPruning: BranchPruning = "off",
): CurveSampleRoutine | null {
  const splinePoints = pointsForSpline(points, kind);
  if (splinePoints.length < 2) return null;

  const variablePoints = splinePoints.map(toVariablePoint);
  const groups = groupSplineInputs(splinePoints, variablePoints);
  const roots: NumNode[] = [];
  let count = 0;
  for (const group of groups) {
    if (group.numericPoints.length < 2) continue;
    const spline = buildDemoSpline(
      group.numericPoints,
      group.variablePoints,
      kind,
    );
    const groupCount = sampleCount(group.numericPoints.length);
    if (roots.length > 0) roots.push(asNum(Number.NaN).n, asNum(Number.NaN).n);
    roots.push(...sampleRoots(spline.segments, groupCount));
    count += groupCount;
  }
  if (roots.length === 0) return null;

  const compileStart = now();
  const routine = compileValueRoutine(roots, {
    backend,
    selectSpecialization: branchPruning === "off" ? false : branchPruning,
  });
  const compileMs = now() - compileStart;

  if (!routine) {
    throw new Error("Could not compile curve sample routine.");
  }

  return {
    backend,
    branchPruning,
    compileMs,
    variableNames: variablePoints.flatMap((point) => [
      point.xName,
      point.yName,
    ]),
    sampleCount: count,
    sample(inputPoints) {
      const values = new Map<string, number>();
      for (const point of inputPoints) {
        values.set(xVariableName(point.id), point.x);
        values.set(yVariableName(point.id), point.y);
      }
      const evalStart = now();
      const sampledPoints = routineValues(routine, values);
      return { points: sampledPoints, evalMs: now() - evalStart };
    },
  };
}

export function sampleSpline(
  points: DemoPoint[],
  kind: SplineKind,
  backend: DemoBackend = "js-codegen",
  branchPruning: BranchPruning = "off",
): XY[] {
  return (
    createCurveSampleRoutine(points, kind, backend, branchPruning)?.sample(
      points,
    ).points ?? []
  );
}

export function labelForKind(kind: SplineKind): string {
  switch (kind) {
    case "catmull-uniform":
      return "Catmull-Rom / uniform";
    case "catmull-centripetal":
      return "Catmull-Rom / centripetal";
    case "catmull-chordal":
      return "Catmull-Rom / chordal";
    case "catmull-straight":
      return "straight Catmull-Rom chords";
    case "natural-uniform":
      return "Natural cubic / uniform";
    case "natural-centripetal":
      return "Natural cubic / centripetal";
    case "natural-chordal":
      return "Natural cubic / chordal";
    case "hobby":
      return "Hobby spline";
    case "kappa":
      return "κ-curves (10 iter)";
    case "kappa-3":
      return "κ-curves (3 iter)";
    case "yuksel":
      return "Yuksel C² spline";
    case "jiang-chen":
      return "Jiang-Chen spline";
    case "pchip":
      return "PCHIP";
  }
}

interface DemoSpline {
  segments: Segment<Point, Vec2>[];
}

function buildDemoSpline(
  numericPoints: DemoPoint[],
  variablePoints: VariablePoint[],
  kind: SplineKind,
): DemoSpline {
  switch (kind) {
    case "catmull-uniform":
    case "catmull-centripetal":
    case "catmull-chordal":
    case "catmull-straight":
      return buildDemoCatmullRom(numericPoints, variablePoints, kind);
    case "pchip":
      return buildDemoPchip(numericPoints, variablePoints);
    case "kappa":
      // κ-curves have no knuckle concept — build straight through, no split.
      return buildKappa(variablePoints.map(({ point }) => point));
    case "kappa-3":
      // Same builder, fewer fixed-point passes — a lighter expression graph
      // to compare against the default 10-iteration κ-curve.
      return buildKappa(
        variablePoints.map(({ point }) => point),
        { iterations: 3 },
      );
    case "jiang-chen":
      return buildJiangChen(
        variablePoints.map(({ point }) => point),
        {
          curvature: numericPoints.map((point) => point.curvature ?? 1),
        },
      );
    case "natural-uniform":
    case "natural-centripetal":
    case "natural-chordal":
    case "hobby":
    case "yuksel":
      return buildSplitOnKnuckles(numericPoints, variablePoints, kind);
  }
}

function buildSplitOnKnuckles(
  numericPoints: DemoPoint[],
  variablePoints: VariablePoint[],
  kind: SplineKind,
): DemoSpline {
  const knuckleIndices = numericPoints
    .map((point, index) => (point.knuckle ? index : -1))
    .filter((index) => index > 0 && index < numericPoints.length - 1);

  if (knuckleIndices.length === 0) {
    return buildSmoothDemoSpline(numericPoints, variablePoints, kind);
  }

  const splitIndices = [0, ...knuckleIndices, numericPoints.length - 1];
  const segments: Segment<Point, Vec2>[] = [];
  for (let i = 0; i < splitIndices.length - 1; i++) {
    const start = splitIndices[i];
    const end = splitIndices[i + 1] + 1;
    const piece = buildSmoothDemoSpline(
      numericPoints.slice(start, end),
      variablePoints.slice(start, end),
      kind,
    );
    segments.push(...piece.segments);
  }

  return { segments };
}

function buildSmoothDemoSpline(
  _numericPoints: DemoPoint[],
  variablePoints: VariablePoint[],
  kind: SplineKind,
): DemoSpline {
  const points = variablePoints.map(({ point }) => point);
  switch (kind) {
    case "natural-uniform":
      return buildNatural(points, { parameterization: "uniform" });
    case "natural-centripetal":
      return buildNatural(points, { parameterization: "centripetal" });
    case "natural-chordal":
      return buildNatural(points, { parameterization: "chordal" });
    case "hobby":
      return buildHobby(points);
    case "kappa":
      return buildKappa(points);
    case "kappa-3":
      return buildKappa(points, { iterations: 3 });
    case "yuksel":
      return buildYuksel(points);
    case "jiang-chen":
      return buildJiangChen(points, {
        curvature: _numericPoints.map((point) => point.curvature ?? 1),
      });
    case "catmull-uniform":
    case "catmull-centripetal":
    case "catmull-chordal":
    case "catmull-straight":
      return buildDemoCatmullRom(_numericPoints, variablePoints, kind);
    case "pchip":
      return buildDemoPchip(_numericPoints, variablePoints);
  }
}

function buildDemoCatmullRom(
  numericPoints: DemoPoint[],
  variablePoints: VariablePoint[],
  kind: SplineKind,
): DemoSpline {
  const points = variablePoints.map(({ point }) => point);
  const parameterization =
    kind === "catmull-chordal"
      ? "chordal"
      : kind === "catmull-centripetal"
        ? "centripetal"
        : "uniform";
  const knots = buildKnots(points, parameterization);
  const scale = kind === "catmull-straight" ? 0 : 1;
  const invSpan = (a: number, b: number) => knots[b].sub(knots[a]).inv();

  const tangents = points.map((_, i) => {
    if (i === 0) return points[0].vecTo(points[1]).scale(invSpan(0, 1));
    if (i === points.length - 1) {
      return points[points.length - 2]
        .vecTo(points[points.length - 1])
        .scale(invSpan(points.length - 2, points.length - 1));
    }

    const a = points[i].vecTo(points[i + 1]).scale(invSpan(i, i + 1));
    const b = points[i - 1].vecTo(points[i + 1]).scale(invSpan(i - 1, i + 1));
    const c = points[i - 1].vecTo(points[i]).scale(invSpan(i - 1, i));
    return a.sub(b).add(c);
  });

  const leaving = tangents.map((tangent) => tangent.scale(scale));
  const arriving = tangents.map((tangent) => tangent.scale(scale));
  if (kind !== "catmull-straight") {
    numericPoints.forEach((point, i) => {
      if (!point.knuckle) return;
      if (i > 0) {
        arriving[i] = points[i - 1].vecTo(points[i]).scale(invSpan(i - 1, i));
      }
      if (i < points.length - 1) {
        leaving[i] = points[i].vecTo(points[i + 1]).scale(invSpan(i, i + 1));
      }
    });
  }

  return splineFromHermite(points, leaving, arriving, knots);
}

function buildDemoPchip(
  numericPoints: DemoPoint[],
  variablePoints: VariablePoint[],
): DemoSpline {
  const knots = numericPoints.map((_, index) => index);
  const xValues = variablePoints.map(({ point }) => point.x);
  const yValues = variablePoints.map(({ point }) => point.y);
  const ks = numericPoints.map((point) => asNum(point.knuckle ? 1 : 0));
  const xSlopes = knuckleSlopes(knots, xValues, ks);
  const ySlopes = knuckleSlopes(knots, yValues, ks);
  const leaving = xSlopes.R.map(
    (xSlope, index) => new Vec2(xSlope, ySlopes.R[index]),
  );
  const arriving = xSlopes.L.map(
    (xSlope, index) => new Vec2(xSlope, ySlopes.L[index]),
  );

  return splineFromHermite(
    variablePoints.map(({ point }) => point),
    leaving,
    arriving,
    knots,
  );
}

function sampleRoots(segments: { at(t: number): Point }[], count: number) {
  const roots: NumNode[] = [];
  if (segments.length === 0 || count < 1) return roots;

  const lastSample = count - 1;
  const lastSegment = segments.length - 1;
  for (let i = 0; i < count; i++) {
    const raw = lastSample === 0 ? 0 : (i / lastSample) * segments.length;
    const segmentIndex = Math.min(lastSegment, Math.floor(raw));
    const t = Math.min(1, raw - segmentIndex);
    const point = segments[segmentIndex].at(t);
    roots.push(point.x.n, point.y.n);
  }
  return roots;
}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function routineValues(
  routine: ValueRoutine | MultiValueRoutine,
  values: Map<string, number>,
): XY[] {
  const raw = routine.eval(values);
  const flat = typeof raw === "number" ? [raw] : raw;
  const out: XY[] = [];
  for (let i = 0; i < flat.length; i += 2) {
    out.push({ x: flat[i], y: flat[i + 1] });
  }
  return out;
}

function toVariablePoint(point: DemoPoint): VariablePoint {
  const xName = xVariableName(point.id);
  const yName = yVariableName(point.id);
  return {
    point: new Point(variableNum(xName), variableNum(yName)),
    xName,
    yName,
  };
}

function xVariableName(id: number): string {
  return `p${id}.x`;
}

function yVariableName(id: number): string {
  return `p${id}.y`;
}
