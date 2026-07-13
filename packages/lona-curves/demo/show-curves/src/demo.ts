import {
  cloneInitialPoints,
  createCurveSampleRoutine,
  labelForKind,
  pointsForSpline,
  samplerCacheKey,
  type BranchPruning,
  type CurveSampleRoutine,
  type DemoBackend,
  type DemoPoint,
  type SplineKind,
  type XY,
} from "./curve-model";
import { curveExamples, type CurveExample } from "./examples";

const canvas = requiredElement("canvas", HTMLCanvasElement);
const exampleKindSelect = requiredElement("example-kind", HTMLSelectElement);
const splineKindSelect = requiredElement("spline-kind", HTMLSelectElement);
const backendKindSelect = requiredElement("backend-kind", HTMLSelectElement);
const branchPruningSelect = requiredElement(
  "branch-pruning",
  HTMLSelectElement,
);
const showTangentsInput = requiredElement("show-tangents", HTMLInputElement);
const showCurvatureInput = requiredElement("show-curvature", HTMLInputElement);
const smallPointsInput = requiredElement("small-points", HTMLInputElement);
const resetButton = requiredElement("reset", HTMLButtonElement);
const deleteButton = requiredElement("delete", HTMLButtonElement);
const clearButton = requiredElement("clear", HTMLButtonElement);
const helpButton = requiredElement("help", HTMLButtonElement);
const helpDialog = requiredElement("help-dialog", HTMLDialogElement);
const pointCount = requiredElement("point-count", HTMLElement);
const selectedPoint = requiredElement("selected-point", HTMLElement);
const message = requiredElement("message", HTMLElement);
const perfIndicator = requiredElement("perf-indicator", HTMLElement);
const compileTime = requiredElement("compile-time", HTMLElement);
const evalTime = requiredElement("eval-time", HTMLElement);
const pruningStatus = requiredElement("pruning-status", HTMLElement);
const ctx = requiredCanvasContext();

// Avoid browser form-value restoration leaving the demo on the straight-line
// reference option after a reload.
populateExampleMenu();
splineKindSelect.value = "catmull-centripetal";
backendKindSelect.value = "wasm-interp";
branchPruningSelect.value = "trace";
updateBranchPruningAvailability();

let points = cloneInitialPoints();
let nextPointId = Math.max(...points.map((point) => point.id)) + 1;
let selectedPointId: number | null = null;
let draggingPointId: number | null = null;
let activePointerId: number | null = null;
let cachedSamplerKey = "";
let cachedSampler: CurveSampleRoutine | null = null;
let lastCurveSamples: XY[] = [];
let lastPointerPosition: XY | null = null;
let shiftKeyDown = false;

function populateExampleMenu(): void {
  for (const example of curveExamples) {
    const option = document.createElement("option");
    option.value = example.id;
    const pointCount = example.contours.reduce(
      (sum, contour) => sum + contour.points.length,
      0,
    );
    option.textContent = `${example.label} (${pointCount} pts)`;
    exampleKindSelect.append(option);
  }
}

function requiredElement<T extends HTMLElement>(
  id: string,
  elementType: new () => T,
): T {
  const element = document.getElementById(id);
  if (!(element instanceof elementType)) {
    throw new Error(`Expected #${id} to be a ${elementType.name}`);
  }
  return element;
}

function requiredCanvasContext(): CanvasRenderingContext2D {
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not create a 2D canvas context");
  }
  return context;
}

function currentSplineKind(): SplineKind {
  return splineKindSelect.value as SplineKind;
}

function currentBackend(): DemoBackend {
  return backendKindSelect.value as DemoBackend;
}

function currentBranchPruning(): BranchPruning {
  if (!backendSupportsBranchPruning(currentBackend())) return "off";
  return branchPruningSelect.value as BranchPruning;
}

function backendSupportsBranchPruning(backend: DemoBackend): boolean {
  // compileValueRoutine select-specialization is currently implemented for
  // these sync value backends. Keep this explicit so unsupported future/backend
  // additions fall back to plain evaluation instead of exposing broken options.
  return (
    backend === "js-codegen" ||
    backend === "js-interp" ||
    backend === "wasm-codegen" ||
    backend === "wasm-interp"
  );
}

function updateBranchPruningAvailability(): void {
  const supported = backendSupportsBranchPruning(currentBackend());
  branchPruningSelect.disabled = !supported;
  branchPruningSelect.title = supported
    ? "Prune unselected select branches during routine compilation/evaluation."
    : "This backend does not support branch pruning.";
  if (!supported) branchPruningSelect.value = "off";
  pruningStatus.textContent = supported
    ? pruningLabel(currentBranchPruning())
    : "Unsupported";
}

function draw(): void {
  const { width, height } = canvasSize();
  ctx.setTransform(
    window.devicePixelRatio || 1,
    0,
    0,
    window.devicePixelRatio || 1,
    0,
    0,
  );
  ctx.clearRect(0, 0, width, height);
  drawGrid(width, height);
  drawControlPolygon();
  drawSpline();
  drawCurveVisualizations();
  drawPoints();
  updatePanel();
}

function isFinitePoint(point: XY): boolean {
  return Number.isFinite(point.x) && Number.isFinite(point.y);
}

function groupedPoints(input: DemoPoint[]): DemoPoint[][] {
  const groups = new Map<number, DemoPoint[]>();
  for (const point of input) {
    const contourId = point.contourId ?? 0;
    const group = groups.get(contourId) ?? [];
    group.push(point);
    groups.set(contourId, group);
  }
  return [...groups.values()];
}

function drawGrid(width: number, height: number): void {
  ctx.save();
  ctx.fillStyle = "#fbfcff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e6ebf5";
  ctx.lineWidth = 1;

  for (let x = 0; x <= width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  for (let y = 0; y <= height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawControlPolygon(): void {
  const orderedPoints = pointsForSpline(points, currentSplineKind());
  const groups = groupedPoints(orderedPoints);
  if (groups.every((group) => group.length < 2)) return;

  ctx.save();
  ctx.setLineDash([8, 8]);
  ctx.strokeStyle = "rgba(82, 97, 126, 0.24)";
  ctx.lineWidth = 1;
  for (const group of groups) {
    if (group.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(group[0].x, group[0].y);
    for (const point of group.slice(1)) {
      ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawSpline(): void {
  let sampled: XY[] = [];
  let text = "Add at least two points to build a spline.";

  try {
    const result = sampleCurrentCurve();
    sampled = result.points;
    lastCurveSamples = sampled;
    updatePerfIndicator(result);
    const finiteSampleCount = sampled.filter(isFinitePoint).length;
    if (finiteSampleCount > 0) {
      text = `${finiteSampleCount} samples from ${labelForKind(currentSplineKind())}.`;
    }
  } catch (error) {
    text = error instanceof Error ? error.message : "Unable to build spline.";
    lastCurveSamples = [];
    updatePerfError();
  }

  message.textContent = text;
  if (sampled.length < 2) return;

  ctx.save();
  ctx.strokeStyle = "#2b68ff";
  ctx.lineWidth = 5;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.shadowColor = "rgba(43, 104, 255, 0.22)";
  ctx.shadowBlur = 14;
  ctx.beginPath();
  let penDown = false;
  for (const point of sampled) {
    if (!isFinitePoint(point)) {
      penDown = false;
      continue;
    }
    if (!penDown) {
      ctx.moveTo(point.x, point.y);
      penDown = true;
    } else {
      ctx.lineTo(point.x, point.y);
    }
  }
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.fillStyle = "#2b68ff";
  const markerStep = Math.max(1, Math.floor(sampled.length / 24));
  for (let i = 0; i < sampled.length; i += markerStep) {
    if (!isFinitePoint(sampled[i])) continue;
    ctx.beginPath();
    ctx.arc(sampled[i].x, sampled[i].y, 2.25, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawCurveVisualizations(): void {
  if (lastCurveSamples.length < 2) return;
  if (showCurvatureInput.checked) drawCurvatureComb();
  if (showTangentsInput.checked) drawTangents();
}

function drawTangents(): void {
  if (points.length < 2) return;

  ctx.save();
  ctx.strokeStyle = "#0f9f6e";
  ctx.fillStyle = "#0f9f6e";
  ctx.lineWidth = 2;
  for (let pointIndex = 0; pointIndex < points.length; pointIndex++) {
    const sampleIndex = sampleIndexForPoint(pointIndex);
    const tangent = tangentAtSample(sampleIndex);
    const length = Math.hypot(tangent.x, tangent.y);
    if (length <= 1e-6) continue;

    const unit = { x: tangent.x / length, y: tangent.y / length };
    const center = points[pointIndex];
    const scale = 48;
    const start = {
      x: center.x - unit.x * scale * 0.5,
      y: center.y - unit.y * scale * 0.5,
    };
    const end = {
      x: center.x + unit.x * scale * 0.5,
      y: center.y + unit.y * scale * 0.5,
    };
    drawArrow(start, end);
  }
  ctx.restore();
}

function drawCurvatureComb(): void {
  if (lastCurveSamples.length < 3) return;

  ctx.save();
  ctx.strokeStyle = "rgba(202, 70, 44, 0.72)";
  ctx.lineWidth = 1.5;
  const step = Math.max(2, Math.floor(lastCurveSamples.length / 64));
  for (let i = step; i < lastCurveSamples.length - step; i += step) {
    const a = lastCurveSamples[i - step];
    const b = lastCurveSamples[i];
    const c = lastCurveSamples[i + step];
    if (!isFinitePoint(a) || !isFinitePoint(b) || !isFinitePoint(c)) continue;
    const curvature = signedCurvature(a, b, c);
    const tangent = { x: c.x - a.x, y: c.y - a.y };
    const tangentLength = Math.hypot(tangent.x, tangent.y);
    if (tangentLength <= 1e-6 || !Number.isFinite(curvature)) continue;

    const normal = {
      x: -tangent.y / tangentLength,
      y: tangent.x / tangentLength,
    };
    const combLength = Math.max(-60, Math.min(60, -curvature * 3200));
    ctx.beginPath();
    ctx.moveTo(b.x, b.y);
    ctx.lineTo(b.x + normal.x * combLength, b.y + normal.y * combLength);
    ctx.stroke();
  }
  ctx.restore();
}

function sampleIndexForPoint(pointIndex: number): number {
  if (points.length <= 1 || lastCurveSamples.length <= 1) return 0;
  return Math.round(
    (pointIndex / (points.length - 1)) * (lastCurveSamples.length - 1),
  );
}

function tangentAtSample(index: number): XY {
  const before = lastCurveSamples[Math.max(0, index - 2)];
  const after =
    lastCurveSamples[Math.min(lastCurveSamples.length - 1, index + 2)];
  if (!isFinitePoint(before) || !isFinitePoint(after)) return { x: 0, y: 0 };
  return { x: after.x - before.x, y: after.y - before.y };
}

function signedCurvature(a: XY, b: XY, c: XY): number {
  const ab = { x: b.x - a.x, y: b.y - a.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const ac = { x: c.x - a.x, y: c.y - a.y };
  const cross = ab.x * bc.y - ab.y * bc.x;
  const denom =
    Math.hypot(ab.x, ab.y) * Math.hypot(bc.x, bc.y) * Math.hypot(ac.x, ac.y);
  return denom <= 1e-6 ? 0 : (2 * cross) / denom;
}

function drawArrow(start: XY, end: XY): void {
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();

  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const head = 8;
  ctx.beginPath();
  ctx.moveTo(end.x, end.y);
  ctx.lineTo(
    end.x - Math.cos(angle - Math.PI / 6) * head,
    end.y - Math.sin(angle - Math.PI / 6) * head,
  );
  ctx.lineTo(
    end.x - Math.cos(angle + Math.PI / 6) * head,
    end.y - Math.sin(angle + Math.PI / 6) * head,
  );
  ctx.closePath();
  ctx.fill();
}

interface SampleCurrentCurveResult {
  points: XY[];
  compileMs: number;
  evalMs: number;
  backend: DemoBackend;
  branchPruning: BranchPruning;
}

function sampleCurrentCurve(): SampleCurrentCurveResult {
  const key = samplerCacheKey(
    points,
    currentSplineKind(),
    currentBackend(),
    currentBranchPruning(),
  );
  if (key !== cachedSamplerKey) {
    cachedSampler = createCurveSampleRoutine(
      points,
      currentSplineKind(),
      currentBackend(),
      currentBranchPruning(),
    );
    cachedSamplerKey = key;
  }
  if (!cachedSampler) {
    return {
      points: [],
      compileMs: 0,
      evalMs: 0,
      backend: currentBackend(),
      branchPruning: currentBranchPruning(),
    };
  }
  const result = cachedSampler.sample(points);
  return {
    points: result.points,
    compileMs: cachedSampler.compileMs,
    evalMs: result.evalMs,
    backend: cachedSampler.backend,
    branchPruning: cachedSampler.branchPruning,
  };
}

function updatePerfIndicator(result: SampleCurrentCurveResult): void {
  perfIndicator.dataset.state = "ok";
  perfIndicator.textContent = `${backendLabel(result.backend)} · ${formatMs(result.evalMs)}`;
  compileTime.textContent = formatMs(result.compileMs);
  evalTime.textContent = formatMs(result.evalMs);
  pruningStatus.textContent = pruningLabel(result.branchPruning);
}

function updatePerfError(): void {
  perfIndicator.dataset.state = "error";
  perfIndicator.textContent = `${backendLabel(currentBackend())} failed`;
  compileTime.textContent = "—";
  evalTime.textContent = "—";
  pruningStatus.textContent = pruningLabel(currentBranchPruning());
}

function backendLabel(backend: DemoBackend): string {
  switch (backend) {
    case "js-codegen":
      return "JS codegen";
    case "js-interp":
      return "JS interp";
    case "wasm-codegen":
      return "Wasm codegen";
    case "wasm-interp":
      return "Wasm interp";
  }
}

function pruningLabel(pruning: BranchPruning): string {
  switch (pruning) {
    case "off":
      return "Off";
    case "trace":
      return "Trace";
    case "full-trace":
      return "Full trace";
  }
}

function formatMs(value: number): string {
  return `${value.toFixed(value < 10 ? 2 : 1)} ms`;
}

function drawPoints(): void {
  for (const point of points) {
    const selected = point.id === selectedPointId;

    ctx.save();
    ctx.fillStyle = selected
      ? "#ff7a1a"
      : point.knuckle
        ? "#f0e7ff"
        : "#ffffff";
    ctx.strokeStyle = selected
      ? "#9f4300"
      : point.knuckle
        ? "#7c3aed"
        : "#172033";
    const radius = smallPointsInput.checked
      ? selected
        ? 6
        : 4
      : selected
        ? 9
        : 7;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (point.knuckle) {
      const markerRadius = smallPointsInput.checked ? 8 : 13;
      ctx.beginPath();
      ctx.moveTo(point.x, point.y - markerRadius);
      ctx.lineTo(point.x + markerRadius, point.y);
      ctx.lineTo(point.x, point.y + markerRadius);
      ctx.lineTo(point.x - markerRadius, point.y);
      ctx.closePath();
      ctx.stroke();
    }

    ctx.restore();
  }
}

function updatePanel(): void {
  pointCount.textContent = String(points.length);
  const selected = selectedPointRecord();
  selectedPoint.textContent = selected
    ? `${Math.round(selected.x)}, ${Math.round(selected.y)}${selected.knuckle ? " · knuckle" : ""}`
    : "None";
  deleteButton.disabled = !selected;
}

function selectedPointRecord(): DemoPoint | undefined {
  return points.find((point) => point.id === selectedPointId);
}

function loadSelectedExample(): void {
  const example = curveExamples.find(
    (candidate) => candidate.id === exampleKindSelect.value,
  );
  points = example ? pointsFromExample(example) : cloneInitialPoints();
  nextPointId = Math.max(0, ...points.map((point) => point.id)) + 1;
  selectedPointId = null;
  draw();
}

function pointsFromExample(example: CurveExample): DemoPoint[] {
  const { width, height } = canvasSize();
  const raw = example.contours.flatMap((contour) => contour.points);
  const minX = Math.min(...raw.map((point) => point.x));
  const maxX = Math.max(...raw.map((point) => point.x));
  const minY = Math.min(...raw.map((point) => point.y));
  const maxY = Math.max(...raw.map((point) => point.y));
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const padding = 56;
  const scale = Math.min(
    (Math.max(120, width) - padding * 2) / spanX,
    (Math.max(120, height) - padding * 2) / spanY,
  );
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;

  let id = 1;
  const out: DemoPoint[] = [];
  example.contours.forEach((contour, contourId) => {
    const source = contour.closed
      ? [...contour.points, contour.points[0]]
      : contour.points;
    for (const point of source) {
      out.push({
        id: id++,
        x: offsetX + (point.x - minX) * scale,
        y: offsetY + (point.y - minY) * scale,
        curvature: point.curvature,
        contourId,
      });
    }
  });
  return out;
}

function canvasSize(): { width: number; height: number } {
  return {
    width: canvas.clientWidth,
    height: canvas.clientHeight,
  };
}

function resizeCanvas(): void {
  const { width, height } = canvasSize();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  draw();
}

function pointerPosition(event: Pick<MouseEvent, "clientX" | "clientY">): XY {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function hitTest(position: XY): DemoPoint | undefined {
  for (const point of [...points].reverse()) {
    const distance = Math.hypot(point.x - position.x, point.y - position.y);
    if (distance <= 14) return point;
  }
  return undefined;
}

function movePoint(id: number, position: XY): void {
  const { width, height } = canvasSize();
  points = points.map((point) =>
    point.id === id
      ? {
          ...point,
          x: Math.min(width, Math.max(0, position.x)),
          y: Math.min(height, Math.max(0, position.y)),
        }
      : point,
  );
}

function addPoint(position: XY, insertAt = points.length): DemoPoint {
  const contourId = points[Math.max(0, insertAt - 1)]?.contourId ?? 0;
  const point = { id: nextPointId++, x: position.x, y: position.y, contourId };
  points = [...points.slice(0, insertAt), point, ...points.slice(insertAt)];
  return point;
}

function nearestCurveSample(
  position: XY,
): { index: number; distance: number } | null {
  if (lastCurveSamples.length === 0) return null;

  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const [index, point] of lastCurveSamples.entries()) {
    if (!isFinitePoint(point)) continue;
    const distance = Math.hypot(point.x - position.x, point.y - position.y);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  }
  return { index: bestIndex, distance: bestDistance };
}

function insertionIndexFromSample(sampleIndex: number): number {
  if (points.length < 2 || lastCurveSamples.length < 2) return points.length;

  const raw =
    (sampleIndex / (lastCurveSamples.length - 1)) * (points.length - 1);
  return Math.min(points.length, Math.floor(raw) + 1);
}

function updateCursor(
  position = lastPointerPosition,
  shiftKey = shiftKeyDown,
): void {
  if (!position) {
    canvas.style.cursor = "crosshair";
    return;
  }
  if (draggingPointId !== null) {
    canvas.style.cursor = "grabbing";
    return;
  }

  const hitPoint = hitTest(position);
  if (hitPoint) {
    canvas.style.cursor = shiftKey ? "not-allowed" : "grab";
    return;
  }

  const nearest = nearestCurveSample(position);
  canvas.style.cursor =
    nearest && nearest.distance <= 18 ? "copy" : "crosshair";
}

canvas.addEventListener("pointerdown", (event) => {
  const position = pointerPosition(event);
  lastPointerPosition = position;
  shiftKeyDown = event.shiftKey;
  const hitPoint = hitTest(position);
  if (!hitPoint) {
    selectedPointId = null;
    draw();
    return;
  }

  if (event.shiftKey) {
    deletePoint(hitPoint.id);
    event.preventDefault();
    updateCursor(position, event.shiftKey);
    return;
  }

  selectedPointId = hitPoint.id;
  draggingPointId = hitPoint.id;
  activePointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  draw();
  updateCursor(position, event.shiftKey);
});

canvas.addEventListener("pointermove", (event) => {
  const position = pointerPosition(event);
  lastPointerPosition = position;
  shiftKeyDown = event.shiftKey;
  if (draggingPointId === null || activePointerId !== event.pointerId) {
    updateCursor(position, event.shiftKey);
    return;
  }
  movePoint(draggingPointId, position);
  draw();
  updateCursor(position, event.shiftKey);
});

canvas.addEventListener("pointerup", (event) => {
  if (activePointerId === event.pointerId) {
    draggingPointId = null;
    activePointerId = null;
    canvas.releasePointerCapture(event.pointerId);
    updateCursor(lastPointerPosition, event.shiftKey);
  }
});

canvas.addEventListener("pointercancel", () => {
  draggingPointId = null;
  activePointerId = null;
  updateCursor();
});

canvas.addEventListener("pointerleave", () => {
  lastPointerPosition = null;
  updateCursor();
});

canvas.addEventListener("dblclick", (event) => {
  const position = pointerPosition(event);
  const hitPoint = hitTest(position);

  if (hitPoint) {
    selectedPointId = hitPoint.id;
    toggleKnuckle(hitPoint.id);
    updateCursor(position, event.shiftKey);
    return;
  }

  const nearest = nearestCurveSample(position);
  const insertAt =
    nearest && nearest.distance <= 18
      ? insertionIndexFromSample(nearest.index)
      : points.length;

  const point = addPoint(position, insertAt);
  selectedPointId = point.id;
  draw();
  updateCursor(position, event.shiftKey);
});

exampleKindSelect.addEventListener("change", loadSelectedExample);
splineKindSelect.addEventListener("change", draw);
backendKindSelect.addEventListener("change", () => {
  updateBranchPruningAvailability();
  draw();
});
branchPruningSelect.addEventListener("change", draw);
showTangentsInput.addEventListener("change", draw);
showCurvatureInput.addEventListener("change", draw);
smallPointsInput.addEventListener("change", draw);

resetButton.addEventListener("click", loadSelectedExample);

clearButton.addEventListener("click", () => {
  points = [];
  selectedPointId = null;
  draw();
});

helpButton.addEventListener("click", () => {
  helpDialog.showModal();
});

deleteButton.addEventListener("click", deleteSelectedPoint);

window.addEventListener("keydown", (event) => {
  if (event.key === "Shift") {
    shiftKeyDown = true;
    updateCursor();
  }
  if (event.key === "Backspace" || event.key === "Delete") {
    deleteSelectedPoint();
  }
});

window.addEventListener("keyup", (event) => {
  if (event.key === "Shift") {
    shiftKeyDown = false;
    updateCursor();
  }
});

function deleteSelectedPoint(): void {
  if (selectedPointId === null) return;
  deletePoint(selectedPointId);
}

function deletePoint(id: number): void {
  points = points.filter((point) => point.id !== id);
  if (selectedPointId === id) selectedPointId = null;
  draw();
}

function toggleKnuckle(id: number): void {
  points = points.map((point) =>
    point.id === id ? { ...point, knuckle: !point.knuckle } : point,
  );
  draw();
}

new ResizeObserver(resizeCanvas).observe(canvas);
resizeCanvas();
