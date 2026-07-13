import {
  profileVariablesFor,
  runOptimisation,
  type BackendChoice,
  type LimitTarget,
  type NumericProfilePoint,
  type OptimisationResult,
  type OptimisationSettings,
  type ProfileVariable,
  type SelectSpecializationChoice,
} from "./solver";
import "./styles.css";

const defaultHalfProfile: NumericProfilePoint[] = [
  { x: 0.0, y: 0.0 },
  { x: 0.55, y: 0.4 },
  { x: 0.9, y: 1.0 },
  { x: 0.95, y: 1.5 },
  // Flat deck/top: this point is mirrored to (-x, y), rather than closing
  // back to the symmetry axis.
  { x: 0.9, y: 2.0 },
];

const defaultTargets: LimitTarget[] = [
  { name: "lower", limitY: 0.7, targetSurface: 0.8 },
  { name: "upper", limitY: 1.4, targetSurface: 2 },
];

const defaultTotalTarget = {
  name: "total",
  targetSurface: 3.4,
};

const defaultOptimisation: OptimisationSettings = {
  iterations: 2500,
  initialDamping: 1e-3,
  regularisationWeight: 1e-6,
  backend: "js-interp",
  selectSpecialization: "off",
};

const PROFILE_MIN_X = 0.001;
const PROFILE_MIN_Y_GAP = 0.001;
const PROFILE_EDITOR_WIDTH = 360;
const PROFILE_EDITOR_HEIGHT = 260;
const PROFILE_EDITOR_PADDING_X = 28;
const PROFILE_EDITOR_PADDING_TOP = 18;
const PROFILE_EDITOR_PADDING_BOTTOM = 34;
const PROFILE_EDITOR_TARGET_COLOURS = [
  "#d97706",
  "#7c3aed",
  "#0284c7",
  "#e11d48",
  "#16a34a",
  "#ea580c",
];

type ProfileEditorGeometry = {
  width: number;
  height: number;
  centerX: number;
  bottomY: number;
  scale: number;
  mapPoint: (point: NumericProfilePoint) => [number, number];
  unmapPoint: (x: number, y: number) => NumericProfilePoint;
};

function lerpNumericProfileX(
  y: number,
  a: NumericProfilePoint,
  b: NumericProfilePoint,
): number {
  const t = (y - a.y) / (b.y - a.y);
  return a.x + t * (b.x - a.x);
}

function halfProfileBelowLimit(
  profile: NumericProfilePoint[],
  limitY: number,
): NumericProfilePoint[] {
  const bottom = profile[0]!;
  if (limitY <= bottom.y) return [bottom, { x: bottom.x, y: bottom.y }];

  const points: NumericProfilePoint[] = [bottom];
  for (let i = 0; i < profile.length - 1; i++) {
    const a = profile[i]!;
    const b = profile[i + 1]!;

    if (limitY >= b.y) {
      points.push(b);
    } else if (limitY > a.y) {
      points.push({ x: lerpNumericProfileX(limitY, a, b), y: limitY });
      break;
    } else {
      break;
    }
  }

  return points;
}

function fullProfilePolygon(
  profile: NumericProfilePoint[],
): NumericProfilePoint[] {
  const right = profile;
  // Mirror every point except the bottom point. This gives a flat segment from
  // the right-most top/cut point to the left-most top/cut point, which matches
  // the boat-hull shape better than closing to a single point on the axis.
  const left = profile
    .slice(1)
    .reverse()
    .map((point) => ({ x: -point.x, y: point.y }));
  return [...right, ...left];
}

function profilePath(
  profile: NumericProfilePoint[],
  mapPoint: (point: NumericProfilePoint) => [number, number],
): string {
  return (
    fullProfilePolygon(profile)
      .map((point, index) => {
        const [x, y] = mapPoint(point);
        return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ") + " Z"
  );
}

function escapeXml(text: string): string {
  return text.replace(
    /[&<>"]/g,
    (char) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
      })[char]!,
  );
}

function buildSvg(result: OptimisationResult): string {
  const { originalProfile, finalProfile, originalSurfaces, finalSurfaces } =
    result;
  const targets = originalSurfaces;
  const surfaceColours = [
    "#fbbf24",
    "#a78bfa",
    "#38bdf8",
    "#fb7185",
    "#4ade80",
    "#f97316",
  ];
  const lineColours = [
    "#d97706",
    "#7c3aed",
    "#0284c7",
    "#e11d48",
    "#16a34a",
    "#ea580c",
  ];
  const colourFor = (palette: string[], index: number) =>
    palette[index % palette.length]!;

  const maxX = Math.max(
    0.1,
    ...originalProfile.map((point) => Math.abs(point.x)),
    ...finalProfile.map((point) => Math.abs(point.x)),
  );
  const minY = 0;
  const maxY = Math.max(
    0.1,
    ...originalProfile.map((point) => point.y),
    ...targets.map((target) => target.limitY),
  );
  const panelBottom = 350;
  const panelHeight = 285;
  const shapeScale = Math.min(220 / (2 * maxX), panelHeight / (maxY - minY));
  const chartStartY = 450;
  const chartRowHeight = 104;
  const svgHeight = Math.max(
    650,
    chartStartY + targets.length * chartRowHeight + 50,
  );

  const mapForPanel = (cx: number) => (point: NumericProfilePoint) =>
    [
      cx + point.x * shapeScale,
      panelBottom - (point.y - minY) * shapeScale,
    ] as [number, number];

  const drawLimitLine = (cx: number, target: LimitTarget, colour: string) => {
    const y = panelBottom - (target.limitY - minY) * shapeScale;
    return `<line x1="${(cx - maxX * shapeScale * 1.15).toFixed(1)}" y1="${y.toFixed(1)}" x2="${(cx + maxX * shapeScale * 1.15).toFixed(1)}" y2="${y.toFixed(1)}" stroke="${colour}" stroke-width="1.5" stroke-dasharray="6 5" />
<text x="${(cx + maxX * shapeScale * 1.2).toFixed(1)}" y="${(y + 4).toFixed(1)}" font-size="12" fill="${colour}">${escapeXml(target.name)} y=${target.limitY}</text>`;
  };

  const pointMarkers = (
    profile: NumericProfilePoint[],
    mapPoint: (point: NumericProfilePoint) => [number, number],
    colour: string,
  ) =>
    profile
      .map((point, index) => {
        const [x, y] = mapPoint(point);
        const labelY = y - 8;
        return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="3.5" fill="${colour}" />
<text x="${(x + 7).toFixed(2)}" y="${labelY.toFixed(2)}" font-size="10" fill="${colour}">${index}</text>`;
      })
      .join("\n");

  const deckLine = (
    profile: NumericProfilePoint[],
    mapPoint: (point: NumericProfilePoint) => [number, number],
    colour: string,
  ) => {
    const top = profile[profile.length - 1]!;
    const [rightX, y] = mapPoint(top);
    const [leftX] = mapPoint({ x: -top.x, y: top.y });
    return `<line x1="${leftX.toFixed(2)}" y1="${y.toFixed(2)}" x2="${rightX.toFixed(2)}" y2="${y.toFixed(2)}" stroke="${colour}" stroke-width="5" stroke-linecap="round" opacity="0.55" />`;
  };

  const shapePanel = (
    title: string,
    profile: NumericProfilePoint[],
    cx: number,
    fill: string,
    stroke: string,
  ) => {
    const mapPoint = mapForPanel(cx);
    const surfaceFills = [...targets]
      .map((target, index) => ({ target, index }))
      .sort((a, b) => b.target.limitY - a.target.limitY)
      .map(({ target, index }) => {
        const below = halfProfileBelowLimit(profile, target.limitY);
        return `<path d="${profilePath(below, mapPoint)}" fill="${colourFor(surfaceColours, index)}" opacity="0.22" />`;
      })
      .join("\n");
    const limitLines = targets
      .map((target, index) =>
        drawLimitLine(cx, target, colourFor(lineColours, index)),
      )
      .join("\n");

    return `<g>
  <text x="${cx}" y="50" text-anchor="middle" font-size="18" font-weight="700">${escapeXml(title)}</text>
  <line x1="${(cx - maxX * shapeScale * 1.18).toFixed(1)}" y1="${panelBottom.toFixed(1)}" x2="${(cx + maxX * shapeScale * 1.18).toFixed(1)}" y2="${panelBottom.toFixed(1)}" stroke="#94a3b8" stroke-width="1" />
  <text x="${(cx - maxX * shapeScale * 1.2).toFixed(1)}" y="${(panelBottom + 16).toFixed(1)}" font-size="11" fill="#64748b">y=0 keel baseline</text>
  ${surfaceFills}
  <path d="${profilePath(profile, mapPoint)}" fill="${fill}" stroke="${stroke}" stroke-width="3" />
  ${deckLine(profile, mapPoint, stroke)}
  ${pointMarkers(profile, mapPoint, stroke)}
  ${limitLines}
</g>`;
  };

  const allSurfaceValues = [
    ...originalSurfaces.map((entry) => entry.surface),
    ...finalSurfaces.map((entry) => entry.surface),
    ...targets.map((entry) => entry.targetSurface),
  ];
  const chartMax = Math.max(0.1, ...allSurfaceValues) * 1.12;
  const barWidth = 600;

  const bar = (label: string, value: number, y: number, colour: string) => {
    const w = (value / chartMax) * barWidth;
    return `<text x="80" y="${y + 14}" font-size="13">${escapeXml(label)}</text>
<rect x="190" y="${y}" width="${w.toFixed(1)}" height="18" rx="4" fill="${colour}" />
<text x="${(200 + w).toFixed(1)}" y="${y + 14}" font-size="13" font-family="monospace">${value.toFixed(4)}</text>`;
  };

  const chartRows = targets
    .map((target, targetIndex) => {
      const y = chartStartY + targetIndex * chartRowHeight;
      const original = originalSurfaces[targetIndex]!.surface;
      const final = finalSurfaces[targetIndex]!.surface;
      const error = final - target.targetSurface;
      return `<g>
  <circle cx="60" cy="${(y - 20).toFixed(1)}" r="5" fill="${colourFor(lineColours, targetIndex)}" />
  <text x="74" y="${y - 16}" font-size="16" font-weight="700">${escapeXml(target.name)} limit: y=${target.limitY}</text>
  <text x="760" y="${y - 16}" font-size="13" font-family="monospace" fill="#475569">final-target=${error.toFixed(4)}</text>
  ${bar("original", original, y, "#60a5fa")}
  ${bar("target", target.targetSurface, y + 28, "#64748b")}
  ${bar("final", final, y + 56, "#34d399")}
</g>`;
    })
    .join("\n");

  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="${svgHeight}" viewBox="0 0 960 ${svgHeight}">
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }
</style>
<rect width="100%" height="100%" fill="#ffffff" />
<text x="480" y="28" text-anchor="middle" font-size="22" font-weight="800">Variable profile surface optimisation</text>
${shapePanel("Original shape", originalProfile, 250, "#dbeafe", "#2563eb")}
${shapePanel("Optimised shape", finalProfile, 710, "#dcfce7", "#059669")}
${chartRows}
</svg>`;
}

function numberInput(name: string, value: number, step = "0.01"): string {
  return `<input name="${name}" type="number" value="${value}" step="${step}" />`;
}

function selectInput(
  name: string,
  options: Array<{ value: string; label: string }>,
  selected: string,
): string {
  return `<select name="${name}">${options
    .map(
      (option) =>
        `<option value="${option.value}"${option.value === selected ? " selected" : ""}>${option.label}</option>`,
    )
    .join("")}</select>`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatProfileNumber(value: number): string {
  return Number(value.toFixed(4)).toString();
}

function profileEditorGeometry(
  profile: NumericProfilePoint[],
  limitYs: number[] = [],
): ProfileEditorGeometry {
  const maxX = Math.max(0.1, ...profile.map((point) => Math.abs(point.x)));
  const maxY = Math.max(
    0.1,
    ...profile.map((point) => point.y),
    ...limitYs.filter((limitY) => Number.isFinite(limitY) && limitY >= 0),
  );
  const centerX = PROFILE_EDITOR_WIDTH / 2;
  const bottomY = PROFILE_EDITOR_HEIGHT - PROFILE_EDITOR_PADDING_BOTTOM;
  const scale = Math.min(
    (centerX - PROFILE_EDITOR_PADDING_X) / maxX,
    (PROFILE_EDITOR_HEIGHT -
      PROFILE_EDITOR_PADDING_TOP -
      PROFILE_EDITOR_PADDING_BOTTOM) /
      maxY,
  );

  return {
    width: PROFILE_EDITOR_WIDTH,
    height: PROFILE_EDITOR_HEIGHT,
    centerX,
    bottomY,
    scale,
    mapPoint: (point) =>
      [centerX + point.x * scale, bottomY - point.y * scale] as [
        number,
        number,
      ],
    unmapPoint: (x, y) => ({
      x: (x - centerX) / scale,
      y: (bottomY - y) / scale,
    }),
  };
}

function profilePolylinePath(
  profile: NumericProfilePoint[],
  mapPoint: (point: NumericProfilePoint) => [number, number],
): string {
  return profile
    .map((point, index) => {
      const [x, y] = mapPoint(point);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function buildProfileEditorSvg(
  profile: NumericProfilePoint[],
  targets: LimitTarget[],
): string {
  const geometry = profileEditorGeometry(
    profile,
    targets.map((target) => target.limitY),
  );
  const baselineY = geometry.bottomY;
  const rightPath = profilePolylinePath(profile, geometry.mapPoint);
  const leftPath = profilePolylinePath(
    profile.map((point) => ({ x: -point.x, y: point.y })),
    geometry.mapPoint,
  );
  const deck = profile[profile.length - 1]!;
  const [deckRightX, deckY] = geometry.mapPoint(deck);
  const [deckLeftX] = geometry.mapPoint({ x: -deck.x, y: deck.y });
  const targetLines = targets
    .filter((target) => Number.isFinite(target.limitY) && target.limitY >= 0)
    .map((target, index) => {
      const [, y] = geometry.mapPoint({ x: 0, y: target.limitY });
      const colour =
        PROFILE_EDITOR_TARGET_COLOURS[
          index % PROFILE_EDITOR_TARGET_COLOURS.length
        ]!;
      return `<g class="profile-editor-target" style="--target-colour: ${colour}">
        <line x1="${PROFILE_EDITOR_PADDING_X}" y1="${y.toFixed(2)}" x2="${(geometry.width - PROFILE_EDITOR_PADDING_X).toFixed(2)}" y2="${y.toFixed(2)}" />
        <text x="${(geometry.width - PROFILE_EDITOR_PADDING_X - 4).toFixed(2)}" y="${(y - 5).toFixed(2)}">${escapeXml(target.name)} y=${formatProfileNumber(target.limitY)}</text>
      </g>`;
    })
    .join("\n");
  const markers = profile
    .map((point, index) => {
      const [x, y] = geometry.mapPoint(point);
      const fixed = index === 0;
      return `<g>
        <circle class="profile-editor-handle${fixed ? " fixed" : ""}" data-point-index="${index}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="7" />
        <text x="${(x + 11).toFixed(2)}" y="${(y - 9).toFixed(2)}" class="profile-editor-label">${index}</text>
      </g>`;
    })
    .join("\n");

  return `<svg class="profile-editor-svg" viewBox="0 0 ${geometry.width} ${geometry.height}" role="img" aria-label="Editable mirrored profile">
    <rect x="0" y="0" width="${geometry.width}" height="${geometry.height}" rx="16" class="profile-editor-bg" />
    <line x1="${geometry.centerX.toFixed(2)}" y1="${PROFILE_EDITOR_PADDING_TOP}" x2="${geometry.centerX.toFixed(2)}" y2="${baselineY.toFixed(2)}" class="profile-editor-axis" />
    <line x1="${PROFILE_EDITOR_PADDING_X}" y1="${baselineY.toFixed(2)}" x2="${(geometry.width - PROFILE_EDITOR_PADDING_X).toFixed(2)}" y2="${baselineY.toFixed(2)}" class="profile-editor-axis" />
    <path d="${profilePath(profile, geometry.mapPoint)}" class="profile-editor-fill" />
    ${targetLines}
    <path d="${leftPath}" class="profile-editor-mirror-line" />
    <path d="${rightPath}" class="profile-editor-line" />
    <line x1="${deckLeftX.toFixed(2)}" y1="${deckY.toFixed(2)}" x2="${deckRightX.toFixed(2)}" y2="${deckY.toFixed(2)}" class="profile-editor-deck" />
    ${markers}
    <text x="${PROFILE_EDITOR_PADDING_X}" y="${(geometry.height - 12).toFixed(2)}" class="profile-editor-caption">drag points · double-click outline to insert</text>
  </svg>`;
}

function profilePointRowHtml(
  point: NumericProfilePoint,
  index: number,
  total: number,
): string {
  const isBottom = index === 0;
  const isTop = index === total - 1;
  const label = isBottom
    ? "bottom / keel"
    : isTop
      ? "top / deck"
      : `point ${index}`;
  return `
    <div class="point-grid" data-profile-row>
      <span class="point-label">${label}</span>
      <input class="point-x" type="number" value="${point.x}" step="0.01" ${isBottom ? "readonly" : ""} />
      <input class="point-y" type="number" value="${point.y}" step="0.01" ${isBottom ? "readonly" : ""} />
      <button type="button" class="small secondary remove-point" ${isBottom || total <= 2 ? "disabled" : ""}>Remove</button>
    </div>`;
}

function renderAppShell(): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (!app) throw new Error("missing #app");

  app.innerHTML = `
    <header>
      <p class="eyebrow">Lona demo</p>
      <h1>Profile surface optimisation</h1>
      <p class="intro">
        The right-side profile is mirrored around x=0. Every point after the
        bottom has its x coordinate driven by a squared Lona variable. A Levenberg–Marquardt optimiser uses the residual
        jacobian to fit target surfaces below horizontal y limits plus the total deck surface.
      </p>
    </header>

    <main class="layout">
      <section class="panel controls">
        <form id="controls-form" novalidate>
          <h2>Surface targets</h2>
          <div class="target-grid header-row">
            <span>Name</span><span>limitY</span><span>target surface</span>
          </div>
          ${defaultTargets
            .map(
              (target, index) => `
              <div class="target-grid">
                <input name="target-${index}-name" value="${target.name}" />
                ${numberInput(`target-${index}-limitY`, target.limitY)}
                ${numberInput(`target-${index}-surface`, target.targetSurface)}
              </div>`,
            )
            .join("")}
          <div class="target-grid">
            <input name="total-target-name" value="${defaultTotalTarget.name}" />
            <input value="top / deck" readonly />
            ${numberInput("total-target-surface", defaultTotalTarget.targetSurface)}
          </div>

          <h2>Profile points</h2>
          <div id="profile-editor" class="profile-editor"></div>
          <p class="hint">Drag the blue points to edit x/y. Double-click the profile outline to insert a point at that position.</p>
          <div id="profile-points" hidden>
            ${defaultHalfProfile
              .map((point, index) =>
                profilePointRowHtml(point, index, defaultHalfProfile.length),
              )
              .join("")}
          </div>

          <h2>Optimiser</h2>
          <label>Backend ${selectInput(
            "backend",
            [
              { value: "js-interp", label: "js-interp (compiled)" },
              { value: "wasm-interp", label: "wasm-interp (compiled)" },
              { value: "direct", label: "direct (forward-autodiff eval)" },
            ],
            defaultOptimisation.backend,
          )}</label>
          <label>Select specialisation ${selectInput(
            "selectSpecialization",
            [
              { value: "off", label: "off" },
              { value: "trace", label: "trace" },
              { value: "full-trace", label: "full-trace" },
            ],
            defaultOptimisation.selectSpecialization,
          )}</label>
          <label>Iterations ${numberInput("iterations", defaultOptimisation.iterations, "1")}</label>
          <label>Initial damping ${numberInput("initialDamping", defaultOptimisation.initialDamping, "0.0001")}</label>
          <label>Regularisation ${numberInput("regularisationWeight", defaultOptimisation.regularisationWeight, "0.000001")}</label>

          <div class="actions">
            <button type="button" id="optimise-button">Optimise</button>
            <button type="button" id="reset-button" class="secondary">Reset</button>
          </div>
        </form>
      </section>

      <section class="panel output">
        <div class="output-header">
          <h2>Result</h2>
        </div>
        <div id="status" class="status">Ready.</div>
        <div id="summary"></div>
        <div id="visualisation-host" class="visualisation-host"></div>
      </section>
    </main>
  `;
}

function readNumber(formData: FormData, name: string): number {
  const value = Number(formData.get(name));
  if (!Number.isFinite(value)) throw new Error(`Invalid number for ${name}`);
  return value;
}

function readBackend(formData: FormData): BackendChoice {
  const backend = String(formData.get("backend"));
  if (
    backend === "js-interp" ||
    backend === "wasm-interp" ||
    backend === "direct"
  ) {
    return backend;
  }
  throw new Error(`Invalid backend: ${backend}`);
}

function readSelectSpecialization(
  formData: FormData,
): SelectSpecializationChoice {
  const mode = String(formData.get("selectSpecialization"));
  if (mode === "off" || mode === "trace" || mode === "full-trace") return mode;
  throw new Error(`Invalid select specialisation: ${mode}`);
}

function readProfile(form: HTMLFormElement): NumericProfilePoint[] {
  const rows = [...form.querySelectorAll<HTMLElement>("[data-profile-row]")];
  const profile = rows.map((row, index) => {
    const x = Number(row.querySelector<HTMLInputElement>(".point-x")?.value);
    const y = Number(row.querySelector<HTMLInputElement>(".point-y")?.value);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new Error("Invalid profile point");
    }
    if (index === 0 ? x !== 0 : x <= 0) {
      throw new Error("Profile x values after the bottom must be positive");
    }
    return { x, y };
  });

  if (profile.length < 2) throw new Error("Profile needs at least 2 points");
  if (profile[0]!.x !== 0 || profile[0]!.y !== 0) {
    throw new Error(
      "The first profile point must be the bottom point at x=0, y=0",
    );
  }
  for (let i = 1; i < profile.length; i++) {
    if (profile[i]!.y <= profile[i - 1]!.y) {
      throw new Error("Profile y values must increase from the bottom upward");
    }
  }

  return profile;
}

function readInputs(form: HTMLFormElement): {
  baseProfile: NumericProfilePoint[];
  targets: LimitTarget[];
  profileVariables: ProfileVariable[];
  settings: OptimisationSettings;
} {
  const data = new FormData(form);
  const baseProfile = readProfile(form);

  const top = baseProfile[baseProfile.length - 1]!;
  const targets = [
    ...defaultTargets.map((target, index) => ({
      name: String(data.get(`target-${index}-name`) || target.name),
      limitY: readNumber(data, `target-${index}-limitY`),
      targetSurface: readNumber(data, `target-${index}-surface`),
    })),
    {
      name: String(data.get("total-target-name") || defaultTotalTarget.name),
      limitY: top.y,
      targetSurface: readNumber(data, "total-target-surface"),
    },
  ];

  return {
    baseProfile,
    targets,
    profileVariables: profileVariablesFor(baseProfile),
    settings: {
      iterations: Math.max(1, Math.round(readNumber(data, "iterations"))),
      initialDamping: Math.max(1e-12, readNumber(data, "initialDamping")),
      regularisationWeight: Math.max(
        0,
        readNumber(data, "regularisationWeight"),
      ),
      backend: readBackend(data),
      selectSpecialization: readSelectSpecialization(data),
    },
  };
}

function renderSummary(
  result: OptimisationResult,
  profileVariables: ProfileVariable[],
): void {
  const summary = document.querySelector<HTMLDivElement>("#summary");
  if (!summary) return;

  const variableRows = profileVariables
    .map((variable) => {
      const original = variable.initialX;
      const final = result.finalVariables[variable.name]! ** 2;
      return `<tr><td><code>${variable.name}</code></td><td>${original.toFixed(4)}</td><td>${final.toFixed(4)}</td></tr>`;
    })
    .join("");

  const surfaceRows = result.originalSurfaces
    .map((target, index) => {
      const final = result.finalSurfaces[index]!;
      return `<tr><td>${escapeXml(target.name)}</td><td>${target.limitY.toFixed(3)}</td><td>${target.surface.toFixed(4)}</td><td>${target.targetSurface.toFixed(4)}</td><td>${final.surface.toFixed(4)}</td></tr>`;
    })
    .join("");

  summary.innerHTML = `
    <div class="metric-grid">
      <div class="metric"><span>Optimise loop</span><strong>${result.stats.optimiseMs.toFixed(2)} ms</strong></div>
      <div class="metric"><span>Steps</span><strong>${result.stats.steps}</strong></div>
      <div class="metric"><span>Compile/setup</span><strong>${result.stats.compileMs.toFixed(2)} ms</strong></div>
      <div class="metric"><span>Total</span><strong>${result.stats.totalMs.toFixed(2)} ms</strong></div>
      <div class="metric"><span>Backend</span><strong>${result.stats.backend}</strong></div>
      <div class="metric"><span>Select specialisation</span><strong>${result.stats.selectSpecialization}</strong></div>
      <div class="metric"><span>Stop reason</span><strong>${result.stats.stoppedReason}</strong></div>
      <div class="metric"><span>LM objective</span><strong>${result.stats.initialLoss.toExponential(2)} → ${result.stats.finalLoss.toExponential(2)}</strong></div>
    </div>
    <div class="summary-grid">
      <div>
        <h3>Variables</h3>
        <table><thead><tr><th>Name</th><th>Original</th><th>Final</th></tr></thead><tbody>${variableRows}</tbody></table>
      </div>
      <div>
        <h3>Surfaces</h3>
        <table><thead><tr><th>Name</th><th>limitY</th><th>Original</th><th>Target</th><th>Final</th></tr></thead><tbody>${surfaceRows}</tbody></table>
        <p class="loss">Expression build time: ${result.stats.expressionBuildMs.toFixed(2)} ms</p>
      </div>
    </div>
  `;
}

let scheduledRunFromForm: number | undefined;

function scheduleRunFromForm(): void {
  if (scheduledRunFromForm !== undefined) {
    window.clearTimeout(scheduledRunFromForm);
  }
  scheduledRunFromForm = window.setTimeout(() => {
    scheduledRunFromForm = undefined;
    runFromForm();
  }, 150);
}

function runFromForm(): void {
  if (scheduledRunFromForm !== undefined) {
    window.clearTimeout(scheduledRunFromForm);
    scheduledRunFromForm = undefined;
  }
  const form = document.querySelector<HTMLFormElement>("#controls-form");
  const status = document.querySelector<HTMLDivElement>("#status");
  const visualisationHost = document.querySelector<HTMLDivElement>(
    "#visualisation-host",
  );
  if (!form || !status || !visualisationHost) return;

  try {
    status.textContent = "Optimising…";
    const { baseProfile, targets, profileVariables, settings } =
      readInputs(form);
    renderProfileEditor(baseProfile, targets);
    const result = runOptimisation(
      baseProfile,
      targets,
      profileVariables,
      settings,
    );
    visualisationHost.innerHTML = buildSvg(result);
    renderSummary(result, profileVariables);
    status.textContent = `Done in ${result.stats.totalMs.toFixed(2)} ms over ${result.stats.steps} step${result.stats.steps === 1 ? "" : "s"}.`;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
    renderProfileEditor();
  }
}

function currentProfileRows(): HTMLElement[] {
  const host = document.querySelector<HTMLDivElement>("#profile-points");
  return host
    ? [...host.querySelectorAll<HTMLElement>("[data-profile-row]")]
    : [];
}

function currentProfileOrNull(): NumericProfilePoint[] | null {
  const form = document.querySelector<HTMLFormElement>("#controls-form");
  if (!form) return null;
  try {
    return readProfile(form);
  } catch {
    return null;
  }
}

function currentEditorTargets(): LimitTarget[] {
  const form = document.querySelector<HTMLFormElement>("#controls-form");
  const profile = currentProfileOrNull() ?? defaultHalfProfile;
  const top = profile[profile.length - 1]!;
  if (!form) {
    return [...defaultTargets, { ...defaultTotalTarget, limitY: top.y }];
  }
  const data = new FormData(form);
  return [
    ...defaultTargets.map((target, index) => {
      const limitY = Number(data.get(`target-${index}-limitY`));
      const targetSurface = Number(data.get(`target-${index}-surface`));
      return {
        name: String(data.get(`target-${index}-name`) || target.name),
        limitY: Number.isFinite(limitY) ? limitY : target.limitY,
        targetSurface: Number.isFinite(targetSurface)
          ? targetSurface
          : target.targetSurface,
      };
    }),
    {
      name: String(data.get("total-target-name") || defaultTotalTarget.name),
      limitY: top.y,
      targetSurface: Number.isFinite(Number(data.get("total-target-surface")))
        ? Number(data.get("total-target-surface"))
        : defaultTotalTarget.targetSurface,
    },
  ];
}

function renderProfileEditor(
  profile = currentProfileOrNull(),
  targets = currentEditorTargets(),
): void {
  const host = document.querySelector<HTMLDivElement>("#profile-editor");
  if (!host) return;
  if (!profile) {
    host.innerHTML = `<div class="profile-editor-error">Enter a valid, bottom-up profile to use the drag editor.</div>`;
    return;
  }
  host.innerHTML = buildProfileEditorSvg(profile, targets);
}

function svgPointFromPointer(
  svg: SVGSVGElement,
  event: Pick<PointerEvent | MouseEvent, "clientX" | "clientY">,
): [number, number] {
  const rect = svg.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) / rect.width) * PROFILE_EDITOR_WIDTH,
    ((event.clientY - rect.top) / rect.height) * PROFILE_EDITOR_HEIGHT,
  ];
}

function writeProfileRowPoint(
  row: HTMLElement,
  point: NumericProfilePoint,
): void {
  const xInput = row.querySelector<HTMLInputElement>(".point-x");
  const yInput = row.querySelector<HTMLInputElement>(".point-y");
  if (xInput) xInput.value = formatProfileNumber(point.x);
  if (yInput) yInput.value = formatProfileNumber(point.y);
}

function constrainedProfilePoint(
  profile: NumericProfilePoint[],
  index: number,
  point: NumericProfilePoint,
): NumericProfilePoint {
  if (index === 0) return { x: 0, y: 0 };

  const previous = profile[index - 1]!;
  const next = profile[index + 1];
  const minY = previous.y + PROFILE_MIN_Y_GAP;
  const maxY = next ? Math.max(minY, next.y - PROFILE_MIN_Y_GAP) : Infinity;
  return {
    x: Math.max(PROFILE_MIN_X, Math.abs(point.x)),
    y: clamp(point.y, minY, maxY),
  };
}

function updateProfilePoint(index: number, point: NumericProfilePoint): void {
  const rows = currentProfileRows();
  const row = rows[index];
  if (!row) return;
  writeProfileRowPoint(row, point);
}

function insertProfilePoint(
  insertBeforeIndex: number,
  point: NumericProfilePoint,
): void {
  const host = document.querySelector<HTMLDivElement>("#profile-points");
  if (!host) return;
  const rows = currentProfileRows();
  const beforeRow = rows[insertBeforeIndex];
  const template = document.createElement("template");
  template.innerHTML = profilePointRowHtml(
    point,
    insertBeforeIndex,
    rows.length + 1,
  );
  const row = template.content.firstElementChild;
  if (!row) return;
  host.insertBefore(row, beforeRow ?? null);
  reindexProfileRows();
  renderProfileEditor();
}

function nearestProfileSegment(
  profile: NumericProfilePoint[],
  targets: LimitTarget[],
  x: number,
  y: number,
): { index: number; t: number; distanceSq: number } | null {
  const geometry = profileEditorGeometry(
    profile,
    targets.map((target) => target.limitY),
  );
  let nearest: { index: number; t: number; distanceSq: number } | null = null;

  for (let index = 0; index < profile.length - 1; index++) {
    for (const side of [1, -1]) {
      const a = profile[index]!;
      const b = profile[index + 1]!;
      const [ax, ay] = geometry.mapPoint({ x: side * a.x, y: a.y });
      const [bx, by] = geometry.mapPoint({ x: side * b.x, y: b.y });
      const dx = bx - ax;
      const dy = by - ay;
      const lengthSq = dx * dx + dy * dy;
      if (lengthSq === 0) continue;
      const t = clamp(((x - ax) * dx + (y - ay) * dy) / lengthSq, 0.05, 0.95);
      const px = ax + t * dx;
      const py = ay + t * dy;
      const distanceSq = (x - px) ** 2 + (y - py) ** 2;
      if (!nearest || distanceSq < nearest.distanceSq) {
        nearest = { index, t, distanceSq };
      }
    }
  }

  return nearest;
}

function addPointFromEditorDoubleClick(
  svg: SVGSVGElement,
  event: MouseEvent,
): void {
  const profile = currentProfileOrNull();
  if (!profile) return;
  const targets = currentEditorTargets();
  const [x, y] = svgPointFromPointer(svg, event);
  const nearest = nearestProfileSegment(profile, targets, x, y);
  if (!nearest || nearest.distanceSq > 24 ** 2) return;

  const a = profile[nearest.index]!;
  const b = profile[nearest.index + 1]!;
  const point = {
    x: a.x + (b.x - a.x) * nearest.t,
    y: a.y + (b.y - a.y) * nearest.t,
  };
  insertProfilePoint(nearest.index + 1, point);
  runFromForm();
}

function startProfilePointDrag(
  svg: SVGSVGElement,
  pointIndex: number,
  event: PointerEvent,
): void {
  if (pointIndex === 0) return;
  const startProfile = currentProfileOrNull();
  if (!startProfile) return;
  const targets = currentEditorTargets();
  const geometry = profileEditorGeometry(
    startProfile,
    targets.map((target) => target.limitY),
  );
  const svgRect = svg.getBoundingClientRect();
  const pointFromDrag = (moveEvent: PointerEvent): [number, number] => [
    ((moveEvent.clientX - svgRect.left) / svgRect.width) * PROFILE_EDITOR_WIDTH,
    ((moveEvent.clientY - svgRect.top) / svgRect.height) *
      PROFILE_EDITOR_HEIGHT,
  ];

  event.preventDefault();

  const move = (moveEvent: PointerEvent) => {
    const profile = currentProfileOrNull();
    if (!profile) return;
    const [x, y] = pointFromDrag(moveEvent);
    const rawPoint = geometry.unmapPoint(x, y);
    const point = constrainedProfilePoint(profile, pointIndex, rawPoint);
    updateProfilePoint(pointIndex, point);
    const nextProfile = currentProfileOrNull();
    renderProfileEditor(nextProfile);
  };
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    runFromForm();
  };

  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up, { once: true });
}

function reindexProfileRows(): void {
  const rows = [
    ...document.querySelectorAll<HTMLElement>("[data-profile-row]"),
  ];
  rows.forEach((row, index) => {
    const label = row.querySelector<HTMLElement>(".point-label");
    const removeButton = row.querySelector<HTMLButtonElement>(".remove-point");
    const isBottom = index === 0;
    const isTop = index === rows.length - 1;
    if (label) {
      label.textContent = isBottom
        ? "bottom / keel"
        : isTop
          ? "top / deck"
          : `point ${index}`;
    }
    if (removeButton) removeButton.disabled = isBottom || rows.length <= 2;
  });
}

function addProfilePoint(): void {
  const host = document.querySelector<HTMLDivElement>("#profile-points");
  if (!host) return;
  const rows = [...host.querySelectorAll<HTMLElement>("[data-profile-row]")];
  const beforeTop = rows[rows.length - 1];
  const previous = rows[rows.length - 2] ?? rows[0];
  const top = rows[rows.length - 1];
  if (!previous || !top) return;

  const previousX = Number(
    previous.querySelector<HTMLInputElement>(".point-x")?.value ?? 0,
  );
  const previousY = Number(
    previous.querySelector<HTMLInputElement>(".point-y")?.value ?? 0,
  );
  const topX = Number(
    top.querySelector<HTMLInputElement>(".point-x")?.value ?? 0,
  );
  const topY = Number(
    top.querySelector<HTMLInputElement>(".point-y")?.value ?? 0,
  );
  const point = {
    x: Number.isFinite(previousX + topX) ? (previousX + topX) / 2 : 0.5,
    y: Number.isFinite(previousY + topY) ? (previousY + topY) / 2 : 1,
  };

  const template = document.createElement("template");
  template.innerHTML = profilePointRowHtml(
    point,
    rows.length - 1,
    rows.length + 1,
  );
  const row = template.content.firstElementChild;
  if (!row) return;
  host.insertBefore(row, beforeTop ?? null);
  reindexProfileRows();
}

function isSurfaceTargetInput(
  target: EventTarget | null,
): target is HTMLInputElement {
  return (
    target instanceof HTMLInputElement &&
    (target.name.startsWith("target-") ||
      target.name.startsWith("total-target-"))
  );
}

function handleSurfaceTargetEdit(event: Event): void {
  if (!isSurfaceTargetInput(event.target)) return;
  renderProfileEditor();
  scheduleRunFromForm();
}

function bindEvents(): void {
  document
    .querySelector<HTMLFormElement>("#controls-form")
    ?.addEventListener("submit", (event) => {
      event.preventDefault();
      runFromForm();
    });

  document.querySelector("#optimise-button")?.addEventListener("click", () => {
    runFromForm();
  });

  document.querySelector("#add-point-button")?.addEventListener("click", () => {
    addProfilePoint();
    runFromForm();
  });

  document
    .querySelector("#profile-points")
    ?.addEventListener("click", (event) => {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>(
        ".remove-point",
      );
      if (!button || button.disabled) return;
      button.closest("[data-profile-row]")?.remove();
      reindexProfileRows();
      runFromForm();
    });

  document
    .querySelector("#profile-points")
    ?.addEventListener("input", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement)) return;
      if (!target.matches(".point-x, .point-y")) return;
      renderProfileEditor();
    });

  const controlsForm = document.querySelector("#controls-form");
  controlsForm?.addEventListener("input", handleSurfaceTargetEdit);
  controlsForm?.addEventListener("change", handleSurfaceTargetEdit);

  document
    .querySelector("#profile-editor")
    ?.addEventListener("dblclick", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest(".profile-editor-handle")) return;
      const svg = target.closest<SVGSVGElement>("svg.profile-editor-svg");
      if (!svg) return;
      addPointFromEditorDoubleClick(svg, event as MouseEvent);
    });

  document
    .querySelector("#profile-editor")
    ?.addEventListener("pointerdown", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const handle = target.closest<SVGCircleElement>(".profile-editor-handle");
      const svg = target.closest<SVGSVGElement>("svg.profile-editor-svg");
      if (!handle || !svg) return;
      const pointIndex = Number(handle.dataset.pointIndex);
      if (!Number.isInteger(pointIndex)) return;
      startProfilePointDrag(svg, pointIndex, event as PointerEvent);
    });

  document.querySelector("#reset-button")?.addEventListener("click", () => {
    renderAppShell();
    bindEvents();
    runFromForm();
  });
}

renderAppShell();
bindEvents();
runFromForm();
