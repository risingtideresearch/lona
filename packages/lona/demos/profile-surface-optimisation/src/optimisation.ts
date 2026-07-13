import type { JacobianRoutine, NumericApi, RoutineBuildContext } from "lona";
import { buildRoutine, direct } from "lona";

export type BackendChoice = "js-interp" | "wasm-interp" | "direct";
export type SelectSpecializationChoice = "off" | "trace" | "full-trace";

/**
 * Builds the residual vector against whatever numeric API the routine builder
 * hands it. Being generic lets a single builder drive both the compiled
 * symbolic backends and the `direct` forward-autodiff backend.
 */
export type ResidualBuilder = <TNum extends NumericApi<TNum>>(
  ctx: RoutineBuildContext<TNum>,
) => TNum[];

export type OptimisationSettings = {
  iterations: number;
  initialDamping: number;
  regularisationWeight: number;
  backend: BackendChoice;
  selectSpecialization: SelectSpecializationChoice;
};

export type OptimisationStopReason =
  "max-iterations" | "small-gradient" | "small-step" | "no-improvement";

export type LevenbergMarquardtStats = {
  steps: number;
  compileMs: number;
  optimiseMs: number;
  initialLoss: number;
  finalLoss: number;
  stoppedReason: OptimisationStopReason;
};

type ResidualEvaluation = {
  vals: number[];
  jacobian: number[][];
};

type ResidualEvaluator = {
  eval(values: Record<string, number>): ResidualEvaluation;
  dispose?: () => void;
};

function objective(vals: number[]): number {
  return vals.reduce((sum, value) => sum + value * value, 0);
}

function buildJacobianRoutine(
  builder: ResidualBuilder,
  variableNames: string[],
  settings: OptimisationSettings,
): JacobianRoutine {
  // The `direct` backend skips compilation entirely: the routine re-runs the
  // builder on every eval, accumulating forward-mode derivatives. The symbolic
  // backends discover the expression once and lower it to a compiled routine.
  const routine =
    settings.backend === "direct"
      ? direct.buildRoutine(builder, { diffVars: variableNames })
      : buildRoutine(builder, {
          diffVars: variableNames,
          backend: settings.backend,
          selectSpecialization:
            settings.selectSpecialization === "off"
              ? undefined
              : settings.selectSpecialization,
        });

  if (routine.shape !== "jacobian") {
    throw new Error("expected a jacobian routine");
  }
  return routine;
}

function buildResidualEvaluator(
  builder: ResidualBuilder,
  variableNames: string[],
  settings: OptimisationSettings,
): { evaluator: ResidualEvaluator; compileMs: number } {
  const compileStart = performance.now();
  const routine = buildJacobianRoutine(builder, variableNames, settings);
  const compileMs = performance.now() - compileStart;

  return {
    compileMs,
    evaluator: {
      eval: (values) => routine.eval(new Map(Object.entries(values))),
      dispose: () => routine.dispose?.(),
    },
  };
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] | null {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]!]);

  for (let pivot = 0; pivot < n; pivot++) {
    let bestRow = pivot;
    for (let row = pivot + 1; row < n; row++) {
      if (Math.abs(a[row]![pivot]!) > Math.abs(a[bestRow]![pivot]!)) {
        bestRow = row;
      }
    }

    if (Math.abs(a[bestRow]![pivot]!) < 1e-14) return null;
    [a[pivot], a[bestRow]] = [a[bestRow]!, a[pivot]!];

    for (let row = pivot + 1; row < n; row++) {
      const factor = a[row]![pivot]! / a[pivot]![pivot]!;
      for (let col = pivot; col <= n; col++) {
        a[row]![col] = a[row]![col]! - factor * a[pivot]![col]!;
      }
    }
  }

  const solution = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let value = a[row]![n]!;
    for (let col = row + 1; col < n; col++) {
      value -= a[row]![col]! * solution[col]!;
    }
    solution[row] = value / a[row]![row]!;
  }

  return solution;
}

function levenbergMarquardtStep(
  evaluation: ResidualEvaluation,
  numVariables: number,
  damping: number,
): { delta: number[]; gradientInfinityNorm: number } | null {
  const normalMatrix = Array.from({ length: numVariables }, () =>
    new Array<number>(numVariables).fill(0),
  );
  const rhs = new Array<number>(numVariables).fill(0);

  for (let row = 0; row < evaluation.vals.length; row++) {
    const residual = evaluation.vals[row]!;
    const jacobianRow = evaluation.jacobian[row]!;
    for (let col = 0; col < numVariables; col++) {
      const jCol = jacobianRow[col]!;
      rhs[col] -= jCol * residual;
      for (let otherCol = 0; otherCol < numVariables; otherCol++) {
        normalMatrix[col]![otherCol] += jCol * jacobianRow[otherCol]!;
      }
    }
  }

  let gradientInfinityNorm = 0;
  for (const value of rhs) {
    gradientInfinityNorm = Math.max(gradientInfinityNorm, Math.abs(value));
  }

  for (let i = 0; i < numVariables; i++) {
    normalMatrix[i]![i] += damping;
  }

  const delta = solveLinearSystem(normalMatrix, rhs);
  return delta ? { delta, gradientInfinityNorm } : null;
}

function candidateValues(
  values: Record<string, number>,
  variableNames: string[],
  delta: number[],
): Record<string, number> {
  const next = { ...values };
  for (let i = 0; i < variableNames.length; i++) {
    const name = variableNames[i]!;
    next[name] = values[name]! + delta[i]!;
  }
  return next;
}

export function optimiseResiduals(
  builder: ResidualBuilder,
  variableNames: string[],
  initialValues: Record<string, number>,
  settings: OptimisationSettings,
): {
  values: Record<string, number>;
  stats: LevenbergMarquardtStats;
} {
  const { evaluator, compileMs } = buildResidualEvaluator(
    builder,
    variableNames,
    settings,
  );

  let values = { ...initialValues };
  let evaluation = evaluator.eval(values);
  const initialLoss = objective(evaluation.vals);
  let currentLoss = initialLoss;
  let damping = settings.initialDamping;
  let steps = 0;
  let stoppedReason: OptimisationStopReason = "max-iterations";

  const optimiseStart = performance.now();
  for (let iteration = 1; iteration <= settings.iterations; iteration++) {
    let accepted = false;
    let stepWasTiny = false;

    for (let attempt = 0; attempt < 12; attempt++) {
      const step = levenbergMarquardtStep(
        evaluation,
        variableNames.length,
        damping,
      );
      if (!step) {
        damping *= 10;
        continue;
      }

      if (step.gradientInfinityNorm < 1e-10) {
        stoppedReason = "small-gradient";
        accepted = false;
        break;
      }

      const maxAbsDelta = Math.max(
        ...step.delta.map((value) => Math.abs(value)),
      );
      if (maxAbsDelta < 1e-10) {
        stoppedReason = "small-step";
        stepWasTiny = true;
        break;
      }

      const proposedValues = candidateValues(values, variableNames, step.delta);
      const proposedEvaluation = evaluator.eval(proposedValues);
      const proposedLoss = objective(proposedEvaluation.vals);

      if (proposedLoss < currentLoss) {
        values = proposedValues;
        evaluation = proposedEvaluation;
        currentLoss = proposedLoss;
        damping = Math.max(damping * 0.3, 1e-12);
        steps++;
        accepted = true;
        break;
      }

      damping *= 10;
    }

    if (stoppedReason === "small-gradient" || stepWasTiny) break;
    if (!accepted) {
      stoppedReason = "no-improvement";
      break;
    }
  }

  const optimiseMs = performance.now() - optimiseStart;
  evaluator.dispose?.();

  return {
    values,
    stats: {
      steps,
      compileMs,
      optimiseMs,
      initialLoss,
      finalLoss: currentLoss,
      stoppedReason,
    },
  };
}
