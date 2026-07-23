/**
 * JS tape evaluator — the hot-path eval loop over a CompiledTape,
 * plus a forward-mode autodiff evaluator that propagates dual numbers
 * through the same tape format.
 *
 * Tape compilation is in tape-opcodes.ts (shared with wasm-tape-eval and
 * gpu-tape-eval).
 */
import type { VarName } from "../../../../core/tree";
import type { GradientResult, JacobianResult } from "../../../tape";
import {
  type CompiledTape,
  OP_LIT,
  OP_VAR,
  OP_SQRT,
  OP_CBRT,
  OP_COS,
  OP_ACOS,
  OP_ASIN,
  OP_TAN,
  OP_ATAN,
  OP_LOG,
  OP_EXP,
  OP_ABS,
  OP_NEG,
  OP_SIN,
  OP_SIGN,
  OP_NOT,
  OP_TANH,
  OP_LOG1P,
  OP_DEBUG,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_MOD,
  OP_ATAN2,
  OP_MIN,
  OP_MAX,
  OP_COMPARE,
  OP_AND,
  OP_OR,
  OP_ASSERT_ZERO,
  OP_ASSERT_NONZERO,
  DIV_BY_ZERO_FALLBACK,
} from "../../../tape";

// Re-export tape compilation for backward compatibility
export {
  type CompiledTape,
  compileTape,
  compileTapeFromSerialized,
} from "../../../tape";
import { assertTapeValue, bindVarMap } from "../../../tape";

// ---------------------------------------------------------------------------
// Scratch buffer cache — avoids allocating per-eval while keeping CompiledTape
// free of mutable state.
// ---------------------------------------------------------------------------

type TapeScratch = { values: Float64Array; varValues: Float64Array };

const scratchCache = new WeakMap<CompiledTape, TapeScratch>();

function getScratch(tape: CompiledTape): TapeScratch {
  let s = scratchCache.get(tape);
  if (!s) {
    s = {
      values: new Float64Array(tape.opcodes.length),
      varValues: new Float64Array(tape.varSlots.length),
    };
    scratchCache.set(tape, s);
  }
  return s;
}

// ---------------------------------------------------------------------------
// interpretTapeValues — the hot loop, decoupled from CompiledTape /
// var-binding / scratch caching. Used by `evalTape` (below) and by
// LiveTape's js-interp sweep backend (live-tape-js-interp-sweep.ts).
//
// `len` is the number of valid entries in the op arrays. The arrays
// may be longer (capacity > length) when called from a growable tape.
// `values` is the destination buffer; the caller owns its lifecycle
// and reads roots / per-slot values out of it after return.
// ---------------------------------------------------------------------------

export function interpretTapeValues(
  opcodes: Uint8Array,
  argA: Int32Array,
  argB: Int32Array,
  literals: Float64Array,
  len: number,
  varValues: Float64Array,
  values: Float64Array,
): void {
  interpretTapeValuesRange(
    opcodes,
    argA,
    argB,
    literals,
    0,
    len,
    varValues,
    values,
  );
}

/** Evaluate only the tape slots in [start, end). */
export function interpretTapeValuesRange(
  opcodes: Uint8Array,
  argA: Int32Array,
  argB: Int32Array,
  literals: Float64Array,
  start: number,
  end: number,
  varValues: Float64Array,
  values: Float64Array,
): void {
  for (let i = start; i < end; i++) {
    const op = opcodes[i]!;
    switch (op) {
      // Leaves
      case OP_LIT:
        values[i] = literals[argA[i]!]!;
        break;
      case OP_VAR:
        values[i] = varValues[argA[i]!]!;
        break;

      // Unary
      case OP_NEG:
        values[i] = -values[argA[i]!]!;
        break;
      case OP_ABS:
        values[i] = Math.abs(values[argA[i]!]!);
        break;
      case OP_SQRT:
        values[i] = Math.sqrt(values[argA[i]!]!);
        break;
      case OP_CBRT:
        values[i] = Math.cbrt(values[argA[i]!]!);
        break;
      case OP_SIN:
        values[i] = Math.sin(values[argA[i]!]!);
        break;
      case OP_COS:
        values[i] = Math.cos(values[argA[i]!]!);
        break;
      case OP_TAN:
        values[i] = Math.tan(values[argA[i]!]!);
        break;
      case OP_ASIN:
        values[i] = Math.asin(values[argA[i]!]!);
        break;
      case OP_ACOS:
        values[i] = Math.acos(values[argA[i]!]!);
        break;
      case OP_ATAN:
        values[i] = Math.atan(values[argA[i]!]!);
        break;
      case OP_EXP:
        values[i] = Math.exp(values[argA[i]!]!);
        break;
      case OP_LOG:
        values[i] = Math.log(values[argA[i]!]!);
        break;
      case OP_LOG1P:
        values[i] = Math.log1p(values[argA[i]!]!);
        break;
      case OP_TANH:
        values[i] = Math.tanh(values[argA[i]!]!);
        break;
      case OP_SIGN:
        values[i] = Math.sign(values[argA[i]!]!);
        break;
      case OP_NOT:
        values[i] = values[argA[i]!]! ? 0 : 1;
        break;
      case OP_DEBUG:
        values[i] = values[argA[i]!]!;
        break;
      case OP_ASSERT_ZERO:
        values[i] = assertTapeValue("zero", values[argA[i]!]!, argB[i]!, i);
        break;
      case OP_ASSERT_NONZERO:
        values[i] = assertTapeValue("nonzero", values[argA[i]!]!, argB[i]!, i);
        break;

      // Binary
      case OP_ADD:
        values[i] = values[argA[i]!]! + values[argB[i]!]!;
        break;
      case OP_SUB:
        values[i] = values[argA[i]!]! - values[argB[i]!]!;
        break;
      case OP_MUL:
        values[i] = values[argA[i]!]! * values[argB[i]!]!;
        break;
      case OP_DIV: {
        const r = values[argB[i]!]!;
        values[i] = r ? values[argA[i]!]! / r : DIV_BY_ZERO_FALLBACK;
        break;
      }
      case OP_MOD:
        values[i] = values[argA[i]!]! % values[argB[i]!]!;
        break;
      case OP_ATAN2:
        values[i] = Math.atan2(values[argA[i]!]!, values[argB[i]!]!);
        break;
      case OP_MIN:
        values[i] = Math.min(values[argA[i]!]!, values[argB[i]!]!);
        break;
      case OP_MAX:
        values[i] = Math.max(values[argA[i]!]!, values[argB[i]!]!);
        break;
      case OP_COMPARE:
        values[i] = Math.sign(values[argA[i]!]! - values[argB[i]!]!);
        break;
      case OP_AND: {
        const l = values[argA[i]!]!;
        values[i] = l === 0 ? l : values[argB[i]!]!;
        break;
      }
      case OP_OR: {
        const l = values[argA[i]!]!;
        values[i] = l === 0 ? values[argB[i]!]! : l;
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// evalTape — public entry point. Thin wrapper over interpretTapeValues
// that handles scratch caching, var-map binding, and root extraction.
// ---------------------------------------------------------------------------

export function evalTape(
  tape: CompiledTape,
  vars: Map<VarName, number>,
  derivatives?: Map<VarName, number>,
): number[] {
  const { opcodes, argA, argB, literals, rootIndices } = tape;
  const { values, varValues } = getScratch(tape);

  bindVarMap(tape, vars, derivatives, varValues);

  interpretTapeValues(
    opcodes,
    argA,
    argB,
    literals,
    opcodes.length,
    varValues,
    values,
  );

  const numRoots = rootIndices.length;
  const result = new Array<number>(numRoots);
  for (let i = 0; i < numRoots; i++) {
    result[i] = values[rootIndices[i]!]!;
  }
  return result;
}

// ---------------------------------------------------------------------------
// evalTapeGrad — forward-mode autodiff over a CompiledTape
// ---------------------------------------------------------------------------
//
// Instead of building symbolic derivative DAGs (which can blow up on
// large expressions), this propagates dual numbers [value, ∂/∂x₁, …, ∂/∂xₙ]
// through the tape during evaluation. Each tape node stores (1 + numDiffVars)
// f64 values: the primal value and one partial derivative per variable.
//
// Cost: O(tape_length × numDiffVars) per evaluation — proportional to the
// number of variables, not the complexity of the derivative expression.

export type ForwardAutodiffFn = (vars: Map<VarName, number>) => GradientResult;
export type ForwardAutodiffMultiFn = (
  vars: Map<VarName, number>,
) => JacobianResult;

/**
 * Internal: compile the forward-mode dual-number tape walk.
 * Returns a function that runs the forward pass and populates `duals`,
 * plus the metadata needed to read results from it.
 */
function compileForwardPass(
  tape: CompiledTape,
  numDiff: number,
): {
  runForwardSeeded: (values: Float64Array, seeds: Float64Array) => void;
  duals: Float64Array;
  stride: number;
} {
  const {
    opcodes,
    argA,
    argB,
    literals,
    varSlots,
    numVars: tapeNumVars,
  } = tape;
  const tapeLen = opcodes.length;
  const stride = 1 + numDiff;

  const duals = new Float64Array(tapeLen * stride);
  const varValues = new Float64Array(varSlots.length);
  let activeSeeds = new Float64Array(0);

  const execute = (): void => {
    for (let i = 0; i < tapeLen; i++) {
      const op = opcodes[i]!;
      const base = i * stride;

      switch (op) {
        case OP_LIT: {
          duals[base] = literals[argA[i]!]!;
          for (let d = 0; d < numDiff; d++) duals[base + 1 + d] = 0;
          break;
        }
        case OP_VAR: {
          const slot = argA[i]!;
          duals[base] = varValues[slot]!;
          for (let d = 0; d < numDiff; d++) {
            duals[base + 1 + d] =
              slot < tapeNumVars ? activeSeeds[slot * numDiff + d]! : 0;
          }
          break;
        }

        // --- Unary ops ---
        case OP_NEG: {
          const aBase = argA[i]! * stride;
          duals[base] = -duals[aBase]!;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = -duals[aBase + 1 + d]!;
          break;
        }
        case OP_ABS: {
          const aBase = argA[i]! * stride;
          const v = duals[aBase]!;
          const s = v > 0 ? 1 : v < 0 ? -1 : 0;
          duals[base] = Math.abs(v);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = s * duals[aBase + 1 + d]!;
          break;
        }
        case OP_SQRT: {
          const aBase = argA[i]! * stride;
          const v = Math.sqrt(duals[aBase]!);
          duals[base] = v;
          const coeff = v !== 0 ? 0.5 / v : 0;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_CBRT: {
          const aBase = argA[i]! * stride;
          const v = Math.cbrt(duals[aBase]!);
          duals[base] = v;
          const coeff = v !== 0 ? 1 / (3 * v * v) : 0;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_SIN: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.sin(a);
          const cosA = Math.cos(a);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = cosA * duals[aBase + 1 + d]!;
          break;
        }
        case OP_COS: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.cos(a);
          const negSinA = -Math.sin(a);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = negSinA * duals[aBase + 1 + d]!;
          break;
        }
        case OP_TAN: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.tan(a);
          const cosA = Math.cos(a);
          const sec2 = 1 / (cosA * cosA);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = sec2 * duals[aBase + 1 + d]!;
          break;
        }
        case OP_ASIN: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.asin(a);
          const coeff = 1 / Math.sqrt(1 - a * a);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_ACOS: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.acos(a);
          const coeff = -1 / Math.sqrt(1 - a * a);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_ATAN: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.atan(a);
          const coeff = 1 / (1 + a * a);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_EXP: {
          const aBase = argA[i]! * stride;
          const v = Math.exp(duals[aBase]!);
          duals[base] = v;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = v * duals[aBase + 1 + d]!;
          break;
        }
        case OP_LOG: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.log(a);
          const coeff = 1 / a;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_LOG1P: {
          const aBase = argA[i]! * stride;
          const a = duals[aBase]!;
          duals[base] = Math.log1p(a);
          const coeff = 1 / (1 + a);
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_TANH: {
          const aBase = argA[i]! * stride;
          const v = Math.tanh(duals[aBase]!);
          duals[base] = v;
          const coeff = 1 - v * v; // sech^2
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = coeff * duals[aBase + 1 + d]!;
          break;
        }
        case OP_SIGN: {
          const aBase = argA[i]! * stride;
          duals[base] = Math.sign(duals[aBase]!);
          for (let d = 0; d < numDiff; d++) duals[base + 1 + d] = 0;
          break;
        }
        case OP_NOT: {
          const aBase = argA[i]! * stride;
          duals[base] = duals[aBase]! ? 0 : 1;
          for (let d = 0; d < numDiff; d++) duals[base + 1 + d] = 0;
          break;
        }
        case OP_DEBUG: {
          const aBase = argA[i]! * stride;
          duals[base] = duals[aBase]!;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[aBase + 1 + d]!;
          break;
        }
        case OP_ASSERT_ZERO:
        case OP_ASSERT_NONZERO: {
          const aBase = argA[i]! * stride;
          duals[base] = assertTapeValue(
            op === OP_ASSERT_ZERO ? "zero" : "nonzero",
            duals[aBase]!,
            argB[i]!,
            i,
          );
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[aBase + 1 + d]!;
          break;
        }

        // --- Binary ops ---
        case OP_ADD: {
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          duals[base] = duals[aBase]! + duals[bBase]!;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[aBase + 1 + d]! + duals[bBase + 1 + d]!;
          break;
        }
        case OP_SUB: {
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          duals[base] = duals[aBase]! - duals[bBase]!;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[aBase + 1 + d]! - duals[bBase + 1 + d]!;
          break;
        }
        case OP_MUL: {
          // d(a*b) = a'*b + a*b'
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          const a = duals[aBase]!;
          const b = duals[bBase]!;
          duals[base] = a * b;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] =
              duals[aBase + 1 + d]! * b + a * duals[bBase + 1 + d]!;
          break;
        }
        case OP_DIV: {
          // d(a/b) = (a'*b - a*b') / b^2
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          const a = duals[aBase]!;
          const b = duals[bBase]!;
          if (b === 0) {
            duals[base] = DIV_BY_ZERO_FALLBACK;
            for (let d = 0; d < numDiff; d++) duals[base + 1 + d] = 0;
          } else {
            duals[base] = a / b;
            const invB2 = 1 / (b * b);
            for (let d = 0; d < numDiff; d++)
              duals[base + 1 + d] =
                (duals[aBase + 1 + d]! * b - a * duals[bBase + 1 + d]!) * invB2;
          }
          break;
        }
        case OP_MOD: {
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          duals[base] = duals[aBase]! % duals[bBase]!;
          // d(a mod b) ≈ da (treating b as constant in most practical uses)
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[aBase + 1 + d]!;
          break;
        }
        case OP_ATAN2: {
          // d(atan2(a,b)) = (b*a' - a*b') / (a^2 + b^2)
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          const a = duals[aBase]!;
          const b = duals[bBase]!;
          duals[base] = Math.atan2(a, b);
          const denom = a * a + b * b;
          if (denom === 0) {
            for (let d = 0; d < numDiff; d++) duals[base + 1 + d] = 0;
          } else {
            const invDenom = 1 / denom;
            for (let d = 0; d < numDiff; d++)
              duals[base + 1 + d] =
                (b * duals[aBase + 1 + d]! - a * duals[bBase + 1 + d]!) *
                invDenom;
          }
          break;
        }
        case OP_MIN: {
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          const a = duals[aBase]!;
          const b = duals[bBase]!;
          const useA = a <= b;
          const srcBase = useA ? aBase : bBase;
          duals[base] = useA ? a : b;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[srcBase + 1 + d]!;
          break;
        }
        case OP_MAX: {
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          const a = duals[aBase]!;
          const b = duals[bBase]!;
          const useA = a >= b;
          const srcBase = useA ? aBase : bBase;
          duals[base] = useA ? a : b;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[srcBase + 1 + d]!;
          break;
        }
        case OP_COMPARE: {
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          duals[base] = Math.sign(duals[aBase]! - duals[bBase]!);
          for (let d = 0; d < numDiff; d++) duals[base + 1 + d] = 0;
          break;
        }
        case OP_AND: {
          // and(a,b) = a == 0 ? a : b
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          const a = duals[aBase]!;
          const srcBase = a === 0 ? aBase : bBase;
          duals[base] = a === 0 ? a : duals[bBase]!;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[srcBase + 1 + d]!;
          break;
        }
        case OP_OR: {
          // or(a,b) = a == 0 ? b : a
          const aBase = argA[i]! * stride;
          const bBase = argB[i]! * stride;
          const a = duals[aBase]!;
          const srcBase = a === 0 ? bBase : aBase;
          duals[base] = a === 0 ? duals[bBase]! : a;
          for (let d = 0; d < numDiff; d++)
            duals[base + 1 + d] = duals[srcBase + 1 + d]!;
          break;
        }
      }
    }
  };

  const runForwardSeeded = (
    values: Float64Array,
    seeds: Float64Array,
  ): void => {
    if (values.length !== tapeNumVars) {
      throw new Error(
        `seeded JVP expected ${tapeNumVars} values, got ${values.length}`,
      );
    }
    if (seeds.length !== tapeNumVars * numDiff) {
      throw new Error(
        `seeded JVP expected ${tapeNumVars * numDiff} seeds, got ${seeds.length}`,
      );
    }
    varValues.fill(0);
    varValues.set(values);
    activeSeeds = seeds;
    execute();
  };

  return { runForwardSeeded, duals, stride };
}

/** Compile a multi-root tape JVP accepting arbitrary packed tangent seeds. */
export function compileSeededJvp(
  tape: CompiledTape,
  numDirections: number,
): (
  values: Float64Array,
  seeds: Float64Array,
) => {
  vals: number[];
  tangents: number[][];
} {
  if (!Number.isInteger(numDirections) || numDirections < 0) {
    throw new Error(
      `seeded JVP direction count must be a non-negative integer, got ${numDirections}`,
    );
  }
  const { runForwardSeeded, duals, stride } = compileForwardPass(
    tape,
    numDirections,
  );
  return (values, seeds) => {
    runForwardSeeded(values, seeds);
    const vals = new Array<number>(tape.rootIndices.length);
    const tangents: number[][] = [];
    for (let root = 0; root < tape.rootIndices.length; root++) {
      const base = tape.rootIndices[root]! * stride;
      vals[root] = duals[base]!;
      tangents.push(
        Array.from(duals.subarray(base + 1, base + 1 + numDirections)),
      );
    }
    return { vals, tangents };
  };
}

/**
 * Compile a tape into a forward-mode autodiff evaluator (single root).
 */
export function compileForwardAutodiff(
  tape: CompiledTape,
  diffVars: VarName[],
): ForwardAutodiffFn {
  const jvp = compileSeededJvp(tape, diffVars.length);
  const inputs = compileIdentityInputs(tape, diffVars);
  return (vars: Map<VarName, number>): GradientResult => {
    const result = jvp(inputs.values(vars), inputs.seeds);
    return { val: result.vals[0]!, gradient: result.tangents[0]! };
  };
}

/**
 * Compile a multi-root tape into a forward-mode autodiff evaluator that
 * returns the full Jacobian. One forward pass computes all values and all
 * partial derivatives for every node; we read each root's dual at the end.
 */
export function compileForwardAutodiffMulti(
  tape: CompiledTape,
  diffVars: VarName[],
): ForwardAutodiffMultiFn {
  const jvp = compileSeededJvp(tape, diffVars.length);
  const inputs = compileIdentityInputs(tape, diffVars);
  return (vars: Map<VarName, number>): JacobianResult => {
    const result = jvp(inputs.values(vars), inputs.seeds);
    return { vals: result.vals, jacobian: result.tangents };
  };
}

function compileIdentityInputs(
  tape: CompiledTape,
  diffVars: readonly VarName[],
): {
  values(vars: Map<VarName, number>): Float64Array;
  seeds: Float64Array;
} {
  const seeds = new Float64Array(tape.numVars * diffVars.length);
  for (let input = 0; input < tape.numVars; input++) {
    const direction = diffVars.indexOf(tape.varSlots[input]!);
    if (direction >= 0) seeds[input * diffVars.length + direction] = 1;
  }
  return {
    values: (vars) =>
      Float64Array.from(
        tape.varSlots.slice(0, tape.numVars),
        (name) => vars.get(name) ?? 0,
      ),
    seeds,
  };
}
