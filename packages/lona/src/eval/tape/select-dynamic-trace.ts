import {
  BinaryOp,
  Derivative,
  KIND_DERIVATIVE,
  KIND_FOREIGN,
  KIND_LIT,
  KIND_SELECT,
  KIND_VAR,
  LiteralNum,
  SelectOp,
  UnaryOp,
  Variable,
  type NumNode,
  type VarName,
  allVariables,
  isBinaryKind,
  isUnaryKind,
} from "../../core/tree";
import { OP_ASSERT_NONZERO, OP_ASSERT_ZERO, OP_VAR } from "./opcodes";
import { GrowableTape } from "./growable-tape";
import type { CompiledTape, TapeAssertion } from "./compiled-tape";
import type { TapeAssertionKind } from "./assertions";
import { interpretTapeValuesRange } from "../routines/backends/js-interp/tape-eval";

export interface DynamicSelectTraceResult {
  tape: CompiledTape;
  selectDecisions: number;
  guardAssertions: number;
}

/**
 * Build a guarded specialized tape by evaluating only select conditions and
 * selected branches. Unlike `traceTapeValues` + `specializeSelectsFromTrace`,
 * this never evaluates unselected select branches while discovering the trace.
 */
export function compileDynamicSelectTraceTape(
  roots: NumNode[],
  vars: Map<VarName, number> | Map<string, number>,
  derivatives?: Map<VarName, number> | Map<string, number>,
  regularVarSlots?: readonly VarName[],
): DynamicSelectTraceResult | null {
  if (roots.length === 0) return null;

  const regularVars = regularVarSlots ?? collectRegularVars(roots);

  const builder = new DynamicSelectTraceBuilder(vars, derivatives, regularVars);
  const rootIndices: number[] = [];
  for (const root of roots) {
    const idx = builder.emitSelected(root);
    if (idx === null) return null;
    rootIndices.push(idx);
  }

  const tape = builder.toCompiledTape(rootIndices as [number, ...number[]]);
  return {
    tape,
    selectDecisions: builder.selectDecisions,
    guardAssertions: builder.guardAssertions,
  };
}

function collectRegularVars(roots: NumNode[]): VarName[] {
  const allRegularVars = new Set<VarName>();
  for (const root of roots) {
    for (const variable of allVariables(root)) allRegularVars.add(variable);
  }
  return [...allRegularVars];
}

class DynamicSelectTraceBuilder {
  private readonly tape = new GrowableTape();
  private readonly memo = new Map<NumNode, number>();
  private values = new Float64Array(64);
  private varValues = new Float64Array(64);
  private evaluatedUntil = 0;
  private derivativeNameToSlot: Map<VarName, number> | null = null;
  private readonly numRegularVars: number;
  private readonly assertions: TapeAssertion[] = [];
  private nextAssertionId = 0;
  private _selectDecisions = 0;

  constructor(
    private readonly vars: Map<VarName, number> | Map<string, number>,
    private readonly derivatives:
      Map<VarName, number> | Map<string, number> | undefined,
    regularVars: readonly VarName[],
  ) {
    this.numRegularVars = regularVars.length;
    this.ensureVarValueCapacity(regularVars.length);
    for (let slot = 0; slot < regularVars.length; slot++) {
      const name = regularVars[slot]!;
      this.tape.varSlots.push(name);
      this.tape.varNameToSlot.set(name, slot);
      this.varValues[slot] = (this.vars as Map<VarName, number>).get(name) ?? 0;
    }
  }

  get selectDecisions(): number {
    return this._selectDecisions;
  }

  get guardAssertions(): number {
    return this.assertions.length;
  }

  emitSelected(node: NumNode): number | null {
    const cached = this.memo.get(node);
    if (cached !== undefined) return cached;

    let idx: number | null;
    const kind = node.kind;

    if (kind === KIND_LIT) {
      idx = this.tape.emitLit((node as LiteralNum).value);
    } else if (kind === KIND_VAR) {
      idx = this.emitVar((node as Variable).name);
    } else if (kind === KIND_DERIVATIVE) {
      idx = this.emitDerivative((node as Derivative).variable.name);
    } else if (kind === KIND_SELECT) {
      idx = this.emitSelect(node as SelectOp);
    } else if (kind === KIND_FOREIGN) {
      return null;
    } else if (isUnaryKind(kind)) {
      const unary = node as UnaryOp;
      const originalIdx = this.emitSelected(unary.original);
      if (originalIdx === null) return null;
      idx = this.tape.emitUnary(kind, originalIdx);
    } else if (isBinaryKind(kind)) {
      const binary = node as BinaryOp;
      const leftIdx = this.emitSelected(binary.left);
      if (leftIdx === null) return null;
      const rightIdx = this.emitSelected(binary.right);
      if (rightIdx === null) return null;
      idx = this.tape.emitBinary(kind, leftIdx, rightIdx);
    } else {
      return null;
    }

    if (idx === null) return null;
    this.memo.set(node, idx);
    this.ensureValueCapacity(this.tape.length);
    return idx;
  }

  toCompiledTape(rootIndices: [number, ...number[]]): CompiledTape {
    const tape = this.tape.toCompiledTape(rootIndices);
    tape.numVars = this.numRegularVars;
    if (this.assertions.length > 0) {
      tape.assertions = [...this.assertions];
    }
    return tape;
  }

  private emitSelect(select: SelectOp): number | null {
    const conditionIdx = this.emitSelected(select.condition);
    if (conditionIdx === null) return null;

    const conditionValue = this.evalSlot(conditionIdx);
    const conditionPassed = conditionValue !== 0;
    this.emitAssertion(
      conditionIdx,
      conditionPassed ? "nonzero" : "zero",
      select,
    );
    this._selectDecisions++;

    const branchIdx = this.emitSelected(
      conditionPassed ? select.ifNonZero : select.ifZero,
    );
    if (branchIdx === null) return null;
    return branchIdx;
  }

  private emitVar(name: VarName): number {
    const idx = this.tape.emitVar(name);
    const slot = this.tape.varNameToSlot.get(name)!;
    this.ensureVarValueCapacity(slot + 1);
    this.varValues[slot] = (this.vars as Map<VarName, number>).get(name) ?? 0;
    return idx;
  }

  /**
   * Preserve `compileTape` semantics for DAGs that explicitly contain
   * `Derivative` nodes. This is not used by `compileGradRoutine`'s forward AD
   * path; it is for value routines whose expression reads derivative slots via
   * the optional `routine.eval(vars, derivatives)` argument.
   */
  private emitDerivative(name: VarName): number {
    if (!this.derivativeNameToSlot) this.derivativeNameToSlot = new Map();

    let slot = this.derivativeNameToSlot.get(name);
    if (slot === undefined) {
      slot = this.tape.varSlots.length;
      this.tape.varSlots.push(name);
      this.derivativeNameToSlot.set(name, slot);
    }

    const idx = this.tape.emitUnary(OP_VAR, 0);
    this.tape.argA[idx] = slot;
    this.tape.argB[idx] = 0;
    this.ensureVarValueCapacity(slot + 1);
    this.varValues[slot] =
      (this.derivatives as Map<VarName, number> | undefined)?.get(name) ?? 0;
    return idx;
  }

  private emitAssertion(
    operandIdx: number,
    kind: TapeAssertionKind,
    source: SelectOp,
  ): number {
    const id = this.nextAssertionId++;
    const idx = this.tape.emitUnary(
      kind === "zero" ? OP_ASSERT_ZERO : OP_ASSERT_NONZERO,
      operandIdx,
    );
    this.tape.argB[idx] = id;
    this.assertions.push({ id, tapeIndex: idx, kind, source });
    this.ensureValueCapacity(this.tape.length);
    return idx;
  }

  private evalSlot(idx: number): number {
    if (idx >= this.evaluatedUntil) {
      this.ensureValueCapacity(this.tape.length);
      interpretTapeValuesRange(
        this.tape.opcodes,
        this.tape.argA,
        this.tape.argB,
        this.tape.literals,
        this.evaluatedUntil,
        idx + 1,
        this.varValues,
        this.values,
      );
      this.evaluatedUntil = idx + 1;
    }
    return this.values[idx]!;
  }

  private ensureValueCapacity(needed: number): void {
    if (needed <= this.values.length) return;
    let cap = this.values.length;
    while (cap < needed) cap *= 2;
    const next = new Float64Array(cap);
    next.set(this.values);
    this.values = next;
  }

  private ensureVarValueCapacity(needed: number): void {
    if (needed <= this.varValues.length) return;
    let cap = this.varValues.length;
    while (cap < needed) cap *= 2;
    const next = new Float64Array(cap);
    next.set(this.varValues);
    this.varValues = next;
  }
}
