export class TapeAssertionError extends Error {
  readonly name = "TapeAssertionError";

  constructor(
    readonly kind: TapeAssertionKind,
    readonly observed: number,
    readonly assertionId = -1,
    readonly tapeIndex = -1,
  ) {
    super(
      `Tape assertion failed: expected ${kind}, observed ${observed}` +
        (assertionId >= 0 ? ` (assertion ${assertionId})` : "") +
        (tapeIndex >= 0 ? ` at tape index ${tapeIndex}` : ""),
    );
  }
}

export type TapeAssertionKind = "zero" | "nonzero";

export function assertTapeValue(
  kind: TapeAssertionKind,
  observed: number,
  assertionId = -1,
  tapeIndex = -1,
): number {
  if (kind === "zero") {
    if (observed !== 0) {
      throw new TapeAssertionError(kind, observed, assertionId, tapeIndex);
    }
  } else if (observed === 0) {
    throw new TapeAssertionError(kind, observed, assertionId, tapeIndex);
  }
  return observed;
}
