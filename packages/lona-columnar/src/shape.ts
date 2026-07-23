import { Num } from "lona";
import type { NumStruct } from "lona";
import type { ColumnValue, ExternalNumValue, StageUsing } from "./types";

export interface ValueShape<T = unknown> {
  readonly kind: "num" | "struct";
  readonly width: number;
  readonly witness?: NumStruct<unknown>;
  flatten(value: unknown, label?: string): readonly Num[];
  rebuild(parts: readonly Num[]): T;
}

function isNumStruct(value: unknown): value is NumStruct<unknown> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<NumStruct<unknown>>;
  return (
    typeof candidate.toNums === "function" &&
    typeof candidate.fromNums === "function"
  );
}

function checkedNums(parts: readonly unknown[], label: string): Num[] {
  return parts.map((part, index) => {
    if (!(part instanceof Num)) {
      throw new Error(`${label} component ${index} is not a Num`);
    }
    return part;
  });
}

export function shapeOfValue<T extends ColumnValue>(
  value: T,
  label = "column value",
): ValueShape<T> {
  if (value instanceof Num) {
    return Object.freeze({
      kind: "num",
      width: 1,
      flatten(candidate: unknown, candidateLabel = label): readonly Num[] {
        if (!(candidate instanceof Num)) {
          throw new Error(`${candidateLabel} does not match scalar Num shape`);
        }
        return [candidate];
      },
      rebuild(parts: readonly Num[]): T {
        if (parts.length !== 1) {
          throw new Error(
            `scalar Num shape requires 1 component, got ${parts.length}`,
          );
        }
        return parts[0] as T;
      },
    });
  }

  if (!isNumStruct(value)) {
    throw new Error(`${label} is neither a Num nor a NumStruct`);
  }

  const witness = value;
  const witnessParts = checkedNums(witness.toNums(), `${label}.toNums()`);
  const width = witnessParts.length;
  if (width === 0) throw new Error(`${label} NumStruct has no components`);

  return Object.freeze({
    kind: "struct",
    width,
    witness,
    flatten(candidate: unknown, candidateLabel = label): readonly Num[] {
      if (!isNumStruct(candidate) || candidate instanceof Num) {
        throw new Error(`${candidateLabel} does not match NumStruct shape`);
      }
      const parts = checkedNums(
        candidate.toNums(),
        `${candidateLabel}.toNums()`,
      );
      if (parts.length !== width) {
        throw new Error(
          `${candidateLabel} has ${parts.length} components; expected ${width}`,
        );
      }
      return parts;
    },
    rebuild(parts: readonly Num[]): T {
      if (parts.length !== width) {
        throw new Error(
          `NumStruct shape requires ${width} components, got ${parts.length}`,
        );
      }
      const rebuilt = witness.fromNums([...parts]);
      if (!isNumStruct(rebuilt)) {
        throw new Error("NumStruct.fromNums() did not return a NumStruct");
      }
      const rebuiltParts = checkedNums(
        rebuilt.toNums(),
        "NumStruct.fromNums().toNums()",
      );
      if (rebuiltParts.length !== width) {
        throw new Error(
          `NumStruct.fromNums() returned ${rebuiltParts.length} components; expected ${width}`,
        );
      }
      return rebuilt as T;
    },
  });
}

export interface FlatUsing {
  readonly names: readonly string[];
  readonly roots: readonly Num[];
  readonly shapes: readonly ValueShape[];
}

export function flattenUsing<TUsing extends StageUsing>(
  using: TUsing,
): FlatUsing {
  const names = Object.keys(using);
  const roots: Num[] = [];
  const shapes: ValueShape[] = [];

  for (const name of names) {
    const value: ExternalNumValue | undefined = using[name];
    if (value === undefined) {
      throw new Error(`using.${name} is undefined`);
    }
    const shape = shapeOfValue(value, `using.${name}`);
    shapes.push(shape);
    roots.push(...shape.flatten(value, `using.${name}`));
  }

  return {
    names: Object.freeze(names),
    roots: Object.freeze(roots),
    shapes: Object.freeze(shapes),
  };
}

export function rebuildUsing<TUsing extends StageUsing>(
  flat: FlatUsing,
  parts: readonly Num[],
): TUsing {
  const rebuilt: Record<string, ExternalNumValue> = {};
  let offset = 0;

  for (let i = 0; i < flat.names.length; i++) {
    const shape = flat.shapes[i]!;
    rebuilt[flat.names[i]!] = shape.rebuild(
      parts.slice(offset, offset + shape.width),
    ) as ExternalNumValue;
    offset += shape.width;
  }

  if (offset !== parts.length) {
    throw new Error(
      `using bindings consumed ${offset} components, got ${parts.length}`,
    );
  }

  return rebuilt as TUsing;
}

export function emptyUsing<TUsing extends StageUsing>(): TUsing {
  return Object.freeze({}) as TUsing;
}
