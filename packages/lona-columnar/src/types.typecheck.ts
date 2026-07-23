import type { Num } from "lona";
import type { NumStruct } from "lona";
import type {
  Column,
  ColumnFactory,
  ReducedColumn,
  ColumnarOutput,
} from "./types";

interface Triangle extends NumStruct<Triangle> {
  readonly ax: Num;
  readonly ay: Num;
  readonly az: Num;
  readonly bx: Num;
  readonly by: Num;
  readonly bz: Num;
  readonly cx: Num;
  readonly cy: Num;
  readonly cz: Num;
}

/** Representative four-component NumStruct used by the hydro API spike. */
interface Contribution extends NumStruct<Contribution> {
  readonly volume: Num;
  readonly mx: Num;
  readonly my: Num;
  readonly mz: Num;
  add(other: Contribution): Contribution;
}

interface Plane extends NumStruct<Plane> {
  readonly z: Num;
}

/**
 * Compile-only API exercise. This function is never called; TypeScript checks
 * callback inference and every assignment below as part of `npm run typecheck`.
 */
export function columnarColumnTypecheck(
  column: ColumnFactory,
  nums: readonly Num[],
  triangles: readonly Triangle[],
  zeroContribution: Contribution,
  waterline: Num,
  density: Num,
  plane: Plane,
): void {
  const scalarColumn: Column<Num> = column(nums);
  const triangleColumn: Column<Triangle> = column(triangles, {
    placement: "gpu",
  });
  const emptyContributions: Column<Contribution> = column([], {
    shape: zeroContribution,
  });
  const scalarSum: ReducedColumn<Num> = scalarColumn.sum({
    placement: "gpu",
  });
  void scalarSum;
  void emptyContributions;

  const contributions: Column<Contribution> = triangleColumn.map({
    using: { waterline, plane },
    build: (triangle, { index, using }) => {
      const indexedVolume = triangle.ax
        .add(triangle.by)
        .add(triangle.cz)
        .sub(using.waterline)
        .add(using.plane.z)
        .add(index.mul(0));

      return zeroContribution.fromNums([
        indexedVolume,
        indexedVolume.mul(triangle.ax),
        indexedVolume.mul(triangle.ay),
        indexedVolume.mul(triangle.az),
      ]);
    },
    placement: "gpu",
  });

  const contributionSum: ReducedColumn<Contribution> = contributions.sum({
    componentWise: true,
    placement: "gpu",
  });
  // @ts-expect-error NumStruct built-ins require explicit component-wise algebra
  contributions.sum();
  void contributionSum;

  const totals: ReducedColumn<Contribution> = contributions.reduce({
    using: { density },
    combine: (left, right, { using }) => {
      const inferredDensity: Num = using.density;
      void inferredDensity;
      return left.add(right);
    },
    initial: zeroContribution,
    associative: true,
    order: "tree",
    placement: "gpu",
  });

  const output: ColumnarOutput<readonly Num[]> = totals.output({
    using: { density, plane },
    build: ([total], { using }) => {
      const safeVolume = total!.volume.max(1e-12);
      return [
        total!.volume.mul(using.density),
        total!.mx.div(safeVolume),
        total!.my.div(safeVolume),
        total!.mz.div(safeVolume).add(using.plane.z),
      ] as const;
    },
    placement: "cpu",
  });
  void output;

  const directReduced: ColumnarOutput<Contribution> = totals.output();
  const directRows: ColumnarOutput<Triangle | readonly Triangle[]> =
    triangleColumn.output();
  const continued: Column<Num> = totals.then(([total]) =>
    column([total!.volume, total!.mx]),
  );
  void directReduced;
  void directRows;
  void continued;

  // A column cannot mix scalar Nums and NumStruct values.
  // @ts-expect-error heterogeneous scalar/struct columns are rejected
  column([waterline, plane]);

  triangleColumn.map({
    using: { waterline },
    build: (triangle, { using }) => {
      const inferredNum: Num = using.waterline;
      const inferredTriangle: Triangle = triangle;
      void inferredNum;
      return inferredTriangle;
    },
  });

  totals.output({
    using: { plane },
    build: ([total], { using }) => {
      const inferredPlane: Plane = using.plane;
      const inferredContribution: Contribution = total!;
      void inferredPlane;
      return inferredContribution;
    },
  });
}
