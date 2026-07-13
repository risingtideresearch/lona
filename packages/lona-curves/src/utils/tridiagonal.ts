import { mapNums, zipNums, type Num, type NumStruct } from "lona";

// ---------------------------------------------------------------------------
// Thomas elimination — the tridiagonal linear solve shared by the spline
// builders. A fixed forward sweep followed by back-substitution, with no
// pivoting, so it stays entirely within `Num` arithmetic and runs on symbolic
// coefficients. Coefficients are scalar `Num`s; the unknowns / right-hand side
// are any `NumStruct` (a vector `Vec2`/`Vec3`, a `Point`, …), combined
// componentwise — so the one routine serves the natural spline's vector
// tangents and the κ-curve's control points alike.
// ---------------------------------------------------------------------------

/**
 * Solve the tridiagonal system `A·x = rhs`, where row `i` is
 * `lower[i]·xᵢ₋₁ + diag[i]·xᵢ + upper[i]·xᵢ₊₁ = rhs[i]`. `lower[0]` and
 * `upper[n−1]` are ignored (the corners have no off-diagonal neighbour), so
 * callers may leave them unset.
 *
 * Stable only for (weakly) diagonally dominant systems — which the spline
 * solves here are. There is deliberately no pivoting: it would have to branch on
 * coefficient values, which the fully-`Num` model forbids.
 */
export function solveTridiagonal<T extends NumStruct<T>>(
  lower: Num[],
  diag: Num[],
  upper: Num[],
  rhs: T[],
): T[] {
  const n = diag.length;
  const sub = (a: T, b: T): T => zipNums(a, b, (x, y) => x.sub(y));
  const scale = (a: T, k: Num): T => mapNums(a, (x) => x.mul(k));

  // Forward sweep: eliminate the sub-diagonal, carrying the modified
  // super-diagonal `cp` (scalar) and right-hand side `dp` (the `NumStruct`).
  const cp: Num[] = new Array(n);
  const dp: T[] = new Array(n);
  let inv = diag[0].inv();
  if (n > 1) cp[0] = upper[0].mul(inv);
  dp[0] = scale(rhs[0], inv);
  for (let i = 1; i < n; i++) {
    inv = diag[i].sub(lower[i].mul(cp[i - 1])).inv();
    if (i < n - 1) cp[i] = upper[i].mul(inv);
    dp[i] = scale(sub(rhs[i], scale(dp[i - 1], lower[i])), inv);
  }

  // Back-substitution.
  const x: T[] = new Array(n);
  x[n - 1] = dp[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    x[i] = sub(dp[i], scale(x[i + 1], cp[i]));
  }
  return x;
}
