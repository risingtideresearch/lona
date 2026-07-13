/**
 * Result shapes produced by graded tape evaluators (forward-mode autodiff).
 */

export type GradientResult = {
  val: number;
  gradient: number[];
};

export type JacobianResult = {
  vals: number[];
  jacobian: number[][];
};
