import {
  NumNode,
  UnaryOp,
  BinaryOp,
  LiteralNum,
  ZERO_NODE as ZERO,
  ONE_NODE as ONE,
  DebugNode,
  SelectOp,
  KIND_LIT,
  KIND_ADD,
  KIND_ABS,
  KIND_MAX,
  KIND_MIN,
  KIND_MUL,
  KIND_NEG,
  KIND_NOT,
  KIND_SUB,
  KIND_DEBUG,
  type VarName,
} from "../../core/tree";
import {
  binaryNode,
  debugNode,
  litNode,
  selectNode,
  unaryNode,
} from "../../core/tree-cons";
import { UnaryOperation, BinaryOperation, NumEvalKernel } from "../../types";
import { evaluateBinaryOp, evaluateUnaryOp } from "./js-eval";

// Delegates to the hash-cons literal factory — `litNode` already dedups
// 0/1/2/-1 to the ZERO/ONE/TWO/NEG_ONE sentinels.
const makeLiteral = (value: number): NumNode => litNode(value);

/**
 * Constant-folding / algebraic rewrite kernel.
 *
 * The public `unaryOp` / `binaryOp` methods are the entry points called by
 * `genericEval`. They dispatch to the private `foldUnary` / `foldBinary`
 * helpers, which are also used *internally* by rewrite rules to re-fold
 * their own output. That cascading is what makes, e.g., `l1 + (l2 + x)`
 * collapse all the way to `x` when `l1 + l2 == 0`, instead of stopping
 * at the intermediate `ADD(0, x)` form.
 *
 * Every rewrite that emits a new binary/unary shape must route through
 * `foldBinary`/`foldUnary` rather than calling `binaryNode`/`unaryNode`
 * directly, so the cascade is complete. The default no-rule-matches case
 * in each helper builds the node via the cons factory, so identity is
 * preserved by hash-consing (`binaryNode(op, l, r) === originalNode`
 * when operands are unchanged).
 *
 * Termination: every rewrite strictly reduces the operand count, unwraps
 * a wrapping node, or replaces a binary shape with a structurally smaller
 * one. There is no rule that grows the expression, so recursive fold
 * calls are bounded by the input depth.
 */
export class ConstantsFoldEval implements NumEvalKernel<NumNode> {
  variable(_: VarName, node: NumNode): NumNode {
    return node;
  }
  derivative(_: VarName, node: NumNode): NumNode {
    return node;
  }
  literal(_: number, node: NumNode): NumNode {
    return node;
  }
  value(): number {
    throw new Error("Method not implemented.");
  }

  unaryOp(
    operation: UnaryOperation,
    simplifiedOperand: NumNode,
    node: NumNode,
  ): NumNode {
    // DebugNode is a unary wrapper that must preserve the debug tag
    // rather than fold into its operand. Handle it separately and leave
    // `foldUnary` free of the wrinkle.
    if (node.kind === KIND_DEBUG) {
      const originalOperand = (node as UnaryOp).original;
      if (simplifiedOperand !== originalOperand) {
        return debugNode(simplifiedOperand, (node as DebugNode).debug);
      }
      return node;
    }
    return this.foldUnary(operation, simplifiedOperand);
  }

  binaryOp(
    operation: BinaryOperation,
    simplifiedLeft: NumNode,
    simplifiedRight: NumNode,
    _node: NumNode,
  ): NumNode {
    return this.foldBinary(operation, simplifiedLeft, simplifiedRight);
  }

  select(
    simplifiedCondition: NumNode,
    simplifiedIfNonZero: NumNode,
    simplifiedIfZero: NumNode,
    _node: SelectOp,
  ): NumNode {
    if (simplifiedCondition.kind === KIND_LIT) {
      return (simplifiedCondition as LiteralNum).value !== 0
        ? simplifiedIfNonZero
        : simplifiedIfZero;
    }
    if (areNodesEqual(simplifiedIfNonZero, simplifiedIfZero)) {
      return simplifiedIfNonZero;
    }
    return selectNode(
      simplifiedCondition,
      simplifiedIfNonZero,
      simplifiedIfZero,
    );
  }

  // --------------------------------------------------------------------
  // foldUnary — pattern-matches a single unary op on already-simplified
  // operands. Safe to call recursively.
  // --------------------------------------------------------------------
  private foldUnary(op: UnaryOperation, operand: NumNode): NumNode {
    const operandKind = operand.kind;

    // Literal fold
    if (operandKind === KIND_LIT) {
      return makeLiteral(evaluateUnaryOp(op, (operand as LiteralNum).value));
    }

    switch (op) {
      case "ABS":
        // abs(abs(x)) = abs(x)
        if (operandKind === KIND_ABS) return operand;
        break;
      case "NEG":
        // -(-x) = x
        if (operandKind === KIND_NEG) {
          return (operand as UnaryOp).original;
        }
        // -(a - b) = b - a — re-fold in case the reversed SUB simplifies.
        if (operandKind === KIND_SUB) {
          const sub = operand as BinaryOp;
          return this.foldBinary("SUB", sub.right, sub.left);
        }
        break;
      case "NOT":
        // !(!(x)) = x
        if (operandKind === KIND_NOT) {
          return (operand as UnaryOp).original;
        }
        break;
    }

    // No rule matched — build the canonical node. Hash-consing makes this
    // a no-op when the operand is unchanged from the kernel's original.
    return unaryNode(op, operand);
  }

  // --------------------------------------------------------------------
  // foldBinary — pattern-matches a single binary op on already-simplified
  // operands. Safe to call recursively; rewrites that emit new binary or
  // unary shapes should call this / foldUnary so cascading folds fire.
  // --------------------------------------------------------------------
  private foldBinary(
    op: BinaryOperation,
    left: NumNode,
    right: NumNode,
  ): NumNode {
    const leftKind = left.kind;
    const rightKind = right.kind;

    // Both-literal fold
    if (leftKind === KIND_LIT && rightKind === KIND_LIT) {
      return makeLiteral(
        evaluateBinaryOp(
          op,
          (left as LiteralNum).value,
          (right as LiteralNum).value,
        ),
      );
    }

    switch (op) {
      case "ADD":
        // x + 0 = x
        if (isZero(right)) return left;
        // 0 + x = x
        if (isZero(left)) return right;
        // Combine literal terms: l1 + (l2 + x) = (l1 + l2) + x. Canonical
        // operand ordering guarantees a literal on the left of each ADD
        // if one exists. Re-fold the outer shape in case l1+l2 == 0.
        if (
          leftKind === KIND_LIT &&
          rightKind === KIND_ADD &&
          (right as BinaryOp).left.kind === KIND_LIT
        ) {
          const rightBin = right as BinaryOp;
          const combined = makeLiteral(
            (left as LiteralNum).value + (rightBin.left as LiteralNum).value,
          );
          return this.foldBinary("ADD", combined, rightBin.right);
        }
        break;

      case "SUB":
        // x - 0 = x
        if (isZero(right)) return left;
        // 0 - x = -x  (re-fold: -(-y) = y falls out automatically)
        if (isZero(left)) return this.foldUnary("NEG", right);
        // x - x = 0
        if (areNodesEqual(left, right)) return ZERO;
        // x - (-y) = x + y  (double-negation through subtraction).
        // Re-fold so that x + y further reduces if either side is 0.
        if (right.kind === KIND_NEG) {
          return this.foldBinary("ADD", left, (right as UnaryOp).original);
        }
        break;

      case "MUL":
        // x * 0 = 0
        if (isZero(left) || isZero(right)) return ZERO;
        // x * 1 = x
        if (isOne(right)) return left;
        // 1 * x = x
        if (isOne(left)) return right;
        // Combine literal factors: l1 * (l2 * x) = (l1*l2) * x.
        // Re-fold so l1*l2 == 0 or == 1 cascades correctly.
        if (
          leftKind === KIND_LIT &&
          rightKind === KIND_MUL &&
          (right as BinaryOp).left.kind === KIND_LIT
        ) {
          const rightBin = right as BinaryOp;
          const combined = makeLiteral(
            (left as LiteralNum).value * (rightBin.left as LiteralNum).value,
          );
          return this.foldBinary("MUL", combined, rightBin.right);
        }
        break;

      case "DIV":
        // x / 1 = x
        if (isOne(right)) return left;
        // 0 / x = 0 (assume x != 0)
        if (isZero(left)) return ZERO;
        // x / x = 1 (assume x != 0)
        if (areNodesEqual(left, right)) return ONE;
        // (a * x) / x = a, (x * a) / x = a. Checks both MUL operands
        // because commutative canonicalisation may have placed the
        // shared factor on either side.
        if (leftKind === KIND_MUL) {
          const mul = left as BinaryOp;
          if (areNodesEqual(mul.left, right)) return mul.right;
          if (areNodesEqual(mul.right, right)) return mul.left;
        }
        break;

      case "MIN":
        // min(x, x) = x
        if (areNodesEqual(left, right)) return left;
        // Absorption: min(x, min(x, y)) = min(x, y). Canonical ordering
        // may place the inner MIN on either side — check both.
        if (rightKind === KIND_MIN) {
          const inner = right as BinaryOp;
          if (
            areNodesEqual(left, inner.left) ||
            areNodesEqual(left, inner.right)
          ) {
            return right;
          }
        }
        if (leftKind === KIND_MIN) {
          const inner = left as BinaryOp;
          if (
            areNodesEqual(right, inner.left) ||
            areNodesEqual(right, inner.right)
          ) {
            return left;
          }
        }
        // Combine literal bounds: min(l1, min(l2, x)) = min(min(l1,l2), x).
        // Re-fold in case the combined min reveals further structure
        // (e.g. absorbs via min(x, min(l, x)) after the rewrite).
        if (
          leftKind === KIND_LIT &&
          rightKind === KIND_MIN &&
          (right as BinaryOp).left.kind === KIND_LIT
        ) {
          const rightBin = right as BinaryOp;
          const combined = makeLiteral(
            Math.min(
              (left as LiteralNum).value,
              (rightBin.left as LiteralNum).value,
            ),
          );
          return this.foldBinary("MIN", combined, rightBin.right);
        }
        break;

      case "MAX":
        // max(x, x) = x
        if (areNodesEqual(left, right)) return left;
        // Absorption: max(x, max(x, y)) = max(x, y).
        if (rightKind === KIND_MAX) {
          const inner = right as BinaryOp;
          if (
            areNodesEqual(left, inner.left) ||
            areNodesEqual(left, inner.right)
          ) {
            return right;
          }
        }
        if (leftKind === KIND_MAX) {
          const inner = left as BinaryOp;
          if (
            areNodesEqual(right, inner.left) ||
            areNodesEqual(right, inner.right)
          ) {
            return left;
          }
        }
        // Combine literal bounds: max(l1, max(l2, x)) = max(max(l1,l2), x)
        if (
          leftKind === KIND_LIT &&
          rightKind === KIND_MAX &&
          (right as BinaryOp).left.kind === KIND_LIT
        ) {
          const rightBin = right as BinaryOp;
          const combined = makeLiteral(
            Math.max(
              (left as LiteralNum).value,
              (rightBin.left as LiteralNum).value,
            ),
          );
          return this.foldBinary("MAX", combined, rightBin.right);
        }
        break;

      case "COMPARE":
        // compare(x, x) = 0
        if (areNodesEqual(left, right)) return ZERO;
        break;

      case "AND":
        // false AND x = false
        if (isZero(left)) return ZERO;
        // true AND x = x
        if (leftKind === KIND_LIT && (left as LiteralNum).value !== 0) {
          return right;
        }
        break;

      case "OR":
        // true OR x = true
        if (leftKind === KIND_LIT && (left as LiteralNum).value !== 0) {
          return left;
        }
        // false OR x = x
        if (isZero(left)) return right;
        break;
    }

    // No rule matched — emit the canonical node. Hash-consing makes this
    // a no-op when the operands are unchanged from the kernel's original.
    return binaryNode(op, left, right);
  }
}

function isZero(node: NumNode): boolean {
  if (node === ZERO) return true;
  return node.kind === KIND_LIT && (node as LiteralNum).value === 0;
}

function isOne(node: NumNode): boolean {
  if (node === ONE) return true;
  return node.kind === KIND_LIT && (node as LiteralNum).value === 1;
}

/**
 * Structural equality on NumNodes. Safe to use as pointer equality because
 * every production code path goes through the hash-cons factories in
 * tree-cons.ts, so two structurally equal sub-DAGs are guaranteed to
 * be the same object.
 */
function areNodesEqual(node1: NumNode, node2: NumNode): boolean {
  return node1 === node2;
}
