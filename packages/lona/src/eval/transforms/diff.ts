import { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";

import {
  BinaryOp,
  type ForeignFn,
  NEG_ONE_NODE,
  ONE_NODE,
  TWO_NODE,
  SelectOp,
  UnaryOp,
  Variable,
  ZERO_NODE,
  NumNode,
} from "../../core/tree";
import {
  binaryNode,
  derivativeNode,
  litNode,
  selectNode,
  unaryNode,
} from "../../core/tree-cons";
import type { VarName } from "../../core/tree";

function nodeMax(a: NumNode, b: NumNode): NumNode {
  return binaryNode("MAX", a, b);
}

function nodeCompare(a: NumNode, b: NumNode): NumNode {
  return binaryNode("COMPARE", a, b);
}

function nodeLessThan(a: NumNode, b: NumNode): NumNode {
  return nodeMax(ZERO_NODE, nodeCompare(b, a));
}

function nodeIfTruthyElse(
  condition: NumNode,
  ifNonZero: NumNode,
  ifZero: NumNode,
): NumNode {
  return selectNode(condition, ifNonZero, ifZero);
}

export class DiffEvalKernel implements NumEvalKernel<NumNode> {
  variable(_: VarName, node: NumNode) {
    return derivativeNode(node as Variable);
  }
  derivative(_: VarName, node: NumNode) {
    return node;
  }
  literal() {
    return ZERO_NODE;
  }
  value(): number {
    throw new Error("Method not implemented.");
  }

  foreignFn(inputDerivatives: NumNode[], node: ForeignFn): NumNode {
    // Multivariate chain rule: Σᵢ (∂f/∂input_i) * (dInput_i/dx)
    let result: NumNode = ZERO_NODE;
    for (let i = 0; i < inputDerivatives.length; i++) {
      const partialWrtInput = node.diffFn(i);
      const dInput = inputDerivatives[i];
      const term = binaryNode("MUL", partialWrtInput, dInput);
      result = result === ZERO_NODE ? term : binaryNode("ADD", result, term);
    }
    return result;
  }

  select(
    _conditionDerivative: NumNode,
    ifNonZeroDerivative: NumNode,
    ifZeroDerivative: NumNode,
    node: SelectOp,
  ): NumNode {
    return selectNode(node.condition, ifNonZeroDerivative, ifZeroDerivative);
  }

  unaryOp(
    operation: UnaryOperation,
    innerDerivative: NumNode,
    node: NumNode,
  ): NumNode {
    if (operation === "NOT" || operation === "SIGN") {
      return ZERO_NODE;
    }

    const dx = (derivative: NumNode) => {
      return binaryNode("MUL", derivative, innerDerivative);
    };

    const operand = (node as UnaryOp).original;

    if (operation === "SQRT") {
      return dx(
        binaryNode(
          "DIV",
          ONE_NODE,
          binaryNode("MUL", TWO_NODE, unaryNode("SQRT", operand)),
        ),
      );
    }

    if (operation === "COS") {
      return dx(unaryNode("NEG", unaryNode("SIN", operand)));
    }

    if (operation === "SIN") {
      return dx(unaryNode("COS", operand));
    }

    if (operation === "TAN") {
      return dx(
        binaryNode(
          "DIV",
          ONE_NODE,
          binaryNode(
            "MUL",
            unaryNode("COS", operand),
            unaryNode("COS", operand),
          ),
        ),
      );
    }

    if (operation === "ACOS") {
      return dx(
        unaryNode(
          "NEG",
          binaryNode(
            "DIV",
            ONE_NODE,
            unaryNode(
              "SQRT",
              binaryNode("SUB", ONE_NODE, binaryNode("MUL", operand, operand)),
            ),
          ),
        ),
      );
    }

    if (operation === "ASIN") {
      return dx(
        binaryNode(
          "DIV",
          ONE_NODE,
          unaryNode(
            "SQRT",
            binaryNode("SUB", ONE_NODE, binaryNode("MUL", operand, operand)),
          ),
        ),
      );
    }

    if (operation === "ATAN") {
      return dx(
        binaryNode(
          "DIV",
          ONE_NODE,
          binaryNode("ADD", ONE_NODE, binaryNode("MUL", operand, operand)),
        ),
      );
    }

    if (operation === "EXP") {
      return dx(unaryNode("EXP", operand));
    }

    if (operation === "LOG") {
      return dx(binaryNode("DIV", ONE_NODE, operand));
    }

    if (operation === "ABS") {
      return dx(unaryNode("SIGN", operand));
    }

    if (operation === "NEG") {
      return dx(NEG_ONE_NODE);
    }

    if (operation === "LOG1P") {
      return dx(
        binaryNode("DIV", ONE_NODE, binaryNode("ADD", ONE_NODE, operand)),
      );
    }

    if (operation === "TANH") {
      const twoCosh = binaryNode(
        "ADD",
        unaryNode("EXP", operand),
        unaryNode("EXP", unaryNode("NEG", operand)),
      );

      return dx(
        binaryNode("DIV", litNode(4), binaryNode("MUL", twoCosh, twoCosh)),
      );
    }

    if (operation === "DEBUG") {
      return innerDerivative;
    }

    if (operation === "CBRT") {
      return dx(
        binaryNode(
          "DIV",
          ONE_NODE,
          binaryNode(
            "MUL",
            litNode(3),
            unaryNode("CBRT", binaryNode("MUL", operand, operand)),
          ),
        ),
      );
    }

    throw new Error(`Unknown unary operation for derivation: ${operation}`);
  }

  binaryOp(
    operation: BinaryOperation,
    lhs: NumNode,
    rhs: NumNode,
    node: NumNode,
  ) {
    const leftDerivative = lhs;
    const rightDerivative = rhs;

    const left = (node as BinaryOp).left;
    const right = (node as BinaryOp).right;

    if (operation === "COMPARE") {
      return ZERO_NODE;
    } else if (operation === "ADD" || operation === "SUB") {
      return binaryNode(operation, leftDerivative, rightDerivative);
    } else if (operation === "MUL") {
      return binaryNode(
        "ADD",
        binaryNode("MUL", leftDerivative, right),
        binaryNode("MUL", left, rightDerivative),
      );
    } else if (operation === "DIV") {
      return binaryNode(
        "DIV",
        binaryNode(
          "SUB",
          binaryNode("MUL", leftDerivative, right),
          binaryNode("MUL", left, rightDerivative),
        ),
        binaryNode("MUL", right, right),
      );
    } else if (operation === "ATAN2") {
      const leftSquared = binaryNode("MUL", left, left);
      const rightSquared = binaryNode("MUL", right, right);
      const sumSquared = binaryNode("ADD", leftSquared, rightSquared);
      return binaryNode(
        "DIV",
        binaryNode(
          "SUB",
          binaryNode("MUL", right, leftDerivative),
          binaryNode("MUL", left, rightDerivative),
        ),
        sumSquared,
      );
    } else if (operation === "MOD") {
      return ONE_NODE;
    } else if (operation === "MAX") {
      const isLeftGreater = nodeLessThan(right, left);
      return nodeIfTruthyElse(isLeftGreater, leftDerivative, rightDerivative);
    } else if (operation === "MIN") {
      const isLeftLess = nodeLessThan(left, right);
      return nodeIfTruthyElse(isLeftLess, leftDerivative, rightDerivative);
    } else if (operation === "AND") {
      // if left is 0 then return left branch else return right branch
      return nodeIfTruthyElse(left, rightDerivative, leftDerivative);
    } else if (operation === "OR") {
      // if left is 0 then return right branch else return left branch
      return nodeIfTruthyElse(left, leftDerivative, rightDerivative);
    } else {
      throw new Error(`Unknown binary operation for derivation: ${operation}`);
    }
  }
}
