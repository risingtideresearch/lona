import { Num } from "../core/num";
import { KIND_LIT, LiteralNum, NumNode, ONE_NODE } from "../core/tree";
import { binaryNode, litNode, unaryNode } from "../core/tree-cons";

type Token =
  | { type: "number"; value: number }
  | { type: "placeholder"; index: number }
  | { type: "op"; op: "+" | "-" | "*" | "/" | "%" | "**" | "^" }
  | { type: "lparen" }
  | { type: "rparen" }
  | { type: "ident"; name: string };

function isDigit(ch: string) {
  return ch >= "0" && ch <= "9";
}

function tokenize(parts: TemplateStringsArray, valuesLength: number): Token[] {
  const tokens: Token[] = [];
  for (let i = 0; i < parts.length; i++) {
    const s = parts[i];
    let j = 0;
    while (j < s.length) {
      const ch = s[j]!;
      if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
        j++;
        continue;
      }
      if (ch === "+" || ch === "-" || ch === "/" || ch === "%") {
        tokens.push({ type: "op", op: ch });
        j++;
        continue;
      }
      if (ch === "*") {
        if (s[j + 1] === "*") {
          tokens.push({ type: "op", op: "**" });
          j += 2;
        } else {
          tokens.push({ type: "op", op: "*" });
          j++;
        }
        continue;
      }
      if (ch === "^") {
        tokens.push({ type: "op", op: "^" });
        j++;
        continue;
      }
      if (ch === "(") {
        tokens.push({ type: "lparen" });
        j++;
        continue;
      }
      if (ch === ")") {
        tokens.push({ type: "rparen" });
        j++;
        continue;
      }
      // identifier (function name like sqrt, log, exp, ...)
      if ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z")) {
        const start = j;
        j++;
        while (j < s.length) {
          const c = s[j]!;
          if (
            (c >= "a" && c <= "z") ||
            (c >= "A" && c <= "Z") ||
            (c >= "0" && c <= "9") ||
            c === "_"
          ) {
            j++;
          } else {
            break;
          }
        }
        const name = s.slice(start, j);
        tokens.push({ type: "ident", name });
        continue;
      }
      // number literal: supports integers and decimals (e.g., 12, 3.4, .5)
      if (isDigit(ch) || (ch === "." && isDigit(s[j + 1] ?? ""))) {
        const start = j;
        let sawDot = false;
        if (ch === ".") {
          sawDot = true;
          j++;
        }
        while (j < s.length && isDigit(s[j]!)) j++;
        if (!sawDot && s[j] === ".") {
          sawDot = true;
          j++;
          while (j < s.length && isDigit(s[j]!)) j++;
        }
        const numStr = s.slice(start, j);
        const val = Number(numStr);
        if (!Number.isFinite(val)) {
          throw new Error(`Invalid number literal: ${numStr}`);
        }
        tokens.push({ type: "number", value: val });
        continue;
      }
      throw new Error(`Unexpected character '${ch}' in expression`);
    }
    // Insert placeholder marker between string parts if there is a value next
    if (i < parts.length - 1 && i < valuesLength) {
      // Placeholder with index i (values[i])
      tokens.push({ type: "placeholder", index: i });
    }
  }
  return tokens;
}

class Parser {
  private pos = 0;
  constructor(
    private tokens: Token[],
    private values: Array<Num | number>,
  ) {}

  // Supported identifier functions mapping to NumNode-level operations
  private readonly IDENT_FUNCS: Record<string, (arg: NumNode) => NumNode> = {
    sqrt: (x) => unaryNode("SQRT", x),
    cbrt: (x) => unaryNode("CBRT", x),
    abs: (x) => unaryNode("ABS", x),
    log: (x) => unaryNode("LOG", x),
    log1p: (x) => unaryNode("LOG1P", x),
    exp: (x) => unaryNode("EXP", x),
    sin: (x) => unaryNode("SIN", x),
    cos: (x) => unaryNode("COS", x),
    tan: (x) => unaryNode("TAN", x),
    asin: (x) => unaryNode("ASIN", x),
    acos: (x) => unaryNode("ACOS", x),
    atan: (x) => unaryNode("ATAN", x),
  };

  // Returns a numeric literal value if the next syntactic unit is a bare number
  // and not part of a larger exponent expression (e.g., not followed by ** or ^).
  // Otherwise returns null.
  private nextIsNumber(): number | null {
    const next = this.peek();
    if (!next || next.type !== "number") return null;
    const after = this.tokens[this.pos + 1];
    if (
      after &&
      after.type === "op" &&
      (after.op === "**" || after.op === "^")
    ) {
      return null;
    }
    return next.value;
  }

  private peek(): Token | undefined {
    return this.tokens[this.pos];
  }
  private consume(): Token {
    const t = this.tokens[this.pos];
    if (!t) throw new Error("Unexpected end of expression");
    this.pos++;
    return t;
  }

  parse(): NumNode {
    const node = this.parseExpr();
    if (this.peek()) {
      throw new Error("Unexpected token after end of expression");
    }
    return node;
  }

  // expr := term (('+' | '-') term)*
  private parseExpr(): NumNode {
    let left = this.parseTerm();
    while (true) {
      const tok = this.peek();
      if (tok && tok.type === "op" && (tok.op === "+" || tok.op === "-")) {
        this.consume();
        const right = this.parseTerm();
        left =
          tok.op === "+"
            ? binaryNode("ADD", left, right)
            : binaryNode("SUB", left, right);
      } else {
        break;
      }
    }
    return left;
  }

  // term := factor (('*' | '/' | '%') factor)*
  private parseTerm(): NumNode {
    let left = this.parseFactor();
    while (true) {
      const tok = this.peek();
      if (
        tok &&
        tok.type === "op" &&
        (tok.op === "*" || tok.op === "/" || tok.op === "%")
      ) {
        this.consume();
        const right = this.parseFactor();
        if (tok.op === "*") {
          left = binaryNode("MUL", left, right);
        } else if (tok.op === "/") {
          left = binaryNode("DIV", left, right);
        } else {
          left = binaryNode("MOD", left, right);
        }
      } else {
        break;
      }
    }
    return left;
  }

  // factor := ('-' factor) | power  (make exponent bind tighter than unary)
  private parseFactor(): NumNode {
    const tok = this.peek();
    if (tok && tok.type === "op" && tok.op === "-") {
      this.consume();
      return unaryNode("NEG", this.parseFactor());
    }
    return this.parsePower();
  }

  // power := primary (("**" | "^") power)?   (right associative)
  private parsePower(): NumNode {
    const left = this.parsePrimary();
    const tok = this.peek();
    if (tok && tok.type === "op" && (tok.op === "**" || tok.op === "^")) {
      this.consume();

      // If the next element is a bare number literal (not an expression), fast-path:
      // - use powi for integers
      // - use exp(log(base) * exponent) for non-integers
      const literal = this.nextIsNumber();
      if (literal !== null) {
        this.consume(); // consume the number token
        if (Number.isInteger(literal)) {
          const k = literal as number;
          if (k === 0) return ONE_NODE;
          if (k > 0) return powiNode(left, k);
          // negative integer power
          return binaryNode("DIV", ONE_NODE, powiNode(left, -k));
        }
        // non-integer literal exponent: exp(log(base) * exponent)
        return unaryNode(
          "EXP",
          binaryNode("MUL", unaryNode("LOG", left), litNode(literal)),
        );
      }

      // Otherwise parse the right-hand side as a power expression (right-assoc)
      const right = this.parsePower();
      return powerNode(left, right);
    }
    return left;
  }

  // primary := number | placeholder | '(' expr ')' | ident '(' expr ')'
  private parsePrimary(): NumNode {
    const tok = this.consume();
    if (tok.type === "number") {
      return litNode(tok.value);
    }
    if (tok.type === "placeholder") {
      const v = this.values[tok.index];
      if (v instanceof Num) return v.n;
      return litNode(v as number);
    }
    if (tok.type === "ident") {
      const fn = this.IDENT_FUNCS[tok.name];
      if (!fn) {
        throw new Error(`Unknown identifier '${tok.name}'`);
      }
      const lp = this.consume();
      if (lp.type !== "lparen") {
        throw new Error(`Expected '(' after ${tok.name}`);
      }
      const inside = this.parseExpr();
      const rp = this.consume();
      if (rp.type !== "rparen") {
        throw new Error(`Expected ')' after ${tok.name}`);
      }
      return fn(inside);
    }
    if (tok.type === "lparen") {
      const inside = this.parseExpr();
      const next = this.consume();
      if (!next || next.type !== "rparen") {
        throw new Error("Expected closing ')' ");
      }
      return inside;
    }
    throw new Error("Expected a number, placeholder, or '(' ");
  }
}

export function nexpr(
  parts: TemplateStringsArray,
  ...values: Array<Num | number>
): Num {
  const tokens = tokenize(parts, values.length);
  const parser = new Parser(tokens, values);
  const node = parser.parse();
  return new Num(node);
}

function powiNode(base: NumNode, k: number): NumNode {
  if (!Number.isInteger(k) || k <= 0) {
    throw new Error(`power must be a positive integer, ${k} received`);
  }
  let result: NumNode = base;
  for (let i = 1; i < k; i++) {
    result = binaryNode("MUL", result, base);
  }
  return result;
}

function powerNode(base: NumNode, exponent: NumNode): NumNode {
  if (
    exponent.kind === KIND_LIT &&
    Number.isInteger((exponent as LiteralNum).value)
  ) {
    const k = (exponent as LiteralNum).value;
    if (k === 0) return ONE_NODE;
    if (k > 0) return powiNode(base, k);
    // negative integer power
    return binaryNode("DIV", ONE_NODE, powiNode(base, -k));
  }
  // exp(log(base) * exponent)
  return unaryNode("EXP", binaryNode("MUL", unaryNode("LOG", base), exponent));
}
