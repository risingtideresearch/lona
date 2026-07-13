import type { DebugNode, NumNode, SelectOp, VarName } from "../../core/tree";
import type { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";

export class DotEvalKernel implements NumEvalKernel<number> {
  private body: string[] = [];
  private currentId = 1;

  private formatter: Intl.NumberFormat;
  private sciFormatter: Intl.NumberFormat;

  constructor(
    public readonly warnPrecision = 6,
    public readonly showPrecision = 4,
  ) {
    this.formatter = new Intl.NumberFormat("en-US", {
      maximumFractionDigits: showPrecision,
    });
    this.sciFormatter = new Intl.NumberFormat("en-US", {
      maximumFractionDigits: showPrecision,
      notation: "scientific",
    });
  }

  private newId() {
    return this.currentId++;
  }

  private formatDotOptions(options: Record<string, string>) {
    if (Object.keys(options).length === 0) return "";
    const str = Object.entries(options)
      .map(([key, value]) => {
        return `${key}="${value}"`;
      })
      .join(" ");

    return `[${str}]`;
  }

  private addNode(
    id: number,
    label: string,
    status: "default" | "info" | "warn" | "error" = "default",
    additionalOptions: Record<string, string> = {},
  ) {
    const options: Record<string, string> = { label, ...additionalOptions };
    if (status !== "default")
      options.fillcolor = { info: "grey", warn: "yellow", error: "red" }[
        status
      ];

    this.body.push(`    node${id} ${this.formatDotOptions(options)};`);
  }

  private addEdge(
    from: number,
    to: number,
    status: "default" | "info" | "warn" | "error" = "default",
    additionalOptions: Record<string, string> = {},
  ) {
    const options: Record<string, string> = { ...additionalOptions };
    if (status !== "default")
      options.color = { info: "grey", warn: "orange", error: "red" }[status];

    this.body.push(
      `    node${from} -> node${to} ${this.formatDotOptions(options)};`,
    );
  }

  private formatNumber(value: number) {
    if (Number.isNaN(value)) return "NaN";
    if (
      Math.abs(value) > 10 ** this.showPrecision ||
      (Math.abs(value) < 0.1 ** this.showPrecision && value !== 0)
    )
      return this.sciFormatter.format(value);
    return this.formatter.format(value);
  }

  private formatLabel(operation: string, value: number) {
    let label = operation.toLowerCase();

    if (value || value === 0 || Number.isNaN(value)) {
      label += ` (${this.formatNumber(value)})`;
    }

    return label;
  }

  private getStatus(value: number) {
    return Number.isNaN(value)
      ? "error"
      : value !== 0 && Math.abs(value) < 10 ** -this.warnPrecision
        ? "warn"
        : Math.abs(value) !== 0 &&
            (Math.abs(value) < 1e-4 || Math.abs(value) > 1e4)
          ? "info"
          : "default";
  }

  value(value: number) {
    return value;
  }

  literal(value: number) {
    const id = this.newId();
    const label = this.formatNumber(value);
    this.addNode(id, label);
    return id;
  }

  variable(name: VarName, node: NumNode & { evalsTo: number }) {
    const id = this.newId();
    const label = this.formatLabel(String(name), node.evalsTo);
    this.addNode(id, label, this.getStatus(node.evalsTo));
    return id;
  }

  derivative(name: VarName, node: NumNode & { evalsTo: number }) {
    const id = this.newId();
    const label = this.formatLabel(`d(${String(name)})`, node.evalsTo);
    this.addNode(id, label, this.getStatus(node.evalsTo));
    return id;
  }

  unaryOp(
    operation: UnaryOperation,
    operand: number,
    node: NumNode & { evalsTo: number },
  ) {
    const id = this.newId();

    const status = this.getStatus(node.evalsTo);

    if (operation === "DEBUG") {
      const str = (node as DebugNode & { evalsTo: number }).debug;
      const label = this.formatLabel(str, node.evalsTo);
      this.addNode(id, label, status, {
        shape: "box",
        fillcolor: "lightgreen",
      });
    } else {
      const label = this.formatLabel(operation, node.evalsTo);
      this.addNode(id, label, status);
    }

    this.addEdge(id, operand, status);
    return id;
  }

  binaryOp(
    operation: BinaryOperation,
    left: number,
    right: number,
    node: NumNode & { evalsTo: number },
  ) {
    const id = this.newId();
    const status = this.getStatus(node.evalsTo);
    const label = this.formatLabel(operation, node.evalsTo);
    this.addNode(id, label, status);
    this.addEdge(id, left, status);
    this.addEdge(id, right, status);
    return id;
  }

  select(
    condition: number,
    ifNonZero: number,
    ifZero: number,
    node: SelectOp & { evalsTo: number },
  ) {
    const id = this.newId();
    const status = this.getStatus(node.evalsTo);
    const label = this.formatLabel("SELECT", node.evalsTo);
    this.addNode(id, label, status);
    this.addEdge(id, condition, status, { label: "cond" });
    this.addEdge(id, ifNonZero, status, { label: "then" });
    this.addEdge(id, ifZero, status, { label: "else" });
    return id;
  }

  getDot() {
    const lines = ["digraph ExpressionTree {"];
    lines.push("    node [shape=circle, style=filled, fillcolor=lightblue];");
    lines.push(...this.body);
    lines.push("}");
    return lines.join("\n");
  }
}
