import { Num, asNum } from "../core/num";
import { variable } from "./value-context";
import { NumNode, type VarName } from "../core/tree";
import { partialDerivative, replaceVariable } from "../core/tree-walks";
import { simpleEval } from "../eval/eval-value";
import { fullDerivative } from "./diff";
import { simplify } from "./simplify";

type NumLike = number | Num;
type DerivativeBuilder = (...args: Num[]) => Num;

export type WrapNumFnOptions = {
  derivatives?: Record<string, DerivativeBuilder>;
};

export type WrappedNumFn<T extends Array<NumLike>> = ((...args: T) => Num) & {
  eval: (...args: number[]) => number;
  tree: () => NumNode;
  vars: () => string[];
  partial: (...args: Array<NumLike | undefined>) => WrappedNumFn<T>;
  partialNamed: (
    values: Record<string, NumLike | undefined>,
  ) => WrappedNumFn<T>;
  derivative: (nameOrIndex: string | number) => WrappedNumFn<T>;
  derivatives: () => Record<string, WrappedNumFn<T>>;
  gradient: () => Array<WrappedNumFn<T>>;
};

class WrappedNumFnClass<T extends Array<NumLike>> {
  private simplifiedRoot: NumNode | null = null;
  private fullDerivativeRoot: NumNode | null = null;

  constructor(
    private readonly root: NumNode,
    private readonly names: string[],
    private readonly nameSymbols: Map<string, VarName>,
    private readonly options: WrapNumFnOptions,
    private readonly placeholders: Num[],
    private readonly replacements: Map<VarName, NumNode> = new Map(),
    private readonly baseDerivativeRoots: Map<VarName, NumNode> = new Map(),
  ) {}

  private getRoot() {
    if (!this.simplifiedRoot) this.simplifiedRoot = simplify(this.root);
    return this.simplifiedRoot;
  }

  private getFullDerivativeRoot() {
    if (!this.fullDerivativeRoot) {
      this.fullDerivativeRoot = fullDerivative(this.getRoot());
    }
    return this.fullDerivativeRoot;
  }

  private getName(nameOrIndex: string | number) {
    if (typeof nameOrIndex === "number") {
      const name = this.names[nameOrIndex];
      if (!name) {
        throw new Error(`Unknown derivative index ${nameOrIndex}`);
      }
      return name;
    }
    return nameOrIndex;
  }

  private getBaseDerivativeRoot(name: VarName) {
    const existing = this.baseDerivativeRoots.get(name);
    if (existing) return existing;
    let builder: DerivativeBuilder | undefined;
    const derivatives = this.options.derivatives;
    if (typeof name === "symbol") {
      builder = derivatives?.[name.description ?? ""];
    } else {
      builder = derivatives?.[name];
    }
    if (!builder) return null;
    const built = builder(...this.placeholders).n;
    this.baseDerivativeRoots.set(name, built);
    return built;
  }

  call(...args: T) {
    const map = new Map<VarName, NumNode>();
    for (let i = 0; i < this.names.length; i++) {
      const value = args[i];
      if (value === undefined) continue;
      map.set(this.nameSymbols.get(this.names[i]!)!, asNum(value).n);
    }
    const replaced = replaceVariable(this.getRoot(), map);
    return new Num(replaced);
  }

  eval(...args: number[]) {
    const values = new Map<VarName, number>();
    for (let i = 0; i < this.names.length; i++) {
      const value = args[i];
      if (value !== undefined) {
        values.set(this.nameSymbols.get(this.names[i]!)!, value);
      }
    }
    return simpleEval(this.getRoot(), values);
  }

  tree() {
    return this.getRoot();
  }

  vars() {
    return this.names.slice();
  }

  partial(...args: Array<NumLike | undefined>) {
    const map = new Map<VarName, NumNode>();
    const remaining: string[] = [];
    for (let i = 0; i < this.names.length; i++) {
      const value = args[i];
      const name = this.names[i]!;
      if (value === undefined) {
        remaining.push(name);
        continue;
      }
      map.set(this.nameSymbols.get(name)!, asNum(value).n);
    }
    const replaced = replaceVariable(this.getRoot(), map);
    const nextReplacements = new Map(this.replacements);
    for (const [key, value] of map) nextReplacements.set(key, value);
    return this.spawn(replaced, remaining, nextReplacements);
  }

  partialNamed(values: Record<string, NumLike | undefined>) {
    const map = new Map<VarName, NumNode>();
    const remaining: string[] = [];
    for (const name of this.names) {
      const value = values[name];
      if (value === undefined) {
        remaining.push(name);
        continue;
      }
      map.set(this.nameSymbols.get(name)!, asNum(value).n);
    }
    const replaced = replaceVariable(this.getRoot(), map);
    const nextReplacements = new Map(this.replacements);
    for (const [key, value] of map) nextReplacements.set(key, value);
    return this.spawn(replaced, remaining, nextReplacements);
  }

  derivative(nameOrIndex: string | number) {
    const name = this.getName(nameOrIndex);
    if (!this.names.includes(name)) {
      throw new Error(`Unknown derivative '${name}'`);
    }
    const sym = this.nameSymbols.get(name)!;
    const baseRoot = this.getBaseDerivativeRoot(sym);
    const derivedRoot =
      baseRoot === null
        ? partialDerivative(this.getFullDerivativeRoot(), sym)
        : replaceVariable(baseRoot, this.replacements);
    return this.spawn(derivedRoot, this.names, this.replacements);
  }

  derivatives() {
    const out: Record<string, WrappedNumFn<T>> = {};
    for (const name of this.names) {
      out[name] = this.derivative(name);
    }
    return out;
  }

  gradient() {
    return this.names.map((name) => this.derivative(name));
  }

  private spawn(
    root: NumNode,
    names: string[],
    replacements: Map<VarName, NumNode>,
  ): WrappedNumFn<T> {
    return makeWrappedNumFn<T>(
      new WrappedNumFnClass<T>(
        root,
        names,
        this.nameSymbols,
        this.options,
        this.placeholders,
        replacements,
        this.baseDerivativeRoots,
      ),
    );
  }
}

function makeWrappedNumFn<T extends Array<NumLike>>(
  instance: WrappedNumFnClass<T>,
): WrappedNumFn<T> {
  const wrapped = (...args: T) => instance.call(...args);
  wrapped.eval = (...args: number[]) => instance.eval(...args);
  wrapped.tree = () => instance.tree();
  wrapped.vars = () => instance.vars();
  wrapped.partial = (...args: Array<NumLike | undefined>) =>
    instance.partial(...args);
  wrapped.partialNamed = (values: Record<string, NumLike | undefined>) =>
    instance.partialNamed(values);
  wrapped.derivative = (nameOrIndex: string | number) =>
    instance.derivative(nameOrIndex);
  wrapped.derivatives = () => instance.derivatives();
  wrapped.gradient = () => instance.gradient();

  return wrapped;
}

export function wrapNumFn<T extends Array<NumLike>>(
  fn: (...args: Num[]) => Num,
  options: WrapNumFnOptions = {},
): WrappedNumFn<T> {
  const names = Array.from({ length: fn.length }, (_, i) => `arg${i}`);
  const nameSymbols = new Map(names.map((name) => [name, Symbol(name)]));
  const placeholders = names.map((name) => variable(nameSymbols.get(name)!, 0));
  const root = fn(...placeholders).n;
  return makeWrappedNumFn<T>(
    new WrappedNumFnClass<T>(root, names, nameSymbols, options, placeholders),
  );
}

export function wrapNumMethods<
  T extends object,
  K extends keyof T & string,
  Args extends unknown[] = unknown[],
>(
  cls: new (...args: Args) => T,
  ...methodNames: K[]
): (instance: T) => Record<K, WrappedNumFn<Array<NumLike>>> {
  void cls;
  return (instance: T) => {
    const out = {} as Record<K, WrappedNumFn<Array<NumLike>>>;
    for (const name of methodNames) {
      const method = instance[name];
      if (typeof method !== "function") {
        throw new Error(`Expected method '${name}' to be a function`);
      }
      out[name] = wrapNumFn((method as (...args: Num[]) => Num).bind(instance));
    }
    return out;
  };
}

export function wrapNumMethodClass<
  Args extends unknown[] = unknown[],
  K extends string = string,
>(
  BaseClass: new (...args: Args) => object,
  ...methodNames: K[]
): new (...args: Args) => object & Record<K, WrappedNumFn<Array<NumLike>>> {
  const Wrapped = class extends (BaseClass as new (...args: Args) => object) {
    constructor(...args: Args) {
      super(...args);
      for (const name of methodNames) {
        const method = (this as Record<string, unknown>)[name];
        if (typeof method !== "function") {
          throw new Error(`Expected method '${name}' to be a function`);
        }
        (this as Record<string, unknown>)[name] = wrapNumFn(
          (method as (...args: Num[]) => Num).bind(this),
        );
      }
    }
  };

  return Wrapped as new (
    ...args: Args
  ) => object & Record<K, WrappedNumFn<Array<NumLike>>>;
}
