import { describe, expect, test } from "vitest";
import { asNum, Num } from "../core/num";
import { wrapNumFn, wrapNumMethods, wrapNumMethodClass } from "./fn";
import { NumValueContext, withContext } from "./value-context";

describe("wrapNumFn", () => {
  test("wrapNumFn lifts numbers to Num", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.mul(y).add(1));
      expect(f(2, 3).eval()).toBe(7);
      expect(f(asNum(2), 3).eval()).toBe(7);
    });
  });

  test("wrapNumFn exposes eval and tree", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.mul(y).add(1));
      expect(f.eval(2, 3)).toBe(7);
      expect(f.tree().operation).toBe("ADD");
    });
  });

  test("wrapNumFn partial replaces variables", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y, z) => x.mul(y).add(z));
      const g = f.partial(2, undefined, 4);
      expect(g.eval(3)).toBe(10);
      expect(g(3).eval()).toBe(10);
    });
  });

  test("wrapNumFn exposes vars", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y, z) => x.mul(y).add(z));
      expect(f.vars()).toEqual(["arg0", "arg1", "arg2"]);
      const g = f.partial(2, undefined, 4);
      expect(g.vars()).toEqual(["arg1"]);
    });
  });

  test("wrapNumFn partialNamed replaces by name", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y, z) => x.mul(y).add(z));
      const g = f.partialNamed({ arg0: 2, arg2: 4 });
      expect(g.vars()).toEqual(["arg1"]);
      expect(g.eval(3)).toBe(10);
    });
  });

  test("wrapNumFn partialNamed accepts Num values", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y, z) => x.mul(y).add(z));
      const g = f.partialNamed({ arg0: asNum(2), arg2: 4 });
      expect(g.eval(3)).toBe(10);
    });
  });

  test("wrapNumFn partial with no values keeps vars", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.add(y));
      const g = f.partial(undefined, undefined);
      expect(g.vars()).toEqual(["arg0", "arg1"]);
    });
  });

  test("wrapNumFn eval ignores extra args", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.mul(y));
      expect(f.eval(2, 3, 4)).toBe(6);
    });
  });

  test("wrapNumFn derivative fallback uses auto-diff", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.mul(y).add(1));
      const dx = f.derivative("arg0");
      const dy = f.derivative("arg1");
      expect(dx.eval(2, 3)).toBe(3);
      expect(dy.eval(2, 3)).toBe(2);
    });
  });

  test("wrapNumFn derivative accepts index", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.mul(y).add(1));
      const dx = f.derivative(0);
      expect(dx.eval(2, 3)).toBe(3);
    });
  });

  test("wrapNumFn custom derivative overrides auto-diff", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.mul(y).add(1), {
        derivatives: {
          arg0: (_x, y) => y.add(1),
        },
      });
      const dx = f.derivative("arg0");
      expect(dx.eval(2, 3)).toBe(4);
    });
  });

  test("wrapNumFn derivatives and gradient agree", () => {
    withContext(new NumValueContext(), () => {
      const f = wrapNumFn((x, y) => x.mul(y).add(1));
      const ds = f.derivatives();
      const grad = f.gradient();
      expect(ds.arg0.eval(2, 3)).toBe(3);
      expect(ds.arg1.eval(2, 3)).toBe(2);
      expect(grad[0]!.eval(2, 3)).toBe(3);
      expect(grad[1]!.eval(2, 3)).toBe(2);
    });
  });
});

describe("wrapNumMethods", () => {
  test("wrapNumMethods wraps class methods with this binding", () => {
    class Foo {
      constructor(private readonly k: number) {}
      f(x: Num, y: Num) {
        return x.mul(y).add(this.k);
      }
    }

    withContext(new NumValueContext(), () => {
      const foo = new Foo(4);
      const { f } = wrapNumMethods(Foo, "f")(foo);
      expect(f.eval(2, 3)).toBe(10);
    });
  });

  test("wrapNumMethodClass returns class with wrapped methods", () => {
    class Foo {
      constructor(private readonly k: number) {}
      f(x: Num, y: Num) {
        return x.mul(y).add(this.k);
      }
    }

    withContext(new NumValueContext(), () => {
      const WrappedFoo = wrapNumMethodClass(Foo, "f");
      const foo = new WrappedFoo(4);
      expect(foo.f.eval(2, 3)).toBe(10);
      expect(foo.f(2, 3).eval()).toBe(10);
    });
  });

  test("wrapNumMethodClass handles multiple methods", () => {
    class Foo {
      constructor(private readonly k: number) {}
      f(x: Num, y: Num) {
        return x.mul(y).add(this.k);
      }
      g(x: Num) {
        return x.add(this.k);
      }
    }

    withContext(new NumValueContext(), () => {
      const WrappedFoo = wrapNumMethodClass(Foo, "f", "g");
      const foo = new WrappedFoo(5);
      expect(foo.f.eval(2, 3)).toBe(11);
      expect(foo.g.eval(2)).toBe(7);
    });
  });

  test("wrapped class instance inside another class", () => {
    class Inner {
      constructor(private readonly k: number) {}
      f(x: Num, y: Num) {
        return x.mul(y).add(this.k);
      }
    }

    const WrappedInner = wrapNumMethodClass(Inner, "f");

    class Outer {
      readonly inner: InstanceType<typeof WrappedInner>;
      constructor(inner: InstanceType<typeof WrappedInner>) {
        this.inner = inner;
      }
      calc(x: Num, y: Num) {
        return this.inner.f(x, y).add(1);
      }
    }

    withContext(new NumValueContext(), () => {
      const inner = new WrappedInner(6);
      const WrappedOuter = wrapNumMethodClass(Outer, "calc");
      const outer = new WrappedOuter(inner);
      expect(outer.calc.eval(2, 3)).toBe(13);
    });
  });
});
