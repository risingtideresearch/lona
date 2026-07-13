# lona-geom

`lona-geom` provides symbolic 2D and 3D geometry primitives built on
[`lona`](../lona). Coordinates, lengths, angles, and the results of geometric
operations are Lona `Num` expressions, so geometry can be evaluated,
differentiated, compiled, simplified, and serialized using the core library.

The package includes:

- 2D points, vectors, unit vectors, angles, and turn-preserving solid angles,
- 3D points, vectors, unit vectors, coordinate planes, and axis rotations,
- embedding and projection between 2D and 3D,
- generic geometry interfaces for algorithms that work with other scalar
  implementations,
- `NumStruct` support for symbolic selection and component-wise operations.

---

## Quick start

```ts
import { asNumber, setVariable, variable } from "lona";
import { Point, angleFromDeg, asVec } from "lona-geom";

const x = variable("x", 0);
const y = variable("y", 0);
const start = new Point(x, y);

const offset = asVec(1, 0).rotate(angleFromDeg(30)).scale(5);
const end = start.add(offset);

asNumber(end.x); // 4.330127018922194
asNumber(end.y); // 2.5

setVariable("x", 2);
setVariable("y", 3);

asNumber(end.x); // 6.330127018922194
asNumber(end.y); // 5.5
```

`end.x` and `end.y` are ordinary `Num` values. No separate geometry evaluator
is needed: use the evaluation and compilation APIs from `lona` directly.

---

## Core model

Geometry values are small immutable wrappers around `Num` components. An
operation such as a dot product or rotation builds a symbolic expression; it
does not immediately reduce the result to a JavaScript number.

```ts
import { asNumber, variable } from "lona";
import { asVec } from "lona-geom";

const width = variable("width", 3);
const height = variable("height", 4);
const diagonal = asVec(width, height).norm();

asNumber(diagonal); // 5
```

Most operations return geometry values of the same general kind:

- point plus vector → point,
- point minus vector → point,
- point to point → vector,
- vector arithmetic → vector,
- dot product, cross product in 2D, and norm → `Num`,
- cross product in 3D → `Vec3`.

Class constructors generally take `Num` components. The vector factory
functions `asVec(...)` and `asVec3(...)` accept both `Num` and JavaScript
numbers and are usually the most convenient way to construct vectors.

---

## 2D geometry

### Points

Construct a `Point` from two `Num` coordinates:

```ts
import { asNum, variable } from "lona";
import { ORIGIN, Point, asVec } from "lona-geom";

const a = new Point(asNum(2), asNum(3));
const b = new Point(variable("bx"), variable("by"));

const moved = a.add(asVec(10, -2));
const displacement = a.vecTo(b);
const midpoint = a.midPoint(b);

ORIGIN; // Point(0, 0)
```

The `Point` API is:

```ts
point.x; // Num
point.y; // Num
point.add(vector); // Point
point.sub(vector); // Point
point.midPoint(other); // Point
point.vecTo(other); // Vec2: other - point
point.vecFrom(other); // Vec2: point - other
point.vecFromOrigin(); // Vec2
point.debug("name"); // Point with labelled components
```

### Vectors

Use `asVec` (an alias of `vecFromCartesianCoords`) when starting from numbers
or mixed numeric/symbolic inputs:

```ts
import { variable } from "lona";
import {
  angleFromDeg,
  asVec,
  vecFromCartesianCoords,
  vecFromPolarCoords,
} from "lona-geom";

const a = asVec(3, 4);
const b = vecFromCartesianCoords(variable("x"), 2);

const length = a.norm(); // Num, sqrt(3² + 4²)
const unit = a.normalize(); // UnitVec2
const rotated = a.rotate(angleFromDeg(90));
const polar = vecFromPolarCoords(5, angleFromDeg(30));
```

`Vec2` supports:

```ts
vector.x; // Num
vector.y; // Num
vector.add(other); // Vec2
vector.sub(other); // Vec2
vector.neg(); // Vec2
vector.scale(k); // Vec2, k may be Num | number
vector.div(k); // Vec2
vector.dot(other); // Num
vector.cross(other); // Num, signed 2D cross product
vector.norm(); // Num
vector.normalize(); // UnitVec2
vector.perp(); // Vec2, (-y, x)
vector.mirrorX(); // Vec2, (-x, y)
vector.mirrorY(); // Vec2, (x, -y)
vector.rotate(angle); // Vec2
vector.asAngle(); // Angle
vector.pointFromOrigin(); // Point
vector.debug("name"); // Vec2 with labelled components
```

A `UnitVec2` is a `Vec2` whose length is assumed to be one. Its `norm()` is the
constant `ONE`, `normalize()` returns the same value, and operations that
preserve length return another `UnitVec2`.

### Angles

An `Angle` is stored as symbolic cosine and sine components rather than as one
radian value. This avoids repeatedly expanding trigonometric addition formulas
and makes angle composition work naturally with symbolic inputs.

Prefer a factory function over calling the constructor directly:

```ts
import { variable } from "lona";
import {
  angleFromDeg,
  angleFromDirection,
  angleFromCos,
  angleFromSin,
  angleRromRad,
  arcTan,
  asVec,
  twoVectorsAngle,
} from "lona-geom";

const degrees = angleFromDeg(45);
const radians = angleRromRad(variable("theta"));
const direction = angleFromDirection(asVec(3, 4));
const between = twoVectorsAngle(asVec(1, 0), asVec(0, 1));
const fromComponents = arcTan(3, 4); // normalized direction (3, 4)

angleFromSin(0.5); // chooses the non-negative cosine branch
angleFromCos(0.5); // chooses the non-negative sine branch
```

> The currently exported radians factory is named `angleRromRad` (with two
> `r`s). The documentation uses the actual public API spelling.

Angle operations and conversions:

```ts
angle.add(other); // Angle
angle.sub(other); // Angle
angle.neg(); // Angle
angle.half(); // Angle; chooses one of the two valid half-angle branches
angle.double(); // Angle
angle.perp(); // Angle, +90°
angle.opposite(); // Angle, +180°
angle.cos(); // Num
angle.sin(); // Num
angle.tan(); // Num
angle.asRad(); // Num, principal atan2(sin, cos) result
angle.asDeg(); // Num, principal angle in degrees
angle.asSortValue(); // Num suitable for ordering directions around a turn
angle.asUnitArcLength(); // Num in radians normalized around a full turn
angle.asVec(); // UnitVec2
angle.debug("name"); // Angle with labelled cosine/sine components
```

Because `Angle` stores only sine and cosine, it represents a direction modulo
one full turn. It cannot distinguish 0°, 360°, and 720°. Use `SolidAngle` when
full-turn count matters.

Available constants:

```ts
NO_TURN;
FULL_TURN; // same direction as NO_TURN
HALF_TURN;
QUARTER_TURN;
THREE_QUARTER_TURN;
EIGHTH_TURN;
```

### Solid angles and turn count

`SolidAngle` stores a symbolic number of turns without wrapping it to one
revolution:

```ts
import { asNumber } from "lona";
import { SolidAngle, angleFromDeg, solidAngleFromAngle } from "lona-geom";

const rotation = new SolidAngle(2).addAngle(angleFromDeg(90));
asNumber(rotation.turns); // 2.25

solidAngleFromAngle(angleFromDeg(180)); // 0.5 turns
```

It supports `addAngle`, `addTurns`, `add`, `sub`, `neg`, and `half`.

---

## 3D geometry

### Points and vectors

```ts
import { asNum, variable } from "lona";
import { ORIGIN_3D, Point3D, X_AXIS, Y_AXIS, Z_AXIS, asVec3 } from "lona-geom";

const point = new Point3D(asNum(1), variable("y"), asNum(3));
const vector = asVec3(2, 0, -1);
const moved = point.add(vector);

const normal = asVec3(1, 2, 3)
  .cross(asVec3(0, 1, 0))
  .normalize();

ORIGIN_3D;
X_AXIS;
Y_AXIS;
Z_AXIS;
```

`Point3D` is the 3D counterpart of `Point`:

```ts
point.x;
point.y;
point.z; // Num components
point.add(vector); // Point3D
point.sub(vector); // Point3D
point.midPoint(other); // Point3D
point.vecTo(other); // Vec3
point.vecFrom(other); // Vec3
point.vecFromOrigin(); // Vec3
point.debug("name"); // Point3D
```

`Vec3` provides the corresponding vector algebra:

```ts
vector.add(other);
vector.sub(other);
vector.neg();
vector.scale(k);
vector.div(k); // Vec3
vector.dot(other); // Num
vector.cross(other); // Vec3
vector.norm(); // Num
vector.normalize(); // UnitVec3
vector.azimuthAngle(); // Angle in the xy plane
vector.elevationAngle(); // Angle measured from the +z direction
vector.rotate(angle, axis); // Vec3, Rodrigues' rotation formula
vector.mirrorX();
vector.mirrorY();
vector.mirrorZ(); // Vec3
vector.pointFromOrigin(); // Point3D
vector.debug("name"); // Vec3
```

`perpX()`, `perpY()`, and `perpZ()` are also available for the package's
axis-oriented component transforms.

A `UnitVec3` behaves like `UnitVec2`: it promises unit length and preserves its
specialized type for operations such as normalization, negation, mirroring,
and rotation.

### Rotations

3D rotation uses an `Angle` and a `UnitVec3` axis:

```ts
import { asNumber } from "lona";
import { X_AXIS, Z_AXIS, angleFromDeg } from "lona-geom";

const rotated = X_AXIS.rotate(angleFromDeg(90), Z_AXIS);

asNumber(rotated.x); // approximately 0
asNumber(rotated.y); // approximately 1
asNumber(rotated.z); // 0
```

The package does not normalize the supplied rotation axis automatically; pass
a valid `UnitVec3`.

### Planes, embedding, and projection

A `Plane` is a local 2D coordinate frame in 3D. It stores an origin, z axis,
and x axis; its y axis is derived as `zAxis.cross(xAxis)`.

```ts
import { asNum } from "lona";
import { Point, XY_PLANE, asVec3, embedPoint, projectPoint } from "lona-geom";

const elevatedPlane = XY_PLANE.translate(asVec3(0, 0, 5));
const local = new Point(asNum(2), asNum(3));
const world = embedPoint(local, elevatedPlane); // (2, 3, 5)
const roundTrip = projectPoint(world, elevatedPlane); // (2, 3)
```

Plane operations:

```ts
plane.origin; // Point3D
plane.xAxis; // UnitVec3
plane.yAxis; // UnitVec3, derived lazily
plane.zAxis; // UnitVec3
plane.translateTo(point); // Plane
plane.translate(vector); // Plane
plane.rotateAroundZ(angle); // Plane
plane.pivot(angle, axis); // Plane
```

Embedding and projection helpers:

```ts
embedPoint(point2D, plane?); // Point3D
embedVec(vector2D, plane?); // Vec3
projectPoint(point3D, plane); // Point
projectVec(vector3D, plane); // Vec2
```

Without a plane, `embedPoint` and `embedVec` place the input directly in the xy
plane with z = 0.

Predefined planes are `XY_PLANE`, `XZ_PLANE`, and `YZ_PLANE`.

---

## Evaluation, differentiation, and compilation

Every exposed coordinate or scalar result is a `Num`, so all core Lona
workflows apply unchanged:

```ts
import { asNumber, compileValueRoutine, variable } from "lona";
import { Point, asVec } from "lona-geom";

const x = variable("x");
const y = variable("y");
const end = new Point(x, y).add(asVec(10, 20));

asNumber(end.x);
end.x.eval({ x: 2, y: 3 }); // 12
end.x.simplify();

const routine = compileValueRoutine(
  end.toNums().map((component) => component.n),
);
if (!routine || routine.shape !== "multi-value") {
  throw new Error("expected a two-component routine");
}

routine.eval({ x: 2, y: 3 }); // [12, 23]
```

Differentiate geometry by differentiating the component expressions. For
example, a point-valued function has a two-row Jacobian, one row for `point.x`
and one for `point.y`. See the core [`lona` documentation](../lona/README.md)
for evaluation contexts, routines, backends, symbolic differentiation, and
serialization.

---

## Symbolic branching with `NumStruct`

`Angle`, `Vec2`, `UnitVec2`, `Point`, `Vec3`, `UnitVec3`, and `Point3D`
implement Lona's `NumStruct<T>` interface. This means generic structure helpers
can select or transform all components together.

### Selecting whole geometry values

```ts
import { selectStruct, variable } from "lona";
import { Point } from "lona-geom";

const condition = variable("condition");
const a = new Point(variable("ax"), variable("ay"));
const b = new Point(variable("bx"), variable("by"));

const selected = selectStruct(condition, a, b);
```

The same symbolic condition is used for every component, so `selected` is
exactly one whole input point. This also preserves constrained types such as
`Angle` and unit vectors.

Longer branches use `whenStruct` and `casesStruct`:

```ts
import { casesStruct, whenStruct } from "lona";

const chosen = whenStruct(c1).then(a).elseIf(c2).then(b).else(fallback);

const switched = casesStruct(selector).case(0, a).case(1, b).default(fallback);
```

### Mapping components

```ts
import { mapNums, zipNums } from "lona";

const doubled = mapNums(point, (component) => component.mul(2));
const midpoint = zipNums(a, b, (x, y) => x.add(y).div(2));
```

`mapNums` and `zipNums` perform arbitrary component-wise operations. They are
appropriate for free-coordinate values such as points and ordinary vectors,
but can violate invariants on `Angle`, `UnitVec2`, and `UnitVec3`. Use those
types' own algebra when an operation must preserve their constraints.

---

## Generic geometry interfaces

The package exports structural interfaces with a generic scalar parameter:

- `PointLike<S>` and `Point3Like<S>`,
- `Vec2Like<S>` and `Vec3Like<S>`,
- `UnitVec2Like<S>` and `UnitVec3Like<S>`,
- `AngleLike<S>`,
- `PlaneLike<S>`,
- `GeomScalar` (`Num | number`).

They are useful for algorithms that only need geometry behavior and should not
be tied to a concrete class:

```ts
import type { PointLike } from "lona-geom";

function distance<S>(a: PointLike<S>, b: PointLike<S>): S {
  return a.vecTo(b).norm();
}
```

The concrete Lona geometry classes use `Num` as their scalar type.

---

## Invariants and edge cases

- `Angle(cos, sin)`, `UnitVec2`, and `UnitVec3` constructors trust the caller;
  they do not verify or normalize their inputs. Prefer angle factories and
  `vector.normalize()` when constructing constrained values.
- Normalizing a zero vector is undefined. Check the norm or select a fallback
  before calling `normalize()` when zero is possible.
- `Angle` is periodic and loses full-turn count. Use `SolidAngle` when turns
  must be preserved.
- `angleFromSin` chooses the non-negative cosine branch;
  `angleFromCos` chooses the non-negative sine branch.
- A `Plane` assumes its x and z axes form a suitable orthonormal frame. The
  constructor does not enforce perpendicular or unit axes.
- Geometry values do not have a separate serialization format. Serialize their
  component `Num` expressions using Lona when needed.

---

## Project layout

```text
src/
├── geom.ts        2D points, vectors, angles, and solid angles
├── geom-3d.ts     3D points, vectors, planes, embedding, and projection
├── geom-types.ts  generic structural geometry interfaces
└── main.ts        public package exports
```

The public API is re-exported from `src/main.ts`. Anything not exported there
is internal and may change without notice.
