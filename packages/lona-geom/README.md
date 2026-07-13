# lona-geom

Symbolic 2D and 3D geometry primitives built on [`lona`](../lona), including
points, vectors, angles, lines, planes, frames, and transformations whose
coordinates can be symbolic `Num` values.

```ts
import { asNum, variable } from "lona";
import { Point, Vec2 } from "lona-geom";

const point = new Point(variable("x"), variable("y"));
const offset = new Vec2(asNum(10), asNum(20));
const moved = point.add(offset);
```
