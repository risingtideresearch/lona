import { Angle, Point, Vec2 } from "./geom";
import { asNum, Num, ONE, ZERO, type NumStruct } from "lona";
import { hypot } from "lona";

export class Vec3 implements NumStruct<Vec3> {
  protected _x: Num;
  protected _y: Num;
  protected _z: Num;

  constructor(x: Num, y: Num, z: Num) {
    this._x = x;
    this._y = y;
    this._z = z;
  }

  toNums(): Num[] {
    return [this._x, this._y, this._z];
  }

  fromNums([x, y, z]: Num[]): Vec3 {
    return new Vec3(x, y, z);
  }

  add(other: Vec3): Vec3 {
    return new Vec3(
      this._x.add(other._x),
      this._y.add(other._y),
      this._z.add(other._z),
    );
  }

  sub(other: Vec3): Vec3 {
    return new Vec3(
      this._x.sub(other._x),
      this._y.sub(other._y),
      this._z.sub(other._z),
    );
  }

  neg(): Vec3 {
    return new Vec3(this._x.neg(), this._y.neg(), this._z.neg());
  }

  scale(other: Num | number): Vec3 {
    return new Vec3(this._x.mul(other), this._y.mul(other), this._z.mul(other));
  }

  div(other: Num | number): Vec3 {
    return new Vec3(this._x.div(other), this._y.div(other), this._z.div(other));
  }

  dot(other: Vec3): Num {
    return this._x
      .mul(other._x)
      .add(this._y.mul(other._y))
      .add(this._z.mul(other._z));
  }

  cross(other: Vec3): Vec3 {
    return new Vec3(
      this._y.mul(other._z).sub(this._z.mul(other._y)),
      this._z.mul(other._x).sub(this._x.mul(other._z)),
      this._x.mul(other._y).sub(this._y.mul(other._x)),
    );
  }

  norm(): Num {
    return this.dot(this).sqrt();
  }

  normalize(): UnitVec3 {
    return this.div(this.norm());
  }

  azimuthAngle(): Angle {
    const normalized = this.normalize();
    return new Angle(normalized._x, normalized._y);
  }

  elevationAngle(): Angle {
    const normalized = this.normalize();
    return new Angle(normalized._z, hypot(normalized._x, normalized._y));
  }

  get x(): Num {
    return this._x;
  }

  get y(): Num {
    return this._y;
  }

  get z(): Num {
    return this._z;
  }

  perpZ(): Vec3 {
    return new Vec3(this._y.neg(), this._x, this._z);
  }

  perpX(): Vec3 {
    return new Vec3(this._x, this._z, this._y.neg());
  }

  perpY(): Vec3 {
    return new Vec3(this._z.neg(), this._y, this._x);
  }

  rotate(angle: Angle, axis: UnitVec3): Vec3 {
    // Rodrigues' rotation formula
    const cos = angle.cos();

    return this.scale(cos)
      .add(axis.cross(this).scale(angle.sin()))
      .add(axis.scale(ONE.sub(cos).mul(this.dot(axis))));
  }

  mirrorX(): Vec3 {
    return new Vec3(this._x.neg(), this._y, this._z);
  }

  mirrorY(): Vec3 {
    return new Vec3(this._x, this._y.neg(), this._z);
  }

  mirrorZ(): Vec3 {
    return new Vec3(this._x, this._y, this._z.neg());
  }

  pointFromOrigin(): Point3D {
    return new Point3D(this._x, this._y, this._z);
  }

  debug(info: string): Vec3 {
    return new Vec3(
      this._x.debug(`${info}.x`),
      this._y.debug(`${info}.y`),
      this._z.debug(`${info}.z`),
    );
  }
}

export class UnitVec3 extends Vec3 implements NumStruct<UnitVec3> {
  // Selecting between unit vectors stays unit, so rebuilding as a UnitVec3 is safe.
  fromNums([x, y, z]: Num[]): UnitVec3 {
    return new UnitVec3(x, y, z);
  }

  azimuthAngle(): Angle {
    return new Angle(this._x, this._y);
  }

  elevationAngle(): Angle {
    return new Angle(this._z, hypot(this._x, this._y));
  }

  norm(): Num {
    return ONE;
  }

  normalize(): UnitVec3 {
    return this;
  }

  neg(): UnitVec3 {
    return new UnitVec3(this._x.neg(), this._y.neg(), this._z.neg());
  }

  perpZ(): UnitVec3 {
    return new UnitVec3(this._y.neg(), this._x, this._z);
  }

  perpX(): UnitVec3 {
    return new UnitVec3(this._x, this._z, this._y.neg());
  }

  perpY(): UnitVec3 {
    return new UnitVec3(this._z.neg(), this._y, this._x);
  }

  mirrorX(): UnitVec3 {
    return new UnitVec3(this._x.neg(), this._y, this._z);
  }

  mirrorY(): UnitVec3 {
    return new UnitVec3(this._x, this._y.neg(), this._z);
  }

  mirrorZ(): UnitVec3 {
    return new UnitVec3(this._x, this._y, this._z.neg());
  }

  rotate(angle: Angle, axis: UnitVec3): UnitVec3 {
    // Rodrigues' rotation formula
    const cos = angle.cos();

    return this.scale(cos)
      .add(axis.cross(this).scale(angle.sin()))
      .add(axis.scale(ONE.sub(cos).mul(this.dot(axis))));
  }

  debug(info: string): UnitVec3 {
    return new UnitVec3(
      this._x.debug(`${info}.x`),
      this._y.debug(`${info}.y`),
      this._z.debug(`${info}.z`),
    );
  }
}

export class Point3D implements NumStruct<Point3D> {
  length() {
    throw new Error("Method not implemented.");
  }
  constructor(
    private _x: Num,
    private _y: Num,
    private _z: Num,
  ) {}

  toNums(): Num[] {
    return [this._x, this._y, this._z];
  }

  fromNums([x, y, z]: Num[]): Point3D {
    return new Point3D(x, y, z);
  }

  add(vec: Vec3): Point3D {
    return new Point3D(
      this._x.add(vec.x),
      this._y.add(vec.y),
      this._z.add(vec.z),
    );
  }

  midPoint(other: Point3D): Point3D {
    return new Point3D(
      this._x.add(other._x).div(2),
      this._y.add(other._y).div(2),
      this._z.add(other._z).div(2),
    );
  }

  sub(vec: Vec3): Point3D {
    return new Point3D(
      this._x.sub(vec.x),
      this._y.sub(vec.y),
      this._z.sub(vec.z),
    );
  }

  vecTo(other: Point3D): Vec3 {
    return new Vec3(
      other._x.sub(this._x),
      other._y.sub(this._y),
      other._z.sub(this._z),
    );
  }

  vecFrom(other: Point3D): Vec3 {
    return new Vec3(
      this._x.sub(other._x),
      this._y.sub(other._y),
      this._z.sub(other._z),
    );
  }

  vecFromOrigin(): Vec3 {
    return new Vec3(this._x, this._y, this._z);
  }

  get x(): Num {
    return this._x;
  }

  get y(): Num {
    return this._y;
  }

  get z(): Num {
    return this._z;
  }

  debug(info: string): Point3D {
    return new Point3D(
      this._x.debug(`${info}.x`),
      this._y.debug(`${info}.y`),
      this._z.debug(`${info}.z`),
    );
  }
}

export function vec3FromCartesianCoords(
  x: Num | number,
  y: Num | number,
  z: Num | number,
): Vec3 {
  return new Vec3(asNum(x), asNum(y), asNum(z));
}
export const asVec3 = vec3FromCartesianCoords;

export function vec3FromPolarCoords(r: Num | number, angle: Angle): Vec2 {
  return new Vec2(angle.cos().mul(r), angle.sin().mul(r));
}

export class Plane {
  private _yAxis: UnitVec3 | null = null;

  constructor(
    public readonly origin: Point3D,
    public readonly zAxis: UnitVec3,
    public readonly xAxis: UnitVec3,
  ) {}

  get yAxis(): UnitVec3 {
    if (!this._yAxis) {
      this._yAxis = this.zAxis.cross(this.xAxis);
    }

    return this._yAxis;
  }

  translateTo(point: Point3D): Plane {
    return new Plane(point, this.zAxis, this.xAxis);
  }

  translate(vec: Vec3): Plane {
    return new Plane(this.origin.add(vec), this.zAxis, this.xAxis);
  }

  rotateAroundZ(angle: Angle): Plane {
    return new Plane(
      this.origin,
      this.zAxis,
      this.xAxis.rotate(angle, this.zAxis),
    );
  }

  pivot(angle: Angle, axis: UnitVec3): Plane {
    return new Plane(
      this.origin,
      this.zAxis.rotate(angle, axis),
      this.xAxis.rotate(angle, axis),
    );
  }
}

export function embedPoint(point: Point, plane?: Plane): Point3D {
  if (!plane) {
    return new Point3D(point.x, point.y, ZERO);
  }

  return plane.origin
    .add(plane.xAxis.scale(point.x))
    .add(plane.yAxis.scale(point.y));
}

export function embedVec(vec: Vec2, plane?: Plane): Vec3 {
  if (!plane) {
    return new Vec3(vec.x, vec.y, ZERO);
  }
  return plane.xAxis.scale(vec.x).add(plane.yAxis.scale(vec.y));
}

export function projectPoint(point: Point3D, plane: Plane): Point {
  const vec = point.vecFrom(plane.origin);
  return new Point(vec.dot(plane.xAxis), vec.dot(plane.yAxis));
}

export function projectVec(vec: Vec3, plane: Plane): Vec2 {
  return new Vec2(vec.dot(plane.xAxis), vec.dot(plane.yAxis));
}

export const ORIGIN_3D = new Point3D(ZERO, ZERO, ZERO);

export const X_AXIS = new UnitVec3(ONE, ZERO, ZERO);
export const Y_AXIS = new UnitVec3(ZERO, ONE, ZERO);
export const Z_AXIS = new UnitVec3(ZERO, ZERO, ONE);

export const XY_PLANE = new Plane(ORIGIN_3D, Z_AXIS, X_AXIS);
export const XZ_PLANE = new Plane(ORIGIN_3D, Y_AXIS, X_AXIS);
export const YZ_PLANE = new Plane(ORIGIN_3D, X_AXIS.neg(), Y_AXIS);
