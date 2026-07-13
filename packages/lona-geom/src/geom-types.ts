import type { Num } from "lona";

export type GeomScalar = Num | number;

export interface UnitVec2Like<S = Num> extends Vec2Like<S> {
  neg(): UnitVec2Like<S>;
}
export interface UnitVec3Like<S = Num> extends Vec3Like<S> {
  neg(): UnitVec3Like<S>;
}

export interface Vec2Like<S = Num> {
  x: S;
  y: S;

  add(other: Vec2Like<S>): Vec2Like<S>;
  sub(other: Vec2Like<S>): Vec2Like<S>;
  neg(): Vec2Like<S>;
  scale(k: GeomScalar): Vec2Like<S>;
  div(k: GeomScalar): Vec2Like<S>;
  dot(other: Vec2Like<S>): S;
  cross(other: Vec2Like<S>): S;
  norm(): S;
  normalize(): UnitVec2Like<S>;
  asAngle(): AngleLike<S>;
  pointFromOrigin(): PointLike<S>;
}

export interface Vec3Like<S = Num> {
  x: S;
  y: S;
  z: S;

  add(other: Vec3Like<S>): Vec3Like<S>;
  sub(other: Vec3Like<S>): Vec3Like<S>;
  neg(): Vec3Like<S>;
  scale(k: GeomScalar): Vec3Like<S>;
  div(k: GeomScalar): Vec3Like<S>;
  dot(other: Vec3Like<S>): S;
  cross(other: Vec3Like<S>): Vec3Like<S>;
  norm(): S;
  normalize(): UnitVec3Like<S>;
  azimuthAngle(): AngleLike<S>;
  elevationAngle(): AngleLike<S>;
  pointFromOrigin(): Point3Like<S>;
}

export interface PointLike<S = Num> {
  x: S;
  y: S;

  add(vec: Vec2Like<S>): PointLike<S>;
  midPoint(other: PointLike<S>): PointLike<S>;
  sub(vec: Vec2Like<S>): PointLike<S>;
  vecTo(other: PointLike<S>): Vec2Like<S>;
  vecFrom(other: PointLike<S>): Vec2Like<S>;
  vecFromOrigin(): Vec2Like<S>;
}

export interface Point3Like<S = Num> {
  x: S;
  y: S;
  z: S;

  add(vec: Vec3Like<S>): Point3Like<S>;
  midPoint(other: Point3Like<S>): Point3Like<S>;
  sub(vec: Vec3Like<S>): Point3Like<S>;
  vecTo(other: Point3Like<S>): Vec3Like<S>;
  vecFrom(other: Point3Like<S>): Vec3Like<S>;
  vecFromOrigin(): Vec3Like<S>;
}

export interface AngleLike<S = Num> {
  add(other: AngleLike<S>): AngleLike<S>;
  sub(other: AngleLike<S>): AngleLike<S>;
  neg(): AngleLike<S>;
  half(): AngleLike<S>;
  double(): AngleLike<S>;
  perp(): AngleLike<S>;
  opposite(): AngleLike<S>;
  cos(): S;
  sin(): S;
  tan(): S;
  asRad(): S;
  asDeg(): S;
  asSortValue(): S;
  asUnitArcLength(): S;
  asVec(): UnitVec2Like<S>;
}

export interface PlaneLike<S = Num> {
  origin: Point3Like<S>;

  xAxis: UnitVec3Like<S>;
  zAxis: UnitVec3Like<S>;
  yAxis: UnitVec3Like<S>;
}
