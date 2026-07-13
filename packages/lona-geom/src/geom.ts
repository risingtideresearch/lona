import { AngleLike, PointLike, UnitVec2Like, Vec2Like } from "./geom-types";
import { Num, ONE, asNum, type NumStruct } from "lona";
import { atan2, hypot, ifTruthyElse } from "lona";
import { nexpr } from "lona";

function nonZeroSign(n: Num): Num {
  return ifTruthyElse(n.lessThan(0), asNum(-1), asNum(1));
}

const _2PI = asNum(Math.PI * 2);

export class Angle implements AngleLike, NumStruct<Angle> {
  private _cos: Num;
  private _sin: Num;

  constructor(cos: Num, sin: Num) {
    this._cos = cos;
    this._sin = sin;
  }

  toNums(): Num[] {
    return [this._cos, this._sin];
  }

  fromNums([cos, sin]: Num[]): Angle {
    return new Angle(cos, sin);
  }

  add(other: Angle): Angle {
    return new Angle(
      nexpr`${this._cos} * ${other._cos} - ${this._sin} * ${other._sin}`,
      nexpr`${this._sin} * ${other._cos} + ${this._cos} * ${other._sin}`,
    );
  }

  sub(other: Angle): Angle {
    return new Angle(
      nexpr`${this._cos} * ${other._cos} + ${this._sin} * ${other._sin}`,
      nexpr`${this._sin} * ${other._cos} - ${this._cos} * ${other._sin}`,
    );
  }

  neg(): Angle {
    return new Angle(this._cos, this._sin.neg());
  }

  half(): Angle {
    const sqrtX = nexpr`sqrt((${this._cos} + 1) / 2)`;
    const sqrtY = nexpr`sqrt((1 - ${this._cos}) / 2)`;
    return new Angle(nexpr`${nonZeroSign(this._sin)} * ${sqrtX}`, sqrtY);
  }

  double(): Angle {
    return new Angle(
      nexpr`${this._cos} * ${this._cos} - ${this._sin} * ${this._sin}`,
      nexpr`${this._cos} * 2 * ${this._sin}`,
    );
  }

  perp(): Angle {
    return new Angle(this._sin.neg(), this._cos);
  }

  opposite(): Angle {
    return new Angle(this._cos.neg(), this._sin.neg());
  }

  cos(): Num {
    return this._cos;
  }

  sin(): Num {
    return this._sin;
  }

  tan(): Num {
    return nexpr`${this._sin} / ${this._cos}`;
  }

  asRad(): Num {
    return atan2(this._sin, this._cos);
  }

  asDeg(): Num {
    return nexpr`${this.asRad()} * 180 / ${Math.PI}`;
  }

  asSortValue(): Num {
    const isQ3Q4 = this._sin.lessThan(0);
    const base = ifTruthyElse(
      isQ3Q4,
      nexpr`${this._cos} + 3`,
      nexpr`1 - ${this._cos}`,
    );
    return nexpr`${base} / 2`;
  }

  asUnitArcLength(): Num {
    return nexpr`(${this.asRad()} + ${_2PI}) % ${_2PI}`;
  }

  asVec(): UnitVec2 {
    return new UnitVec2(this._cos, this._sin);
  }

  debug(info: string): Angle {
    return new Angle(
      this._cos.debug(`${info}.cos`),
      this._sin.debug(`${info}.sin`),
    );
  }
}

export function angleRromRad(rad: Num | number): Angle {
  const val = asNum(rad);
  return new Angle(val.cos(), val.sin());
}

export function angleFromDeg(deg: Num | number): Angle {
  const val = asNum(deg);
  return angleRromRad(nexpr`${val} * ${Math.PI} / 180`);
}

export function angleFromSin(sin: Num | number): Angle {
  const val = asNum(sin);
  return new Angle(nexpr`sqrt(1 - ${val} * ${val})`, val);
}

export function angleFromCos(cos: Num | number): Angle {
  const val = asNum(cos);
  return new Angle(val, nexpr`sqrt(1 - ${val} * ${val})`);
}

export function angleFromDirection(direction: Vec2): Angle {
  return direction.asAngle();
}

export function twoVectorsAngle(v1: Vec2, v2: Vec2): Angle {
  const u1 = v1.normalize();
  const u2 = v2.normalize();

  const cos = u1.dot(u2);
  const sin = u1.cross(u2);

  return new Angle(cos, sin);
}

export function arcTan(x: Num | number, y: Num | number): Angle {
  const norm = hypot(x, y);
  return new Angle(nexpr`${asNum(x)} / ${norm}`, nexpr`${asNum(y)} / ${norm}`);
}

export const NO_TURN = new Angle(asNum(1), asNum(0));
export const FULL_TURN = new Angle(asNum(1), asNum(0));
export const HALF_TURN = new Angle(asNum(-1), asNum(0));
export const QUARTER_TURN = new Angle(asNum(0), asNum(1));
export const THREE_QUARTER_TURN = new Angle(asNum(0), asNum(-1));
export const EIGHTH_TURN = new Angle(asNum(Math.SQRT1_2), asNum(Math.SQRT1_2));

export class Vec2 implements Vec2Like, NumStruct<Vec2> {
  protected _x: Num;
  protected _y: Num;

  constructor(x: Num, y: Num) {
    this._x = x;
    this._y = y;
  }

  toNums(): Num[] {
    return [this._x, this._y];
  }

  fromNums([x, y]: Num[]): Vec2 {
    return new Vec2(x, y);
  }

  add(other: Vec2): Vec2 {
    return new Vec2(
      nexpr`${this._x} + ${other._x}`,
      nexpr`${this._y} + ${other._y}`,
    );
  }

  sub(other: Vec2): Vec2 {
    return new Vec2(
      nexpr`${this._x} - ${other._x}`,
      nexpr`${this._y} - ${other._y}`,
    );
  }

  neg(): Vec2 {
    return new Vec2(this._x.neg(), this._y.neg());
  }

  scale(other: Num | number): Vec2 {
    return new Vec2(nexpr`${this._x} * ${other}`, nexpr`${this._y} * ${other}`);
  }

  div(other: Num | number): Vec2 {
    return new Vec2(nexpr`${this._x} / ${other}`, nexpr`${this._y} / ${other}`);
  }

  dot(other: Vec2): Num {
    return nexpr`${this._x} * ${other._x} + ${this._y} * ${other._y}`;
  }

  cross(other: Vec2): Num {
    return nexpr`${this._x} * ${other._y} - ${this._y} * ${other._x}`;
  }

  norm(): Num {
    return nexpr`sqrt(${this.dot(this)})`;
  }

  normalize(): UnitVec2 {
    return this.div(this.norm());
  }

  get x(): Num {
    return this._x;
  }

  get y(): Num {
    return this._y;
  }

  perp(): Vec2 {
    return new Vec2(this._y.neg(), this._x);
  }

  mirrorX(): Vec2 {
    return new Vec2(this._x.neg(), this._y);
  }

  mirrorY(): Vec2 {
    return new Vec2(this._x, this._y.neg());
  }

  rotate(angle: Angle): Vec2 {
    return new Vec2(
      nexpr`${angle.cos()} * ${this._x} - ${angle.sin()} * ${this._y}`,
      nexpr`${angle.sin()} * ${this._x} + ${angle.cos()} * ${this._y}`,
    );
  }

  asAngle(): Angle {
    const normalized = this.normalize();
    return new Angle(normalized._x, normalized._y);
  }

  pointFromOrigin(): Point {
    return new Point(this._x, this._y);
  }

  debug(info: string): Vec2 {
    return new Vec2(this._x.debug(`${info}.x`), this._y.debug(`${info}.y`));
  }
}

export class UnitVec2
  extends Vec2
  implements UnitVec2Like, NumStruct<UnitVec2>
{
  // Selecting between unit vectors stays unit (one whole input is chosen), so
  // rebuilding as a UnitVec2 is safe even though raw components needn't be unit.
  fromNums([x, y]: Num[]): UnitVec2 {
    return new UnitVec2(x, y);
  }

  asAngle(): Angle {
    return new Angle(this._x, this._y);
  }

  norm(): Num {
    return ONE;
  }

  normalize(): UnitVec2 {
    return this;
  }

  neg(): UnitVec2 {
    return new UnitVec2(this._x.neg(), this._y.neg());
  }

  perp(): UnitVec2 {
    return new UnitVec2(this._y.neg(), this._x);
  }

  mirrorX(): UnitVec2 {
    return new UnitVec2(this._x.neg(), this._y);
  }

  mirrorY(): UnitVec2 {
    return new UnitVec2(this._x, this._y.neg());
  }

  rotate(angle: Angle): UnitVec2 {
    return new UnitVec2(
      nexpr`${angle.cos()} * ${this._x} - ${angle.sin()} * ${this._y}`,
      nexpr`${angle.sin()} * ${this._x} + ${angle.cos()} * ${this._y}`,
    );
  }

  debug(info: string): UnitVec2 {
    return new UnitVec2(this._x.debug(`${info}.x`), this._y.debug(`${info}.y`));
  }
}

export class Point implements PointLike, NumStruct<Point> {
  constructor(
    private _x: Num,
    private _y: Num,
  ) {}

  toNums(): Num[] {
    return [this._x, this._y];
  }

  fromNums([x, y]: Num[]): Point {
    return new Point(x, y);
  }

  add(vec: Vec2): Point {
    return new Point(
      nexpr`${this._x} + ${vec.x}`,
      nexpr`${this._y} + ${vec.y}`,
    );
  }

  midPoint(other: Point): Point {
    return new Point(
      nexpr`(${this._x} + ${other._x}) / 2`,
      nexpr`(${this._y} + ${other._y}) / 2`,
    );
  }

  sub(vec: Vec2): Point {
    return new Point(
      nexpr`${this._x} - ${vec.x}`,
      nexpr`${this._y} - ${vec.y}`,
    );
  }

  vecTo(other: Point): Vec2 {
    return new Vec2(
      nexpr`${other._x} - ${this._x}`,
      nexpr`${other._y} - ${this._y}`,
    );
  }

  vecFrom(other: Point): Vec2 {
    return new Vec2(
      nexpr`${this._x} - ${other._x}`,
      nexpr`${this._y} - ${other._y}`,
    );
  }

  vecFromOrigin(): Vec2 {
    return new Vec2(this._x, this._y);
  }

  get x(): Num {
    return this._x;
  }

  get y(): Num {
    return this._y;
  }

  debug(info: string): Point {
    return new Point(this._x.debug(`${info}.x`), this._y.debug(`${info}.y`));
  }
}

export const ORIGIN = new Point(asNum(0), asNum(0));

export function vecFromCartesianCoords(x: Num | number, y: Num | number): Vec2 {
  return new Vec2(asNum(x), asNum(y));
}
export const asVec = vecFromCartesianCoords;

export function vecFromPolarCoords(r: Num | number, angle: Angle): Vec2 {
  return new Vec2(nexpr`${angle.cos()} * ${r}`, nexpr`${angle.sin()} * ${r}`);
}

export class SolidAngle {
  private _turns: Num;

  constructor(turns: Num | number) {
    this._turns = asNum(turns);
  }

  get turns(): Num {
    return this._turns;
  }

  addAngle(angle: Angle): SolidAngle {
    return new SolidAngle(
      nexpr`${this._turns} + ${angle.asRad()} / ${Math.PI * 2}`,
    );
  }

  addTurns(turns: Num | number): SolidAngle {
    return new SolidAngle(nexpr`${this._turns} + ${asNum(turns)}`);
  }

  add(other: SolidAngle): SolidAngle {
    return new SolidAngle(nexpr`${this._turns} + ${other._turns}`);
  }

  sub(other: SolidAngle): SolidAngle {
    return new SolidAngle(nexpr`${this._turns} - ${other._turns}`);
  }

  neg(): SolidAngle {
    return new SolidAngle(this._turns.neg());
  }

  half(): SolidAngle {
    return new SolidAngle(nexpr`${this._turns} / 2`);
  }
}

export function solidAngleFromAngle(angle: Angle): SolidAngle {
  return new SolidAngle(0).addAngle(angle);
}

// The `ifTruthyElseForPoints` / `…Vec2s` / `…Angles` selects moved to
// `./num-layouts`, where they are one-liners over the shared `NumLayout`
// instances (and joined by 3D variants). They remain exported from the package
// barrel, so existing imports are unaffected.
