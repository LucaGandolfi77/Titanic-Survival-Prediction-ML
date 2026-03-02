export class Vector2 {
  constructor(x = 0, y = 0) {
    this.x = x;
    this.y = y;
  }

  set(x, y) {
    this.x = x;
    this.y = y;
    return this;
  }

  copy(v) {
    this.x = v.x;
    this.y = v.y;
    return this;
  }

  add(v) {
    this.x += v.x;
    this.y += v.y;
    return this;
  }

  sub(v) {
    this.x -= v.x;
    this.y -= v.y;
    return this;
  }

  multiplyScalar(s) {
    this.x *= s;
    this.y *= s;
    return this;
  }

  mag() {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }

  magSq() {
    return this.x * this.x + this.y * this.y;
  }

  normalize() {
    const m = this.mag();
    if (m > 0) {
      this.multiplyScalar(1 / m);
    }
    return this;
  }

  distanceTo(v) {
    const dx = this.x - v.x;
    const dy = this.y - v.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  distanceToSquared(v) {
    const dx = this.x - v.x;
    const dy = this.y - v.y;
    return dx * dx + dy * dy;
  }

  dot(v) {
    return this.x * v.x + this.y * v.y;
  }

  angle() {
    return Math.atan2(this.y, this.x);
  }

  rotate(angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const nx = this.x * cos - this.y * sin;
    const ny = this.x * sin + this.y * cos;
    this.x = nx;
    this.y = ny;
    return this;
  }

  clone() {
    return new Vector2(this.x, this.y);
  }
}

export function lerp(start, end, amt) {
  return (1 - amt) * start + amt * end;
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function randomRange(min, max) {
  return Math.random() * (max - min) + min;
}

export function angledVector(angle, length = 1) {
  return new Vector2(Math.cos(angle) * length, Math.sin(angle) * length);
}

// Format seconds into MM:SS
export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// Circle-circle collision
export function checkCircleCollision(x1, y1, r1, x2, y2, r2) {
  const dx = x1 - x2;
  const dy = y1 - y2;
  const distSq = dx * dx + dy * dy;
  const radSum = r1 + r2;
  return distSq < radSum * radSum;
}