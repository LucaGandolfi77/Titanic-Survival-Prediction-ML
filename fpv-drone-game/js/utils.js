/**
 * Utility helpers — DRONE STRIKE
 * Object pool, math utils, helpers.
 * @module utils
 */

/* ═══════════ Object Pool ═══════════ */

export class ObjectPool {
  /**
   * @param {() => any} createFn  – factory for new items
   * @param {(obj:any) => void} resetFn – called when recycling
   * @param {number} initialSize
   */
  constructor(createFn, resetFn, initialSize = 20) {
    this._create = createFn;
    this._reset = resetFn;
    /** @type {any[]} */ this._pool = [];
    /** @type {Set<any>} */ this._active = new Set();
    for (let i = 0; i < initialSize; i++) {
      const obj = this._create();
      obj.__poolActive = false;
      this._pool.push(obj);
    }
  }

  /** Get an inactive item (or create a new one). */
  get() {
    let obj = this._pool.find(o => !o.__poolActive);
    if (!obj) {
      obj = this._create();
      this._pool.push(obj);
    }
    obj.__poolActive = true;
    this._active.add(obj);
    return obj;
  }

  /** Release an item back to the pool. */
  release(obj) {
    obj.__poolActive = false;
    this._active.delete(obj);
    this._reset(obj);
  }

  /** All currently active items. */
  get active() { return this._active; }

  /** Release every active item. */
  releaseAll() {
    for (const obj of [...this._active]) this.release(obj);
  }
}

/* ═══════════ Math Utilities ═══════════ */

export const MathUtils = {
  lerp:  (a, b, t) => a + (b - a) * t,
  clamp: (v, lo, hi) => Math.max(lo, Math.min(hi, v)),
  randomRange: (min, max) => Math.random() * (max - min) + min,
  randomInt: (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
  
  /** Random Vector3 within a sphere of given radius around center. */
  randomVector3InRadius(center, radius) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = radius * Math.cbrt(Math.random());
    return new THREE.Vector3(
      center.x + r * Math.sin(phi) * Math.cos(theta),
      center.y + r * Math.sin(phi) * Math.sin(theta),
      center.z + r * Math.cos(phi)
    );
  },

  normalizeAngle(a) {
    while (a > Math.PI) a -= Math.PI * 2;
    while (a < -Math.PI) a += Math.PI * 2;
    return a;
  },

  /** Distance from a point to an AABB (THREE.Box3). Returns 0 if inside. */
  distToBox(point, box) {
    const dx = Math.max(box.min.x - point.x, 0, point.x - box.max.x);
    const dy = Math.max(box.min.y - point.y, 0, point.y - box.max.y);
    const dz = Math.max(box.min.z - point.z, 0, point.z - box.max.z);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
};

export function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function degToRad(d) { return d * (Math.PI / 180); }
export function radToDeg(r) { return r * (180 / Math.PI); }
