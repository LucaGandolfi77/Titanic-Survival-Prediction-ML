import * as THREE from 'three';

export class ObjectPool {
  constructor(factory, initialSize = 20) {
    this._factory = factory;
    this._pool = [];
    this._active = new Set();
    for (let i = 0; i < initialSize; i++) {
      const obj = this._factory();
      obj.visible = false;
      obj.userData.__poolActive = false;
      this._pool.push(obj);
    }
  }

  get() {
    let obj = this._pool.pop();
    if (!obj) obj = this._factory();
    obj.visible = true;
    obj.userData.__poolActive = true;
    this._active.add(obj);
    return obj;
  }

  release(obj) {
    obj.visible = false;
    obj.userData.__poolActive = false;
    this._active.delete(obj);
    this._pool.push(obj);
  }

  releaseAll() {
    for (const obj of this._active) {
      obj.visible = false;
      obj.userData.__poolActive = false;
      this._pool.push(obj);
    }
    this._active.clear();
  }

  get activeCount() {
    return this._active.size;
  }

  get active() {
    return this._active;
  }
}

export const MathUtils = {
  lerp(a, b, t) {
    return a + (b - a) * t;
  },

  lerpAngle(a, b, t) {
    let diff = b - a;
    while (diff > Math.PI) diff -= Math.PI * 2;
    while (diff < -Math.PI) diff += Math.PI * 2;
    return a + diff * t;
  },

  clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  },

  randomRange(min, max) {
    return min + Math.random() * (max - min);
  },

  randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  },

  smoothDamp(current, target, velocity, smoothTime, maxSpeed, delta) {
    smoothTime = Math.max(0.0001, smoothTime);
    const omega = 2 / smoothTime;
    const x = omega * delta;
    const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    const change = current - target;
    const temp = (velocity + omega * change) * delta;
    velocity = (velocity - omega * temp) * exp;
    let output = target + (change + temp) * exp;
    if (target - current > 0 === output > target) {
      output = target;
      velocity = 0;
    }
    return { value: output, velocity };
  }
};

export function createBox(w, h, d, color) {
  const geo = new THREE.BoxGeometry(w, h, d);
  const mat = new THREE.MeshStandardMaterial({ color });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export function createCylinder(rTop, rBot, h, color, segments = 12) {
  const geo = new THREE.CylinderGeometry(rTop, rBot, h, segments);
  const mat = new THREE.MeshStandardMaterial({ color });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export function createSphere(r, color, wSeg = 12, hSeg = 8) {
  const geo = new THREE.SphereGeometry(r, wSeg, hSeg);
  const mat = new THREE.MeshStandardMaterial({ color });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}
