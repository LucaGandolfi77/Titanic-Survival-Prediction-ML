import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.min.js';

export class MathUtils {
  static lerp(a, b, t) {
    return a + (b - a) * Math.max(0, Math.min(1, t));
  }

  static clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  static distance(v1, v2) {
    const dx = v1.x - v2.x;
    const dy = v1.y - v2.y;
    const dz = v1.z - v2.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  static randRange(min, max) {
    return min + Math.random() * (max - min);
  }

  static remap(val, inMin, inMax, outMin, outMax) {
    const t = (val - inMin) / (inMax - inMin);
    return outMin + t * (outMax - outMin);
  }

  static smoothstep(edge0, edge1, x) {
    const t = this.clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  // Quaternion normalization
  static quatNormalize(q) {
    let len = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (len === 0) return q;
    return new THREE.Quaternion(q.x / len, q.y / len, q.z / len, q.w / len);
  }

  // Quaternion conjugate
  static quatConj(q) {
    return new THREE.Quaternion(-q.x, -q.y, -q.z, q.w);
  }

  // Quaternion multiply
  static quatMult(a, b) {
    const result = new THREE.Quaternion();
    result.multiplyQuaternions(a, b);
    return result;
  }

  // Check if point is in AABB
  static checkPointInAABB(point, boxMin, boxMax) {
    return point.x >= boxMin.x && point.x <= boxMax.x &&
           point.y >= boxMin.y && point.y <= boxMax.y &&
           point.z >= boxMin.z && point.z <= boxMax.z;
  }

  // Check AABB collision
  static checkAABBCollision(pos1, size1, pos2, size2) {
    return pos1.x - size1.x < pos2.x + size2.x &&
           pos1.x + size1.x > pos2.x - size2.x &&
           pos1.y - size1.y < pos2.y + size2.y &&
           pos1.y + size1.y > pos2.y - size2.y &&
           pos1.z - size1.z < pos2.z + size2.z &&
           pos1.z + size1.z > pos2.z - size2.z;
  }

  // Point to plane distance
  static pointToPlaneDistance(point, planePoint, planeNormal) {
    const v = new THREE.Vector3().subVectors(point, planePoint);
    return v.dot(planeNormal);
  }

  // Reflection of vector across plane
  static reflect(v, n) {
    const result = v.clone();
    result.sub(n.clone().multiplyScalar(2 * v.dot(n)));
    return result;
  }
}

// Stencil utilities
export class StencilUtils {
  static createStencilRenderTarget(width, height) {
    return new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      stencilBuffer: true,
      format: THREE.RGBFormat,
      type: THREE.UnsignedByteType
    });
  }

  static setupStencilPass(renderer, id) {
    renderer.state.setStencilTest(true);
    renderer.state.setStencilFunc(THREE.AlwaysStencilFunc, id, 0xff);
    renderer.state.setStencilOp(THREE.ReplaceStencilOp, THREE.ReplaceStencilOp, THREE.ReplaceStencilOp);
    renderer.state.setColorWrite(false);
    renderer.state.setDepthTest(false);
  }

  static setupStencilTestPass(renderer, id) {
    renderer.state.setStencilTest(true);
    renderer.state.setStencilFunc(THREE.EqualStencilFunc, id, 0xff);
    renderer.state.setColorWrite(true);
    renderer.state.setDepthTest(true);
  }

  static disableStencil(renderer) {
    renderer.state.setStencilTest(false);
  }
}

// Random string/ID generation
export function generateID() {
  return Math.random().toString(36).substr(2, 9);
}

// Clamp angle to 0-2PI
export function normalizeAngle(angle) {
  while (angle < 0) angle += Math.PI * 2;
  while (angle > Math.PI * 2) angle -= Math.PI * 2;
  return angle;
}

// Linear interpolation between two angles (shortest path)
export function lerpAngle(a, b, t) {
  let delta = b - a;
  if (delta > Math.PI) delta -= Math.PI * 2;
  if (delta < -Math.PI) delta += Math.PI * 2;
  return a + delta * MathUtils.clamp(t, 0, 1);
}

// Queue-based event system
export class EventBus {
  constructor() {
    this.listeners = {};
  }

  on(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  off(event, callback) {
    if (!this.listeners[event]) return;
    this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
  }

  emit(event, data) {
    if (!this.listeners[event]) return;
    this.listeners[event].forEach(cb => cb(data));
  }

  clear() {
    this.listeners = {};
  }
}

// Vector rotation utilities
export function rotateVectorAroundAxis(vector, axis, angle) {
  const quat = new THREE.Quaternion();
  quat.setFromAxisAngle(axis, angle);
  const v = new THREE.Vector3().copy(vector);
  v.applyQuaternion(quat);
  return v;
}