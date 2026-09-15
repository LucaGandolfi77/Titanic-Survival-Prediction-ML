import * as THREE from 'three';

export class ShadowSystem {
  constructor() {
    this.history = [];
    this.maxHistory = 600;
    this.ghostMesh = null;
    this.ghostDelay = 5.0;
    this.maxDelay = 10.0;
    this.minDelay = 5.0;
    this.enabled = true;
    this._lastRecordTime = 0;
    this._recordInterval = 1 / 60;
  }

  createGhostMesh(scene) {
    if (this.ghostMesh) {
      if (this.ghostMesh.parent === scene) return;
      scene.add(this.ghostMesh);
      return;
    }
    const geo = new THREE.BoxGeometry(0.25, 1.75, 0.25);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.15,
      depthWrite: false
    });
    this.ghostMesh = new THREE.Mesh(geo, mat);
    this.ghostMesh.visible = false;
    scene.add(this.ghostMesh);
  }

  disposeGhostMesh(scene) {
    if (this.ghostMesh) {
      scene.remove(this.ghostMesh);
      this.ghostMesh.geometry.dispose();
      this.ghostMesh.material.dispose();
      this.ghostMesh = null;
    }
  }

  record(x, y, z, yaw, pitch) {
    const now = performance.now() / 1000;
    if (now - this._lastRecordTime < this._recordInterval) return;
    this._lastRecordTime = now;

    this.history.push({ x, y, z, yaw, pitch, time: now });
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }
  }

  setSanity(sanity) {
    const ratio = Math.max(0, Math.min(1, (sanity + 100) / 200));
    this.ghostDelay = this.maxDelay - ratio * (this.maxDelay - this.minDelay);
  }

  getGhostState(currentTime) {
    if (this.history.length === 0) return null;
    const targetTime = currentTime - this.ghostDelay;
    let best = null;
    let bestDiff = Infinity;
    for (let i = this.history.length - 1; i >= 0; i--) {
      const entry = this.history[i];
      const diff = targetTime - entry.time;
      if (diff < 0) continue;
      if (diff < bestDiff) {
        bestDiff = diff;
        best = entry;
      } else {
        break;
      }
    }
    if (!best) {
      for (let i = this.history.length - 1; i >= 0; i--) {
        if (this.history[i].time <= currentTime) {
          best = this.history[i];
          break;
        }
      }
    }
    return best;
  }

  update(x, y, z, yaw, pitch, sanity, currentTime) {
    if (!this.enabled) return null;
    this.record(x, y, z, yaw, pitch);
    this.setSanity(sanity);
    return this.getGhostState(currentTime);
  }

  updateGhostMesh(state) {
    if (!this.ghostMesh) return;
    if (!state) {
      this.ghostMesh.visible = false;
      return;
    }
    this.ghostMesh.visible = true;
    this.ghostMesh.position.set(state.x, state.y + 0.9, state.z);
    this.ghostMesh.rotation.y = state.yaw;
  }

  reset() {
    this.history.length = 0;
    this._lastRecordTime = 0;
    this.ghostDelay = this.minDelay;
  }
}
