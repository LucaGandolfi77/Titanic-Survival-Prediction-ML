/* ── js/effects.js ── Particle effects: explosions, smoke, muzzle flash, sparks ── */

import { MathUtils } from './utils.js';

const THREE = globalThis.THREE;

/* ── Shared particle material ── */
const particleTex = (() => {
  const size = 32;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.4, 'rgba(255,255,255,0.6)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
})();

/* ══════════════════════════════════════════════════════════
   EffectsManager — manages all visual effects
   ══════════════════════════════════════════════════════════ */
export class EffectsManager {
  constructor(scene) {
    this.scene = scene;
    this.explosions  = [];
    this.smokeTrails = [];
    this.flashes     = [];
  }

  /* ── Explosion — burst of particles ── */
  spawnExplosion(position, scale = 1) {
    const count = 60;
    const positions  = new Float32Array(count * 3);
    const colors     = new Float32Array(count * 3);
    const velocities = [];

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      positions[i3]     = position.x;
      positions[i3 + 1] = position.y;
      positions[i3 + 2] = position.z;

      // Random velocity (spherical)
      const speed = MathUtils.randomRange(5, 25) * scale;
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      velocities.push(new THREE.Vector3(
        speed * Math.sin(phi) * Math.cos(theta),
        speed * Math.sin(phi) * Math.sin(theta),
        speed * Math.cos(phi)
      ));

      // Color: orange to yellow
      const r = MathUtils.randomRange(0.8, 1);
      const g = MathUtils.randomRange(0.3, 0.8);
      const b = MathUtils.randomRange(0, 0.2);
      colors[i3]     = r;
      colors[i3 + 1] = g;
      colors[i3 + 2] = b;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
      size: 1.2 * scale,
      map: particleTex,
      transparent: true,
      opacity: 1,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
    });

    const points = new THREE.Points(geo, mat);
    this.scene.add(points);

    this.explosions.push({
      points,
      velocities,
      life: 0,
      maxLife: 1.2,
      scale,
    });

    // Flash light
    const light = new THREE.PointLight(0xff6600, 8 * scale, 30 * scale);
    light.position.copy(position);
    this.scene.add(light);
    this.flashes.push({ light, life: 0, maxLife: 0.15 });
  }

  /* ── Muzzle flash — brief point light + small particles ── */
  spawnMuzzleFlash(position, direction) {
    const light = new THREE.PointLight(0x00ff41, 3, 8);
    light.position.copy(position);
    this.scene.add(light);
    this.flashes.push({ light, life: 0, maxLife: 0.06 });

    // Small particle burst (6 particles)
    const count = 6;
    const positions = new Float32Array(count * 3);
    const vels = [];
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      positions[i3]     = position.x;
      positions[i3 + 1] = position.y;
      positions[i3 + 2] = position.z;
      vels.push(new THREE.Vector3(
        direction.x * 15 + MathUtils.randomRange(-3, 3),
        direction.y * 15 + MathUtils.randomRange(-3, 3),
        direction.z * 15 + MathUtils.randomRange(-3, 3)
      ));
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.3,
      map: particleTex,
      transparent: true,
      opacity: 1,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      color: 0x00ff41,
    });
    const pts = new THREE.Points(geo, mat);
    this.scene.add(pts);

    this.explosions.push({
      points: pts,
      velocities: vels,
      life: 0,
      maxLife: 0.2,
      scale: 0.3,
    });
  }

  /* ── Smoke trail particle ── */
  spawnSmoke(position) {
    const count = 8;
    const positions = new Float32Array(count * 3);
    const vels = [];
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      positions[i3]     = position.x + MathUtils.randomRange(-0.3, 0.3);
      positions[i3 + 1] = position.y + MathUtils.randomRange(-0.3, 0.3);
      positions[i3 + 2] = position.z + MathUtils.randomRange(-0.3, 0.3);
      vels.push(new THREE.Vector3(
        MathUtils.randomRange(-1, 1),
        MathUtils.randomRange(1, 4),
        MathUtils.randomRange(-1, 1)
      ));
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.8,
      map: particleTex,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      blending: THREE.NormalBlending,
      color: 0x888888,
    });
    const pts = new THREE.Points(geo, mat);
    this.scene.add(pts);

    this.smokeTrails.push({
      points: pts,
      velocities: vels,
      life: 0,
      maxLife: 1.5,
    });
  }

  /* ── Engine sparks (when boosting) ── */
  spawnSparks(position, direction) {
    const count = 4;
    const positions = new Float32Array(count * 3);
    const vels = [];
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      positions[i3]     = position.x;
      positions[i3 + 1] = position.y;
      positions[i3 + 2] = position.z;
      vels.push(new THREE.Vector3(
        -direction.x * 8 + MathUtils.randomRange(-3, 3),
        -direction.y * 8 + MathUtils.randomRange(-1, 3),
        -direction.z * 8 + MathUtils.randomRange(-3, 3)
      ));
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.4,
      map: particleTex,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      color: 0x00ff88,
    });
    const pts = new THREE.Points(geo, mat);
    this.scene.add(pts);

    this.explosions.push({
      points: pts,
      velocities: vels,
      life: 0,
      maxLife: 0.4,
      scale: 0.3,
    });
  }

  /* ── Update all effects ── */
  update(delta) {
    // Update explosions & muzzle particles
    for (let i = this.explosions.length - 1; i >= 0; i--) {
      const exp = this.explosions[i];
      exp.life += delta;

      if (exp.life >= exp.maxLife) {
        this.scene.remove(exp.points);
        exp.points.geometry.dispose();
        exp.points.material.dispose();
        this.explosions.splice(i, 1);
        continue;
      }

      const progress = exp.life / exp.maxLife;
      exp.points.material.opacity = 1 - progress;

      // Move particles
      const posAttr = exp.points.geometry.getAttribute('position');
      for (let j = 0; j < exp.velocities.length; j++) {
        const j3 = j * 3;
        posAttr.array[j3]     += exp.velocities[j].x * delta;
        posAttr.array[j3 + 1] += exp.velocities[j].y * delta;
        posAttr.array[j3 + 2] += exp.velocities[j].z * delta;
        // Gravity on particles
        exp.velocities[j].y -= 9.8 * delta * 0.5;
      }
      posAttr.needsUpdate = true;
    }

    // Update smoke
    for (let i = this.smokeTrails.length - 1; i >= 0; i--) {
      const sm = this.smokeTrails[i];
      sm.life += delta;

      if (sm.life >= sm.maxLife) {
        this.scene.remove(sm.points);
        sm.points.geometry.dispose();
        sm.points.material.dispose();
        this.smokeTrails.splice(i, 1);
        continue;
      }

      const progress = sm.life / sm.maxLife;
      sm.points.material.opacity = 0.5 * (1 - progress);
      sm.points.material.size = 0.8 + progress * 2; // expand

      const posAttr = sm.points.geometry.getAttribute('position');
      for (let j = 0; j < sm.velocities.length; j++) {
        const j3 = j * 3;
        posAttr.array[j3]     += sm.velocities[j].x * delta;
        posAttr.array[j3 + 1] += sm.velocities[j].y * delta;
        posAttr.array[j3 + 2] += sm.velocities[j].z * delta;
      }
      posAttr.needsUpdate = true;
    }

    // Update light flashes
    for (let i = this.flashes.length - 1; i >= 0; i--) {
      const f = this.flashes[i];
      f.life += delta;
      if (f.life >= f.maxLife) {
        this.scene.remove(f.light);
        f.light.dispose();
        this.flashes.splice(i, 1);
        continue;
      }
      f.light.intensity *= 0.85;
    }
  }

  /* ── Clear all effects ── */
  clear() {
    for (const exp of this.explosions) {
      this.scene.remove(exp.points);
      exp.points.geometry.dispose();
      exp.points.material.dispose();
    }
    this.explosions.length = 0;

    for (const sm of this.smokeTrails) {
      this.scene.remove(sm.points);
      sm.points.geometry.dispose();
      sm.points.material.dispose();
    }
    this.smokeTrails.length = 0;

    for (const f of this.flashes) {
      this.scene.remove(f.light);
      f.light.dispose();
    }
    this.flashes.length = 0;
  }
}
