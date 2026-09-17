/* ── js/weapons.js ── Player weapon systems — cannon + homing missiles ── */

import { ObjectPool, MathUtils } from './utils.js';

const THREE = globalThis.THREE;

/* ── Projectile factory ── */
function createBullet() {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.08, 4, 4),
    new THREE.MeshBasicMaterial({ color: 0x00ff41 })
  );
  mesh.visible = false;
  return {
    mesh,
    velocity: new THREE.Vector3(),
    life: 0,
    maxLife: 2.5,
    damage: 20,
    active: false,
  };
}

/* ── Plasma beam factory ── */
function createPlasma() {
  const length = 8;
  const geo = new THREE.CylinderGeometry(0.15, 0.25, length, 8);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x00ffcc,
    transparent: true,
    opacity: 0.8,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.visible = false;

  // Inner glow
  const glowGeo = new THREE.CylinderGeometry(0.3, 0.45, length, 8);
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0x00ff88,
    transparent: true,
    opacity: 0.3,
  });
  const glowMesh = new THREE.Mesh(glowGeo, glowMat);
  mesh.add(glowMesh);

  // Point light at tip
  const tipLight = new THREE.PointLight(0x00ffcc, 3, 15);
  tipLight.position.set(0, -length / 2, 0);
  mesh.add(tipLight);

  return {
    mesh,
    tipLight,
    velocity: new THREE.Vector3(0, 0, -1),
    life: 0,
    maxLife: 0.3,
    damage: 60,
    active: false,
    length,
  };
}

function createMissile() {
  const group = new THREE.Group();
  // Body
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.08, 0.6, 6),
    new THREE.MeshStandardMaterial({ color: 0xff8800, emissive: 0xff4400, emissiveIntensity: 0.5 })
  );
  body.rotation.x = Math.PI / 2;
  group.add(body);
  // Exhaust light
  const light = new THREE.PointLight(0xff4400, 2, 5);
  light.position.set(0, 0, 0.35);
  group.add(light);
  group.visible = false;

  return {
    mesh: group,
    velocity: new THREE.Vector3(),
    life: 0,
    maxLife: 5,
    damage: 80,
    active: false,
    target: null,     // THREE.Vector3 or enemy ref
    speed: 120,
    turnRate: 3.0,
  };
}

/* ══════════════════════════════════════════════════════════
   WeaponSystem — manages player weapons
   ══════════════════════════════════════════════════════════ */
export class WeaponSystem {
  constructor(scene) {
    this.scene = scene;

    // ── Primary cannon settings ──
    this.fireRate      = 0.1;   // 10 rounds/sec
    this.bulletSpeed   = 200;
    this._fireTimer    = 0;

    // ── Missile settings ──
    this.maxMissiles      = 8;
    this.missileCount     = 8;
    this.missileRecharge  = 15;
    this._missileTimer    = 0;

    // ── Plasma settings ──
    this.plasmaCooldown  = 1.5;
    this._plasmaTimer    = 0;

    // ── Object pools ──
    this.bulletPool = new ObjectPool(
      () => createBullet(),
      (b) => { b.active = false; b.mesh.visible = false; b.life = 0; },
      50
    );
    this.missilePool = new ObjectPool(
      () => createMissile(),
      (m) => { m.active = false; m.mesh.visible = false; m.life = 0; m.target = null; },
      16
    );
    this.plasmaPool = new ObjectPool(
      () => createPlasma(),
      (p) => { p.active = false; p.mesh.visible = false; p.life = 0; },
      4
    );

    // Tracking
    this.isFiring = false;
  }

  /* ── Fire primary cannon ── */
  firePrimary(origin, direction) {
    if (this._fireTimer > 0) return false;
    this._fireTimer = this.fireRate;

    const bullet = this.bulletPool.get();
    if (!bullet) return false;

    bullet.mesh.position.copy(origin).addScaledVector(direction, 1.5);
    bullet.velocity.copy(direction).multiplyScalar(this.bulletSpeed);
    bullet.life     = 0;
    bullet.active   = true;
    bullet.mesh.visible = true;
    if (!bullet.mesh.parent) this.scene.add(bullet.mesh);

    this.isFiring = true;
    return true;
  }

  /* ── Fire plasma beam ── */
  firePlasma(origin, direction) {
    if (this._plasmaTimer > 0) return false;
    this._plasmaTimer = this.plasmaCooldown;

    const plasma = this.plasmaPool.get();
    if (!plasma) return false;

    plasma.mesh.position.copy(origin);
    plasma.velocity.copy(direction).multiplyScalar(200);
    plasma.life = 0;
    plasma.active = true;
    plasma.mesh.visible = true;

    // Orient beam along direction
    const lookAt = plasma.mesh.position.clone().add(direction);
    plasma.mesh.lookAt(lookAt);
    plasma.mesh.rotateX(Math.PI / 2);

    if (!plasma.mesh.parent) this.scene.add(plasma.mesh);

    return true;
  }

  /* ── Fire homing missile ── */
  fireMissile(origin, direction, enemies) {
    if (this.missileCount <= 0) return false;

    const missile = this.missilePool.get();
    if (!missile) return false;

    this.missileCount--;

    missile.mesh.position.copy(origin).addScaledVector(direction, 2);
    missile.velocity.copy(direction).multiplyScalar(missile.speed);
    missile.life   = 0;
    missile.active = true;
    missile.mesh.visible = true;
    if (!missile.mesh.parent) this.scene.add(missile.mesh);

    // Find nearest enemy for homing
    let nearest = null;
    let minDist = Infinity;
    for (const enemy of enemies) {
      const d = enemy.position.distanceTo(origin);
      if (d < minDist) {
        minDist = d;
        nearest = enemy;
      }
    }
    missile.target = nearest;

    return true;
  }

  /* ── Update all projectiles ── */
  update(delta) {
    this._fireTimer -= delta;
    this.isFiring = false;
    this._plasmaTimer -= delta;
    if (this.missileCount < this.maxMissiles) {
      this._missileTimer += delta;
      if (this._missileTimer >= this.missileRecharge) {
        this._missileTimer = 0;
        this.missileCount = Math.min(this.missileCount + 1, this.maxMissiles);
      }
    }

    // Update bullets
    for (const bullet of [...this.bulletPool.active]) {
      bullet.life += delta;
      if (bullet.life >= bullet.maxLife) {
        this.bulletPool.release(bullet);
        continue;
      }
      bullet.mesh.position.addScaledVector(bullet.velocity, delta);
    }

    // Update missiles (homing)
    for (const missile of [...this.missilePool.active]) {
      missile.life += delta;
      if (missile.life >= missile.maxLife) {
        this.missilePool.release(missile);
        continue;
      }
      if (missile.target && missile.target.alive) {
        const toTarget = new THREE.Vector3()
          .subVectors(missile.target.position, missile.mesh.position)
          .normalize();
        const currentDir = missile.velocity.clone().normalize();
        currentDir.lerp(toTarget, missile.turnRate * delta);
        currentDir.normalize();
        missile.velocity.copy(currentDir).multiplyScalar(missile.speed);
      }
      missile.mesh.position.addScaledVector(missile.velocity, delta);
      const lookAt = missile.mesh.position.clone().add(missile.velocity);
      missile.mesh.lookAt(lookAt);
    }

    // Update plasma beams
    for (const plasma of [...this.plasmaPool.active]) {
      plasma.life += delta;
      if (plasma.life >= plasma.maxLife) {
        this.plasmaPool.release(plasma);
        continue;
      }
      plasma.mesh.position.addScaledVector(plasma.velocity, delta);
      // Fade beam
      const fade = 1 - (plasma.life / plasma.maxLife);
      plasma.mesh.material.opacity = 0.8 * fade;
      plasma.mesh.children[0].material.opacity = 0.3 * fade;
      plasma.tipLight.intensity = 3 * fade;
    }
  }

  /* ── Getters ── */
  getPlasma() {
    return [...this.plasmaPool.active];
  }

  /* ── Getters ── */
  getBullets() {
    return [...this.bulletPool.active];
  }

  getMissiles() {
    return [...this.missilePool.active];
  }

  getAllProjectiles() {
    return [...this.bulletPool.active, ...this.missilePool.active, ...this.plasmaPool.active];
  }

  /* ── Remove a specific projectile ── */
  removePlasma(plasma) {
    this.plasmaPool.release(plasma);
  }

  /* ── Remove a specific projectile ── */
  removeBullet(bullet) {
    this.bulletPool.release(bullet);
  }

  removeMissile(missile) {
    this.missilePool.release(missile);
  }

  removeProjectile(proj) {
    if (this.bulletPool.active.has(proj)) {
      this.bulletPool.release(proj);
    } else if (this.missilePool.active.has(proj)) {
      this.missilePool.release(proj);
    } else if (this.plasmaPool.active.has(proj)) {
      this.plasmaPool.release(proj);
    }
  }

  /* ── Reset ── */
  reset() {
    this.bulletPool.releaseAll();
    this.missilePool.releaseAll();
    this.plasmaPool.releaseAll();
    this.missileCount  = this.maxMissiles;
    this._missileTimer = 0;
    this._fireTimer    = 0;
    this._plasmaTimer  = 0;
  }
}
