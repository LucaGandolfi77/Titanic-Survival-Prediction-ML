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
    this.missileRecharge  = 15;  // seconds per missile
    this._missileTimer    = 0;

    // ── Object pools (pre-allocate) ──
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

    // Missile recharge
    if (this.missileCount < this.maxMissiles) {
      this._missileTimer += delta;
      if (this._missileTimer >= this.missileRecharge) {
        this._missileTimer = 0;
        this.missileCount = Math.min(this.missileCount + 1, this.maxMissiles);
      }
    }

    // Update bullets (copy Set to allow safe mutation)
    for (const bullet of [...this.bulletPool.active]) {
      bullet.life += delta;
      if (bullet.life >= bullet.maxLife) {
        this.bulletPool.release(bullet);
        continue;
      }
      bullet.mesh.position.addScaledVector(bullet.velocity, delta);
    }

    // Update missiles (homing) — copy Set
    for (const missile of [...this.missilePool.active]) {
      missile.life += delta;
      if (missile.life >= missile.maxLife) {
        this.missilePool.release(missile);
        continue;
      }

      // Homing logic
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

      // Orient missile along velocity
      const lookAt = missile.mesh.position.clone().add(missile.velocity);
      missile.mesh.lookAt(lookAt);
    }
  }

  /* ── Getters ── */
  getBullets() {
    return [...this.bulletPool.active];
  }

  getMissiles() {
    return [...this.missilePool.active];
  }

  getAllProjectiles() {
    return [...this.bulletPool.active, ...this.missilePool.active];
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
    }
  }

  /* ── Reset ── */
  reset() {
    this.bulletPool.releaseAll();
    this.missilePool.releaseAll();
    this.missileCount  = this.maxMissiles;
    this._missileTimer = 0;
    this._fireTimer    = 0;
  }
}
