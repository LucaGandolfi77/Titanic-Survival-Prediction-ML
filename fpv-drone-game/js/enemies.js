/* ── js/enemies.js ── Enemy types, AI, wave system ── */

import { ObjectPool, MathUtils } from './utils.js';

const THREE = globalThis.THREE;

/* ── Helper: build simple enemy meshes ── */
function buildTurretMesh() {
  const g = new THREE.Group();
  // Base
  const base = new THREE.Mesh(
    new THREE.CylinderGeometry(1.2, 1.4, 1.5, 8),
    new THREE.MeshStandardMaterial({ color: 0x884422, metalness: 0.5, roughness: 0.6 })
  );
  g.add(base);
  // Barrel
  const barrel = new THREE.Mesh(
    new THREE.CylinderGeometry(0.15, 0.15, 2, 6),
    new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.7 })
  );
  barrel.rotation.x = Math.PI / 2;
  barrel.position.set(0, 0.5, -1.2);
  barrel.name = 'barrel';
  g.add(barrel);
  g.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
  return g;
}

function buildPatrolDroneMesh() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(0.7, 8, 6),
    new THREE.MeshStandardMaterial({ color: 0xcc3333, metalness: 0.4, roughness: 0.5 })
  );
  g.add(body);
  // Wings
  [-1, 1].forEach(s => {
    const wing = new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 0.08, 0.5),
      new THREE.MeshStandardMaterial({ color: 0x992222 })
    );
    wing.position.set(s * 1.1, 0, 0);
    g.add(wing);
  });
  g.traverse(c => { if (c.isMesh) c.castShadow = true; });
  return g;
}

function buildGunshipMesh() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(3, 1.2, 2),
    new THREE.MeshStandardMaterial({ color: 0x2244aa, metalness: 0.5, roughness: 0.4 })
  );
  g.add(body);
  const cockpit = new THREE.Mesh(
    new THREE.SphereGeometry(0.6, 8, 6),
    new THREE.MeshStandardMaterial({ color: 0x66aaff, transparent: true, opacity: 0.6 })
  );
  cockpit.position.set(0, 0.4, -0.9);
  g.add(cockpit);
  // Guns
  [-1, 1].forEach(s => {
    const gun = new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.1, 1.2, 6),
      new THREE.MeshStandardMaterial({ color: 0x333333 })
    );
    gun.rotation.x = Math.PI / 2;
    gun.position.set(s * 1.2, -0.3, -1.3);
    g.add(gun);
  });
  g.traverse(c => { if (c.isMesh) c.castShadow = true; });
  return g;
}

/* ── Enemy projectile mesh (reused from pool) ── */
function createEnemyProjectile() {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 4, 4),
    new THREE.MeshBasicMaterial({ color: 0xff4444 })
  );
  mesh.visible = false;
  return {
    mesh,
    velocity: new THREE.Vector3(),
    life: 0,
    maxLife: 3,
    damage: 8,
    active: false,
  };
}

/* ══════════════════════════════════════════════════════════
   Enemy base class
   ══════════════════════════════════════════════════════════ */
class Enemy {
  constructor(mesh, hp, scoreValue, radius) {
    this.mesh       = mesh;
    this.hp         = hp;
    this.maxHp      = hp;
    this.scoreValue = scoreValue;
    this.radius     = radius;
    this.alive      = true;
    this.position   = mesh.position;
    this.fireTimer  = 0;
  }

  takeDamage(amount) {
    this.hp -= amount;
    // Flash red
    this.mesh.traverse(c => {
      if (c.isMesh && c.material) {
        c.material.emissive = new THREE.Color(0xff0000);
        c.material.emissiveIntensity = 0.8;
      }
    });
    setTimeout(() => {
      this.mesh.traverse(c => {
        if (c.isMesh && c.material) {
          c.material.emissiveIntensity = 0;
        }
      });
    }, 100);
    if (this.hp <= 0) {
      this.alive = false;
    }
  }

  dispose(scene) {
    scene.remove(this.mesh);
  }
}

/* ── Turret — static, tracks & fires ── */
class Turret extends Enemy {
  constructor(scene, pos) {
    const mesh = buildTurretMesh();
    mesh.position.copy(pos);
    scene.add(mesh);
    super(mesh, 60, 50, 1.5);
    this.type       = 'turret';
    this.fireRate   = 2.5;  // seconds between shots
    this.range      = 150;
    this.projSpeed  = 80;
  }

  update(delta, dronePos, projectilePool, scene) {
    if (!this.alive) return;
    const dir = new THREE.Vector3().subVectors(dronePos, this.position);
    const dist = dir.length();

    // Rotate to face player
    if (dist < this.range) {
      const angle = Math.atan2(dir.x, dir.z);
      this.mesh.rotation.y = MathUtils.lerp(
        this.mesh.rotation.y, angle, 1 - Math.pow(0.05, delta)
      );
    }

    // Fire
    this.fireTimer -= delta;
    if (this.fireTimer <= 0 && dist < this.range) {
      this.fireTimer = this.fireRate;
      this._fire(dir.normalize(), projectilePool, scene);
    }
  }

  _fire(direction, pool, scene) {
    const proj = pool.get();
    if (!proj) return;
    proj.mesh.position.copy(this.position).add(new THREE.Vector3(0, 0.5, 0));
    proj.velocity.copy(direction).multiplyScalar(this.projSpeed);
    proj.life = 0;
    proj.active = true;
    proj.mesh.visible = true;
    if (!proj.mesh.parent) scene.add(proj.mesh);
  }
}

/* ── PatrolDrone — waypoints, chase, fire ── */
class PatrolDrone extends Enemy {
  constructor(scene, pos) {
    const mesh = buildPatrolDroneMesh();
    mesh.position.copy(pos);
    scene.add(mesh);
    super(mesh, 40, 75, 1.0);
    this.type       = 'patrol';
    this.fireRate   = 1.5;
    this.chaseRange = 80;
    this.fireRange  = 40;
    this.moveSpeed  = 18;
    this.projSpeed  = 100;

    // Generate random patrol waypoints
    this.waypoints = [];
    for (let i = 0; i < 4; i++) {
      this.waypoints.push(new THREE.Vector3(
        pos.x + MathUtils.randomRange(-60, 60),
        MathUtils.randomRange(15, 50),
        pos.z + MathUtils.randomRange(-60, 60)
      ));
    }
    this.wpIndex = 0;
    this.state   = 'patrol'; // 'patrol' | 'chase'
    this._bobOffset = Math.random() * Math.PI * 2;
  }

  update(delta, dronePos, projectilePool, scene) {
    if (!this.alive) return;

    const dir = new THREE.Vector3().subVectors(dronePos, this.position);
    const dist = dir.length();

    if (dist < this.chaseRange) {
      this.state = 'chase';
    } else {
      this.state = 'patrol';
    }

    if (this.state === 'chase') {
      // Move toward player (but stop at fireRange)
      if (dist > this.fireRange * 0.8) {
        const moveDir = dir.clone().normalize();
        this.position.addScaledVector(moveDir, this.moveSpeed * delta);
      }
      // Look at player
      this.mesh.lookAt(dronePos);
    } else {
      // Patrol — move toward current waypoint
      const wp = this.waypoints[this.wpIndex];
      const toWp = new THREE.Vector3().subVectors(wp, this.position);
      if (toWp.length() < 3) {
        this.wpIndex = (this.wpIndex + 1) % this.waypoints.length;
      } else {
        this.position.addScaledVector(toWp.normalize(), this.moveSpeed * 0.5 * delta);
      }
      this.mesh.lookAt(wp);
    }

    // Gentle bob
    this.position.y += Math.sin(performance.now() * 0.002 + this._bobOffset) * 0.01;

    // Fire
    this.fireTimer -= delta;
    if (this.fireTimer <= 0 && dist < this.fireRange) {
      this.fireTimer = this.fireRate;
      this._fire(dir.normalize(), projectilePool, scene);
    }
  }

  _fire(direction, pool, scene) {
    const proj = pool.get();
    if (!proj) return;
    proj.mesh.position.copy(this.position);
    proj.velocity.copy(direction).multiplyScalar(this.projSpeed);
    proj.life = 0;
    proj.active = true;
    proj.mesh.visible = true;
    if (!proj.mesh.parent) scene.add(proj.mesh);
  }
}

/* ── Gunship — orbits, burst fire, dash ── */
class Gunship extends Enemy {
  constructor(scene, pos) {
    const mesh = buildGunshipMesh();
    mesh.position.copy(pos);
    scene.add(mesh);
    super(mesh, 250, 200, 2.5);
    this.type        = 'gunship';
    this.fireRate    = 0.3;    // between burst shots
    this.burstCount  = 5;
    this.burstPause  = 4;     // seconds between bursts
    this._burstLeft  = 0;
    this._burstTimer = 2;
    this.orbitRadius = 60;
    this.orbitSpeed  = 0.4;
    this.orbitAngle  = Math.random() * Math.PI * 2;
    this.orbitCenter = pos.clone();
    this.moveSpeed   = 12;
    this.projSpeed   = 90;
    this.dashTimer   = 8;
    this.dashing     = false;
    this.dashVel     = new THREE.Vector3();
  }

  update(delta, dronePos, projectilePool, scene) {
    if (!this.alive) return;

    const dir = new THREE.Vector3().subVectors(dronePos, this.position);
    const dist = dir.length();

    // Recenter orbit on player roughly
    this.orbitCenter.lerp(dronePos, 0.01);

    // Orbit
    if (!this.dashing) {
      this.orbitAngle += this.orbitSpeed * delta;
      const targetX = this.orbitCenter.x + Math.cos(this.orbitAngle) * this.orbitRadius;
      const targetZ = this.orbitCenter.z + Math.sin(this.orbitAngle) * this.orbitRadius;
      const targetY = MathUtils.clamp(dronePos.y + 10, 20, 100);
      const target = new THREE.Vector3(targetX, targetY, targetZ);
      this.position.lerp(target, 1 - Math.pow(0.1, delta));
    } else {
      // Dash toward player
      this.position.addScaledVector(this.dashVel, delta);
      this.dashTimer -= delta;
      if (this.dashTimer <= 0) {
        this.dashing = false;
        this.dashTimer = MathUtils.randomRange(6, 12);
      }
    }

    // Look at player
    this.mesh.lookAt(dronePos);

    // Dash trigger
    if (!this.dashing) {
      this.dashTimer -= delta;
      if (this.dashTimer <= 0 && dist > 30) {
        this.dashing = true;
        this.dashVel = dir.normalize().multiplyScalar(35);
        this.dashTimer = 1.5; // dash duration
      }
    }

    // Burst fire
    this._burstTimer -= delta;
    if (this._burstTimer <= 0) {
      if (this._burstLeft <= 0) {
        this._burstLeft = this.burstCount;
      }
      if (this._burstLeft > 0) {
        this.fireTimer -= delta;
        if (this.fireTimer <= 0) {
          this.fireTimer = this.fireRate;
          this._burstLeft--;
          if (this._burstLeft <= 0) this._burstTimer = this.burstPause;
          this._fire(dir.clone().normalize(), projectilePool, scene);
        }
      }
    }
  }

  _fire(direction, pool, scene) {
    const proj = pool.get();
    if (!proj) return;
    proj.mesh.position.copy(this.position);
    proj.velocity.copy(direction).multiplyScalar(this.projSpeed);
    proj.life = 0;
    proj.active = true;
    proj.mesh.visible = true;
    if (!proj.mesh.parent) scene.add(proj.mesh);
  }
}

/* ══════════════════════════════════════════════════════════
   EnemyManager — wave system + update loop
   ══════════════════════════════════════════════════════════ */
export class EnemyManager {
  constructor(scene) {
    this.scene    = scene;
    this.enemies  = [];
    this.wave     = 0;

    // Enemy projectile pool (pre-allocate 100)
    this.projectilePool = new ObjectPool(
      () => createEnemyProjectile(),
      (p) => { p.active = false; p.mesh.visible = false; p.life = 0; },
      100
    );
  }

  /* ── Wave definitions ── */
  static WAVES = [
    // wave 1
    { turrets: 4, patrols: 0, gunships: 0 },
    // wave 2
    { turrets: 3, patrols: 2, gunships: 0 },
    // wave 3
    { turrets: 2, patrols: 4, gunships: 0 },
    // wave 4
    { turrets: 1, patrols: 5, gunships: 0 },
    // wave 5
    { turrets: 2, patrols: 4, gunships: 1 },
  ];

  getWaveDef(waveNum) {
    if (waveNum <= EnemyManager.WAVES.length) {
      return EnemyManager.WAVES[waveNum - 1];
    }
    // Scale beyond defined waves
    const base = EnemyManager.WAVES[EnemyManager.WAVES.length - 1];
    const scale = 1 + (waveNum - EnemyManager.WAVES.length) * 0.3;
    return {
      turrets:  Math.floor(base.turrets * scale),
      patrols:  Math.floor(base.patrols * scale),
      gunships: Math.floor(base.gunships * scale),
    };
  }

  /* ── Spawn a wave ── */
  spawnWave(waveNum, dronePos) {
    this.wave = waveNum;
    const def = this.getWaveDef(waveNum);

    // Clear old enemies
    this.clearAll();

    const spawnPos = () => {
      const angle = Math.random() * Math.PI * 2;
      const dist  = MathUtils.randomRange(100, 250);
      return new THREE.Vector3(
        MathUtils.clamp(dronePos.x + Math.cos(angle) * dist, -900, 900),
        MathUtils.randomRange(1, 5),
        MathUtils.clamp(dronePos.z + Math.sin(angle) * dist, -900, 900)
      );
    };

    const spawnAirPos = () => {
      const p = spawnPos();
      p.y = MathUtils.randomRange(20, 60);
      return p;
    };

    for (let i = 0; i < def.turrets; i++) {
      this.enemies.push(new Turret(this.scene, spawnPos()));
    }
    for (let i = 0; i < def.patrols; i++) {
      this.enemies.push(new PatrolDrone(this.scene, spawnAirPos()));
    }
    for (let i = 0; i < def.gunships; i++) {
      this.enemies.push(new Gunship(this.scene, spawnAirPos()));
    }
  }

  /* ── Update all enemies ── */
  update(delta, dronePos) {
    // Update enemies
    for (const enemy of this.enemies) {
      if (enemy.alive) {
        enemy.update(delta, dronePos, this.projectilePool, this.scene);
      }
    }

    // Update enemy projectiles (copy Set to avoid mutation during iteration)
    for (const proj of [...this.projectilePool.active]) {
      proj.life += delta;
      if (proj.life >= proj.maxLife) {
        this.projectilePool.release(proj);
        continue;
      }
      proj.mesh.position.addScaledVector(proj.velocity, delta);
    }
  }

  /* ── Getters ── */
  get aliveCount() {
    return this.enemies.filter(e => e.alive).length;
  }

  get allEnemies() {
    return this.enemies.filter(e => e.alive);
  }

  getEnemyProjectiles() {
    return [...this.projectilePool.active];
  }

  /* ── Remove enemy (after death anim) ── */
  removeEnemy(enemy) {
    enemy.dispose(this.scene);
    const idx = this.enemies.indexOf(enemy);
    if (idx >= 0) this.enemies.splice(idx, 1);
  }

  /* ── Clear all ── */
  clearAll() {
    for (const e of this.enemies) {
      e.dispose(this.scene);
    }
    this.enemies.length = 0;
    this.projectilePool.releaseAll();
  }
}
