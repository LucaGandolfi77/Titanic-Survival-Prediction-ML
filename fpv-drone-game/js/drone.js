/* ── js/drone.js ── Drone flight physics & FPV camera ── */

import { MathUtils } from './utils.js';

const THREE = globalThis.THREE;
const _v  = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _euler = new THREE.Euler(0, 0, 0, 'YXZ');
const _quat  = new THREE.Quaternion();

/* ── Simple low-poly drone mesh ── */
function buildDroneMesh() {
  const group = new THREE.Group();

  // Body
  const bodyGeo = new THREE.BoxGeometry(1.2, 0.3, 0.8);
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.6, roughness: 0.4 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  group.add(body);

  // Arms (4)
  const armGeo = new THREE.BoxGeometry(0.1, 0.08, 0.7);
  const armMat = new THREE.MeshStandardMaterial({ color: 0x333333 });
  const offsets = [
    { x:  0.5, z:  0.35, rY: 0.3 },
    { x: -0.5, z:  0.35, rY: -0.3 },
    { x:  0.5, z: -0.35, rY: -0.3 },
    { x: -0.5, z: -0.35, rY: 0.3 },
  ];
  offsets.forEach(o => {
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.position.set(o.x, 0, o.z);
    arm.rotation.y = o.rY;
    group.add(arm);
  });

  // Rotors (4 discs)
  const rotorGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.02, 16);
  const rotorMat = new THREE.MeshStandardMaterial({ color: 0x00ff41, transparent: true, opacity: 0.5 });
  const rotorPositions = [
    { x:  0.55, z:  0.55 },
    { x: -0.55, z:  0.55 },
    { x:  0.55, z: -0.55 },
    { x: -0.55, z: -0.55 },
  ];
  const rotors = [];
  rotorPositions.forEach(p => {
    const rotor = new THREE.Mesh(rotorGeo, rotorMat);
    rotor.position.set(p.x, 0.18, p.z);
    rotors.push(rotor);
    group.add(rotor);
  });

  // FPV camera housing
  const camGeo = new THREE.BoxGeometry(0.15, 0.12, 0.12);
  const camMat = new THREE.MeshStandardMaterial({ color: 0xff3333 });
  const cam = new THREE.Mesh(camGeo, camMat);
  cam.position.set(0, 0.12, -0.45);
  group.add(cam);

  group.castShadow = true;
  group.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });

  return { group, rotors };
}

/* ══════════════════════════════════════════════════════════
   Drone class — arcade flight physics + FPV camera
   ══════════════════════════════════════════════════════════ */
export class Drone {
  constructor(scene, camera) {
    this.scene  = scene;
    this.camera = camera;

    // Build mesh
    const { group, rotors } = buildDroneMesh();
    this.mesh   = group;
    this.rotors = rotors;
    scene.add(this.mesh);

    // ── Physics constants ──
    this.maxThrust   = 25;
    this.mass        = 1.5;
    this.gravity     = 9.8;
    this.dragFactor  = 0.92;   // per-frame at 60 fps
    this.angDrag     = 0.85;
    this.maxSpeed    = 60;
    this.boostMul    = 1.6;
    this.pitchMax    = Math.PI / 4;   // ±45°
    this.rollMax     = Math.PI / 4;
    this.yawRate     = 3.0;           // rad/s at full deflection
    this.pitchRate   = 4.0;
    this.rollRate    = 5.0;
    this.minAlt      = 0.5;
    this.maxAlt      = 300;

    // ── State ──
    this.position    = new THREE.Vector3(0, 20, 0);
    this.velocity    = new THREE.Vector3();
    this.yaw         = 0;
    this.pitch       = 0;   // current pitch angle
    this.roll        = 0;   // current roll angle
    this.throttle    = 0;   // 0-1

    this.health      = 100;
    this.maxHealth   = 100;
    this.isAlive     = true;
    this.invincible  = 0;   // seconds of invincibility after hit

    this.speed       = 0;   // scalar speed (for HUD)
    this.altitude    = 20;

    // Engine glow
    this.engineLight = new THREE.PointLight(0x00ff41, 0, 3);
    this.engineLight.position.set(0, -0.2, 0);
    this.mesh.add(this.engineLight);
  }

  /* ── Reset to start position ── */
  reset(pos) {
    this.position.copy(pos || new THREE.Vector3(0, 20, 0));
    this.velocity.set(0, 0, 0);
    this.yaw      = 0;
    this.pitch    = 0;
    this.roll     = 0;
    this.throttle = 0;
    this.health   = this.maxHealth;
    this.isAlive  = true;
    this.invincible = 0;
    this._syncTransform(1);
  }

  /* ── Take damage ── */
  takeDamage(amount) {
    if (!this.isAlive || this.invincible > 0) return;
    this.health = Math.max(0, this.health - amount);
    this.invincible = 0.25;       // brief invincibility
    if (this.health <= 0) {
      this.isAlive = false;
    }
  }

  /* ── Main update ── */
  update(delta, input) {
    if (!this.isAlive) return;

    // Invincibility countdown
    if (this.invincible > 0) this.invincible -= delta;

    // ── Throttle ──
    const throttleRate = 1.5; // units/sec
    this.throttle = MathUtils.clamp(
      this.throttle + input.throttle * throttleRate * delta, 0, 1
    );

    const boost = input.boost ? this.boostMul : 1;

    // ── Rotation ──
    // Yaw — accumulate
    this.yaw += input.yaw * this.yawRate * delta;
    this.yaw = MathUtils.normalizeAngle(this.yaw);

    // Pitch — lerp toward target angle
    const targetPitch = input.pitch * this.pitchMax;
    this.pitch = MathUtils.lerp(this.pitch, targetPitch, 1 - Math.pow(0.05, delta));

    // Roll — lerp toward target angle
    const targetRoll = -input.roll * this.rollMax;
    this.roll = MathUtils.lerp(this.roll, targetRoll, 1 - Math.pow(0.05, delta));

    // ── Forces ──
    const force = _v.set(0, 0, 0);

    // Gravity
    force.y -= this.gravity * this.mass;

    // Lift (upward, modulated by pitch)
    const lift = this.throttle * this.maxThrust * boost * (1 + Math.abs(this.pitch) * 0.2);
    force.y += lift;

    // Forward thrust — in drone's facing direction
    const fwd = _v2.set(0, 0, -1).applyAxisAngle(
      new THREE.Vector3(0, 1, 0), this.yaw
    );
    const fwdForce = this.throttle * this.maxThrust * boost * Math.sin(this.pitch);
    force.addScaledVector(fwd, -fwdForce);  // negative because pitch-down = forward

    // Lateral thrust from roll
    const right = new THREE.Vector3(1, 0, 0).applyAxisAngle(
      new THREE.Vector3(0, 1, 0), this.yaw
    );
    const latForce = this.throttle * this.maxThrust * boost * Math.sin(this.roll) * 0.7;
    force.addScaledVector(right, -latForce);

    // ── Integrate velocity ──
    const accel = force.divideScalar(this.mass);
    this.velocity.addScaledVector(accel, delta);

    // Drag (frame-rate independent)
    const drag = Math.pow(this.dragFactor, delta * 60);
    this.velocity.multiplyScalar(drag);

    // Clamp speed
    const spd = this.velocity.length();
    if (spd > this.maxSpeed * boost) {
      this.velocity.multiplyScalar((this.maxSpeed * boost) / spd);
    }

    // ── Integrate position ──
    this.position.addScaledVector(this.velocity, delta);

    // Altitude clamp
    if (this.position.y < this.minAlt) {
      this.position.y = this.minAlt;
      if (this.velocity.y < 0) this.velocity.y *= -0.3; // bounce
    }
    if (this.position.y > this.maxAlt) {
      this.position.y = this.maxAlt;
      if (this.velocity.y > 0) this.velocity.y = 0;
    }

    // World bounds (keep inside ±950)
    this.position.x = MathUtils.clamp(this.position.x, -950, 950);
    this.position.z = MathUtils.clamp(this.position.z, -950, 950);

    // Derived values for HUD
    this.speed    = this.velocity.length();
    this.altitude = this.position.y;

    // ── Sync mesh & camera ──
    this._syncTransform(delta);
  }

  /* ── Sync mesh & FPV camera ── */
  _syncTransform(delta) {
    // Mesh
    this.mesh.position.copy(this.position);
    _euler.set(this.pitch, this.yaw, this.roll, 'YXZ');
    this.mesh.rotation.copy(_euler);

    // Spin rotors based on throttle
    const rotorSpeed = this.throttle * 40 + 5;
    this.rotors.forEach((r, i) => {
      r.rotation.y += rotorSpeed * delta * (i % 2 === 0 ? 1 : -1);
    });

    // Engine glow
    this.engineLight.intensity = this.throttle * 2;

    // ── FPV Camera ──
    // Target position: move target to the front of the drone so it's not inside the body
    const camOffset = new THREE.Vector3(0, 0.15, -0.4);
    camOffset.applyEuler(_euler);
    const targetPos = this.position.clone().add(camOffset);

    // Apply fixed FPV camera uptilt (20 degrees) so players can see forward while pitched down
    const cameraUptilt = 20 * (Math.PI / 180);
    _euler.set(this.pitch + cameraUptilt, this.yaw, this.roll, 'YXZ');
    _quat.setFromEuler(_euler);

    // Snap camera directly to drone to avoid FPV lag/nausea
    this.camera.position.copy(targetPos);
    this.camera.quaternion.copy(_quat);
  }

  /* ── Get forward direction (for weapons) ── */
  getForward() {
    return this.camera.getWorldDirection(new THREE.Vector3());
  }

  /* ── Push drone away from collision ── */
  pushOut(normal, depth) {
    this.position.addScaledVector(normal, depth + 0.1);
    // Kill velocity in the collision normal direction
    const dot = this.velocity.dot(normal);
    if (dot < 0) {
      this.velocity.addScaledVector(normal, -dot);
    }
  }
}
