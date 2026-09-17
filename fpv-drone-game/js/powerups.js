/* ── js/powerups.js ── PowerUp system ── */

import * as THREE from 'three';

/* ── PowerUp type definitions ── */
const POWERUP_TYPES = {
  shield: {
    color: 0x4488ff,
    duration: 8,
    label: 'SHIELD',
    icon: '🛡️',
  },
  overdrive: {
    color: 0xffaa00,
    duration: 5,
    label: 'OVERDRIVE',
    icon: '⚡',
  },
  ghost: {
    color: 0xaa88ff,
    duration: 6,
    label: 'GHOST',
    icon: '👻',
  },
  nanoRegen: {
    color: 0x00ff41,
    duration: 10,
    label: 'NANO-REGEN',
    icon: '🔄',
  },
};

/* ── Create a power-up mesh ── */
function createPowerUpMesh(type, position) {
  const config = POWERUP_TYPES[type];
  const group = new THREE.Group();

  // Outer rotating ring
  const ringGeo = new THREE.TorusGeometry(1.2, 0.08, 8, 24);
  const ringMat = new THREE.MeshBasicMaterial({
    color: config.color,
    transparent: true,
    opacity: 0.7,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  group.add(ring);

  // Inner core
  const coreGeo = new THREE.OctahedronGeometry(0.5, 0);
  const coreMat = new THREE.MeshBasicMaterial({
    color: config.color,
    transparent: true,
    opacity: 0.9,
  });
  const core = new THREE.Mesh(coreGeo, coreMat);
  group.add(core);

  // Point light
  const light = new THREE.PointLight(config.color, 2, 10);
  group.add(light);

  group.position.copy(position);
  group.userData = {
    type,
    config,
    ring,
    core,
    light,
  };

  return group;
}

/* ══════════════════════════════════════════════════════════
   PowerUpManager — spawning, collection, effects
   ══════════════════════════════════════════════════════════ */
export class PowerUpManager {
  constructor(scene) {
    this.scene = scene;
    this.active = [];
    this.spawnTimer = 0;
    this.spawnInterval = 15;
  }

  /* ── Spawn a random power-up ── */
  spawn(position) {
    const types = Object.keys(POWERUP_TYPES);
    const type = types[Math.floor(Math.random() * types.length)];
    const mesh = createPowerUpMesh(type, position);
    this.scene.add(mesh);
    this.active.push(mesh);
  }

  /* ── Update (call every frame) ── */
  update(delta, dronePosition) {
    this.spawnTimer += delta;

    // Auto-spawn at random intervals
    if (this.spawnTimer >= this.spawnInterval) {
      this.spawnTimer = 0;
      this.spawn(new THREE.Vector3(
        (Math.random() - 0.5) * 800,
        Math.random() * 40 + 5,
        (Math.random() - 0.5) * 800
      ));
    }

    // Animate active power-ups
    for (let i = this.active.length - 1; i >= 0; i--) {
      const pu = this.active[i];
      const ud = pu.userData;

      // Bobbing
      pu.position.y += Math.sin(performance.now() * 0.003 + i) * 0.02;

      // Rotate ring
      ud.ring.rotation.z += delta * 2;

      // Spin core
      ud.core.rotation.y += delta * 3;
      ud.core.rotation.x += delta * 1.5;

      // Pulse light
      ud.light.intensity = 2 + Math.sin(performance.now() * 0.005) * 1;

      // Check collection
      const dist = pu.position.distanceTo(dronePosition);
      if (dist < 4) {
        this.collect(pu);
      }
    }
  }

  /* ── Collect power-up ── */
  collect(pu) {
    const type = pu.userData.type;
    this.scene.remove(pu);
    pu.userData.ring.geometry.dispose();
    pu.userData.ring.material.dispose();
    pu.userData.core.geometry.dispose();
    pu.userData.core.material.dispose();
    pu.userData.light.dispose();

    const idx = this.active.indexOf(pu);
    if (idx >= 0) this.active.splice(idx, 1);

    return type;
  }

  /* ── Clear all ── */
  clear() {
    for (const pu of this.active) {
      this.scene.remove(pu);
    }
    this.active.length = 0;
  }
}

export { POWERUP_TYPES };
