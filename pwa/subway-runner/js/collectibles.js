import * as THREE from 'three';
import { createSphere, createBox } from './utils.js';
import { LANE_WIDTH } from './track.js';

const POWERUP_TYPES = {
  MAGNET: 'magnet',
  MULTIPLIER: 'multiplier',
  SHIELD: 'shield',
  JETPACK: 'jetpack'
};

const POWERUP_COLORS = {
  magnet: 0xff1744,
  multiplier: 0xffd600,
  shield: 0x00e5ff,
  jetpack: 0xd500f9
};

const POWERUP_ICONS = {
  magnet: 'MAGNET',
  multiplier: 'x2',
  shield: 'SHIELD',
  jetpack: 'JET'
};

export class CollectibleManager {
  constructor(scene) {
    this.scene = scene;
    this.coins = [];
    this.powerups = [];
    this.activePowerups = {};
    this.lastSpawnZ = 0;
    this.coinValue = 1;
    this.magnetActive = false;
    this.magnetTimer = 0;
    this.nextSpawnZ = -20;
  }

  createCoin(laneX, y, z) {
    const group = new THREE.Group();
    const coinGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.08, 16);
    const coinMat = new THREE.MeshStandardMaterial({
      color: 0xffd600,
      emissive: 0xff8f00,
      emissiveIntensity: 0.3,
      metalness: 0.8,
      roughness: 0.2
    });
    const coinMesh = new THREE.Mesh(coinGeo, coinMat);
    coinMesh.rotation.x = Math.PI / 2;
    coinMesh.castShadow = true;
    group.add(coinMesh);

    const innerGeo = new THREE.CylinderGeometry(0.15, 0.15, 0.09, 8);
    const innerMat = new THREE.MeshStandardMaterial({ color: 0xff8f00, metalness: 0.9, roughness: 0.1 });
    const inner = new THREE.Mesh(innerGeo, innerMat);
    inner.rotation.x = Math.PI / 2;
    group.add(inner);

    group.position.set(laneX, y, z);
    group.userData.collected = false;
    group.userData.isCoin = true;
    this.scene.add(group);
    this.coins.push(group);
    return group;
  }

  createPowerup(type, laneX, y, z) {
    const group = new THREE.Group();
    const color = POWERUP_COLORS[type];

    const outerGeo = new THREE.IcosahedronGeometry(0.5, 0);
    const outerMat = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0.8,
      wireframe: true
    });
    const outer = new THREE.Mesh(outerGeo, outerMat);
    group.add(outer);

    const innerGeo = new THREE.IcosahedronGeometry(0.25, 0);
    const innerMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: color,
      emissiveIntensity: 0.8
    });
    const inner = new THREE.Mesh(innerGeo, innerMat);
    group.add(inner);

    group.position.set(laneX, y + 0.5, z);
    group.userData.type = type;
    group.userData.collected = false;
    group.userData.isPowerup = true;
    group.userData.icon = POWERUP_ICONS[type];
    this.scene.add(group);
    this.powerups.push(group);
    return group;
  }

  spawnCoins(laneX, startZ, count, spacing, y) {
    for (let i = 0; i < count; i++) {
      this.createCoin(laneX, y || 1.0, startZ - i * spacing);
    }
  }

  spawnArc(laneX, startZ) {
    const heights = [1.0, 1.8, 2.5, 2.5, 1.8, 1.0];
    for (let i = 0; i < heights.length; i++) {
      this.createCoin(laneX, heights[i], startZ - i * 1.5);
    }
  }

  spawnLine(startZ, count) {
    const lane = Math.floor(Math.random() * 3) - 1;
    const laneX = lane * LANE_WIDTH;
    this.spawnCoins(laneX, startZ, count, 1.5);
  }

  spawnPowerup(type, laneX, z) {
    this.createPowerup(type, laneX, 1.5, z);
  }

  update(playerZ, playerX, playerWorld, delta) {
    while (this.nextSpawnZ > playerZ - 200) {
      const gap = 10 + Math.random() * 15;
      this.nextSpawnZ -= gap;

      const spawnType = Math.random();
      if (spawnType < 0.15) {
        const types = Object.values(POWERUP_TYPES);
        const type = types[Math.floor(Math.random() * types.length)];
        const lane = Math.floor(Math.random() * 3) - 1;
        this.spawnPowerup(type, lane * LANE_WIDTH, this.nextSpawnZ);
      } else if (spawnType < 0.4) {
        this.spawnArc(Math.floor(Math.random() * 3 - 1) * LANE_WIDTH, this.nextSpawnZ);
      } else {
        this.spawnLine(this.nextSpawnZ, 3 + Math.floor(Math.random() * 5));
      }
    }

    if (this.magnetActive) {
      this.magnetTimer -= delta;
      if (this.magnetTimer <= 0) {
        this.magnetActive = false;
      }
    }

    const now = performance.now();
    for (const key in this.activePowerups) {
      if (now > this.activePowerups[key].endTime) {
        delete this.activePowerups[key];
      }
    }

    for (let i = this.coins.length - 1; i >= 0; i--) {
      const coin = this.coins[i];
      if (coin.position.z > playerZ + 30) {
        this.scene.remove(coin);
        coin.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
        this.coins.splice(i, 1);
        continue;
      }
      if (!coin.userData.collected) {
        coin.rotation.y += 3 * delta;
        coin.position.y = 1.0 + Math.sin(performance.now() * 0.003 + coin.position.z) * 0.15;

        if (this.magnetActive) {
          const dist = coin.position.distanceTo(playerWorld);
          if (dist < 6) {
            const dir = new THREE.Vector3().subVectors(playerWorld, coin.position).normalize();
            coin.position.addScaledVector(dir, 15 * delta);
          }
        }
      }
    }

    for (let i = this.powerups.length - 1; i >= 0; i--) {
      const pu = this.powerups[i];
      if (pu.position.z > playerZ + 30) {
        this.scene.remove(pu);
        pu.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
        this.powerups.splice(i, 1);
        continue;
      }
      pu.rotation.y += 2 * delta;
      pu.children.forEach(c => { c.rotation.x += 1.5 * delta; c.rotation.z += 1 * delta; });
    }
  }

  collectCoin(coin) {
    if (coin.userData.collected) return 0;
    coin.userData.collected = true;
    coin.visible = false;
    return this.coinValue;
  }

  collectPowerup(powerup) {
    if (powerup.userData.collected) return null;
    powerup.userData.collected = true;
    powerup.visible = false;
    const type = powerup.userData.type;
    const duration = this.getPowerupDuration(type);
    this.activePowerups[type] = {
      endTime: performance.now() + duration * 1000,
      duration
    };
    if (type === 'magnet') {
      this.magnetActive = true;
      this.magnetTimer = duration;
    }
    return type;
  }

  getPowerupDuration(type) {
    switch (type) {
      case 'magnet': return 8;
      case 'multiplier': return 10;
      case 'shield': return 10;
      case 'jetpack': return 5;
      default: return 5;
    }
  }

  hasPowerup(type) {
    return !!this.activePowerups[type];
  }

  getPowerupTimeLeft(type) {
    const pu = this.activePowerups[type];
    if (!pu) return 0;
    return Math.max(0, (pu.endTime - performance.now()) / 1000);
  }

  reset() {
    for (const coin of this.coins) {
      this.scene.remove(coin);
      coin.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
    }
    for (const pu of this.powerups) {
      this.scene.remove(pu);
      pu.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
    }
    this.coins = [];
    this.powerups = [];
    this.activePowerups = {};
    this.magnetActive = false;
    this.magnetTimer = 0;
    this.nextSpawnZ = -20;
  }
}

export { POWERUP_TYPES, POWERUP_ICONS };
