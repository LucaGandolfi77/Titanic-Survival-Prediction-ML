import * as THREE from 'three';
import { ObjectPool, createBox, createCylinder, createSphere } from './utils.js';
import { LANE_WIDTH, CHUNK_LENGTH } from './track.js';

const OBSTACLE_TYPES = {
  BARRIER_LOW: 'barrier_low',
  BARRIER_HIGH: 'barrier_high',
  TRAIN: 'train',
  SIGN: 'sign'
};

const COLORS = {
  barrierRed: 0xd32f2f,
  barrierYellow: 0xfbc02d,
  barrierStripe: 0x212121,
  trainBody: [0x1565c0, 0xc62828, 0x2e7d32, 0xf57f17, 0x6a1b9a],
  trainWindow: 0x90caf9,
  trainRoof: 0x616161,
  signPole: 0x757575,
  signBoard: 0xff8f00
};

export class ObstacleManager {
  constructor(scene) {
    this.scene = scene;
    this.obstacles = [];
    this.lastSpawnZ = 0;
    this.minGap = 12;
    this.maxGap = 25;
    this.nextSpawnZ = -30;
  }

  createBarrierLow(laneX, z) {
    const group = new THREE.Group();
    const bar = createBox(1.8, 0.8, 0.4, COLORS.barrierRed);
    bar.position.y = 0.4;
    bar.castShadow = true;
    group.add(bar);

    const stripe = createBox(1.8, 0.15, 0.42, COLORS.barrierYellow);
    stripe.position.y = 0.5;
    group.add(stripe);

    const postL = createCylinder(0.04, 0.04, 0.8, COLORS.barrierYellow, 6);
    postL.position.set(-0.8, 0.4, 0);
    group.add(postL);
    const postR = createCylinder(0.04, 0.04, 0.8, COLORS.barrierYellow, 6);
    postR.position.set(0.8, 0.4, 0);
    group.add(postR);

    group.position.set(laneX, 0, z);
    group.userData.type = OBSTACLE_TYPES.BARRIER_LOW;
    group.userData.hitbox = new THREE.Box3().setFromObject(group);
    return group;
  }

  createBarrierHigh(laneX, z) {
    const group = new THREE.Group();
    const bar = createBox(1.8, 2.2, 0.4, COLORS.barrierRed);
    bar.position.y = 1.1;
    bar.castShadow = true;
    group.add(bar);

    const stripe1 = createBox(1.8, 0.15, 0.42, COLORS.barrierYellow);
    stripe1.position.y = 0.5;
    group.add(stripe1);
    const stripe2 = createBox(1.8, 0.15, 0.42, COLORS.barrierYellow);
    stripe2.position.y = 1.5;
    group.add(stripe2);

    group.position.set(laneX, 0, z);
    group.userData.type = OBSTACLE_TYPES.BARRIER_HIGH;
    group.userData.hitbox = new THREE.Box3().setFromObject(group);
    return group;
  }

  createTrain(laneX, z) {
    const group = new THREE.Group();
    const bodyColor = COLORS.trainBody[Math.floor(Math.random() * COLORS.trainBody.length)];

    const body = createBox(2, 2.5, 8, bodyColor);
    body.position.y = 1.75;
    body.castShadow = true;
    group.add(body);

    const roof = createBox(2.1, 0.2, 8.1, COLORS.trainRoof);
    roof.position.y = 3.1;
    group.add(roof);

    for (let w = -2; w <= 2; w++) {
      const window = createBox(0.08, 0.8, 0.8, COLORS.trainWindow);
      window.position.set(1.01, 2, w * 1.5);
      group.add(window);
      const window2 = createBox(0.08, 0.8, 0.8, COLORS.trainWindow);
      window2.position.set(-1.01, 2, w * 1.5);
      group.add(window2);
    }

    const wheelGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.15, 8);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x212121 });
    for (const xOff of [-0.8, 0.8]) {
      for (const zOff of [-2.5, 0, 2.5]) {
        const wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(xOff, 0.25, zOff);
        group.add(wheel);
      }
    }

    group.position.set(laneX, 0, z);
    group.userData.type = OBSTACLE_TYPES.TRAIN;
    group.userData.hitbox = new THREE.Box3().setFromObject(group);
    return group;
  }

  createSign(laneX, z) {
    const group = new THREE.Group();
    const pole = createCylinder(0.05, 0.05, 3, COLORS.signPole, 6);
    pole.position.y = 1.5;
    pole.castShadow = true;
    group.add(pole);

    const board = createBox(1.5, 1, 0.1, COLORS.signBoard);
    board.position.set(0, 2.8, 0);
    board.castShadow = true;
    group.add(board);

    group.position.set(laneX, 0, z);
    group.userData.type = OBSTACLE_TYPES.SIGN;
    group.userData.hitbox = new THREE.Box3().setFromObject(group);
    return group;
  }

  spawnObstacle(z, occupiedLanes) {
    const types = [OBSTACLE_TYPES.BARRIER_LOW, OBSTACLE_TYPES.BARRIER_HIGH, OBSTACLE_TYPES.TRAIN, OBSTACLE_TYPES.SIGN];
    const type = types[Math.floor(Math.random() * types.length)];
    const availableLanes = [-1, 0, 1].filter(l => !occupiedLanes.includes(l));
    if (availableLanes.length === 0) return null;

    const lane = availableLanes[Math.floor(Math.random() * availableLanes.length)];
    const laneX = lane * LANE_WIDTH;
    let obstacle;

    switch (type) {
      case OBSTACLE_TYPES.BARRIER_LOW:
        obstacle = this.createBarrierLow(laneX, z);
        break;
      case OBSTACLE_TYPES.BARRIER_HIGH:
        obstacle = this.createBarrierHigh(laneX, z);
        break;
      case OBSTACLE_TYPES.TRAIN:
        obstacle = this.createTrain(laneX, z);
        break;
      case OBSTACLE_TYPES.SIGN:
        obstacle = this.createSign(laneX, z);
        break;
    }

    if (obstacle) {
      this.scene.add(obstacle);
      this.obstacles.push(obstacle);
      occupiedLanes.push(lane);
    }

    return obstacle;
  }

  generateSection(startZ, difficulty) {
    const count = 1 + Math.floor(Math.random() * Math.min(3, difficulty));
    const usedLanes = [];
    let z = startZ;

    for (let i = 0; i < count; i++) {
      this.spawnObstacle(z, usedLanes);
      z -= 3 + Math.random() * 4;
    }

    return z;
  }

  update(playerZ, difficulty) {
    while (this.nextSpawnZ > playerZ - 200) {
      const gap = MathUtils_randomRange(this.minGap, this.maxGap) / Math.max(0.5, difficulty * 0.3);
      this.nextSpawnZ -= gap;
      this.generateSection(this.nextSpawnZ, difficulty);
    }

    for (let i = this.obstacles.length - 1; i >= 0; i--) {
      const obs = this.obstacles[i];
      if (obs.position.z > playerZ + 40) {
        this.scene.remove(obs);
        obs.traverse(child => {
          if (child.geometry) child.geometry.dispose();
          if (child.material) {
            if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
            else child.material.dispose();
          }
        });
        this.obstacles.splice(i, 1);
      }
    }
  }

  getObstacles() {
    return this.obstacles;
  }

  reset() {
    for (const obs of this.obstacles) {
      this.scene.remove(obs);
      obs.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
          else child.material.dispose();
        }
      });
    }
    this.obstacles = [];
    this.nextSpawnZ = -30;
  }
}

function MathUtils_randomRange(min, max) {
  return min + Math.random() * (max - min);
}

export { OBSTACLE_TYPES };
