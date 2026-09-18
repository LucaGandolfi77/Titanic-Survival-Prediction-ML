import * as THREE from 'three';
import { createBox, createCylinder } from './utils.js';

const LANE_WIDTH = 2.5;
const LANE_POSITIONS = [-LANE_WIDTH, 0, LANE_WIDTH];
const CHUNK_LENGTH = 80;
const VISIBLE_CHUNKS = 5;
const GROUND_WIDTH = 12;

const COLORS = {
  track: 0x37474f,
  rail: 0x90a4ae,
  sleepers: 0x5d4037,
  ground: 0x4caf50,
  groundDark: 0x388e3c,
  concrete: 0x9e9e9e,
  building: [0x78909c, 0x607d8b, 0x546e7a, 0x455a64, 0x37474f],
  fence: 0x8d6e63,
  signal: [0xff1744, 0x00e676, 0xffd600]
};

export class Track {
  constructor(scene) {
    this.scene = scene;
    this.chunks = [];
    this.nextChunkZ = 0;
    this.playerZ = 0;
    this.buildings = [];
    this.build();
  }

  build() {
    for (let i = 0; i < VISIBLE_CHUNKS; i++) {
      this.addChunk();
    }
  }

  addChunk() {
    const group = new THREE.Group();
    group.position.z = this.nextChunkZ;

    const groundGeo = new THREE.PlaneGeometry(GROUND_WIDTH, CHUNK_LENGTH);
    const groundMat = new THREE.MeshStandardMaterial({ color: COLORS.ground, side: THREE.DoubleSide });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    group.add(ground);

    const trackBase = createBox(GROUND_WIDTH * 0.6, 0.15, CHUNK_LENGTH, COLORS.track);
    trackBase.position.y = 0.075;
    trackBase.receiveShadow = true;
    group.add(trackBase);

    for (let lane = -1; lane <= 1; lane++) {
      const xPos = lane * LANE_WIDTH;

      for (let rail = -1; rail <= 1; rail += 2) {
        const railMesh = createBox(0.08, 0.12, CHUNK_LENGTH, COLORS.rail);
        railMesh.position.set(xPos + rail * 0.7, 0.21, 0);
        group.add(railMesh);
      }

      const sleeperCount = Math.floor(CHUNK_LENGTH / 1.2);
      for (let s = 0; s < sleeperCount; s++) {
        const sleeper = createBox(1.6, 0.08, 0.2, COLORS.sleepers);
        sleeper.position.set(xPos, 0.16, -CHUNK_LENGTH / 2 + s * 1.2 + 0.6);
        group.add(sleeper);
      }
    }

    this.addScenery(group);

    this.scene.add(group);
    this.chunks.push({ group, z: this.nextChunkZ });
    this.nextChunkZ -= CHUNK_LENGTH;
  }

  addScenery(group) {
    const side = Math.random() > 0.5 ? 1 : -1;
    const baseX = side * (GROUND_WIDTH / 2 + 2);

    const buildingCount = Math.floor(Math.random() * 3) + 1;
    for (let i = 0; i < buildingCount; i++) {
      const bWidth = 3 + Math.random() * 5;
      const bHeight = 5 + Math.random() * 20;
      const bDepth = 4 + Math.random() * 6;
      const color = COLORS.building[Math.floor(Math.random() * COLORS.building.length)];
      const building = createBox(bWidth, bHeight, bDepth, color);
      building.position.set(
        baseX + (Math.random() - 0.5) * 6,
        bHeight / 2,
        (Math.random() - 0.5) * CHUNK_LENGTH * 0.7
      );
      building.castShadow = true;
      building.receiveShadow = true;
      group.add(building);
      this.buildings.push(building);

      for (let w = 0; w < Math.floor(bHeight / 3); w++) {
        for (let h = 0; h < 2; h++) {
          const win = createBox(0.6, 0.8, 0.1, Math.random() > 0.3 ? 0xfff9c4 : 0x37474f);
          win.position.set(
            building.position.x + (h - 0.5) * 1.5,
            2 + w * 3,
            building.position.z + bDepth / 2 + 0.05
          );
          group.add(win);
        }
      }
    }

    const otherSide = -side;
    const otherBaseX = otherSide * (GROUND_WIDTH / 2 + 1.5);
    const fenceCount = Math.floor(CHUNK_LENGTH / 4);
    for (let i = 0; i < fenceCount; i++) {
      const post = createCylinder(0.05, 0.05, 1.2, COLORS.fence, 6);
      post.position.set(otherBaseX, 0.6, -CHUNK_LENGTH / 2 + i * 4 + 2);
      group.add(post);

      if (i < fenceCount - 1) {
        const bar = createBox(0.05, 0.05, 4, COLORS.fence);
        bar.position.set(otherBaseX, 0.9, -CHUNK_LENGTH / 2 + i * 4 + 4);
        group.add(bar);
      }
    }

    if (Math.random() > 0.6) {
      const signalX = side * (GROUND_WIDTH / 2 - 0.5);
      const signalColor = COLORS.signal[Math.floor(Math.random() * COLORS.signal.length)];
      const pole = createCylinder(0.06, 0.06, 3.5, 0x424242, 6);
      pole.position.set(signalX, 1.75, -CHUNK_LENGTH / 2 + Math.random() * CHUNK_LENGTH);
      group.add(pole);
      const light = new THREE.Mesh(
        new THREE.SphereGeometry(0.15, 8, 8),
        new THREE.MeshStandardMaterial({ color: signalColor, emissive: signalColor, emissiveIntensity: 0.8 })
      );
      light.position.set(signalX, 3.5, pole.position.z);
      group.add(light);
    }
  }

  update(playerWorldZ) {
    this.playerZ = playerWorldZ;

    while (this.chunks.length > 0 && this.chunks[0].z > playerWorldZ + CHUNK_LENGTH) {
      const old = this.chunks.shift();
      this.scene.remove(old.group);
      old.group.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
          else child.material.dispose();
        }
      });
    }

    while (this.chunks.length < VISIBLE_CHUNKS) {
      this.addChunk();
    }
  }

  getLaneX(lane) {
    return LANE_POSITIONS[lane + 1];
  }

  getBuildings() {
    return this.buildings;
  }
}

export { LANE_WIDTH, LANE_POSITIONS, CHUNK_LENGTH };
