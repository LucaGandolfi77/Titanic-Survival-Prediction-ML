/**
 * World / environment — DRONE STRIKE
 * Terrain, buildings, trees, grid.
 * @module world
 */

import { MathUtils } from './utils.js';

const THREE = globalThis.THREE;

/**
 * Build the entire environment.
 * @param {THREE.Scene} scene
 * @returns {{ buildings: {mesh:THREE.Mesh, box:THREE.Box3}[], terrain:THREE.Mesh }}
 */
export function buildWorld(scene) {
  const terrain = createTerrain(scene);
  const buildings = createBuildings(scene);
  createTrees(scene);
  createGrid(scene);
  return { buildings, terrain };
}

/* ── Terrain ───────────────────────────────────────── */
function createTerrain(scene) {
  const geo = new THREE.PlaneGeometry(2000, 2000, 64, 64);
  geo.rotateX(-Math.PI / 2);

  // Gentle hills via vertex displacement
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const y = Math.sin(x * 0.008) * Math.cos(z * 0.012) * 4 +
              Math.sin(x * 0.02 + z * 0.015) * 1.5;
    pos.setY(i, y);
  }
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: 0x3d5a2a,
    roughness: 0.95,
    metalness: 0.05,
    flatShading: false,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  scene.add(mesh);
  return mesh;
}

/* ── Grid helper ───────────────────────────────────── */
function createGrid(scene) {
  const grid = new THREE.GridHelper(2000, 200, 0x224422, 0x224422);
  grid.position.y = 0.05;
  grid.material.opacity = 0.08;
  grid.material.transparent = true;
  scene.add(grid);
}

/* ── Buildings ─────────────────────────────────────── */
function createBuildings(scene) {
  const buildings = [];
  const occupied = [];
  const mat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.9, metalness: 0.1 });

  for (let i = 0; i < 30; i++) {
    const w = MathUtils.randomRange(10, 25);
    const h = MathUtils.randomRange(15, 50);
    const d = MathUtils.randomRange(10, 25);

    let x, z, tooClose;
    let attempts = 0;
    do {
      x = MathUtils.randomRange(-900, 900);
      z = MathUtils.randomRange(-900, 900);
      tooClose = occupied.some(p => Math.hypot(p.x - x, p.z - z) < 40);
      attempts++;
    } while (tooClose && attempts < 50);

    occupied.push({ x, z });

    const geo = new THREE.BoxGeometry(w, h, d);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, h / 2, z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);

    const box = new THREE.Box3().setFromObject(mesh);
    buildings.push({ mesh, box });
  }

  return buildings;
}

/* ── Trees ─────────────────────────────────────────── */
function createTrees(scene) {
  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x5c3a1e, roughness: 0.9 });
  const leafMat  = new THREE.MeshStandardMaterial({ color: 0x2d7a2d, roughness: 0.8 });

  for (let i = 0; i < 20; i++) {
    const x = MathUtils.randomRange(-900, 900);
    const z = MathUtils.randomRange(-900, 900);

    const group = new THREE.Group();
    group.position.set(x, 0, z);

    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.5, 0.8, 5, 6),
      trunkMat
    );
    trunk.position.y = 2.5;
    trunk.castShadow = true;
    group.add(trunk);

    const foliage = new THREE.Mesh(
      new THREE.ConeGeometry(4, 8, 6),
      leafMat
    );
    foliage.position.y = 9;
    foliage.castShadow = true;
    group.add(foliage);

    scene.add(group);
  }
}
