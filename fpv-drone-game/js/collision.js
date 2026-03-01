/**
 * Collision detection — DRONE STRIKE
 * Sphere-AABB, sphere-sphere, ground checks.
 * @module collision
 */

const THREE = globalThis.THREE;
const _v = new THREE.Vector3();

export class CollisionSystem {
  /**
   * @param {{mesh:THREE.Mesh, box:THREE.Box3}[]} buildings
   */
  constructor(buildings) {
    this.buildings = buildings;
  }

  /* ── Drone vs buildings (sphere vs AABB) ── */
  checkDroneVsBuildings(dronePos, radius = 1.5) {
    for (const b of this.buildings) {
      const box = b.box;
      // Closest point on AABB to sphere center
      _v.set(
        Math.max(box.min.x, Math.min(dronePos.x, box.max.x)),
        Math.max(box.min.y, Math.min(dronePos.y, box.max.y)),
        Math.max(box.min.z, Math.min(dronePos.z, box.max.z))
      );
      const dist = _v.distanceTo(dronePos);
      if (dist < radius) {
        // Compute push-out normal and penetration depth
        const normal = new THREE.Vector3().subVectors(dronePos, _v);
        const depth = radius - dist;
        if (normal.lengthSq() < 0.001) normal.set(0, 1, 0);
        else normal.normalize();
        return { normal, depth, building: b };
      }
    }
    return null;
  }

  /* ── Projectiles vs enemies (sphere-sphere) ── */
  checkProjectileVsEnemies(projectiles, enemies) {
    const hits = [];
    for (const p of projectiles) {
      if (!p.__poolActive) continue;
      for (const e of enemies) {
        if (!e.alive) continue;
        const dist = p.mesh.position.distanceTo(e.mesh.position);
        if (dist < (p.radius || 0.5) + (e.radius || 2)) {
          hits.push({ enemy: e, projectile: p, damage: p.damage || 20 });
          break; // one hit per projectile
        }
      }
    }
    return hits;
  }

  /* ── Enemy projectiles vs drone (sphere-sphere) ── */
  checkEnemyProjectileVsDrone(projectiles, dronePos, radius = 1.5) {
    const hits = [];
    for (const p of projectiles) {
      if (!p.__poolActive) continue;
      const dist = p.mesh.position.distanceTo(dronePos);
      if (dist < radius + (p.radius || 0.3)) {
        hits.push(p);
      }
    }
    return hits;
  }

  /* ── Drone vs ground ── */
  checkDroneVsGround(dronePos) {
    return dronePos.y < 0.5;
  }
}
