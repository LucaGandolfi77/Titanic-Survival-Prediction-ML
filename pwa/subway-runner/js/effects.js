import * as THREE from 'three';

const MAX_PARTICLES = 200;

export class EffectsManager {
  constructor(scene) {
    this.scene = scene;
    this.particles = [];
    this.trails = [];
    this.screenShake = { intensity: 0, duration: 0 };
  }

  spawnCoinBurst(position) {
    const count = 8;
    for (let i = 0; i < count; i++) {
      const geo = new THREE.SphereGeometry(0.05, 4, 4);
      const mat = new THREE.MeshBasicMaterial({ color: 0xffd600 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(position);
      const angle = (i / count) * Math.PI * 2;
      const speed = 3 + Math.random() * 2;
      mesh.userData.velocity = new THREE.Vector3(
        Math.cos(angle) * speed,
        4 + Math.random() * 3,
        Math.sin(angle) * speed * 0.3
      );
      mesh.userData.life = 0.6;
      mesh.userData.maxLife = 0.6;
      this.scene.add(mesh);
      this.particles.push(mesh);
    }
  }

  spawnHitEffect(position) {
    const count = 12;
    for (let i = 0; i < count; i++) {
      const geo = new THREE.SphereGeometry(0.08, 4, 4);
      const mat = new THREE.MeshBasicMaterial({ color: 0xff1744 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(position);
      mesh.position.y += 1;
      const angle = (i / count) * Math.PI * 2;
      const speed = 4 + Math.random() * 3;
      mesh.userData.velocity = new THREE.Vector3(
        Math.cos(angle) * speed,
        3 + Math.random() * 4,
        Math.sin(angle) * speed * 0.5 - 2
      );
      mesh.userData.life = 0.8;
      mesh.userData.maxLife = 0.8;
      this.scene.add(mesh);
      this.particles.push(mesh);
    }
  }

  spawnPowerupEffect(position, color) {
    const count = 16;
    for (let i = 0; i < count; i++) {
      const geo = new THREE.SphereGeometry(0.06, 4, 4);
      const mat = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(position);
      const angle = (i / count) * Math.PI * 2;
      const speed = 2 + Math.random() * 2;
      mesh.userData.velocity = new THREE.Vector3(
        Math.cos(angle) * speed,
        2 + Math.random() * 5,
        Math.sin(angle) * speed
      );
      mesh.userData.life = 1.0;
      mesh.userData.maxLife = 1.0;
      this.scene.add(mesh);
      this.particles.push(mesh);
    }
  }

  spawnTrail(position, color) {
    if (this.trails.length > 30) return;
    const geo = new THREE.SphereGeometry(0.1, 4, 4);
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(position);
    mesh.position.y = 0.1;
    mesh.userData.life = 0.4;
    mesh.userData.maxLife = 0.4;
    this.scene.add(mesh);
    this.trails.push(mesh);
  }

  triggerScreenShake(intensity, duration) {
    this.screenShake.intensity = intensity;
    this.screenShake.duration = duration;
  }

  update(delta) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.userData.life -= delta;
      if (p.userData.life <= 0) {
        this.scene.remove(p);
        p.geometry.dispose();
        p.material.dispose();
        this.particles.splice(i, 1);
        continue;
      }
      p.position.addScaledVector(p.userData.velocity, delta);
      p.userData.velocity.y -= 10 * delta;
      const t = p.userData.life / p.userData.maxLife;
      p.material.opacity = t;
      p.material.transparent = true;
      p.scale.setScalar(t);
    }

    for (let i = this.trails.length - 1; i >= 0; i--) {
      const t = this.trails[i];
      t.userData.life -= delta;
      if (t.userData.life <= 0) {
        this.scene.remove(t);
        t.geometry.dispose();
        t.material.dispose();
        this.trails.splice(i, 1);
        continue;
      }
      const ratio = t.userData.life / t.userData.maxLife;
      t.material.opacity = ratio * 0.6;
      t.scale.setScalar(ratio);
    }

    if (this.screenShake.duration > 0) {
      this.screenShake.duration -= delta;
    }
  }

  getShakeOffset() {
    if (this.screenShake.duration <= 0) return { x: 0, y: 0 };
    const t = this.screenShake.duration;
    return {
      x: (Math.random() - 0.5) * 2 * this.screenShake.intensity * t,
      y: (Math.random() - 0.5) * 2 * this.screenShake.intensity * t
    };
  }

  reset() {
    for (const p of this.particles) {
      this.scene.remove(p);
      p.geometry.dispose();
      p.material.dispose();
    }
    for (const t of this.trails) {
      this.scene.remove(t);
      t.geometry.dispose();
      t.material.dispose();
    }
    this.particles = [];
    this.trails = [];
    this.screenShake = { intensity: 0, duration: 0 };
  }
}
