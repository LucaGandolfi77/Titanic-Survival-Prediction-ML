import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.min.js';

export class Particle {
  constructor(position, velocity, lifetime, type = 'dust') {
    this.position = position.clone();
    this.velocity = velocity.clone();
    this.lifetime = lifetime;
    this.age = 0;
    this.type = type;
    
    this.mesh = this.createMesh();
  }

  createMesh() {
    let geo, mat;
    
    if (this.type === 'dust') {
      geo = new THREE.SphereGeometry(0.05, 4, 4);
      mat = new THREE.MeshBasicMaterial({ color: 0x999999, transparent: true });
    } else if (this.type === 'spark') {
      geo = new THREE.BoxGeometry(0.02, 0.02, 0.02);
      mat = new THREE.MeshBasicMaterial({ color: 0xffff00, emissive: 0xff8800 });
    } else if (this.type === 'note') {
      geo = new THREE.PlaneGeometry(0.1, 0.1);
      mat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    }
    
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(this.position);
    return mesh;
  }

  update(dt) {
    this.age += dt;
    const progress = this.age / this.lifetime;
    
    if (progress >= 1) {
      return false; // Dead
    }
    
    // Apply physics
    this.velocity.y -= 9.8 * dt; // Gravity
    this.position.add(this.velocity.clone().multiplyScalar(dt));
    
    this.mesh.position.copy(this.position);
    
    // Fade out
    this.mesh.material.opacity = 1 - progress;
    
    // Rotation
    this.mesh.rotation.x += this.velocity.x * dt * 0.1;
    this.mesh.rotation.y += this.velocity.z * dt * 0.1;
    
    return true; // Still alive
  }
}

export class ParticleSystem {
  constructor(scene) {
    this.scene = scene;
    this.particles = [];
  }

  emit(position, velocity, lifetime, type = 'dust') {
    const particle = new Particle(position, velocity, lifetime, type);
    this.particles.push(particle);
    this.scene.add(particle.mesh);
    return particle;
  }

  emitBurst(position, count = 10, type = 'dust') {
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const speed = 3 + Math.random() * 2;
      const vel = new THREE.Vector3(
        Math.cos(angle) * speed,
        Math.random() * 5,
        Math.sin(angle) * speed
      );
      
      this.emit(position, vel, 1 + Math.random(), type);
    }
  }

  update(dt) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const alive = this.particles[i].update(dt);
      if (!alive) {
        this.scene.remove(this.particles[i].mesh);
        this.particles.splice(i, 1);
      }
    }
  }

  clear() {
    this.particles.forEach(p => this.scene.remove(p.mesh));
    this.particles = [];
  }
}

// Floating note particle
export class FloatingNote {
  constructor(scene, text) {
    this.scene = scene;
    this.text = text;
    this.position = new THREE.Vector3(
      (Math.random() - 0.5) * 10,
      Math.random() * 5 + 2,
      (Math.random() - 0.5) * 10
    );
    
    this.group = new THREE.Group();
    this.group.position.copy(this.position);
    
    this.buildMesh();
    scene.add(this.group);
    
    this.floatSpeed = Math.random() * 2 + 1;
    this.time = 0;
  }

  buildMesh() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = '#ffff00';
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = '#000000';
    ctx.font = '20px Arial';
    ctx.fillText(this.text, 10, 128);
    
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.01);
    
    const mesh = new THREE.Mesh(geometry, material);
    this.group.add(mesh);
  }

  update(dt) {
    this.time += dt;
    
    // Gentle floating motion
    this.group.position.y += Math.sin(this.time * this.floatSpeed) * 0.1;
    this.group.rotation.z += dt * 0.5;
    
    return true;
  }

  remove() {
    this.scene.remove(this.group);
  }
}

// Printer debris (sparks)
export class PrinterSparks {
  constructor(scene, position) {
    this.scene = scene;
    this.position = position;
    this.active = true;
    this.time = 0;
    this.duration = 2; // seconds
  }

  update(dt, particleSystem) {
    if (!this.active) return;
    
    this.time += dt;
    
    if (this.time > this.duration) {
      this.active = false;
      return;
    }
    
    // Emit random sparks
    if (Math.random() < 0.3) {
      const vel = new THREE.Vector3(
        (Math.random() - 0.5) * 5,
        Math.random() * 10,
        (Math.random() - 0.5) * 5
      );
      particleSystem.emit(this.position, vel, 0.5, 'spark');
    }
  }
}