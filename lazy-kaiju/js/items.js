import * as THREE from 'three';
import { MathUtils } from './utils.js';

export class TrashManager {
    constructor(scene, city) {
        this.scene = scene;
        this.city = city;
        this.trashItems = [];
        this.particles = [];
        
        // Settings
        this.baseCapacity = 50;
        this.maxCapacity = 200;
        this.spawnTimer = 0;
        
        // Object geometries/materials purely reused for perf
        this.geometries = [
             new THREE.CylinderGeometry(0.3, 0.4, 0.8), // can
             new THREE.BoxGeometry(0.6, 0.6, 0.6),     // crate
             new THREE.DodecahedronGeometry(0.5)        // abstract junk
        ];
        
        this.materials = [
             new THREE.MeshLambertMaterial({color: 0x999999}), // silver
             new THREE.MeshLambertMaterial({color: 0x8b4513}), // wood
             new THREE.MeshLambertMaterial({color: 0xff4500})  // red rust
        ];
        
        // Debris particles
        this.particleGeo = new THREE.BoxGeometry(0.2, 0.2, 0.2);
        this.particleMat = new THREE.MeshBasicMaterial({color: 0x555555});
        
        // Karma/Score
        this.trashCleared = 0;
        
        // Initial spawn
        this.initialSpawn(this.baseCapacity);
    }
    
    initialSpawn(amount) {
        let spawned = 0;
        // Search roads, place trash
        for(let i=0; i<amount; i++) {
           this.spawnSingle();
        }
    }
    
    spawnSingle() {
        if(this.city.roadTiles.length === 0) return;
        
        // Pick random road
        let r = this.city.roadTiles[Math.floor(Math.random() * this.city.roadTiles.length)];
        
        // Random offset within the 10x10 road block (-5 to +5)
        let ox = MathUtils.randRange(-4, 4);
        let oz = MathUtils.randRange(-4, 4);
        
        // Create Mesh
        let geo = this.geometries[Math.floor(Math.random() * this.geometries.length)];
        let mat = this.materials[Math.floor(Math.random() * this.materials.length)];
        let mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        
        mesh.position.set(r.x + ox, 0.5, r.z + oz);
        
        this.scene.add(mesh);
        
        this.trashItems.push({
            mesh: mesh,
            v: new THREE.Vector3(),
            active: true
        });
    }
    
    update(dt, tailSpheres) {
        // Continuous spawn logic
        this.spawnTimer += dt;
        if(this.spawnTimer > 1.5 && this.trashItems.length < this.maxCapacity) {
            this.spawnTimer = 0;
            this.spawnSingle();
        }
        
        // Iterate trash backward for clean removal
        for(let i = this.trashItems.length - 1; i >= 0; i--) {
            let t = this.trashItems[i];
            
            // Apply velocity
            t.mesh.position.add(t.v.clone().multiplyScalar(dt));
            t.v.multiplyScalar(0.9); // friction/damping
            
            // Check tail collision
            let hit = false;
            let sweepForce = 15;
            
            for(let s of tailSpheres) {
                // simple sphere overlap
                let dx = t.mesh.position.x - s.x;
                let dz = t.mesh.position.z - s.z;
                let distSq = dx*dx + dz*dz;
                let radSum = 0.5 + s.r; // trash radius approx 0.5
                
                if(distSq < radSum*radSum) {
                    // Normalize hit vector
                    let dist = Math.sqrt(distSq);
                    let nx = dx / (dist || 1);
                    let nz = dz / (dist || 1);
                    
                    // Add impulse to trash
                    t.v.x += nx * sweepForce;
                    t.v.z += nz * sweepForce;
                    t.v.y += 5; // pop up
                    hit = true;
                }
            }
            
            // Gravity on trash
            if(t.mesh.position.y > 0.5) {
                t.v.y -= 20 * dt; 
            } else {
                t.mesh.position.y = 0.5;
                if(t.v.y < 0) {
                    // bounce
                    t.v.y *= -0.3;
                }
            }
            
            // Map boundaries - if pushed off map (<-100, >100), consider "Swept away"
            if(Math.abs(t.mesh.position.x) > 100 || Math.abs(t.mesh.position.z) > 100) {
                this.removeTrash(i);
                continue;
            }
            
            // Rotation based on velocity
            if(t.v.lengthSq() > 1) {
                t.mesh.rotation.x += t.v.z * dt;
                t.mesh.rotation.z -= t.v.x * dt;
            }
        }
        
        // Update particles
        for(let i = this.particles.length - 1; i >= 0; i--) {
            let p = this.particles[i];
            p.mesh.position.add(p.v.clone().multiplyScalar(dt));
            p.v.y -= 25 * dt; // gravity
            p.life -= dt;
            p.mesh.scale.setScalar(p.life);
            
            if(p.life <= 0 || p.mesh.position.y < 0) {
                this.scene.remove(p.mesh);
                this.particles.splice(i, 1);
            }
        }
    }
    
    removeTrash(index) {
        let t = this.trashItems[index];
        this.scene.remove(t.mesh);
        this.trashItems.splice(index, 1);
        this.trashCleared++;
        
        // Spawn success particle effect
        this.spawnExplosion(t.mesh.position, 0x00ff00, 3);
        
        // Emit logic event for UI update
        const ev = new CustomEvent('trashCleared', {detail: {count: this.trashCleared}});
        window.dispatchEvent(ev);
    }
    
    spawnExplosion(pos, colorHex, count) {
        let mat = new THREE.MeshBasicMaterial({color: colorHex});
        for(let i=0; i<count; i++) {
            let p = new THREE.Mesh(this.particleGeo, mat);
            p.position.copy(pos);
            this.scene.add(p);
            
            this.particles.push({
                mesh: p,
                v: new THREE.Vector3(
                    MathUtils.randRange(-10, 10),
                    MathUtils.randRange(5, 15),
                    MathUtils.randRange(-10, 10)
                ),
                life: 1.0 + Math.random() * 0.5
            });
        }
    }
}