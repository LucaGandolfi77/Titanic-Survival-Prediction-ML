import * as THREE from 'three';

export class Tail {
    constructor(scene, attachTarget, numSegments, segmentLength) {
        this.scene = scene;
        this.attachTarget = attachTarget; // object to pull from
        this.numSegments = numSegments;
        this.segmentLength = segmentLength;
        
        this.points = [];
        this.meshes = [];
        this.radii = [];
        
        // Verlet config
        this.gravity = new THREE.Vector3(0, -9.8, 0);
        this.damping = 0.95;
        this.stiffness = 2; // how rigidly it pulls back to straight
        this.groundFriction = 0.8;
        
        this.init();
    }
    
    init() {
        const mat = new THREE.MeshLambertMaterial({color: 0x4a7a2a});
        const tipMat = new THREE.MeshLambertMaterial({color: 0x2d5a27});
        
        // start position at attach
        let startPos = this.attachTarget.position.clone();
        
        for(let i=0; i<this.numSegments; i++) {
            // Taper effect for tail width
            const t = i / (this.numSegments - 1);
            const r = 1.5 * (1 - t * 0.7); // 1.5 to 0.45
            this.radii.push(r);
            
            // Initial constraint points
            this.points.push({
                x: startPos.x,
                y: startPos.y,
                z: startPos.z - (i * this.segmentLength),
                oldx: startPos.x,
                oldy: startPos.y,
                oldz: startPos.z - (i * this.segmentLength),
                mass: 1.0 - t*0.5 // tip is lighter
            });
            
            // Visuals
            const geo = i === this.numSegments - 1 
                ? new THREE.TetrahedronGeometry(r * 2) // spiky tip
                : new THREE.BoxGeometry(r*2, r*2, r*2);
            
            const mesh = new THREE.Mesh(geo, i === this.numSegments-1 ? tipMat : mat);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            this.scene.add(mesh);
            this.meshes.push(mesh);
        }
    }
    
    update(dt) {
        if(dt > 0.1) dt = 0.1; // clamp dt for physics stability
        
        // 1. Verlet Integration (update positions based on velocity)
        for (let i = 1; i < this.numSegments; i++) { // skip root
            let p = this.points[i];
            
            // Vel = pos - oldpos
            let vx = (p.x - p.oldx) * this.damping;
            let vy = (p.y - p.oldy) * this.damping;
            let vz = (p.z - p.oldz) * this.damping;
            
            p.oldx = p.x;
            p.oldy = p.y;
            p.oldz = p.z;
            
            // Add Forces (Gravity)
            p.x += vx + this.gravity.x * dt * dt;
            p.y += vy + this.gravity.y * dt * dt;
            p.z += vz + this.gravity.z * dt * dt;
            
            // Ground Collision
            if (p.y < this.radii[i]) {
                p.y = this.radii[i]; // sit on ground
                
                // apply friction if hitting ground
                p.x -= (p.x - p.oldx) * (1-this.groundFriction);
                p.z -= (p.z - p.oldz) * (1-this.groundFriction);
            }
        }
        
        // 2. Satisfy Constraints (3 iterations for stability)
        for (let iter = 0; iter < 3; iter++) {
            // Anchor root perfectly to attach point
            let anchor = this.attachTarget.getTailAttachPoint();
            this.points[0].x = anchor.x;
            this.points[0].y = anchor.y;
            this.points[0].z = anchor.z;
            
            // Distance constraints
            for (let i = 0; i < this.numSegments - 1; i++) {
                let p1 = this.points[i];
                let p2 = this.points[i + 1];
                
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let dz = p2.z - p1.z;
                let dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                
                // Avoid division by zero
                if(dist === 0) continue; 
                
                let diff = this.segmentLength - dist;
                let percent = diff / dist / 2; // Split adjustment between p1 and p2
                let offsetX = dx * percent;
                let offsetY = dy * percent;
                let offsetZ = dz * percent;
                
                // Do not move the root
                if (i !== 0) {
                    p1.x -= offsetX;
                    p1.y -= offsetY;
                    p1.z -= offsetZ;
                }
                
                p2.x += offsetX;
                p2.y += offsetY;
                p2.z += offsetZ;
            }
        }

        // Apply back to meshes and calculate rotation to look at next segment
        for(let i=0; i<this.numSegments; i++) {
            this.meshes[i].position.set(this.points[i].x, this.points[i].y, this.points[i].z);
            
            if(i < this.numSegments - 1) {
                // look at next joint
                this.meshes[i].lookAt(
                    this.points[i+1].x,
                    this.points[i+1].y,
                    this.points[i+1].z
                );
            } else {
                // last joint looks along previous trajectory
                const vx = this.points[i].x - this.points[i-1].x;
                const vy = this.points[i].y - this.points[i-1].y;
                const vz = this.points[i].z - this.points[i-1].z;
                
                this.meshes[i].lookAt(
                    this.points[i].x + vx,
                    this.points[i].y + vy,
                    this.points[i].z + vz
                );
            }
        }
    }
    
    // Sweeping attack manual offset
    applySweepImpulse(forceVec) {
          // Send force down the chain but ignore root
          for(let i=2; i<this.numSegments; i++) {
              let p = this.points[i];
              let forceMult = i / this.numSegments; // more force at tip
              
              p.x += forceVec.x * forceMult;
              p.y += forceVec.y * forceMult;
              p.z += forceVec.z * forceMult;
          }
    }

    // Get collision spheres for AABB overlap
    getCollisionSpheres() {
        return this.points.map((p, i) => ({
            x: p.x, y: p.y, z: p.z, r: this.radii[i]
        }));
    }
}