import * as THREE from 'three';

export class PhysicsWorld {
    constructor() {
        this.gravity = new THREE.Vector3(0, -9.8, 0);
        this.bodies = [];
        this.planes = []; // Collision planes defined by normal and constant: n·x + d = 0 -> distance = -d
                          // In Three.js Plane: distance is distance from origin along normal
    }

    addBody(body) {
        this.bodies.push(body);
    }

    removeBody(body) {
        const index = this.bodies.indexOf(body);
        if (index > -1) {
            this.bodies.splice(index, 1);
        }
    }

    addPlane(normal, distance) {
        this.planes.push(new THREE.Plane(normal.clone().normalize(), distance));
    }

    update(dt) {
        // Cap dt to avoid huge jumps
        dt = Math.min(dt, 0.05);

        // 1. Integration
        for (const body of this.bodies) {
            if (body.isKinematic) continue;

            // Apply gravity
            body.velocity.addScaledVector(this.gravity, dt);

            // Apply drag
            body.velocity.multiplyScalar(1 - body.drag * dt);
            body.angularVelocity.multiplyScalar(1 - body.angularDrag * dt);

            // Integrate position
            body.position.addScaledVector(body.velocity, dt);

            // Integrate rotation
            if (body.angularVelocity.lengthSq() > 0) {
                const angle = body.angularVelocity.length() * dt;
                const axis = body.angularVelocity.clone().normalize();
                const qDelta = new THREE.Quaternion().setFromAxisAngle(axis, angle);
                body.quaternion.premultiply(qDelta).normalize();
            }

            // Sync visual mesh
            if (body.mesh) {
                body.mesh.position.copy(body.position);
                body.mesh.quaternion.copy(body.quaternion);
            }
        }

        // 2. Collision Detection & Resolution
        // Plane Collisions (Walls/Floors)
        for (const body of this.bodies) {
            if (body.isKinematic) continue;

            for (const plane of this.planes) {
                this.checkPlaneCollision(body, plane);
            }
        }

        // Body-Body Collisions (Sphere approx for now if not Box)
        for (let i = 0; i < this.bodies.length; i++) {
            for (let j = i + 1; j < this.bodies.length; j++) {
                const b1 = this.bodies[i];
                const b2 = this.bodies[j];
                
                // Broadphase
                if (b1.position.distanceToSquared(b2.position) > 9) continue;
                
                if (b1.isKinematic && b2.isKinematic) continue;

                // Simple sphere-sphere collision for all objects to keep it lightweight
                // or specific logic. We fallback to sphere bounding.
                const r1 = b1.radius || Math.max(b1.dimensions.x, b1.dimensions.y, b1.dimensions.z);
                const r2 = b2.radius || Math.max(b2.dimensions.x, b2.dimensions.y, b2.dimensions.z);
                
                const diff = new THREE.Vector3().subVectors(b1.position, b2.position);
                const distSq = diff.lengthSq();
                const minSq = (r1 + r2) * (r1 + r2);

                if (distSq < minSq) {
                    const dist = Math.sqrt(distSq);
                    const normal = diff.clone().divideScalar(dist || 1);
                    const penetration = (r1 + r2) - dist;
                    
                    this.resolveCollision(b1, b2, normal, penetration);
                }
            }
        }
    }

    checkPlaneCollision(body, plane) {
        // Simplified: treat body as sphere for plane collision extent
        const r = body.radius || Math.max(body.dimensions.x, body.dimensions.y, body.dimensions.z);
        const dist = plane.distanceToPoint(body.position);

        if (dist < r) {
            const penetration = r - dist;
            // Position correction
            body.position.addScaledVector(plane.normal, penetration);
            
            // Velocity projection along normal
            const velNormal = body.velocity.dot(plane.normal);
            if (velNormal < 0) {
                // Bounce
                const restitution = body.restitution;
                const j = -(1 + restitution) * velNormal;
                body.velocity.addScaledVector(plane.normal, j);
                
                // Friction
                const velTangent = body.velocity.clone().sub(plane.normal.clone().multiplyScalar(velNormal));
                velTangent.multiplyScalar(1 - body.friction);
                
                const correctedVelocity = plane.normal.clone().multiplyScalar(body.velocity.dot(plane.normal)).add(velTangent);
                 body.velocity.copy(correctedVelocity);

                // Add some pseudo-angular spin on bounce
                body.angularVelocity.add(new THREE.Vector3(
                    (Math.random() - 0.5) * j * 10,
                    (Math.random() - 0.5) * j * 10,
                    (Math.random() - 0.5) * j * 10
                ));

                if(window.gameAudio && j > 1.0) window.gameAudio.playCollision(j);
            }
            
            if (body.mesh) body.mesh.position.copy(body.position);
        }
    }

    resolveCollision(b1, b2, normal, penetration) {
        // Positional correction
        const totalMass = (b1.isKinematic ? 0 : b1.mass) + (b2.isKinematic ? 0 : b2.mass);
        const m1Ratio = b1.isKinematic ? 0 : b1.mass / totalMass;
        const m2Ratio = b2.isKinematic ? 0 : b2.mass / totalMass;

        if (!b1.isKinematic) b1.position.addScaledVector(normal, penetration * m2Ratio);
        if (!b2.isKinematic) b2.position.addScaledVector(normal, -penetration * m1Ratio);

        // Velocity resolution
        const relVel = new THREE.Vector3().subVectors(b1.velocity, b2.velocity);
        const velAlongNormal = relVel.dot(normal);

        if (velAlongNormal > 0) return; // Moving apart

        const e = Math.min(b1.restitution, b2.restitution);
        let j = -(1 + e) * velAlongNormal;
        
        const invM1 = b1.isKinematic ? 0 : 1 / b1.mass;
        const invM2 = b2.isKinematic ? 0 : 1 / b2.mass;
        
        j /= (invM1 + invM2);

        const impulse = normal.clone().multiplyScalar(j);

        if (!b1.isKinematic) b1.velocity.addScaledVector(impulse, invM1);
        if (!b2.isKinematic) b2.velocity.addScaledVector(impulse, -invM2);
        
        if(window.gameAudio && j > 0.5) window.gameAudio.playCollision(j);
    }
}

export class PhysicsBody {
    constructor(options) {
        this.position = options.position || new THREE.Vector3();
        this.velocity = new THREE.Vector3();
        this.angularVelocity = new THREE.Vector3();
        this.quaternion = options.quaternion || new THREE.Quaternion();
        this.mass = options.mass || 1;
        this.restitution = options.restitution || 0.4;
        this.friction = options.friction || 0.5;
        this.isKinematic = options.isKinematic || false;
        this.drag = options.drag || 0.1;
        this.angularDrag = options.angularDrag || 0.1;
        
        this.boundingShape = options.shape || "sphere";
        this.dimensions = options.dimensions || new THREE.Vector3(0.5, 0.5, 0.5);
        this.radius = options.radius || 0.5; // for sphere/cylinder approx
        
        this.mesh = options.mesh || null;
        if(this.mesh) {
            this.mesh.position.copy(this.position);
            this.mesh.quaternion.copy(this.quaternion);
        }
    }
}