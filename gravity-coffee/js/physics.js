import * as THREE from 'three';

export class PhysicsWorld {
    constructor() {
        this.gravity = new THREE.Vector3(0, -9.8, 0);
        this.bodies = [];
        this.planes = [];
        this.cellSize = 3.0;
        this.grid = {};
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

    _gridKey(pos) {
        const x = Math.floor(pos.x / this.cellSize);
        const y = Math.floor(pos.y / this.cellSize);
        const z = Math.floor(pos.z / this.cellSize);
        return `${x},${y},${z}`;
    }

    _buildGrid() {
        this.grid = {};
        for (const body of this.bodies) {
            if (body.isKinematic) continue;
            const key = this._gridKey(body.position);
            if (!this.grid[key]) this.grid[key] = [];
            this.grid[key].push(body);
        }
    }

    _getNeighbors(body) {
        const key = this._gridKey(body.position);
        const parts = key.split(',');
        const neighbors = [];
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dz = -1; dz <= 1; dz++) {
                    const nk = `${Number(parts[0])+dx},${Number(parts[1])+dy},${Number(parts[2])+dz}`;
                    if (this.grid[nk]) {
                        for (const b of this.grid[nk]) {
                            if (b !== body) neighbors.push(b);
                        }
                    }
                }
            }
        }
        return neighbors;
    }

    update(dt) {
        dt = Math.min(dt, 0.05);

        // 1. Integration
        for (const body of this.bodies) {
            if (body.isKinematic) continue;
            body.velocity.addScaledVector(this.gravity, dt);
            body.velocity.multiplyScalar(Math.exp(-body.drag * dt));
            body.angularVelocity.multiplyScalar(Math.exp(-body.angularDrag * dt));
            body.position.addScaledVector(body.velocity, dt);

            if (body.angularVelocity.lengthSq() > 0) {
                const angle = body.angularVelocity.length() * dt;
                const axis = body.angularVelocity.clone().normalize();
                const qDelta = new THREE.Quaternion().setFromAxisAngle(axis, angle);
                body.quaternion.premultiply(qDelta).normalize();
            }

            if (body.mesh) {
                body.mesh.position.copy(body.position);
                body.mesh.quaternion.copy(body.quaternion);
            }
        }

        // Plane collisions
        for (const body of this.bodies) {
            if (body.isKinematic) continue;
            for (const plane of this.planes) {
                this.checkPlaneCollision(body, plane);
            }
        }

        // Body-Body with spatial hash
        this._buildGrid();
        const checked = new Set();
        for (const body of this.bodies) {
            if (body.isKinematic) continue;
            const neighbors = this._getNeighbors(body);
            for (const other of neighbors) {
                const pairKey = body.position.x < other.position.x ? `${body.position.x.toFixed(2)},${other.position.x.toFixed(2)}` : `${other.position.x.toFixed(2)},${body.position.x.toFixed(2)}`;
                if (checked.has(pairKey)) continue;
                checked.add(pairKey);

                if (other.isKinematic && body.isKinematic) continue;

                const r1 = body.radius || Math.max(body.dimensions.x, body.dimensions.y, body.dimensions.z);
                const r2 = other.radius || Math.max(other.dimensions.x, other.dimensions.y, other.dimensions.z);
                const diff = new THREE.Vector3().subVectors(body.position, other.position);
                const distSq = diff.lengthSq();
                const minSq = (r1 + r2) * (r1 + r2);

                if (distSq < minSq) {
                    const dist = Math.sqrt(distSq);
                    const normal = diff.clone().divideScalar(dist || 1);
                    const penetration = (r1 + r2) - dist;
                    this.resolveCollision(body, other, normal, penetration);
                }
            }
        }
    }

    checkPlaneCollision(body, plane) {
        const r = body.radius || Math.max(body.dimensions.x, body.dimensions.y, body.dimensions.z);
        const dist = plane.distanceToPoint(body.position);

        if (dist < r) {
            const penetration = r - dist;
            body.position.addScaledVector(plane.normal, penetration);

            const velNormal = body.velocity.dot(plane.normal);
            if (velNormal < 0) {
                const restitution = body.restitution;
                const j = -(1 + restitution) * velNormal;
                body.velocity.addScaledVector(plane.normal, j);

                const velTangent = body.velocity.clone().sub(plane.normal.clone().multiplyScalar(velNormal));
                velTangent.multiplyScalar(1 - body.friction);

                const correctedVelocity = plane.normal.clone().multiplyScalar(body.velocity.dot(plane.normal)).add(velTangent);
                body.velocity.copy(correctedVelocity);

                body.angularVelocity.add(new THREE.Vector3(
                    (Math.random() - 0.5) * j * 10,
                    (Math.random() - 0.5) * j * 10,
                    (Math.random() - 0.5) * j * 10
                ));

                if(window.gameEngine && window.gameEngine.audioManager && j > 1.0) window.gameEngine.audioManager.playCollision(j);
            }

            if (body.mesh) body.mesh.position.copy(body.position);
        }
    }

    resolveCollision(b1, b2, normal, penetration) {
        const totalMass = (b1.isKinematic ? 0 : b1.mass) + (b2.isKinematic ? 0 : b2.mass);
        const m1Ratio = b1.isKinematic ? 0 : b1.mass / totalMass;
        const m2Ratio = b2.isKinematic ? 0 : b2.mass / totalMass;

        if (!b1.isKinematic) b1.position.addScaledVector(normal, penetration * m2Ratio);
        if (!b2.isKinematic) b2.position.addScaledVector(normal, -penetration * m1Ratio);

        const relVel = new THREE.Vector3().subVectors(b1.velocity, b2.velocity);
        const velAlongNormal = relVel.dot(normal);

        if (velAlongNormal > 0) return;

        const e = Math.min(b1.restitution, b2.restitution);
        let j = -(1 + e) * velAlongNormal;

        const invM1 = b1.isKinematic ? 0 : 1 / b1.mass;
        const invM2 = b2.isKinematic ? 0 : 1 / b2.mass;

        j /= (invM1 + invM2);

        const impulse = normal.clone().multiplyScalar(j);

        if (!b1.isKinematic) b1.velocity.addScaledVector(impulse, invM1);
        if (!b2.isKinematic) b2.velocity.addScaledVector(impulse, -invM2);

        if(window.gameEngine && window.gameEngine.audioManager && j > 0.5) window.gameEngine.audioManager.playCollision(j);
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
        this.radius = options.radius || 0.5;

        this.mesh = options.mesh || null;
        if(this.mesh) {
            this.mesh.position.copy(this.position);
            this.mesh.quaternion.copy(this.quaternion);
        }
    }
}
