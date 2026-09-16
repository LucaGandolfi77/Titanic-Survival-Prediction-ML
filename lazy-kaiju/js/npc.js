import * as THREE from 'three';
import { MathUtils } from './utils.js';

const NPC_STATE = { IDLE: 'idle', CHASING: 'chasing', CAUGHT: 'caught' };

export class ActivistNPC {
    constructor(scene, startPos, color = 0xef4444) {
        this.scene = scene;
        this.position = startPos.clone();
        this.velocity = new THREE.Vector3();
        this.state = NPC_STATE.IDLE;
        this.speed = 3.5;
        this.detectionRange = 40;
        this.catchRange = 3;
        this.time = 0;

        this.buildMesh(color);
        this.gridX = Math.floor(startPos.x / 10) + 10;
        this.gridZ = Math.floor(startPos.z / 10) + 10;
        this.patrolTarget = null;
        this.patrolTimer = 0;
    }

    buildMesh(color) {
        this.group = new THREE.Group();

        const bodyMat = new THREE.MeshLambertMaterial({ color });
        const body = new THREE.Mesh(new THREE.BoxGeometry(1, 2, 0.6), bodyMat);
        body.position.y = 1;
        body.castShadow = true;
        this.group.add(body);

        const headMat = new THREE.MeshLambertMaterial({ color: 0xfbbf24 });
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.4, 8, 8), headMat);
        head.position.y = 2.3;
        this.group.add(head);

        const signMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const sign = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.5), signMat);
        sign.position.set(0, 1.8, 0.4);
        this.group.add(sign);

        this.group.position.copy(this.position);
        this.scene.add(this.group);
    }

    update(dt, kaijuPos, cityGrid, buildings) {
        this.time += dt;
        this.state = NPC_STATE.CHASING;

        const distToKaiju = this.position.distanceTo(kaijuPos);

        if (distToKaiju < this.catchRange) {
            return { caught: true, distance: distToKaiju };
        }

        const direction = new THREE.Vector3().subVectors(kaijuPos, this.position).normalize();
        const desiredVel = direction.multiplyScalar(this.speed);

        const obstacleAvoid = this._avoidObstacles(dt, buildings);
        this.velocity.lerp(desiredVel.add(obstacleAvoid), 0.1);

        this.position.add(this.velocity.clone().multiplyScalar(dt));

        this.position.x = MathUtils.clamp(this.position.x, -95, 95);
        this.position.z = MathUtils.clamp(this.position.z, -95, 95);

        if (this.velocity.lengthSq() > 0.1) {
            const targetAngle = Math.atan2(this.velocity.x, this.velocity.z);
            this.group.rotation.y = MathUtils.lerpAngle(this.group.rotation.y, targetAngle, 5 * dt);
        }

        this.group.position.copy(this.position);

        const bob = Math.sin(this.time * 8) * 0.05;
        this.group.position.y = bob;

        return { caught: false, distance: distToKaiju };
    }

    _avoidObstacles(dt, buildings) {
        const avoidForce = new THREE.Vector3();
        const lookAhead = this.velocity.clone().multiplyScalar(3);
        const checkPos = this.position.clone().add(lookAhead);

        for (const b of buildings) {
            if (!b || !b.pos || b.userData?.destroyed) continue;
            const dx = checkPos.x - b.pos.x;
            const dz = checkPos.z - b.pos.z;
            const dist = Math.sqrt(dx * dx + dz * dz);
            const radius = 4;

            if (dist < radius && dist > 0) {
                const push = new THREE.Vector3(dx, 0, dz).normalize().multiplyScalar(5);
                avoidForce.add(push);
            }
        }

        return avoidForce;
    }

    _findPathAStar(kaijuPos, buildings) {
        const gridW = 20;
        const gridH = 20;
        const startX = Math.floor((this.position.x + 100) / 10);
        const startZ = Math.floor((this.position.z + 100) / 10);
        const endX = Math.floor((kaijuPos.x + 100) / 10);
        const endZ = Math.floor((kaijuPos.z + 100) / 10);

        const blocked = new Set();
        for (const b of buildings) {
            if (!b?.pos) continue;
            const bx = Math.floor((b.pos.x + 100) / 10);
            const bz = Math.floor((b.pos.z + 100) / 10);
            blocked.add(`${bx},${bz}`);
        }

        const open = [{ x: startX, z: startZ, g: 0, f: 0, parent: null }];
        const closed = new Set();
        const heuristic = (x, z) => Math.abs(x - endX) + Math.abs(z - endZ);

        open[0].f = heuristic(startX, startZ);

        const dirs = [{dx:0,dz:-1},{dx:0,dz:1},{dx:-1,dz:0},{dx:1,dz:0}];
        let iterations = 0;

        while (open.length > 0 && iterations < 100) {
            iterations++;
            open.sort((a, b) => a.f - b.f);
            const current = open.shift();

            if (current.x === endX && current.z === endZ) {
                const path = [];
                let node = current;
                while (node) {
                    path.unshift({ x: node.x, z: node.z });
                    node = node.parent;
                }
                return path;
            }

            closed.add(`${current.x},${current.z}`);

            for (const d of dirs) {
                const nx = current.x + d.dx;
                const nz = current.z + d.dz;
                const key = `${nx},${nz}`;

                if (nx < 0 || nx >= gridW || nz < 0 || nz >= gridH) continue;
                if (closed.has(key) || blocked.has(key)) continue;

                const g = current.g + 1;
                const existing = open.find(n => n.x === nx && n.z === nz);
                if (!existing) {
                    open.push({ x: nx, z: nz, g, f: g + heuristic(nx, nz), parent: current });
                } else if (g < existing.g) {
                    existing.g = g;
                    existing.f = g + heuristic(nx, nz);
                    existing.parent = current;
                }
            }
        }

        return null;
    }

    dispose() {
        if (this.group) {
            this.scene.remove(this.group);
            this.group.traverse(child => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
                    else child.material.dispose();
                }
            });
        }
    }
}