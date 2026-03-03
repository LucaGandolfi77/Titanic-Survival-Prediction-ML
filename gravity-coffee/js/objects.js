import * as THREE from 'three';
import { PhysicsBody } from './physics.js';
import { createCoffeeMaterial } from './coffee.js';

export class ItemManager {
    constructor(scene, physicsWorld) {
        this.scene = scene;
        this.physicsWorld = physicsWorld;
        this.items = [];
        this.coffeePot = null;
    }

    createCoffeeCup(x, y, z) {
        const geo = new THREE.CylinderGeometry(0.18, 0.15, 0.3, 16);
        // Create cup with open top (hollow) via CSG normally, but here we just use 
        // a basic cylinder and put the liquid plane slightly below top.
        // To see inside, we need DoubleSide or two meshes.
        const mat = new THREE.MeshStandardMaterial({ 
            color: 0xffffff, 
            roughness: 0.2, 
            side: THREE.DoubleSide 
        });
        const mesh = new THREE.Mesh(geo, mat);
        
        // Add liquid plane
        const liquidGeo = new THREE.PlaneGeometry(0.34, 0.34);
        const liquidMat = createCoffeeMaterial();
        const liquidMesh = new THREE.Mesh(liquidGeo, liquidMat);
        liquidMesh.rotation.x = -Math.PI / 2;
        liquidMesh.position.y = -0.14; // start empty at bottom
        mesh.add(liquidMesh);

        mesh.position.set(x, y, z);
        this.scene.add(mesh);

        const body = new PhysicsBody({
            position: mesh.position.clone(),
            mass: 0.3,
            restitution: 0.2,
            friction: 0.6,
            shape: "cylinder",
            dimensions: new THREE.Vector3(0.18, 0.15, 0.18),
            radius: 0.18,
            mesh: mesh
        });
        
        body.type = 'cup';
        body.name = "Coffee Cup";
        body.fillLevel = 0.0;
        body.sugarCount = 0;
        body.liquidMesh = liquidMesh;

        this.physicsWorld.addBody(body);
        this.items.push(body);
        return body;
    }

    createCoffeePot(x, y, z) {
        const group = new THREE.Group();
        
        // Main body
        const geo = new THREE.CylinderGeometry(0.25, 0.3, 0.6, 16);
        const mat = new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.8, roughness: 0.2 });
        const mesh = new THREE.Mesh(geo, mat);
        group.add(mesh);
        
        // Spout
        const spoutGeo = new THREE.CylinderGeometry(0.02, 0.05, 0.3);
        const spout = new THREE.Mesh(spoutGeo, mat);
        spout.rotation.z = -Math.PI / 4;
        spout.position.set(-0.3, 0.1, 0);
        group.add(spout);

        // Handle
        const handleGeo = new THREE.BoxGeometry(0.1, 0.4, 0.1);
        const handle = new THREE.Mesh(handleGeo, mat);
        handle.position.set(0.3, 0, 0);
        group.add(handle);

        group.position.set(x, y, z);
        this.scene.add(group);

        const body = new PhysicsBody({
            position: group.position.clone(),
            mass: 0.8,
            restitution: 0.1,
            friction: 0.8,
            shape: "cylinder",
            radius: 0.3,
            mesh: group
        });
        
        body.type = 'pot';
        body.name = "Coffee Pot";
        this.coffeePot = body;

        this.physicsWorld.addBody(body);
        this.items.push(body);
        return body;
    }
    
    createSugarCube(x, y, z) {
        const geo = new THREE.BoxGeometry(0.05, 0.05, 0.05);
        const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, y, z);
        this.scene.add(mesh);

        const body = new PhysicsBody({
            position: mesh.position.clone(),
            mass: 0.05,
            restitution: 0.1,
            friction: 0.5,
            shape: "box",
            dimensions: new THREE.Vector3(0.025, 0.025, 0.025),
            radius: 0.04,
            mesh: mesh
        });
        
        body.type = 'sugar';
        body.name = "Sugar Cube";

        this.physicsWorld.addBody(body);
        this.items.push(body);
        return body;
    }
    
    update(dt, gravityDir) {
        // Update liquid shaders for all cups
        for (const item of this.items) {
            if (item.type === 'cup') {
                const liquidMat = item.liquidMesh.material;
                liquidMat.uniforms.uTime.value += dt;
                liquidMat.uniforms.uGravityDir.value.copy(gravityDir);
                
                // Calculate tilt relative to gravity
                const up = new THREE.Vector3(0, 1, 0).applyQuaternion(item.quaternion);
                const tilt = up.angleTo(gravityDir.clone().negate()); // angle to actual "up"
                liquidMat.uniforms.uCupTilt.value = tilt;
                
                // Adjust plane height based on fill level
                item.liquidMesh.position.y = -0.14 + (item.fillLevel * 0.28);
                
                // Handle spilling
                if (tilt > Math.PI / 3 && item.fillLevel > 0) { // > 60 degrees
                    item.fillLevel -= dt * 0.5;
                    item.fillLevel = Math.max(0, item.fillLevel);
                    if(item.fillLevel > 0 && Math.random() < 0.2) {
                        // Emit splill particle
                        if(window.gameEngine) window.gameEngine.emitSpillParticle(item.position);
                    }
                }
            }
        }
    }
}