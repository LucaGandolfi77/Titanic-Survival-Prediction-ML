import * as THREE from 'three';
import { MathUtils } from './utils.js';

export class Kaiju {
    constructor(scene) {
        this.group = new THREE.Group();
        this.scene = scene;
        scene.add(this.group);
        
        this.velocity = new THREE.Vector3();
        this.speed = 8;
        this.accel = 6;
        this.decel = 10;
        this.turnSpeed = 2.0;
        this.targetYaw = 0;
        
        this.isMoving = false;
        this.time = 0;
        this.yawnTimer = 0;
        
        this.buildMesh();
        
        // Position variables
        this.boundingBoxOffset = new THREE.Vector3(0, 3, 0);
        this.boundingRadius = 3;
    }
    
    buildMesh() {
        const matGreen = new THREE.MeshLambertMaterial({color: 0x2d5a27});
        const matLight = new THREE.MeshLambertMaterial({color: 0x4a7a2a});
        
        // Body (origin is ground level)
        this.body = new THREE.Mesh(new THREE.BoxGeometry(4, 6, 3), matGreen);
        this.body.position.y = 4;
        this.body.rotation.x = 0.1; // hunched
        this.body.castShadow = true;
        this.group.add(this.body);
        
        // Belly
        const belly = new THREE.Mesh(new THREE.BoxGeometry(3.5, 4, 1), matLight);
        belly.position.set(0, -0.5, 1.2);
        this.body.add(belly);
        
        // Head
        this.headParent = new THREE.Group();
        this.headParent.position.set(0, 3, 1);
        this.body.add(this.headParent);
        
        const head = new THREE.Mesh(new THREE.BoxGeometry(3, 2.5, 2.5), matGreen);
        head.castShadow = true;
        this.headParent.add(head);
        
        const snout = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1, 1.5), matGreen);
        snout.position.set(0, -0.5, 1.5);
        this.headParent.add(snout);
        
        // Eyes
        const eyeGeo = new THREE.SphereGeometry(0.35);
        const eyeMat = new THREE.MeshLambertMaterial({color: 0xffd700, emissive: 0x664400});
        const pupilMat = new THREE.MeshBasicMaterial({color: 0x000000});
        
        const rEye = new THREE.Mesh(eyeGeo, eyeMat);
        rEye.position.set(1, 0.2, 1);
        const lEye = rEye.clone();
        lEye.position.set(-1, 0.2, 1);
        
        this.headParent.add(rEye, lEye);
        
        // Eyelids (lazy)
        this.eyelids = [];
        const lidGeo = new THREE.BoxGeometry(0.8, 0.4, 0.2);
        const rLid = new THREE.Mesh(lidGeo, matGreen);
        rLid.position.set(1, 0.4, 1.2);
        const lLid = rLid.clone();
        lLid.position.set(-1, 0.4, 1.2);
        this.headParent.add(rLid, lLid);
        this.eyelids.push(rLid, lLid);
        
        // Legs
        this.legs = [];
        const legGeo = new THREE.BoxGeometry(1.8, 4, 2);
        const rLeg = new THREE.Mesh(legGeo, matGreen);
        rLeg.position.set(1.5, 2, 0);
        const lLeg = rLeg.clone();
        lLeg.position.set(-1.5, 2, 0);
        this.group.add(rLeg, lLeg);
        this.legs.push(lLeg, rLeg); // left, right
        
        // Arms
        this.arms = [];
        const armGeo = new THREE.BoxGeometry(1.2, 3, 1);
        const rArm = new THREE.Mesh(armGeo, matGreen);
        rArm.position.set(2.5, 1, 0);
        const lArm = rArm.clone();
        lArm.position.set(-2.5, 1, 0);
        this.body.add(rArm, lArm);
        this.arms.push(lArm, rArm);
        
        // Spikes
        for(let i=0; i<4; i++) {
            const spike = new THREE.Mesh(new THREE.TetrahedronGeometry(0.6), matLight);
            spike.position.set(0, 2 - i, -1.8);
            spike.rotation.set(-0.5, 0, 0);
            this.body.add(spike);
        }
        
        // Jaw for yawn
        this.jaw = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.5, 1.5), matGreen);
        this.jaw.position.set(0, -1.2, 1.5);
        this.headParent.add(this.jaw);
        
        // Attach point for tail
        this.tailBase = new THREE.Object3D();
        this.tailBase.position.set(0, 2, -1.5);
        this.group.add(this.tailBase);
    }
    
    update(dt, inputDir) {
        this.time += dt;
        this.yawnTimer -= dt;
        
        if(this.yawnTimer < 0 && Math.random() < 0.005) {
            this.yawn();
        }
        
        // Movement
        if (inputDir.lengthSq() > 0) {
            this.targetYaw = Math.atan2(inputDir.x, inputDir.z); // note Z is FWD
            this.isMoving = true;
            
            // Accelerate
            const targetVel = inputDir.clone().multiplyScalar(this.speed);
            this.velocity.lerp(targetVel, this.accel * dt);
        } else {
            this.isMoving = false;
            // Decelerate
            this.velocity.lerp(new THREE.Vector3(), this.decel * dt);
        }

        // Apply pos bounds to city (200x200)
        this.group.position.add(this.velocity.clone().multiplyScalar(dt));
        this.group.position.x = MathUtils.clamp(this.group.position.x, -95, 95);
        this.group.position.z = MathUtils.clamp(this.group.position.z, -95, 95);

        // Rotation
        if(this.isMoving) {
            this.group.rotation.y = MathUtils.lerpAngle(this.group.rotation.y, this.targetYaw, this.turnSpeed * dt);
        }
        
        this.animate(dt);
    }
    
    animate(dt) {
        // Breathing
        this.body.scale.y = 1.0 + Math.sin(this.time * 2.5) * 0.03;
        
        // Head bobbing (nodding off)
        let yawnOffset = this.yawnTimer > 8 ? (10 - this.yawnTimer) * 0.2 : 0; // If yawning, tilt back
        this.headParent.rotation.x = Math.sin(this.time * 1.8) * 0.05 - yawnOffset;
        
        // Eyelid droop
        let droop = this.yawnTimer > 0 ? 0.5 : (Math.sin(this.time * 0.5) * 0.1 + 0.3);
        this.eyelids[0].position.y = droop;
        this.eyelids[1].position.y = droop;
        
        // Arms dangle
        this.arms[0].rotation.z = 0.1 + Math.sin(this.time * 3) * 0.05;
        this.arms[1].rotation.z = -0.1 - Math.sin(this.time * 3) * 0.05;
        
        // Walk cycle
        if(this.isMoving && this.velocity.lengthSq() > 1) {
            const walkSpeed = this.time * 10;
            this.legs[0].rotation.x = Math.sin(walkSpeed) * 0.4;
            this.legs[1].rotation.x = Math.sin(walkSpeed + Math.PI) * 0.4;
            
            this.legs[0].position.y = 2 + Math.max(0, Math.sin(walkSpeed)) * 0.5;
            this.legs[1].position.y = 2 + Math.max(0, Math.sin(walkSpeed + Math.PI)) * 0.5;
            
            this.body.rotation.z = Math.sin(walkSpeed * 0.5) * 0.05; // slight wobble
        } else {
            this.legs[0].rotation.x = MathUtils.lerp(this.legs[0].rotation.x, 0, 10*dt);
            this.legs[1].rotation.x = MathUtils.lerp(this.legs[1].rotation.x, 0, 10*dt);
            this.legs[0].position.y = MathUtils.lerp(this.legs[0].position.y, 2, 10*dt);
            this.legs[1].position.y = MathUtils.lerp(this.legs[1].position.y, 2, 10*dt);
            this.body.rotation.z = MathUtils.lerp(this.body.rotation.z, 0, 5*dt);
        }
    }
    
    yawn() {
        this.yawnTimer = 10; // State timer
        // Animation handles jaw dropping
        if(window.gameAudio) window.gameAudio.playYawn();
        const popupEvent = new CustomEvent('kaijuPopup', {detail: {text: "😴 GROGG YAWNS..."}});
        window.dispatchEvent(popupEvent);
    }
    
    getPosition() {
        return this.group.position;
    }
    
    getTailAttachPoint() {
        const v = new THREE.Vector3();
        this.tailBase.getWorldPosition(v);
        return v;
    }
}