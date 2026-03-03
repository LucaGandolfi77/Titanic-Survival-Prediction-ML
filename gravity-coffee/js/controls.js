import * as THREE from 'three';

export class Controls {
    constructor(cameraObj, physicsWorld, hud) {
        this.cameraObj = cameraObj;
        this.camera = cameraObj.children[0]; // actual camera inside parent
        this.physicsWorld = physicsWorld;
        this.hud = hud;
        
        this.pitch = 0;
        this.yaw = 0;
        this.sensitivity = 0.002;
        
        this.isLocked = false;
        
        this.raycaster = new THREE.Raycaster();
        this.highlightMesh = null;
        this.hoveredBody = null;
        
        this.grabbedBody = null;
        this.isTilting = false;
        this.tiltAngle = 0;
        
        this.initEventListeners();
        this.createHighlightMesh();
    }
    
    createHighlightMesh() {
        const geo = new THREE.SphereGeometry(1, 16, 16);
        const mat = new THREE.MeshBasicMaterial({ 
            color: 0xffffff, 
            wireframe: true, 
            transparent: true, 
            opacity: 0.5 
        });
        this.highlightMesh = new THREE.Mesh(geo, mat);
        this.highlightMesh.visible = false;
        this.cameraObj.parent.add(this.highlightMesh);
    }
    
    initEventListeners() {
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('mousedown', (e) => this.onMouseDown(e));
        document.addEventListener('mouseup', (e) => this.onMouseUp(e));
        document.addEventListener('wheel', (e) => this.onWheel(e), {passive: false});
        
        document.addEventListener('pointerlockchange', () => {
            this.isLocked = document.pointerLockElement === document.body;
            if(!this.isLocked && window.gameEngine && window.gameEngine.state === 'playing') {
                window.gameEngine.pauseGame();
            }
        });
    }
    
    lock() {
        document.body.requestPointerLock();
    }
    
    unlock() {
        document.exitPointerLock();
    }
    
    onMouseMove(e) {
        if (!this.isLocked) return;
        
        const mx = e.movementX || 0;
        const my = e.movementY || 0;
        
        this.yaw -= mx * this.sensitivity;
        this.pitch -= my * this.sensitivity;
        
        // Clamp pitch
        this.pitch = Math.max(-Math.PI/2.5, Math.min(Math.PI/2.5, this.pitch));
        
        this.cameraObj.rotation.y = this.yaw;
        this.camera.rotation.x = this.pitch;
    }
    
    onMouseDown(e) {
        if (!this.isLocked) return;
        
        if (e.button === 0) { // Left click
            if (this.grabbedBody) {
                // Drop
                this.dropObject();
            } else if (this.hoveredBody) {
                // Grab
                this.grabObject(this.hoveredBody);
            }
        } else if (e.button === 2) { // Right click
            if(this.grabbedBody) {
                this.isTilting = true;
            }
        }
    }
    
    onMouseUp(e) {
        if (e.button === 2) {
            this.isTilting = false;
        }
    }
    
    onWheel(e) {
        if(!this.isLocked) return;
        if(this.isTilting && this.grabbedBody) {
            this.tiltAngle += e.deltaY * 0.01;
            this.tiltAngle = Math.max(0, Math.min(Math.PI, this.tiltAngle)); // 0 to 180 deg
            e.preventDefault();
        }
    }
    
    update(dt) {
        if(!this.isLocked) return;
        
        // Raycast for hover
        if(!this.grabbedBody) {
            const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(this.cameraObj.quaternion).applyQuaternion(this.camera.quaternion);
            this.raycaster.set(this.cameraObj.position, dir);
            
            let closest = null;
            let minDist = 2.0; // Max reach distance
            
            for(const body of this.physicsWorld.bodies) {
                if(body.isStatic || body.isKinematic) continue; // Note: Ensure only grabbable things are picked
                
                const dist = this.raycaster.ray.distanceToPoint(body.position);
                const projectedDist = this.cameraObj.position.distanceTo(body.position);
                
                if(projectedDist < minDist && dist < body.radius * 1.5) {
                    minDist = projectedDist;
                    closest = body;
                }
            }
            
            this.hoveredBody = closest;
            
            if(closest) {
                this.highlightMesh.visible = true;
                this.highlightMesh.position.copy(closest.position);
                const s = closest.radius * 1.1;
                this.highlightMesh.scale.set(s,s,s);
                this.hud.setCrosshairHover(true);
            } else {
                this.highlightMesh.visible = false;
                this.hud.setCrosshairHover(false);
            }
        } else {
            this.highlightMesh.visible = false;
            
            // Hold mechanics
            const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.cameraObj.quaternion).applyQuaternion(this.camera.quaternion);
            const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.cameraObj.quaternion);
            
            // Position in front of camera
            const holdPos = this.cameraObj.position.clone().add(forward.multiplyScalar(0.7));
            // Move slightly down and right based on tilt? Just center for now.
            holdPos.add(new THREE.Vector3(0, -0.2, 0).applyQuaternion(this.cameraObj.quaternion));
            
            // Lerp object to hold pos
            this.grabbedBody.position.lerp(holdPos, 15 * dt);
            
            // Rotation
            const camQuat = new THREE.Quaternion().multiplyQuaternions(this.cameraObj.quaternion, this.camera.quaternion);
            const targetQuat = camQuat.clone();
            
            if(this.tiltAngle > 0) {
                 const tiltQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), this.tiltAngle);
                 targetQuat.multiply(tiltQuat);
            }
            
            this.grabbedBody.quaternion.slerp(targetQuat, 15 * dt);
            
            // Sync mesh
            if(this.grabbedBody.mesh) {
                this.grabbedBody.mesh.position.copy(this.grabbedBody.position);
                this.grabbedBody.mesh.quaternion.copy(this.grabbedBody.quaternion);
            }
            
            // HUD Update
            this.hud.updateHeldInfo(this.grabbedBody.name, this.tiltAngle, this.grabbedBody.type === 'cup' ? this.grabbedBody.fillLevel : null);
        }
    }
    
    grabObject(body) {
        this.grabbedBody = body;
        body.isKinematic = true;
        this.tiltAngle = 0;
        this.hud.showHeldInfo(true);
    }
    
    dropObject() {
        this.grabbedBody.isKinematic = false;
        
        // Throw roughly in camera direction
        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.cameraObj.quaternion).applyQuaternion(this.camera.quaternion);
        this.grabbedBody.velocity.copy(forward.multiplyScalar(2));
        
        // Reset state
        this.grabbedBody = null;
        this.isTilting = false;
        this.tiltAngle = 0;
        this.hud.showHeldInfo(false);
    }
}