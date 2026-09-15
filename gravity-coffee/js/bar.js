import * as THREE from 'three';
import { MathUtils } from './utils.js';

export class SpaceBar {
    constructor(scene, physicsWorld) {
        this.scene = scene;
        this.physicsWorld = physicsWorld;
        
        this.roomSize = new THREE.Vector3(12, 8, 10);
        
        this.createProceduralTextures();
        this.buildRoom();
        this.buildBarCounter();
        this.buildStools();
        this.buildShelves();
        this.addDecorations();
    }
    
    createProceduralTextures() {
        // Wood texture for counter
        const woodCanvas = document.createElement('canvas');
        woodCanvas.width = 256; woodCanvas.height = 256;
        const wctx = woodCanvas.getContext('2d');
        wctx.fillStyle = '#2c1810';
        wctx.fillRect(0, 0, 256, 256);
        for(let i = 0; i < 60; i++) {
            wctx.strokeStyle = `rgba(${60 + Math.random()*40}, ${30 + Math.random()*20}, ${10 + Math.random()*15}, ${0.3 + Math.random()*0.4})`;
            wctx.lineWidth = 1 + Math.random() * 3;
            wctx.beginPath();
            const x = Math.random() * 256;
            wctx.moveTo(x, 0);
            for(let y = 0; y < 256; y += 8) {
                wctx.lineTo(x + Math.sin(y * 0.05 + i) * 4, y);
            }
            wctx.stroke();
        }
        this.woodTexture = new THREE.CanvasTexture(woodCanvas);
        this.woodTexture.wrapS = this.woodTexture.wrapT = THREE.RepeatWrapping;
        this.woodTexture.repeat.set(2, 1);

        // Concrete texture for walls
        const concreteCanvas = document.createElement('canvas');
        concreteCanvas.width = 256; concreteCanvas.height = 256;
        const cctx = concreteCanvas.getContext('2d');
        cctx.fillStyle = '#0a0a1a';
        cctx.fillRect(0, 0, 256, 256);
        for(let i = 0; i < 2000; i++) {
            const x = Math.random() * 256;
            const y = Math.random() * 256;
            const brightness = Math.random() * 30;
            cctx.fillStyle = `rgba(${brightness},${brightness},${brightness+10},${0.1 + Math.random()*0.2})`;
            cctx.fillRect(x, y, 1 + Math.random()*2, 1 + Math.random()*2);
        }
        this.concreteTexture = new THREE.CanvasTexture(concreteCanvas);
        this.concreteTexture.wrapS = this.concreteTexture.wrapT = THREE.RepeatWrapping;
        this.concreteTexture.repeat.set(3, 2);

        // Apply seasonal theme
        this.applySeasonalTheme();
    }

    applySeasonalTheme() {
        const theme = MathUtils.getSeasonalTheme();
        if(!theme) return;
        this.seasonalTheme = theme;
        // Update dust color
        if(this.dust && this.dust.material) {
            this.dust.material.color.set(theme.dustColor);
        }
    }

    buildRoom() {
        const hW = this.roomSize.x / 2;
        const hH = this.roomSize.y / 2;
        const hD = this.roomSize.z / 2;
        
        // Physics Planes
        // Floor
        this.physicsWorld.addPlane(new THREE.Vector3(0, 1, 0), hH); // Floor is at -hH, distance from origin is -(-hH)
        // Ceiling
        this.physicsWorld.addPlane(new THREE.Vector3(0, -1, 0), hH);
        // Walls
        this.physicsWorld.addPlane(new THREE.Vector3(1, 0, 0), hW); // Left
        this.physicsWorld.addPlane(new THREE.Vector3(-1, 0, 0), hW); // Right
        this.physicsWorld.addPlane(new THREE.Vector3(0, 0, 1), hD); // Back
        this.physicsWorld.addPlane(new THREE.Vector3(0, 0, -1), hD); // Front
        
        // Visual Room (Inverted box so we see inside)
        const roomGeo = new THREE.BoxGeometry(this.roomSize.x, this.roomSize.y, this.roomSize.z);
        
        // Custom texture attempt using basic material array
        const darkWall = new THREE.MeshStandardMaterial({ map: this.concreteTexture, roughness: 0.9, side: THREE.BackSide });
        const floorMat = new THREE.MeshStandardMaterial({ color: 0x141428, roughness: 0.7, side: THREE.BackSide });
        const ceilingMat = new THREE.MeshStandardMaterial({ color: 0x050510, roughness: 0.8, side: THREE.BackSide });
        
        const materials = [
            darkWall, // Right
            darkWall, // Left
            ceilingMat, // Top
            floorMat, // Bottom
            darkWall, // Front
            darkWall  // Back
        ];
        
        const roomMesh = new THREE.Mesh(roomGeo, materials);
        roomMesh.frustumCulled = false;
        this.scene.add(roomMesh);
        
        // Grid pattern on floor manually
        const gridGeo = new THREE.PlaneGeometry(this.roomSize.x, this.roomSize.z);
        const gridMat = new THREE.MeshBasicMaterial({color: 0x00e5ff, transparent: true, opacity: 0.05, wireframe: true});
        const grid = new THREE.Mesh(gridGeo, gridMat);
        grid.rotation.x = -Math.PI / 2;
        grid.position.y = -hH + 0.01;
        grid.frustumCulled = true;
        this.scene.add(grid);
    }
    
    buildBarCounter() {
        // Counter base
        const baseGeo = new THREE.BoxGeometry(8, 1.2, 1.5);
        const baseMat = new THREE.MeshStandardMaterial({ map: this.woodTexture, roughness: 0.8 });
        const base = new THREE.Mesh(baseGeo, baseMat);
        base.position.set(0, -this.roomSize.y/2 + 0.6, -1);
        base.frustumCulled = true;
        this.scene.add(base);
        
        // Counter top
        const topGeo = new THREE.BoxGeometry(8.2, 0.1, 1.7);
        const topMat = new THREE.MeshStandardMaterial({ map: this.woodTexture, roughness: 0.5, metalness: 0.1 });
        const top = new THREE.Mesh(topGeo, topMat);
        top.position.set(0, -this.roomSize.y/2 + 1.2 + 0.05, -1);
        top.frustumCulled = true;
        this.scene.add(top);
        
        // Add lights under bar
        const light1 = new THREE.PointLight(0xff8c00, 1.5, 5);
        light1.position.set(-3, -this.roomSize.y/2 + 1.0, -0.2);
        this.scene.add(light1);
        
        const light2 = new THREE.PointLight(0xff8c00, 1.5, 5);
        light2.position.set(3, -this.roomSize.y/2 + 1.0, -0.2);
        this.scene.add(light2);
        
        // Add physics body for counter (Static)
        // Simplified as kinematic/static Box
        // We'll just define planes for the top and sides for simplicity, 
        // or actually let's implement a Static Box body loosely.
        // Actually, to keep custom physics simple, we'll add AABB checks in physics, 
        // but for now, we'll just add the top plane as a "table" surface.
        
        // Top surface of bar
        this.physicsWorld.addPlane(new THREE.Vector3(0, 1, 0), this.roomSize.y/2 - 1.25);
    }
    
    buildStools() {
        const stoolMat = new THREE.MeshStandardMaterial({ color: 0x888888, metalness: 0.8, roughness: 0.2 });
        const seatMat = new THREE.MeshStandardMaterial({ color: 0xff1744, roughness: 0.6 });
        
        const poleGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.8);
        const seatGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.1);
        
        // InstancedMesh for stool poles
        const poleMesh = new THREE.InstancedMesh(poleGeo, stoolMat, 4);
        const poleMatrix = new THREE.Matrix4();
        for(let i=0; i<4; i++) {
            const x = -3 + i*2;
            const z = 0.5;
            poleMatrix.makeTranslation(x, -this.roomSize.y/2 + 0.4, z);
            poleMesh.setMatrixAt(i, poleMatrix);
        }
        poleMesh.instanceMatrix.needsUpdate = true;
        poleMesh.frustumCulled = true;
        poleMesh.computeBoundingSphere();
        this.scene.add(poleMesh);
        
        // InstancedMesh for stool seats
        const seatMesh = new THREE.InstancedMesh(seatGeo, seatMat, 4);
        for(let i=0; i<4; i++) {
            const x = -3 + i*2;
            const z = 0.5;
            poleMatrix.makeTranslation(x, -this.roomSize.y/2 + 0.85, z);
            seatMesh.setMatrixAt(i, poleMatrix);
        }
        seatMesh.instanceMatrix.needsUpdate = true;
        seatMesh.frustumCulled = true;
        seatMesh.computeBoundingSphere();
        this.scene.add(seatMesh);
    }
    
    buildShelves() {
        const z = -this.roomSize.z/2 + 0.2;
        const color = 0x111111;
        const mat = new THREE.MeshStandardMaterial({color});
        
        for(let i=0; i<3; i++) {
            const y = -this.roomSize.y/4 + i * 1.0;
            const geo = new THREE.BoxGeometry(6, 0.1, 0.5);
            const shelf = new THREE.Mesh(geo, mat);
            shelf.position.set(0, y, z);
            shelf.frustumCulled = true;
            this.scene.add(shelf);
            
            // Add physics plane for shelf
            this.physicsWorld.addPlane(new THREE.Vector3(0, 1, 0), -y);
        }
    }
    
    addDecorations() {
        // Neon Sign
        const neonGeo = new THREE.PlaneGeometry(4, 1);
        
        // Create canvas for neon text
        const canvas = document.createElement('canvas');
        canvas.width = 512; canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 512, 128);
        ctx.font = '60px Orbitron';
        ctx.fillStyle = '#00e5ff';
        ctx.textAlign = 'center';
        ctx.fillText('THE FLOATING BEAN', 256, 80);
        
        const tex = new THREE.CanvasTexture(canvas);
        const neonMat = new THREE.MeshBasicMaterial({ map: tex, transparent:true, blending: THREE.AdditiveBlending });
        
        const neon = new THREE.Mesh(neonGeo, neonMat);
        neon.position.set(0, 2, -this.roomSize.z/2 + 0.1);
        neon.frustumCulled = false;
        this.scene.add(neon);
        
        // Neon Light
        const neonLight = new THREE.PointLight(0x00e5ff, 1.0, 10);
        neonLight.position.set(0, 2, -this.roomSize.z/2 + 0.5);
        this.scene.add(neonLight);
    }
}