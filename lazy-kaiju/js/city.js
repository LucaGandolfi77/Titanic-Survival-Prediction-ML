import * as THREE from 'three';
import { MathUtils } from './utils.js';

export class CityGenerator {
    constructor(sceneManager) {
        this.scene = sceneManager.scene;
        this.cityGroup = new THREE.Group();
        this.scene.add(this.cityGroup);
        
        this.buildings = [];
        this.roadTiles = [];
        
        this.gridSize = 200;
        this.blockSize = 25; // 20 block + 5 road
        
        this.palettes = [0xe8d5b0, 0xc4a882, 0x8fb4c8, 0xc8c8d4, 0xb8a090, 0xd4c4b0];
    }

    generate(levelData) {
        // Clear previous
        while(this.cityGroup.children.length > 0) {
            const child = this.cityGroup.children[0];
            this.cityGroup.remove(child);
            if(child.geometry) child.geometry.dispose();
            if(child.material) child.material.dispose();
        }
        this.buildings = [];
        this.roadTiles = [];

        // Ground
        const groundGeo = new THREE.PlaneGeometry(this.gridSize, this.gridSize);
        const groundMat = new THREE.MeshLambertMaterial({color: 0x3a3a3a});
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.cityGroup.add(ground);

        const blocksX = Math.floor(this.gridSize / this.blockSize);
        const blocksZ = Math.floor(this.gridSize / this.blockSize);
        
        const halfGrid = this.gridSize / 2;

        let ecoPlaced = 0;
        const totalEco = levelData.ecoCount;
        const totalBlocks = blocksX * blocksZ;
        const ecoProbability = totalEco / totalBlocks;

        for (let i = 0; i < blocksX; i++) {
            for (let j = 0; j < blocksZ; j++) {
                const cx = -halfGrid + (i * this.blockSize) + (this.blockSize/2);
                const cz = -halfGrid + (j * this.blockSize) + (this.blockSize/2);
                
                // Save road centers for trash spawning
                this.roadTiles.push(new THREE.Vector3(cx - 10, 0, cz - 10));

                const isEcoBlock = (Math.random() < ecoProbability && ecoPlaced < totalEco) || 
                                   (ecoPlaced < totalEco && i*j > totalBlocks - totalEco);
                
                if (isEcoBlock) ecoPlaced++;

                this.buildBlock(cx, cz, isEcoBlock, levelData);
            }
        }
    }

    buildBlock(cx, cz, isEco, levelData) {
        // Safe inner 20x20 area
        const numBuildings = MathUtils.randInt(3, 6);
        
        for (let b = 0; b < numBuildings; b++) {
            const w = MathUtils.randRange(3, 8);
            const d = MathUtils.randRange(3, 8);
            const h = MathUtils.randRange(5, levelData.maxHeight);
            
            const bx = cx + MathUtils.randRange(-7, 7);
            const bz = cz + MathUtils.randRange(-7, 7);
            
            const geo = new THREE.BoxGeometry(w, h, d);
            geo.computeVertexNormals(); // flat look
            
            let mat, type;

            if (isEco && b === 0) {
                // Force at least one building to be eco
                mat = new THREE.MeshLambertMaterial({color: 0x22c55e});
                type = 'eco';
            } else {
                mat = new THREE.MeshLambertMaterial({color: MathUtils.randItem(this.palettes)});
                type = 'normal';
            }

            const building = new THREE.Mesh(geo, mat);
            building.position.set(bx, h/2, bz);
            building.castShadow = true;
            building.receiveShadow = true;
            
            // Add details
            if (type === 'eco') {
                this.addEcoDetails(building, w, h, d);
            } else if (Math.random() < 0.5) {
                this.addACUnit(building, w, h, d);
            }

            this.cityGroup.add(building);
            
            this.buildings.push({
                mesh: building,
                type: type,
                health: h < 10 ? 1 : (h < 18 ? 2 : 3),
                pos: building.position.clone(),
                size: new THREE.Vector3(w, h, d),
                active: true
            });
        }
    }

    addEcoDetails(parent, w, h, d) {
        // Solar panel
        const solarGeo = new THREE.PlaneGeometry(w*0.6, d*0.6);
        const solarMat = new THREE.MeshBasicMaterial({color: 0x003366, side: THREE.DoubleSide});
        const solar = new THREE.Mesh(solarGeo, solarMat);
        solar.position.set(0, h/2 + 0.1, 0);
        solar.rotation.x = -Math.PI / 2;
        parent.add(solar);
        
        // Green light glow
        const light = new THREE.PointLight(0x22c55e, 1.0, 10);
        light.position.set(0, h/2 + 1, 0);
        parent.add(light);
    }
    
    addACUnit(parent, w, h, d) {
        const acGeo = new THREE.BoxGeometry(1, 1, 1);
        const acMat = new THREE.MeshLambertMaterial({color: 0x888888});
        const ac = new THREE.Mesh(acGeo, acMat);
        ac.position.set(w/4, h/2 + 0.5, -d/4);
        parent.add(ac);
    }

    getRoadTiles() {
        return this.roadTiles;
    }
}