// Procedural world building for each cell
import { CELL_THEMES, PORTAL_POSITIONS, HYPERCUBE_GRAPH } from './hypercube.js';

export class WorldBuilder {
    constructor() {
        this.materialCache = {};
    }

    getMaterial(color, roughness = 0.8) {
        const key = `${color}_${roughness}`;
        if (!this.materialCache[key]) {
            this.materialCache[key] = new THREE.MeshStandardMaterial({ 
                color: new THREE.Color(color),
                roughness: roughness,
                metalness: 0.2
            });
        }
        return this.materialCache[key];
    }

    buildCell(cellId, levelConfig) {
        const group = new THREE.Group();
        group.userData = { cellId: cellId };
        
        const theme = CELL_THEMES[cellId];
        const type = theme.type;

        // Base Room boundaries (100x100x100 minus walls)
        const floorGeo = new THREE.PlaneGeometry(100, 100);
        const floorMat = this.getMaterial(this.lightenColor(theme.bg, 20));
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        group.add(floor);

        // Add walls based on adjacency
        this.buildWalls(group, cellId, theme);

        // Procedural content
        if (type === 'urban') this.buildUrban(group, theme);
        else if (type === 'park') this.buildPark(group, theme);
        else if (type === 'tunnel') this.buildTunnel(group, theme);
        else if (type === 'void') this.buildVoid(group, theme);
        else if (type === 'floating') this.buildFloating(group, theme);
        else if (type === 'rooftop') this.buildRooftop(group, theme);
        else this.buildWarehouse(group, theme);

        return group;
    }

    buildWalls(group, cellId, theme) {
        const connections = HYPERCUBE_GRAPH[cellId];
        const wallMat = this.getMaterial(theme.bg, 0.9);
        const wallMatTrans = new THREE.MeshStandardMaterial({
            color: theme.bg, 
            transparent: true, 
            opacity: 0.5,
            wireframe: true
        });

        connections.forEach(conn => {
            // we leave a hole for portals, or just represent boundaries
            const wallMesh = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), wallMatTrans);
            const pos = PORTAL_POSITIONS[conn.dir].pos;
            wallMesh.position.copy(pos).multiplyScalar(2); // push to edge
            wallMesh.lookAt(new THREE.Vector3(0,0,0));
            group.add(wallMesh);
        });
    }

    buildUrban(group, theme) {
        const bldgGeo = new THREE.BoxGeometry(1, 1, 1);
        bldgGeo.translate(0, 0.5, 0); // origin at bottom
        
        const bldgMat = this.getMaterial('#555555');
        const instancedBuildings = new THREE.InstancedMesh(bldgGeo, bldgMat, 100);
        
        const dummy = new THREE.Object3D();
        let idx = 0;
        
        for (let x = -40; x <= 40; x += 20) {
            for (let z = -40; z <= 40; z += 20) {
                if (Math.abs(x) < 10 && Math.abs(z) < 10) continue; // leave center open
                // Random building block
                dummy.position.set(x + (Math.random() * 5 - 2.5), 0, z + (Math.random() * 5 - 2.5));
                dummy.scale.set(8 + Math.random() * 4, 10 + Math.random() * 30, 8 + Math.random() * 4);
                dummy.updateMatrix();
                instancedBuildings.setMatrixAt(idx++, dummy.matrix);
            }
        }
        instancedBuildings.count = idx;
        instancedBuildings.castShadow = true;
        instancedBuildings.receiveShadow = true;
        group.add(instancedBuildings);
    }

    buildPark(group, theme) {
        // Simple trees
        const trunkGeo = new THREE.CylinderGeometry(0.5, 0.7, 4);
        trunkGeo.translate(0, 2, 0);
        const leavesGeo = new THREE.ConeGeometry(3, 6);
        leavesGeo.translate(0, 6, 0);
        
        const trunkMat = this.getMaterial('#5c4033');
        const leavesMat = this.getMaterial(theme.accent);
        
        for(let i=0; i<20; i++) {
            const x = (Math.random() - 0.5) * 80;
            const z = (Math.random() - 0.5) * 80;
            if (Math.abs(x) < 15 && Math.abs(z) < 15) continue; // center open
            
            const trunk = new THREE.Mesh(trunkGeo, trunkMat);
            const leaves = new THREE.Mesh(leavesGeo, leavesMat);
            trunk.position.set(x, 0, z);
            leaves.position.set(x, 0, z);
            group.add(trunk);
            group.add(leaves);
        }
    }

    buildTunnel(group, theme) {
        const mat = new THREE.MeshStandardMaterial({color: theme.accent, wireframe: true});
        const geo = new THREE.CylinderGeometry(15, 15, 100, 16, 1, true);
        geo.rotateX(Math.PI/2);
        const tunnel = new THREE.Mesh(geo, mat);
        tunnel.position.y = 10;
        group.add(tunnel);
    }

    buildVoid(group, theme) {
        // Just floating platforms
        for(let i=0; i<10; i++) {
            const geo = new THREE.BoxGeometry(10+Math.random()*10, 1, 10+Math.random()*10);
            const mesh = new THREE.Mesh(geo, this.getMaterial(theme.accent));
            mesh.position.set((Math.random()-0.5)*60, Math.random()*20, (Math.random()-0.5)*60);
            group.add(mesh);
        }
    }

    buildFloating(group, theme) {
        this.buildVoid(group, theme);
    }

    buildRooftop(group, theme) {
        const geo = new THREE.BoxGeometry(70, 40, 70);
        const mesh = new THREE.Mesh(geo, this.getMaterial('#333333'));
        mesh.position.y = -20; // top aligns with 0
        group.add(mesh);
    }

    buildWarehouse(group, theme) {
        // Shelves
        const geo = new THREE.BoxGeometry(4, 20, 20);
        const mat = this.getMaterial('#888888');
        for(let x=-30; x<=30; x+=20) {
            if(Math.abs(x) < 5) continue;
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(x, 10, -20);
            group.add(mesh);
            const mesh2 = new THREE.Mesh(geo, mat);
            mesh2.position.set(x, 10, 20);
            group.add(mesh2);
        }
    }

    lightenColor(hex, percent) {
        // Dummy lighten just returns accent for simplicity if needed
        return hex;
    }
}