// Portal zones and transition triggers
import { PORTAL_POSITIONS, HYPERCUBE_GRAPH } from './hypercube.js';

export class PortalManager {
    constructor(sceneContainer) {
        this.container = sceneContainer;
        this.activePortals = []; // The visual portal objects
    }

    createVisualsForCell(cellId) {
        // Clear old ones
        this.activePortals.forEach(p => this.container.remove(p));
        this.activePortals = [];

        const connections = HYPERCUBE_GRAPH[cellId];
        
        connections.forEach(conn => {
            const width = 12;
            const height = 10;
            const geo = new THREE.PlaneGeometry(width, height);
            
            // animated portal material
            const mat = new THREE.MeshBasicMaterial({
                color: 0x00e5ff,
                transparent: true,
                opacity: 0.6,
                side: THREE.DoubleSide,
                additiveBlending: true
            });
            
            const mesh = new THREE.Mesh(geo, mat);
            
            const pInfo = PORTAL_POSITIONS[conn.dir];
            // Push slightly towards center to avoid Z-fighting with walls
            const pos = pInfo.pos.clone().multiplyScalar(0.99);
            mesh.position.copy(pos);
            mesh.rotation.copy(pInfo.rot);
            
            mesh.userData = { targetCell: conn.target, targetWall: conn.targetWall, axis: conn.axis, angle: conn.angle, entryDir: conn.dir };
            
            this.container.add(mesh);
            this.activePortals.push(mesh);
        });
    }

    checkIntersections(vanPosition) {
        // Simple distance check to the portal centers
        for (const portal of this.activePortals) {
            // Distance on the plane of the portal
            const dist = vanPosition.distanceTo(portal.position);
            // If very close to the center of a portal
            if (dist < 6) {
                return portal.userData;
            }
        }
        return null;
    }
}