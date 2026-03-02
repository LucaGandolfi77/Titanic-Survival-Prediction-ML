// Handles the Inception-style world flipping animation
import { easeInOutCubic } from './utils.js';

export class TransitionSystem {
    constructor(worldContainer, camera, gameCore) {
        this.world = worldContainer; // The THREE.Group containing the level
        this.camera = camera;
        this.gameCore = gameCore;
        this.isTransitioning = false;
        this.progress = 0;
        
        this.startQuat = new THREE.Quaternion();
        this.targetQuat = new THREE.Quaternion();
        
        this.duration = 0.8; // seconds
        this.transitionData = null;
    }

    startTransition(portalData, van) {
        this.isTransitioning = true;
        this.progress = 0;
        this.transitionData = portalData;
        
        // Target cell
        const nextCell = portalData.targetCell;
        
        // Determine target rotation for the world
        // If we go through +x portal, we rotate world around Y by -90 deg
        const axis = portalData.axis.clone().normalize();
        const angle = portalData.angle;
        
        this.startQuat.copy(this.world.quaternion);
        
        const qRot = new THREE.Quaternion();
        qRot.setFromAxisAngle(axis, angle);
        this.targetQuat.copy(this.startQuat).multiply(qRot);
        
        // Reposition van logically to the opposite side of the new cell
        // e.g., enter +x portal, exit from -x portal of new cell
        // BUT because we rotate the world, the relative coordinate system 
        // handles most of this. We just need to snap the van to the exit wall 
        // to prevent immediate re-triggering.
        this.gameCore.prepareNextCell(nextCell);
        
        // Screen flash
        const gameCanvas = document.getElementById('game-canvas');
        gameCanvas.classList.add('flash-white');
        setTimeout(() => gameCanvas.classList.remove('flash-white'), 300);
    }

    update(dt) {
        if (!this.isTransitioning) return;

        this.progress += dt / this.duration;
        
        if (this.progress >= 1.0) {
            this.progress = 1.0;
            this.isTransitioning = false;
            this.world.quaternion.copy(this.targetQuat);
            this.gameCore.finalizeTransition(this.transitionData.targetCell);
            return;
        }

        const t = easeInOutCubic(this.progress);
        this.world.quaternion.slerpQuaternions(this.startQuat, this.targetQuat, t);
    }
}