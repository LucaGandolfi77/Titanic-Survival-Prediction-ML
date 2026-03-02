// Game loop and state manager
import { SceneManager } from './scene.js';
import { WorldBuilder } from './world.js';
import { Van } from './van.js';
import { PortalManager } from './portals.js';
import { TransitionSystem } from './transitions.js';
import { PackageManager } from './packages.js';
import { Controls } from './controls.js';
import { HUD } from './hud.js';
import { AudioSystem } from './audio.js';
import { UIManager } from './ui.js';
import { CELL_THEMES } from './hypercube.js';

class GameCore {
    constructor() {
        this.sceneMgr = new SceneManager('game-canvas');
        this.worldBuilder = new WorldBuilder();
        this.portals = new PortalManager(this.sceneMgr.worldContainer);
        this.transitions = new TransitionSystem(this.sceneMgr.worldContainer, this.sceneMgr.camera, this);
        this.packages = new PackageManager(this.sceneMgr.worldContainer);
        
        this.van = new Van();
        this.sceneMgr.worldContainer.add(this.van.group);
        
        this.controls = new Controls();
        this.hud = new HUD();
        this.audio = new AudioSystem();
        this.ui = new UIManager(this);
        
        this.isRunning = false;
        this.isPaused = false;
        
        this.currentLevel = 1;
        this.score = 0;
        
        this.activeCells = [];
        this.currentCellId = 0;
        this.visualCellGroups = {}; // Cache of loaded cell groups
        
        this.deliveriesDone = 0;
        this.deliveriesTarget = 0;
        
        this.clock = new THREE.Clock();
        
        // Setup initial camera
        this.cameraOffset = new THREE.Vector3(0, 5, -12); // behind van
        
        // Start loop
        requestAnimationFrame(() => this.gameLoop());
    }
    
    startLevel(level) {
        this.audio.init();
        this.audio.resume();
        
        this.currentLevel = level;
        this.isRunning = true;
        this.isPaused = false;
        this.deliveriesDone = 0;
        this.deliveriesTarget = 2 + level;
        
        // Clear old cells
        Object.values(this.visualCellGroups).forEach(g => {
            this.sceneMgr.worldContainer.remove(g);
        });
        this.visualCellGroups = {};
        this.sceneMgr.worldContainer.quaternion.identity(); // reset rotation
        this.van.group.position.set(0, 0, 0); // reset van
        this.van.group.rotation.set(0, 0, 0);
        this.van.speed = 0;
        
        // Determine active cells based on level
        const numCells = Math.min(8, 2 + level * 2);
        this.activeCells = Array.from({length: numCells}, (_, i) => i);
        
        this.currentCellId = 0;
        this.loadCell(this.currentCellId);
        this.setEnvironmentForCell(this.currentCellId);
        
        this.packages.activePackages = []; // Clear
        this.spawnDelivery();
        
        this.hud.updateScore(this.score, this.currentLevel, this.deliveriesTarget);
        this.hud.updateMinimap(this.currentCellId, this.packages.activePackages);
        
        this.clock.start();
        this.ui.showToast(`Level ${level} Started!`);
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        if (!this.isPaused) this.clock.getDelta(); // clear accumulated time
    }
    
    spawnDelivery() {
        // Can have multiple active packages at higher levels
        const desiredActive = Math.min(3, Math.ceil(this.currentLevel / 2));
        while(this.packages.activePackages.length < desiredActive) {
            this.packages.generatePackageForLevel(this.currentLevel, this.activeCells);
        }
        this.packages.createVisuals(this.currentCellId);
        this.hud.updatePackages(this.packages.activePackages);
        this.hud.updateMinimap(this.currentCellId, this.packages.activePackages);
    }
    
    loadCell(cellId) {
        // Load target cell if not cached
        if (!this.visualCellGroups[cellId]) {
            const group = this.worldBuilder.buildCell(cellId, this.currentLevel);
            this.visualCellGroups[cellId] = group;
        }
        
        // Set visibility
        Object.keys(this.visualCellGroups).forEach(id => {
            this.visualCellGroups[id].visible = (parseInt(id) === cellId);
        });
        
        this.sceneMgr.worldContainer.add(this.visualCellGroups[cellId]);
        this.portals.createVisualsForCell(cellId);
        
        // Re-create package meshes for the new cell
        this.packages.createVisuals(cellId);
    }
    
    setEnvironmentForCell(cellId) {
        const theme = CELL_THEMES[cellId];
        this.sceneMgr.setEnvironmentColor(theme.bg, theme.fog, 0.015);
    }

    prepareNextCell(nextCellId) {
        // Build/Add next cell to world container BEFORE flip
        // We set it visible, so both are visible during the flip
        if (!this.visualCellGroups[nextCellId]) {
            this.visualCellGroups[nextCellId] = this.worldBuilder.buildCell(nextCellId, this.currentLevel);
            this.sceneMgr.worldContainer.add(this.visualCellGroups[nextCellId]);
        }
        this.visualCellGroups[nextCellId].visible = true;
        
        // Position van to coming into the new cell (invert entry pos)
        // If entry is +x wall (50,0,0), new position is -x wall (-48,0,0)
        // Note: The world container rotation handles the axis alignment! 
        // We just need to pop the van backwards slightly relative to its own facing direction
        const bounceBack = new THREE.Vector3(0,0,-8).applyQuaternion(this.van.group.quaternion);
        this.van.group.position.add(bounceBack);
        
        this.audio.playPortalWhoosh();
    }

    finalizeTransition(newCellId) {
        // Hide the old cell
        if (this.currentCellId !== newCellId) {
            this.visualCellGroups[this.currentCellId].visible = false;
        }
        
        this.currentCellId = newCellId;
        this.setEnvironmentForCell(newCellId);
        this.portals.createVisualsForCell(newCellId);
        this.packages.createVisuals(newCellId);
        
        this.ui.showToast(CELL_THEMES[newCellId].name);
        this.hud.updateMinimap(this.currentCellId, this.packages.activePackages);
    }

    gameLoop() {
        requestAnimationFrame(() => this.gameLoop());
        
        const dt = Math.min(this.clock.getDelta(), 0.1); // Cap dt
        
        if (!this.isRunning || this.isPaused) return;

        // 1. Controls & Van update
        this.controls.update();
        if (!this.transitions.isTransitioning) {
            this.van.update(dt, this.controls);
        }
        
        // 2. Camera follow (smooth)
        const idealCameraPos = this.van.group.position.clone()
            .add(this.cameraOffset.clone().applyQuaternion(this.van.group.quaternion));
        
        this.sceneMgr.camera.position.lerp(idealCameraPos, 10 * dt);
        
        const lookAtTarget = this.van.group.position.clone().add(new THREE.Vector3(0, 1.5, 0));
        this.sceneMgr.camera.lookAt(lookAtTarget);

        // 3. Portals
        if (!this.transitions.isTransitioning) {
            const hitPortal = this.portals.checkIntersections(this.van.group.position);
            if (hitPortal) {
                // Cannot go to inactive cells
                if (this.activeCells.includes(hitPortal.targetCell)) {
                    this.transitions.startTransition(hitPortal, this.van);
                } else {
                    // Bounce off
                    this.van.speed *= -0.5;
                    this.van.group.position.sub(this.van.velocity.clone().multiplyScalar(0.1));
                    this.ui.showToast("Portal Locked!");
                }
            }
        }
        
        // 4. Transitions update
        this.transitions.update(dt);
        
        // 5. Packages & HUD
        const pkgResult = this.packages.update(dt, this.van.group.position);
        
        if (pkgResult.stateChanged) {
            if (pkgResult.deliveredCount > 0) {
                this.deliveriesDone += pkgResult.deliveredCount;
                this.score += pkgResult.scoreGained;
                this.audio.playDeliverySuccess();
                this.ui.showToast("+ Delivery!");
                this.spawnDelivery();
            }
            // Check fail
            if (this.packages.activePackages.some(p => p.state === 'failed')) {
                this.ui.showGameOver({level: this.currentLevel, score: this.score});
                this.isRunning = false;
            }
            
            this.hud.updateScore(this.score, this.currentLevel, this.deliveriesTarget);
            this.hud.updatePackages(this.packages.activePackages);
            this.hud.updateMinimap(this.currentCellId, this.packages.activePackages);
        }
        
        // Check Level win
        if (this.deliveriesDone >= this.deliveriesTarget && this.isRunning) {
            this.isRunning = false;
            this.ui.showLevelComplete({level: this.currentLevel, deliveries: this.deliveriesDone, score: this.score});
        }
        
        // Audio & Visual updates
        this.audio.updateEngine(this.van.speed, this.van.maxSpeed);
        this.hud.updateSpeed(this.van.speed, this.van.maxSpeed);
        
        if (this.controls.keys.h && !this._hornDown) {
            this.audio.playHonk();
            this._hornDown = true;
        } else if (!this.controls.keys.h) {
            this._hornDown = false;
        }
        
        // Gravity arrow update (to keep 'down' pointing correctly relative to camera)
        // Since camera is attached to world, we don't need complex math for HUD arrow,
        // it just always points down, but during flip it would rotate.
        const arrow = document.getElementById('gravity-arrow');
        if (arrow) {
            // Simplified logic: normal state down is 0
            if (this.transitions.isTransitioning) {
                arrow.style.transform = `rotate(${this.transitions.progress * 90}deg)`;
            } else {
                arrow.style.transform = `rotate(0deg)`;
            }
        }

        // Render
        this.sceneMgr.render();
    }
}

// Bootstrap
window.onload = () => {
    window.game = new GameCore();
};