import * as THREE from 'three';

import { SceneManager } from './scene.js';
import { PhysicsWorld } from './physics.js';
import { SpaceBar } from './bar.js';
import { ItemManager } from './objects.js';
import { GravityDirector } from './gravity.js';
import { CustomerSystem } from './customers.js';
import { Controls } from './controls.js';
import { HUD } from './hud.js';
import { AudioSystem } from './audio.js';
import { UI } from './ui.js';

class GameEngine {
    constructor() {
        this.state = 'menu'; // menu, playing, paused, gameover
        
        this.hudManager = new HUD();
        this.audioManager = new AudioSystem();
        window.gameAudio = this.audioManager;
        
        this.sceneManager = new SceneManager();
        this.physicsWorld = new PhysicsWorld();
        
        this.bar = new SpaceBar(this.sceneManager.scene, this.physicsWorld);
        this.itemManager = new ItemManager(this.sceneManager.scene, this.physicsWorld);
        this.gravity = new GravityDirector(this.physicsWorld, this.hudManager);
        this.customers = new CustomerSystem(this.sceneManager.scene, this.hudManager);
        this.controls = new Controls(this.sceneManager.cameraObj, this.physicsWorld, this.hudManager);
        
        this.ui = new UI(this);
        
        this.score = 0;
        this.spills = 0;
        this.maxSpills = 5;
        
        this.clock = new THREE.Clock();
        
        // Spawn initial items
        this.itemManager.createCoffeePot(1, -2.5, -0.5); // counter height is ~ -2.75
        this.itemManager.createCoffeeCup(-1, -2.5, -0.5);
        this.itemManager.createSugarCube(-0.5, -2.5, -0.6);
        this.itemManager.createSugarCube(-0.4, -2.5, -0.6);

        // Menu attract mode
        this.sceneManager.cameraObj.position.set(0, 0, 5);
        
        requestAnimationFrame(this.loop.bind(this));
    }
    
    startGame() {
        this.state = 'playing';
        this.ui.showHUD();
        this.controls.lock();
        this.audioManager.resume();
        
        this.score = 0;
        this.spills = 0;
        this.hudManager.updateScore(this.score);
        this.hudManager.updateSpills(`💧 ${this.spills} / ${this.maxSpills}`);
        
        // Reset camera
        this.sceneManager.cameraObj.position.set(0, 1.7, 0);
        this.sceneManager.cameraObj.rotation.set(0, 0, 0);
        this.controls.pitch = 0;
        this.controls.yaw = 0;
        
        this.gravity.timer = 0;
        this.gravity.nextShiftTime = 30;
        
        this.clock.start();
    }
    
    pauseGame() {
        this.state = 'paused';
        this.controls.unlock();
        this.ui.showPause(this.score);
    }
    
    resumeGame() {
        this.state = 'playing';
        this.ui.showHUD();
        this.controls.lock();
        this.clock.getDelta(); // clear delta
    }
    
    endGame() {
        this.state = 'gameover';
        this.controls.unlock();
        if(this.spills >= this.maxSpills) {
            this.ui.showGameOver(this.score, this.spills);
        }
    }
    
    addScore(pts) {
        this.score += pts;
        this.hudManager.updateScore(this.score);
        this.gravity.setDifficulty(this.score);
    }
    
    addSpill() {
        this.spills++;
        this.hudManager.updateSpills(`💧 ${this.spills} / ${this.maxSpills}`);
        if(this.spills >= this.maxSpills) {
            this.endGame();
            this.ui.showGameOver(this.score, this.spills);
        }
    }
    
    handlePouring(dt) {
        const held = this.controls.grabbedBody;
        if(!held || held.type !== 'pot') {
            this.hudManager.showPourMeter(false);
            this.audioManager.setPouring(false);
            return;
        }
        
        const tilt = this.controls.tiltAngle;
        if(tilt > Math.PI / 4) { // > 45 deg
            // Check for cup underneath
            let cupToFill = null;
            for(const item of this.itemManager.items) {
                if(item.type === 'cup') {
                    // Simple distance check in X/Z
                    const d = Math.hypot(item.position.x - held.position.x, item.position.z - held.position.z);
                    if(d < 0.3 && item.position.y < held.position.y) {
                        cupToFill = item;
                        break;
                    }
                }
            }
            
            if(cupToFill) {
                // Pouring logic
                cupToFill.fillLevel += dt * 0.1; // Fill rate
                cupToFill.fillLevel = Math.min(1.0, cupToFill.fillLevel);
                
                // Show pouring HUD target based on current active order?
                let target = 0.8;
                if(this.customers.customers.length > 0) target = this.customers.customers[0].order.targetFill;
                
                this.hudManager.showPourMeter(true, cupToFill.fillLevel, target);
                this.audioManager.setPouring(true);
                return;
            }
        }
        
        this.hudManager.showPourMeter(false);
        this.audioManager.setPouring(false);
    }
    
    emitSpillParticle(pos) {
        const geo = new THREE.SphereGeometry(0.05);
        const mat = new THREE.MeshBasicMaterial({color: 0x3d1a00});
        const mesh = new THREE.Mesh(geo, mat);
        
        mesh.position.copy(pos);
        // Random velocity
        const vel = new THREE.Vector3(
            (Math.random()-0.5)*2,
            Math.random()*2,
            (Math.random()-0.5)*2
        );
        
        this.sceneManager.addParticle(mesh, vel, 2.0); // 2s lifetime
    }
    
    loop() {
        requestAnimationFrame(this.loop.bind(this));
        
        const dt = this.clock.getDelta();
        
        if (this.state === 'playing') {
            this.physicsWorld.update(dt);
            this.gravity.update(dt);
            this.itemManager.update(dt, this.gravity.currentVector);
            this.controls.update(dt);
            this.customers.update(dt);
            this.handlePouring(dt);
            this.sceneManager.updateParticles(dt, this.gravity.currentVector);
            
            // Check delivery
            if(this.controls.grabbedBody === null) {
                // Check if any loose cup is near customer
                for(let i=this.itemManager.items.length-1; i>=0; i--) {
                    const item = this.itemManager.items[i];
                    if(item.type === 'cup') {
                        if(this.customers.checkDelivery(item)) {
                            // Delivered successfully, remove cup and spawn new one
                            this.sceneManager.scene.remove(item.mesh);
                            this.physicsWorld.removeBody(item);
                            this.itemManager.items.splice(i, 1);
                            
                            // Respawn a new cup on counter
                            setTimeout(() => {
                                this.itemManager.createCoffeeCup(-1.5 + Math.random(), -2.5, -0.5);
                            }, 2000);
                        }
                    }
                }
            }
            
            // Camera effect based on gravity shift (roll)
            // We apply an inverse subtle roll to parent based on gravity direction
            const gDir = this.gravity.currentVector.clone().normalize();
            if(gDir.y > -0.9) { // If not straight down
                // Very rudimentary roll toward shift
                const roll = -gDir.x * 0.2; 
                this.sceneManager.cameraObj.rotation.z = Math.sin(Date.now() * 0.001) * 0.02 + roll; // wobble
            } else {
                this.sceneManager.cameraObj.rotation.z = Math.sin(Date.now() * 0.001) * 0.01; 
            }
            
        } else if (this.state === 'menu') {
            // Attract mode spin
            this.sceneManager.cameraObj.position.x = Math.sin(Date.now() * 0.0005) * 2;
            this.sceneManager.cameraObj.position.z = Math.cos(Date.now() * 0.0005) * 5;
            this.sceneManager.cameraObj.lookAt(0,0,0);
        }
        
        this.sceneManager.render();
    }
}

// Init
window.onload = () => {
    window.gameEngine = new GameEngine();
};