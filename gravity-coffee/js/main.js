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
import { bus } from './event-bus.js';

class GameEngine {
    constructor() {
        this.state = 'menu'; // menu, playing, paused, gameover
        
        this.hudManager = new HUD();
        this.audioManager = new AudioSystem();
        bus.emit("audio:ready", this.audioManager);
        
        this.sceneManager = new SceneManager();
        this.physicsWorld = new PhysicsWorld();

        // Hide loading screen after first render
        requestAnimationFrame(() => {
            const loading = document.getElementById('loading-screen');
            if (loading) {
                loading.classList.add('hidden');
                setTimeout(() => loading.remove(), 600);
            }
        });
        
        this.bar = new SpaceBar(this.sceneManager.scene, this.physicsWorld);
        this.itemManager = new ItemManager(this.sceneManager.scene, this.physicsWorld);
        this.gravity = new GravityDirector(this.physicsWorld, this.hudManager);
        this.customers = new CustomerSystem(this.sceneManager.scene, this.hudManager);
        this.controls = new Controls(this.sceneManager.cameraObj, this.physicsWorld, this.hudManager);
        
        this.ui = new UI(this);
        
        this.score = 0;
        this.spills = 0;
        this.maxSpills = 5;
        this.speedrunActive = false;
        this.speedrunTime = 0;
        this.speedrunSplits = [];
        this.achievements = {
            served_cups: 0, perfect_pours: 0, total_spills: 0,
            stress_mode: false, high_score_500: false, high_score_1000: false,
            speedrun_splits: 0
        };
        this.activePowerUps = {};
        this.powerUpSpawnTimer = 20;
        this.loadAchievements();
        
        this.clock = new THREE.Clock();
        this.particlePool = [];
        for(let i = 0; i < 20; i++) {
            const geo = new THREE.SphereGeometry(0.05);
            const mat = new THREE.MeshBasicMaterial({color: 0x3d1a00, transparent: true, opacity: 1.0});
            const mesh = new THREE.Mesh(geo, mat);
            mesh.visible = false;
            this.sceneManager.scene.add(mesh);
            this.particlePool.push({mesh, active: false, velocity: new THREE.Vector3(), lifeTime: 0, age: 0});
        }
        
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
        this.audioManager.startMusic();
        this.speedrunActive = true;
        this.speedrunTime = 0;
        this.speedrunSplits = [];
        this.speedrunSplits.push({ name: "Start", time: 0 });
        this.ui.showHUD();
        this.controls.lock();
        this.audioManager.resume();
        
        this.score = 0;
        this.spills = 0;
        this.activePowerUps = {};
        this.powerUpSpawnTimer = 20;
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
        this.audioManager.stopMusic();
        this.state = 'gameover';
        this.speedrunActive = false;
        if(this.score >= 500) this.achievements.high_score_500 = true;
        if(this.score >= 1000) this.achievements.high_score_1000 = true;
        this.saveAchievements();
        if(this.spills >= this.maxSpills) {
            this.saveHighScore(this.score);
        }
        this.ui.showGameOver(this.score, this.spills);
        this.controls.unlock();
    }

    getHighScores() {
        try {
            return JSON.parse(localStorage.getItem('gc_highscores') || '[]');
        } catch(e) { return []; }
    }

    saveHighScore(score) {
        const scores = this.getHighScores();
        scores.push({ score, date: new Date().toLocaleDateString() });
        scores.sort((a, b) => b.score - a.score);
        localStorage.setItem('gc_highscores', JSON.stringify(scores.slice(0, 10)));
    }
    
    addScore(pts) {
        this.score += pts;
        this.hudManager.updateScore(this.score);
        this.gravity.setDifficulty(this.score);
    }
    
    addSpill() {
        this.spills++;
        this.hudManager.updateSpills(`💧 ${this.spills} / ${this.maxSpills}`);
        if(this.achievements) this.achievements.total_spills++;
        if(this.spills >= this.maxSpills) {
            this.endGame();
            this.ui.showGameOver(this.score, this.spills);
        }
    }

    toggleStressMode() {
        if(!this.gravity.stressMode) {
            this.gravity.stressMode = true;
            this.achievements.stress_mode = true;
            this.saveAchievements();
            this.gravity.stressTimer = 0;
            this.hudManager.showMessage("Stress Mode: 10 customers, 5s shifts!");
            this.audioManager.playSound && this.audioManager.playSound("alarm");
        } else {
            this.gravity.stressMode = false;
            this.gravity.stressTimer = 0;
            this.hudManager.showMessage("Stress Mode Ended");
        }
    }

    activatePowerUp(type) {
        const durations = { anchor: 10, grip: 15, time: 8 };
        this.activePowerUps[type] = { expiresAt: this.clock.getElapsedTime() + durations[type] };
        this.audioManager.playHappy();
        this.hudManager.showMessage(type.toUpperCase() + " activated!");
    }

    // Speedrun timer
    updateSpeedrun(dt) {
        if(!this.speedrunActive) return;
        this.speedrunTime += dt;
    }

    addSpeedrunSplit(name) {
        if(!this.speedrunActive) return;
        this.speedrunSplits.push({ name, time: this.speedrunTime });
    }

    // Achievement tracking
    trackAchievement(type) {
        if(!this.achievements) return;
        if(type in this.achievements) {
            this.achievements[type]++;
        }
    }

    loadAchievements() {
        try {
            const saved = localStorage.getItem('gc_achievements');
            if(saved) {
                const data = JSON.parse(saved);
                if(data.served_cups) this.achievements.served_cups = data.served_cups;
                if(data.perfect_pours) this.achievements.perfect_pours = data.perfect_pours;
                if(data.total_spills !== undefined) this.achievements.total_spills = data.total_spills;
            }
        } catch(e) { /* ignore */ }
    }

    saveAchievements() {
        try {
            localStorage.setItem('gc_achievements', JSON.stringify({
                served_cups: this.achievements.served_cups,
                perfect_pours: this.achievements.perfect_pours,
                total_spills: this.achievements.total_spills
            }));
        } catch(e) { /* ignore */ }
    }
    
    handlePouring(dt) {
        const held = this.controls.grabbedBody;
        if(!held || held.type !== 'pot') {
            this.hudManager.showPourMeter(false);
            this.audioManager.setPouring(false);
            return;
        }

        const tilt = this.controls.tiltAngle;
        if(tilt > Math.PI / 3) { // > 60 deg - consistent with objects.js threshold
            let cupToFill = null;
            for(const item of this.itemManager.items) {
                if(item.type === 'cup') {
                    const d = Math.hypot(item.position.x - held.position.x, item.position.z - held.position.z);
                    if(d < 0.3 && item.position.y < held.position.y) {
                        cupToFill = item;
                        break;
                    }
                }
            }

            if(cupToFill) {
                const rate = cupToFill.fillRate || 0.1;
                cupToFill.fillLevel = Math.min(1.0, cupToFill.fillLevel + dt * rate);
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
        if(!this.particlePool) return;
        const item = this.particlePool.find(p => !p.active);
        if(!item) return;
        item.active = true;
        item.mesh.visible = true;
        item.mesh.position.copy(pos);
        item.velocity.set(
            (Math.random()-0.5)*2,
            Math.random()*2,
            (Math.random()-0.5)*2
        );
        item.age = 0;
        item.lifeTime = 2.0;
        if(item.mesh.material) item.mesh.material.opacity = 1.0;
    }
    
    loop() {
        requestAnimationFrame(this.loop.bind(this));
        
        const dt = this.clock.getDelta();
        
        if (this.state === 'playing') {
            if(this.speedrunActive) {
                this.speedrunTime += dt;
                // Auto-split at milestones
                if(this.speedrunSplits.length < 5) {
                    const lastSplit = this.speedrunSplits[this.speedrunSplits.length - 1];
                    if(this.speedrunTime - lastSplit.time > 15) {
                        this.speedrunSplits.push({ name: `Split ${this.speedrunSplits.length}`, time: this.speedrunTime });
                        this.achievements.speedrun_splits = this.speedrunSplits.length - 1;
                        this.saveAchievements();
                    }
                }
            }
            this.physicsWorld.update(dt);
            this.gravity.update(dt, this.gravity.stressMode);
            this.customers.update(dt, this.gravity.stressMode);
            this.itemManager.update(dt, this.gravity.currentVector);
            this.controls.update(dt, this.state === 'paused');
            if(this.controls.spectatorMode) this.controls.updateSpectator(dt);

            // Power-up spawning
            this.powerUpSpawnTimer -= dt;
            if(this.powerUpSpawnTimer <= 0 && Object.keys(this.activePowerUps).length < 3) {
                const types = ['anchor', 'grip', 'time'];
                const type = types[Math.floor(Math.random() * types.length)];
                const x = -2 + Math.random() * 4;
                const z = -1 + Math.random() * 2;
                const pu = this.itemManager.createPowerUp(x, -2.5, z, type);
                this.activePowerUps[pu.id] = pu;
                this.powerUpSpawnTimer = 25 + Math.random() * 15;
            }

            // Check power-up pickup
            for(let i = this.itemManager.items.length - 1; i >= 0; i--) {
                const item = this.itemManager.items[i];
                if(item.type === 'powerup') {
                    const camPos = this.sceneManager.cameraObj.position;
                    const d = Math.hypot(item.position.x - camPos.x, item.position.z - camPos.z);
                    if(d < 1.0) {
                        this.activatePowerUp(item.powerUpType);
                        this.sceneManager.scene.remove(item.mesh);
                        this.physicsWorld.removeBody(item);
                        this.itemManager.items.splice(i, 1);
                        break;
                    }
                }
            }

            this.hudManager.updateSpeedrun(this.speedrunTime, this.speedrunSplits);
            this.sceneManager.updateParticles(dt, this.gravity.currentVector);
            
            // Check delivery
            if(this.controls.grabbedBody === null) {
                // Check if any loose cup is near customer
                for(let i=this.itemManager.items.length-1; i>=0; i--) {
                    const item = this.itemManager.items[i];
                    if(item.type === 'cup') {
                        if(this.customers.checkDelivery(item)) {
                            // Delivered successfully, track achievement
                            this.achievements.served_cups++;
                            if(item.fillLevel > 0.95) this.achievements.perfect_pours++;
                            this.saveAchievements();
                            this.sceneManager.scene.remove(item.mesh);
                            this.physicsWorld.removeBody(item);
                            this.itemManager.items.splice(i, 1);
                            
                            // Respawn a new cup on counter (game-clock scheduled)
                            this.pendingSpawns = this.pendingSpawns || [];
                            this.pendingSpawns.push({ delay: 2.0, action: () => {
                                if(this.state !== 'gameover') {
                                    this.itemManager.createCoffeeCup(-1.5 + Math.random(), -2.5, -0.5);
                                }
                            }});
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
                this.sceneManager.cameraObj.rotation.z = Math.sin(this.clock.getElapsedTime() * 0.5) * 0.02 + roll; // wobble
            } else {
                this.sceneManager.cameraObj.rotation.z = Math.sin(this.clock.getElapsedTime() * 0.3) * 0.01; 
            }
            
            // Process pending spawns
            if(this.pendingSpawns) {
                for(let i = this.pendingSpawns.length - 1; i >= 0; i--) {
                    this.pendingSpawns[i].delay -= dt;
                    if(this.pendingSpawns[i].delay <= 0) {
                        this.pendingSpawns[i].action();
                        this.pendingSpawns.splice(i, 1);
                    }
                }
            }
            
        } else if (this.state === 'menu') {
            // Attract mode spin
            this.sceneManager.cameraObj.position.x = Math.sin(Date.now() * 0.0005) * 2;
            this.sceneManager.cameraObj.position.z = Math.cos(Date.now() * 0.0005) * 5;
            this.sceneManager.cameraObj.lookAt(0,0,0);

            // Animated menu background
            this.sceneManager.updateMenuCup(this.clock.getElapsedTime());
            this.sceneManager.updateMenuConfetti(this.clock.getDelta());
        }
        
        this.sceneManager.updateDust(this.clock.getElapsedTime() * 1000);
        this.sceneManager.render();
    }
    // === Simple FSM ===
    setState(newState) {
        const oldState = this.state;
        if (oldState === newState) return;
        bus.emit("state:exit", { from: oldState, to: newState });
        this.state = newState;
        bus.emit("state:enter", { from: oldState, to: newState });
    }

}

// Init
window.onload = () => {
    window.gameEngine = new GameEngine();
    bus.emit("engine:ready", window.gameEngine);
};