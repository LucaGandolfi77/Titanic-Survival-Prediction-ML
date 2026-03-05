// Main Entry Point
import * as THREE from 'three';
import { MathUtils } from './utils.js';
import { SceneManager } from './scene.js';
import { CityGenerator } from './city.js';
import { Kaiju } from './kaiju.js';
import { Tail } from './tail.js';
import { TrashManager } from './items.js';
import { GameAudio } from './audio.js';

class GameController {
    constructor() {
        this.gameState = 'start'; // 'start', 'playing', 'gameover'
        this.score = 0;
        this.karma = 100; // max 100, min 0
        this.stamina = 100;
        this.maxStamina = 100;
        
        // Setup modules
        this.sceneMgr = new SceneManager();
        // Create city and immediately generate layout so road tiles exist for TrashManager
        this.city = new CityGenerator(this.sceneMgr);
        this.city.generate({ ecoCount: 8, maxHeight: 22 });

        this.kaiju = new Kaiju(this.sceneMgr.scene);
        this.tail = new Tail(this.sceneMgr.scene, this.kaiju, 12, 1.5);
        // Create TrashManager after city.generate so it can spawn items on roads
        this.trashMgr = new TrashManager(this.sceneMgr.scene, this.city);
        
        // Single Audio Singleton handling context
        window.gameAudio = new GameAudio();
        
        // Controls / Inputs
        this.keys = { W:false, A:false, S:false, D:false, SPACE:false };
        this.inputVec = new THREE.Vector3();
        
        // Cooldowns
        this.sweepCooldown = 0;
        this.isSweeping = false;
        
        this.lastTime = performance.now();
        this.setupBindings();
        
        // Start Loop
        requestAnimationFrame(t => this.loop(t));
    }
    
    setupBindings() {
        // Keyboard
        window.addEventListener('keydown', e => {
            const key = e.key.toUpperCase();
            if(this.keys.hasOwnProperty(key)) this.keys[key] = true;
            if(e.code === 'Space') this.keys.SPACE = true;
        });
        window.addEventListener('keyup', e => {
            const key = e.key.toUpperCase();
            if(this.keys.hasOwnProperty(key)) this.keys[key] = false;
            if(e.code === 'Space') this.keys.SPACE = false;
        });
        
        // UI Start (legacy id) and compatibility bindings for menu buttons in HTML
        const startBtn = document.getElementById('start-btn');
        if (startBtn) startBtn.addEventListener('click', () => {
             const ss = document.getElementById('start-screen'); if (ss) ss.classList.add('hidden');
             if (window.gameAudio && typeof window.gameAudio.resume === 'function') window.gameAudio.resume();
             this.gameState = 'playing';
             this.resetStats();
        });

        // New menu IDs present in index.html: wire them up if present
        const btnPlay = document.getElementById('btn-play');
        if (btnPlay) btnPlay.addEventListener('click', () => {
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
            const hud = document.getElementById('hud'); if (hud) hud.classList.remove('hidden');
            const mobile = document.getElementById('mobile-controls'); if (mobile) mobile.classList.remove('hidden');
            if (window.gameAudio && typeof window.gameAudio.resume === 'function') window.gameAudio.resume();
            this.gameState = 'playing';
            this.resetStats();
            if (this.trashMgr && typeof this.trashMgr.initialSpawn === 'function') this.trashMgr.initialSpawn(50);
        });

        const btnHowTo = document.getElementById('btn-howto');
        if (btnHowTo) btnHowTo.addEventListener('click', () => {
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
            const how = document.getElementById('screen-howto'); if (how) how.classList.remove('hidden');
        });
        const btnHowToBack = document.getElementById('btn-howto-back');
        if (btnHowToBack) btnHowToBack.addEventListener('click', () => {
            const how = document.getElementById('screen-howto'); if (how) how.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnHiscores = document.getElementById('btn-hiscores');
        if (btnHiscores) btnHiscores.addEventListener('click', () => {
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
            const hs = document.getElementById('screen-hiscores'); if (hs) hs.classList.remove('hidden');
        });
        const btnHsBack = document.getElementById('btn-hs-back');
        if (btnHsBack) btnHsBack.addEventListener('click', () => {
            const hs = document.getElementById('screen-hiscores'); if (hs) hs.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnSettings = document.getElementById('btn-settings');
        if (btnSettings) btnSettings.addEventListener('click', () => {
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.add('hidden');
            const s = document.getElementById('screen-settings'); if (s) s.classList.remove('hidden');
        });
        const btnSetBack = document.getElementById('btn-set-back');
        if (btnSetBack) btnSetBack.addEventListener('click', () => {
            const s = document.getElementById('screen-settings'); if (s) s.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });

        const btnLsBack = document.getElementById('btn-ls-back');
        if (btnLsBack) btnLsBack.addEventListener('click', () => {
            const ls = document.getElementById('screen-level-select'); if (ls) ls.classList.add('hidden');
            const menu = document.getElementById('screen-menu'); if (menu) menu.classList.remove('hidden');
        });
        
           // UI Restart
           const restartBtn = document.getElementById('restart-btn');
           if (restartBtn) restartBtn.addEventListener('click', () => {
               const go = document.getElementById('game-over-screen'); if (go) go.classList.add('hidden');
               this.gameState = 'playing';
               this.resetStats();
               if (this.city && typeof this.city.rebuild === 'function') this.city.rebuild();
               if (this.trashMgr && typeof this.trashMgr.initialSpawn === 'function') this.trashMgr.initialSpawn(50);
               if (this.kaiju && this.kaiju.group && this.kaiju.group.position) this.kaiju.group.position.set(0, 0, 0);
           });
        
        // Attack overlay
        const sweepBtn = document.getElementById('sweep-btn');
        if (sweepBtn) {
            sweepBtn.addEventListener('touchstart', (e) => {
                e.preventDefault();
                this.keys.SPACE = true;
            });
            sweepBtn.addEventListener('touchend', (e) => {
                e.preventDefault();
                this.keys.SPACE = false;
            });
        }
        
        // Listeners
           window.addEventListener('trashCleared', (e) => {
               this.score++;
               const tc = document.getElementById('trash-count'); if (tc) tc.innerText = this.score;
               if (window.gameAudio && typeof window.gameAudio.playClick === 'function') window.gameAudio.playClick();
               // Gain karma for sweeping off map
               this.modKarma(2);
           });
        
        window.addEventListener('kaijuPopup', (e) => {
            const hud = document.getElementById('hud-center');
            if (hud) {
                hud.innerText = e.detail.text;
                hud.style.opacity = 1;
                setTimeout(() => { if (hud) hud.style.opacity = 0; }, 2000);
            }
        });
    }
    
    resetStats() {
        this.score = 0;
        this.karma = 100;
        this.stamina = 100;
        const trashCountEl = document.getElementById('trash-count');
        if (trashCountEl) trashCountEl.innerText = "0";
        this.updateKarmaUI();
    }
    
    updateKarmaUI() {
        // Bar update
        const barFill = document.querySelector('.bar-fill');
        if (barFill) {
            barFill.style.width = `${this.karma}%`;
            // Color transition
            if(this.karma > 60) barFill.style.background = 'var(--eco-green)';
            else if(this.karma > 30) barFill.style.background = 'var(--karma-gold)';
            else barFill.style.background = 'magenta';
        }

        if(this.karma <= 0) {
            this.gameState = 'gameover';
            const go = document.getElementById('game-over-screen'); if (go) go.classList.remove('hidden');
        }
    }
    
    modKarma(amount) {
        this.karma = MathUtils.clamp(this.karma + amount, 0, 100);
        this.updateKarmaUI();
        
        if(amount < 0) {
            // Flash screen red
            const flash = document.getElementById('karma-flash');
            if (flash) {
                flash.style.animation = 'none';
                void flash.offsetWidth; // trigger reflow
                flash.style.animation = 'flashFade 0.5s ease-out';
            }
        }
    }
    
    handleInput() {
        this.inputVec.set(0, 0, 0);
        
        if(this.keys.W) this.inputVec.z -= 1;
        if(this.keys.S) this.inputVec.z += 1;
        if(this.keys.A) this.inputVec.x -= 1;
        if(this.keys.D) this.inputVec.x += 1;
        
        if(this.inputVec.lengthSq() > 0) this.inputVec.normalize();
        
        // Handle Sweep logic (cooldown 1.5s, cost 15 stamina)
        if(this.keys.SPACE && this.sweepCooldown <= 0 && this.stamina >= 15) {
            this.performSweep();
        }
    }
    
    performSweep() {
        this.sweepCooldown = 1.5;
        this.stamina -= 15;
        this.isSweeping = true;
        
        // Visual cooldown sweep arc JS update
        const cooldownInner = document.querySelector('.cooldown-inner');
        if (cooldownInner) cooldownInner.style.transform = 'scale(1)';
        setTimeout(() => { 
            if (cooldownInner) cooldownInner.style.transform = 'scale(0)'; 
            this.isSweeping = false;
        }, 300); // quick sweep duration
        
        // Physics Impulse to tail based on facing direction
        let fwd = new THREE.Vector3(0,0,1).applyAxisAngle(new THREE.Vector3(0,1,0), this.kaiju.targetYaw);
        
        // Apply huge rapid rotational torque to the tail segments
        // We apply a "whip" force sideways based on facing direction
        let right = new THREE.Vector3(fwd.z, 0, -fwd.x).multiplyScalar(50); // perpendicular twist
        this.tail.applySweepImpulse(right);
        
        window.gameAudio.playSweep();
    }
    
    checkDestruction() {
        // Destructible logic - check if heavy tail overlaps buildings
        const spheres = this.tail.getCollisionSpheres();
        let blocks = this.city.buildings; // flat array of building groups
        
        for(let i = blocks.length - 1; i >= 0; i--) {
            let b = blocks[i];
            if(!b || !b.userData || b.userData.destroyed) continue;
            
            let pos = b.position;
            // Check radius against tail chain
            for(let s of spheres) {
                let dx = pos.x - s.x;
                let dz = pos.z - s.z;
                // simplified check: buildings are 10x10 approx
                if(dx*dx + dz*dz < (s.r + 5)*(s.r + 5)) {
                    // Destroy!
                    b.userData.destroyed = true;
                    this.sceneMgr.scene.remove(b);
                    blocks.splice(i, 1);
                    
                    window.gameAudio.playCrunch();
                    
                    // Penalty!
                    if(b.userData.isEco) {
                        this.modKarma(-20); // Big penalty for eco buildings
                        window.dispatchEvent(new CustomEvent('kaijuPopup', {detail: {text: "-20 KARMA! Eco-Building Destroyed!"}}));
                    } else {
                        this.modKarma(-5);  // Small penalty for standard
                    }
                    
                    // Spawn particles
                    this.trashMgr.spawnExplosion(pos, 0x555555, 10);
                    break; 
                }
            }
        }
    }
    
    loop(time) {
        requestAnimationFrame(t => this.loop(t));
        
        // Delta time computation
        const dt = (time - this.lastTime) / 1000;
        this.lastTime = time;
        
        // Clamp heavily to avoid absurd physics leaps on tab switches
        const safeDt = Math.min(dt, 0.1);
        
        // Always step visual/animation systems so Kaiju can doze/yawn and tail animates
        // (this allows idle/menu attract mode animations to run)
        this.kaiju.update(safeDt, this.inputVec);
        this.tail.update(safeDt);

        // Only run gameplay mechanics while playing
        if (this.gameState === 'playing') {
            // Regain stamina
            this.stamina = MathUtils.clamp(this.stamina + 5 * safeDt, 0, this.maxStamina);

            // Cooldown sweep
            if(this.sweepCooldown > 0) {
                this.sweepCooldown -= safeDt;
            }

            // Inputs and physics
            this.handleInput();
            if (this.trashMgr && typeof this.trashMgr.update === 'function') {
                this.trashMgr.update(safeDt, this.tail.getCollisionSpheres()); // pass physics state to trash objects
            }

            if(this.isSweeping) {
                this.checkDestruction();
            }
        }
        
        // Camera Follows Kaiju smoothly
        const kPos = this.kaiju.getPosition();
        // Offset: Back and Up
        let fwd = new THREE.Vector3(0,0,1).applyAxisAngle(new THREE.Vector3(0,1,0), this.kaiju.targetYaw);
        
        const offset = new THREE.Vector3(
           -fwd.x * 20,
           15,
           -fwd.z * 20
        );
        
        // Default iso cam when stationary
        if(!this.kaiju.isMoving) {
            offset.set(0, 20, -25);
        }
        
        const targetCamPos = new THREE.Vector3(
            kPos.x + offset.x,
            kPos.y + offset.y,
            kPos.z + offset.z
        );
        
        this.sceneMgr.camera.position.lerp(targetCamPos, 3 * safeDt);
        this.sceneMgr.camera.lookAt(kPos.x, kPos.y + 5, kPos.z);
        
        this.sceneMgr.render();
    }
}

// Bootstrapper
window.onload = () => {
    const game = new GameController();
};