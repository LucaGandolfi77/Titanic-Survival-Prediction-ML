import * as THREE from 'three';
import { MathUtils } from './utils.js';

export const GRAVITY_STATES = {
    "normal":    { vec: new THREE.Vector3(0, -9.8, 0), name: "NORMAL" },
    "reversed":  { vec: new THREE.Vector3(0, 9.8, 0), name: "REVERSED" },
    "left":      { vec: new THREE.Vector3(-9.8, 0, 0), name: "LEFT" },
    "right":     { vec: new THREE.Vector3(9.8, 0, 0), name: "RIGHT" },
    "forward":   { vec: new THREE.Vector3(0, 0, -9.8), name: "FORWARD" },
    "backward":  { vec: new THREE.Vector3(0, 0, 9.8), name: "BACKWARD" },
    "diagonal1": { vec: new THREE.Vector3(-6.93, -6.93, 0), name: "DIAG-L" },
    "diagonal2": { vec: new THREE.Vector3(6.93, -6.93, 0), name: "DIAG-R" },
    "zerog":     { vec: new THREE.Vector3(0, 0, 0), name: "ZERO-G" },
    "micro":     { vec: new THREE.Vector3(0, -1.5, 0), name: "MICRO-G" },
    "heavy":     { vec: new THREE.Vector3(0, -19.6, 0), name: "HEAVY" }
};

export class GravityDirector {
    constructor(physicsWorld, hud) {
        this.physicsWorld = physicsWorld;
        this.hud = hud;
        
        this.currentStateId = "normal";
        this.currentVector = GRAVITY_STATES["normal"].vec.clone();
        this.targetVector = GRAVITY_STATES["normal"].vec.clone();
        
        this.timer = 0;
        this.nextShiftTime = 30; // initial shift time
        
        this.isTransitioning = false;
        this.transitionProgress = 0;
        this.transitionDuration = 2.0;

        this.isWarning = false;
        
        this.availableStates = ["normal"];
        this.chaosMode = false;
        this.chaosTimer = 0;
        
        // Setup initial gravity
        this.physicsWorld.gravity.copy(this.currentVector);
    }
    
    setDifficulty(score) {
        if (score < 500) {
            this.availableStates = ["normal", "normal"]; // Mostly normal
            this.maxTime = 40;
        } else if (score < 1500) {
            this.availableStates = ["normal", "reversed", "left", "right", "micro"];
            this.maxTime = 30;
        } else if (score < 3000) {
            this.availableStates = Object.keys(GRAVITY_STATES).filter(k => k !== "heavy");
            this.maxTime = 20;
        } else {
            this.availableStates = Object.keys(GRAVITY_STATES);
            this.maxTime = 15;
            if(Math.random() < 0.2) this.chaosMode = true; // 20% chance of chaos per shift wait
        }
    }

    update(dt) {
        if (this.chaosMode) {
            this.updateChaos(dt);
            return;
        }

        if (this.isTransitioning) {
            this.transitionProgress += dt / this.transitionDuration;
            if (this.transitionProgress >= 1.0) {
                this.transitionProgress = 1.0;
                this.isTransitioning = false;
            }
            
            // Lerp gravity
            this.currentVector.lerpVectors(this.physicsWorld.gravity, this.targetVector, MathUtils.smoothstep(0, 1, this.transitionProgress));
            this.physicsWorld.gravity.copy(this.currentVector);
            
            this.hud.updateGravityDisplay(this.currentVector, GRAVITY_STATES[this.currentStateId].name, 0, 1);
            return;
        }

        this.timer += dt;
        const timeRemaining = this.nextShiftTime - this.timer;
        
        this.hud.updateGravityDisplay(this.currentVector, GRAVITY_STATES[this.currentStateId].name, timeRemaining, this.nextShiftTime);

        // Warning phase
        if (timeRemaining <= 3 && timeRemaining > 0 && !this.isWarning) {
            this.isWarning = true;
            this.hud.showWarning(true);
            if(window.gameAudio) window.gameAudio.playAlarm();
        }

        // Trigger shift
        if (Math.ceil(this.timer) >= this.nextShiftTime) {
            this.triggerShift();
        }
    }
    
    updateChaos(dt) {
        this.chaosTimer -= dt;
        if(this.chaosTimer <= 0) {
            this.chaosTimer = 3.0; // shifts every 3s
            this.triggerShift(true);
        }
        
        // fast lerp
        this.currentVector.lerp(this.targetVector, dt * 2.0);
        this.physicsWorld.gravity.copy(this.currentVector);
        
        this.hud.updateGravityDisplay(this.currentVector, "CHAOS!", 0, 1);
        this.hud.showWarning(true);
    }

    triggerShift(isChaos = false) {
        this.timer = 0;
        this.isWarning = false;
        this.hud.showWarning(false);
        
        let nextState;
        do {
            nextState = MathUtils.randomChoice(this.availableStates);
        } while (nextState === this.currentStateId && this.availableStates.length > 1);
        
        this.currentStateId = nextState;
        this.targetVector = GRAVITY_STATES[this.currentStateId].vec.clone();
        
        if(!isChaos) {
            this.isTransitioning = true;
            this.transitionProgress = 0;
            this.nextShiftTime = MathUtils.randomRange(15, this.maxTime);
            if(window.gameAudio) window.gameAudio.playShift();
        }
    }
}