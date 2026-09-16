export class MusicEngine {
    constructor() {
        this.ctx = null;
        this.masterGain = null;
        this.isPlaying = false;
        this.currentState = 'calm';
        this.previousState = 'calm';
        this.transitionTime = 0;
        this.noteIndex = 0;
        this.oscillators = [];
        this.stepInterval = 0.4;
        this.stepTimer = 0;
        this.level = 1;

        this.musicEnabled = true;
        this.volume = 0.12;

        this.markovChain = {
            calm: { calm: 0.7, tense: 0.2, danger: 0.05, gameover: 0.05 },
            tense: { calm: 0.3, tense: 0.5, danger: 0.15, gameover: 0.05 },
            danger: { calm: 0.1, tense: 0.3, danger: 0.5, gameover: 0.1 },
            gameover: { calm: 0.5, tense: 0.1, danger: 0.1, gameover: 0.3 }
        };

        this.notePools = {
            calm: [261.63, 329.63, 392.00, 523.25, 440.00],
            tense: [293.66, 369.99, 440.00, 554.37, 493.88],
            danger: [311.13, 392.00, 466.16, 587.33, 523.25],
            gameover: [220.00, 261.63, 196.00, 246.94, 174.61]
        };
    }

    init() {
        try {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
            this.masterGain = this.ctx.createGain();
            this.masterGain.gain.value = 0;
            this.masterGain.connect(this.ctx.destination);
            return true;
        } catch {
            return false;
        }
    }

    setState(state) {
        this.previousState = this.currentState;
        this.currentState = state;
        this.transitionTime = 0;
    }

    setLevel(level) {
        this.level = level;
        this.stepInterval = Math.max(0.15, 0.4 - level * 0.02);
    }

    start() {
        if (!this.ctx && !this.init()) return;
        if (this.ctx.state === 'suspended') this.ctx.resume();
        this.isPlaying = true;
        this.masterGain.gain.linearRampToValueAtTime(this.volume, this.ctx.currentTime + 1);
    }

    stop() {
        if (!this.ctx) return;
        this.isPlaying = false;
        this.masterGain.gain.linearRampToValueAtTime(0, this.ctx.currentTime + 0.5);
    }

    update(dt, gameState) {
        if (!this.isPlaying || !this.musicEnabled) return;

        this.transitionTime += dt;

        let newState = this.currentState;
        if (gameState === 'danger') newState = 'danger';
        else if (gameState === 'tense') newState = 'tense';
        else if (gameState === 'gameover') newState = 'gameover';
        else if (gameState === 'idle') newState = 'calm';

        if (newState !== this.currentState) {
            this.setState(newState);
        }

        this.stepTimer += dt;
        if (this.stepTimer >= this.stepInterval) {
            this.stepTimer = 0;
            this._playStep();
        }
    }

    _playStep() {
        if (!this.ctx) return;
        const notes = this.notePools[this.currentState];
        const noteFreq = notes[this.noteIndex % notes.length];
        this.noteIndex++;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.masterGain);

        const isMelodic = this.currentState !== 'danger';
        osc.type = isMelodic ? 'sine' : 'sawtooth';
        osc.frequency.setValueAtTime(noteFreq, this.ctx.currentTime);

        const dur = this.stepInterval * 0.9;
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(this.volume * 0.6, this.ctx.currentTime + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + dur);

        osc.start(this.ctx.currentTime);
        osc.stop(this.ctx.currentTime + dur);
        this.oscillators.push(osc);
    }

    setVolume(vol) {
        this.volume = MathUtils.clamp(vol, 0, 0.3);
        if (this.masterGain && this.ctx) {
            this.masterGain.gain.linearRampToValueAtTime(this.volume, this.ctx.currentTime + 0.1);
        }
    }

    toggle() {
        this.musicEnabled = !this.musicEnabled;
        if (!this.musicEnabled && this.masterGain) {
            this.masterGain.gain.linearRampToValueAtTime(0, this.ctx.currentTime + 0.2);
        } else if (this.musicEnabled && this.masterGain) {
            this.masterGain.gain.linearRampToValueAtTime(this.volume, this.ctx.currentTime + 0.2);
        }
    }
}