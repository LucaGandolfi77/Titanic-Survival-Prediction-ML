import * as THREE from 'three';

export class GameAudio {
    constructor() {
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.value = 0.5; // Initial volume
        this.masterGain.connect(this.ctx.destination);
    }

    resume() {
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    // A low thud when the tail sweeps or hits something heavy
    playThud(intensity = 1.0) {
        if (!this.ctx) return;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.connect(gain);
        gain.connect(this.masterGain);

        osc.type = 'sine'; // deeply resonant thud
        
        let freq = 150 * (0.8 + Math.random() * 0.4);
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
        // Rapid pitch drop for impact impact punch
        osc.frequency.exponentialRampToValueAtTime(10, this.ctx.currentTime + 0.3);

        const dur = 0.4;
        let v = 0.6 * intensity;
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(v, this.ctx.currentTime + 0.05); // sharp attack
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + dur); // long decay

        osc.start(this.ctx.currentTime);
        osc.stop(this.ctx.currentTime + dur);
    }

    // Crunching sound when a building is destroyed or trash is hit
    playCrunch() {
        if (!this.ctx) return;
        const dur = 0.3;
        
        // Use an AudioBuffer for noise
        const bufferSize = this.ctx.sampleRate * dur;
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);

        // Brown noise generation
        let lastOut = 0;
        for (let i = 0; i < bufferSize; i++) {
            const white = Math.random() * 2 - 1;
            lastOut = (lastOut + (0.02 * white)) / 1.02;
            data[i] = lastOut * 3.5; 
        }

        const noiseNode = this.ctx.createBufferSource();
        noiseNode.buffer = buffer;

        // Bandpass filter to make it sound "stony/crunchy"
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = 800 + Math.random() * 400; // 800-1200 hz
        filter.Q.value = 1.0;
        
        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(0.5, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + dur);

        noiseNode.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);

        noiseNode.start(this.ctx.currentTime);
    }

    // Snore / Yawn
    playYawn() {
        if(!this.ctx) return;
        
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(200, this.ctx.currentTime);
        // Groan dropping pitch
        osc.frequency.exponentialRampToValueAtTime(60, this.ctx.currentTime + 1.2);
        
        const dur = 1.5;
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.3, this.ctx.currentTime + 0.4);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + dur);
        
        osc.start(this.ctx.currentTime);
        osc.stop(this.ctx.currentTime + dur);
    }
    
    // UI Click
    playClick() {
        if(!this.ctx) return;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.type = 'square';
        osc.frequency.setValueAtTime(800, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(100, this.ctx.currentTime + 0.1);
        
        gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.1);
        
        osc.start(this.ctx.currentTime);
        osc.stop(this.ctx.currentTime + 0.1);
    }
    
    // Whoosh for sweeping
    playSweep() {
        if(!this.ctx) return;
        
        const dur = 0.5;
        const bufferSize = this.ctx.sampleRate * dur;
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for(let i=0; i<bufferSize; i++) {
            data[i] = Math.random() * 2 - 1; // pure white noise
        }
        
        const noiseNode = this.ctx.createBufferSource();
        noiseNode.buffer = buffer;
        
        // Sweeping filter
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(100, this.ctx.currentTime);
        filter.frequency.exponentialRampToValueAtTime(2000, this.ctx.currentTime + 0.2); // sweep up
        filter.frequency.exponentialRampToValueAtTime(100, this.ctx.currentTime + dur); // sweep down
        
        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.2, this.ctx.currentTime + 0.1);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + dur);
        
        noiseNode.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);
        
        noiseNode.start();
    }
    
    // Positive eco-reward ding
    playDing() {
      if(!this.ctx) return;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, this.ctx.currentTime); // A5
        osc.frequency.setValueAtTime(1318.51, this.ctx.currentTime + 0.1); // E6
        
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.3, this.ctx.currentTime + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.4);
        
        osc.start(this.ctx.currentTime);
        osc.stop(this.ctx.currentTime + 0.4);
    }
}