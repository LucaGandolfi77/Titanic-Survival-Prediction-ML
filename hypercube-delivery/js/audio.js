// Web Audio API procedural synthesis 
export class AudioSystem {
    constructor() {
        this.ctx = null;
        this.masterGain = null;
        this.engineOsc = null;
        this.engineGain = null;
        this.initialized = false;
    }
    
    init() {
        if (this.initialized) return;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioContext();
        
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.value = 0.3;
        this.masterGain.connect(this.ctx.destination);
        
        // Engine sound
        this.engineOsc = this.ctx.createOscillator();
        this.engineOsc.type = 'sawtooth';
        
        this.engineGain = this.ctx.createGain();
        this.engineGain.gain.value = 0;
        
        // Filter for engine
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = 400;
        
        this.engineOsc.connect(filter);
        filter.connect(this.engineGain);
        this.engineGain.connect(this.masterGain);
        
        this.engineOsc.start();
        this.initialized = true;
    }
    
    updateEngine(speed, maxSpeed) {
        if (!this.initialized) return;
        
        const ratio = Math.abs(speed) / maxSpeed;
        
        // Frequency shifts with speed
        const freq = 40 + (ratio * 120);
        this.engineOsc.frequency.setTargetAtTime(freq, this.ctx.currentTime, 0.1);
        
        // Volume
        const vol = 0.05 + (ratio * 0.15);
        this.engineGain.gain.setTargetAtTime(vol, this.ctx.currentTime, 0.1);
    }
    
    playPortalWhoosh() {
        if (!this.initialized) return;
        
        const duration = 1.0;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        const filter = this.ctx.createBiquadFilter();
        
        osc.type = 'sine';
        
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(200, this.ctx.currentTime);
        filter.frequency.exponentialRampToValueAtTime(800, this.ctx.currentTime + duration/2);
        filter.frequency.exponentialRampToValueAtTime(100, this.ctx.currentTime + duration);
        
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.5, this.ctx.currentTime + 0.2);
        gain.gain.linearRampToValueAtTime(0, this.ctx.currentTime + duration);
        
        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);
        
        osc.start();
        osc.stop(this.ctx.currentTime + duration);
    }
    
    playDeliverySuccess() {
        if (!this.initialized) return;
        
        const t = this.ctx.currentTime;
        const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
        
        notes.forEach((freq, i) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            
            osc.type = 'triangle';
            osc.frequency.value = freq;
            
            gain.gain.setValueAtTime(0, t + i*0.1);
            gain.gain.linearRampToValueAtTime(0.2, t + i*0.1 + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.01, t + i*0.1 + 0.3);
            
            osc.connect(gain);
            gain.connect(this.masterGain);
            
            osc.start(t + i*0.1);
            osc.stop(t + i*0.1 + 0.3);
        });
    }

    playHonk() {
        if (!this.initialized) return;
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc.type = 'square';
        osc.frequency.setValueAtTime(300, t);
        osc.frequency.linearRampToValueAtTime(280, t + 0.2);
        
        gain.gain.setValueAtTime(0.2, t);
        gain.gain.linearRampToValueAtTime(0, t + 0.3);
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.start(t);
        osc.stop(t + 0.3);
    }
    
    resume() {
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }
}