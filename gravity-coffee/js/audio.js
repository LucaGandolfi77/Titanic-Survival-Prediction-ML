export class AudioSystem {
    constructor() {
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.enabled = true;
        this.masterGain = this.ctx.createGain();
        this.masterGain.connect(this.ctx.destination);
        this.masterGain.gain.value = 0.5;
        
        // Base nodes to keep active
        this.pourNoise = null;
        this.pourGain = null;
    }

    resume() {
        if(this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    toggle() {
        this.enabled = !this.enabled;
        this.masterGain.gain.value = this.enabled ? 0.5 : 0;
        return this.enabled;
    }

    createNoiseBuffer() {
        const bufferSize = this.ctx.sampleRate * 2.0; // 2 seconds
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            data[i] = Math.random() * 2 - 1;
        }
        return buffer;
    }

    playAlarm() {
        if(!this.enabled) return;
        this.resume();
        
        const t = this.ctx.currentTime;
        
        for(let i=0; i<3; i++) {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'square';
            osc.frequency.setValueAtTime(880, t + i*0.4);
            osc.frequency.setValueAtTime(660, t + i*0.4 + 0.2);
            
            gain.gain.setValueAtTime(0, t + i*0.4);
            gain.gain.linearRampToValueAtTime(0.3, t + i*0.4 + 0.05);
            gain.gain.linearRampToValueAtTime(0.3, t + i*0.4 + 0.35);
            gain.gain.linearRampToValueAtTime(0, t + i*0.4 + 0.4);
            
            osc.connect(gain);
            gain.connect(this.masterGain);
            
            osc.start(t + i*0.4);
            osc.stop(t + i*0.4 + 0.4);
        }
    }

    playShift() {
        if(!this.enabled) return;
        this.resume();
        
        const noise = this.ctx.createBufferSource();
        noise.buffer = this.createNoiseBuffer();
        
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'highpass';
        filter.frequency.setValueAtTime(200, this.ctx.currentTime);
        filter.frequency.exponentialRampToValueAtTime(2000, this.ctx.currentTime + 0.8);
        
        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.5, this.ctx.currentTime + 0.1);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.8);
        
        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);
        
        noise.start();
        noise.stop(this.ctx.currentTime + 0.8);
    }
    
    playCollision(force) {
        if(!this.enabled) return;
        this.resume();
        
        const noise = this.ctx.createBufferSource();
        noise.buffer = this.createNoiseBuffer();
        
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = 300 + (force * 100);
        
        const gain = this.ctx.createGain();
        const v = Math.min(1.0, force * 0.1);
        gain.gain.setValueAtTime(v, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.1);
        
        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);
        
        noise.start();
        noise.stop(this.ctx.currentTime + 0.1);
    }
    
    playHappy() {
        if(!this.enabled) return;
        this.resume();
        
        const t = this.ctx.currentTime;
        const freqs = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
        
        freqs.forEach((freq, i) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.value = freq;
            
            gain.gain.setValueAtTime(0, t + i*0.08);
            gain.gain.linearRampToValueAtTime(0.3, t + i*0.08 + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.01, t + i*0.08 + 0.15);
            
            osc.connect(gain);
            gain.connect(this.masterGain);
            
            osc.start(t + i*0.08);
            osc.stop(t + i*0.08 + 0.15);
        });
    }

    playAngry() {
        if(!this.enabled) return;
        this.resume();
        const t = this.ctx.currentTime;
        
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sawtooth';
        
        osc.frequency.setValueAtTime(300, t);
        osc.frequency.linearRampToValueAtTime(150, t + 0.5);
        
        gain.gain.setValueAtTime(0.4, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.5);
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.start(t);
        osc.stop(t + 0.5);
    }
    
    setPouring(isPouring) {
        if(!this.enabled) return;
        this.resume();
        
        if (isPouring && !this.pourNoise) {
            this.pourNoise = this.ctx.createBufferSource();
            this.pourNoise.buffer = this.createNoiseBuffer();
            this.pourNoise.loop = true;
            
            const filter = this.ctx.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.value = 1000;
            
            this.pourGain = this.ctx.createGain();
            this.pourGain.gain.setValueAtTime(0, this.ctx.currentTime);
            this.pourGain.gain.linearRampToValueAtTime(0.3, this.ctx.currentTime + 0.1);
            
            this.pourNoise.connect(filter);
            filter.connect(this.pourGain);
            this.pourGain.connect(this.masterGain);
            
            this.pourNoise.start();
        } else if (!isPouring && this.pourNoise) {
            this.pourGain.gain.linearRampToValueAtTime(0.01, this.ctx.currentTime + 0.1);
            setTimeout(() => {
                if(this.pourNoise) {
                    this.pourNoise.stop();
                    this.pourNoise.disconnect();
                    this.pourNoise = null;
                }
            }, 100);
        }
    }
}