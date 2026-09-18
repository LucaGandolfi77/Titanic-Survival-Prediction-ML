export class AudioManager {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.muted = false;
    this.initialized = false;
    this.engineOsc = null;
    this.engineGain = null;
    this.engineFilter = null;
  }

  init() {
    if (this.initialized) return;
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.3;
      this.masterGain.connect(this.ctx.destination);
      this.initialized = true;
    } catch (e) {
      console.warn('AudioContext not available');
    }
  }

  resume() {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  toggleMute() {
    this.muted = !this.muted;
    if (this.masterGain) {
      this.masterGain.gain.setTargetAtTime(this.muted ? 0 : 0.3, this.ctx.currentTime, 0.05);
    }
    return this.muted;
  }

  startEngine() {
    if (!this.ctx) return;
    this.engineOsc = this.ctx.createOscillator();
    this.engineOsc.type = 'sawtooth';
    this.engineOsc.frequency.value = 80;

    this.engineFilter = this.ctx.createBiquadFilter();
    this.engineFilter.type = 'lowpass';
    this.engineFilter.frequency.value = 300;

    this.engineGain = this.ctx.createGain();
    this.engineGain.gain.value = 0;

    this.engineOsc.connect(this.engineFilter);
    this.engineFilter.connect(this.engineGain);
    this.engineGain.connect(this.masterGain);
    this.engineOsc.start();
  }

  updateEngine(speed) {
    if (!this.ctx || !this.engineOsc) return;
    const t = this.ctx.currentTime;
    const normalizedSpeed = Math.min(speed / 30, 1);
    this.engineOsc.frequency.setTargetAtTime(80 + normalizedSpeed * 140, t, 0.1);
    this.engineGain.gain.setTargetAtTime(normalizedSpeed * 0.1, t, 0.05);
    this.engineFilter.frequency.setTargetAtTime(300 + normalizedSpeed * 800, t, 0.1);
  }

  stopEngine() {
    if (this.engineOsc) {
      this.engineOsc.stop();
      this.engineOsc = null;
      this.engineGain = null;
      this.engineFilter = null;
    }
  }

  playCoinCollect() {
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1320, this.ctx.currentTime + 0.08);
    gain.gain.setValueAtTime(0.15, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.15);
    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.15);
  }

  playPowerup() {
    if (!this.ctx) return;
    const notes = [523, 659, 784, 1047];
    notes.forEach((freq, i) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0, this.ctx.currentTime + i * 0.08);
      gain.gain.linearRampToValueAtTime(0.12, this.ctx.currentTime + i * 0.08 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + i * 0.08 + 0.2);
      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(this.ctx.currentTime + i * 0.08);
      osc.stop(this.ctx.currentTime + i * 0.08 + 0.2);
    });
  }

  playJump() {
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(200, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(600, this.ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.1, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.2);
    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.2);
  }

  playSlide() {
    if (!this.ctx) return;
    const bufferSize = this.ctx.sampleRate * 0.2;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
    }
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 2000;
    filter.Q.value = 2;
    const gain = this.ctx.createGain();
    gain.gain.value = 0.08;
    source.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);
    source.start();
  }

  playHit() {
    if (!this.ctx) return;
    const bufferSize = this.ctx.sampleRate * 0.4;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / bufferSize, 2);
    }
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(800, this.ctx.currentTime);
    filter.frequency.exponentialRampToValueAtTime(100, this.ctx.currentTime + 0.3);
    const gain = this.ctx.createGain();
    gain.gain.value = 0.2;
    source.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);
    source.start();
  }

  playGameOver() {
    if (!this.ctx) return;
    const notes = [392, 330, 262, 196];
    notes.forEach((freq, i) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0, this.ctx.currentTime + i * 0.2);
      gain.gain.linearRampToValueAtTime(0.15, this.ctx.currentTime + i * 0.2 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + i * 0.2 + 0.4);
      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(this.ctx.currentTime + i * 0.2);
      osc.stop(this.ctx.currentTime + i * 0.2 + 0.4);
    });
  }

  dispose() {
    this.stopEngine();
    if (this.ctx) {
      this.ctx.close();
      this.ctx = null;
    }
  }
}
