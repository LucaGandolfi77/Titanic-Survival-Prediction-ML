class AudioEngine {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.masterGain = null;
  }

  init() {
    if (this.ctx) return;
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioContext();
      this.masterGain = this.ctx.createGain();
      this.masterGain.connect(this.ctx.destination);
      this.masterGain.gain.value = 1.0;
    } catch (e) {
      console.warn("Web Audio API not supported", e);
      this.enabled = false;
    }
  }

  setEnabled(val) {
    this.enabled = val;
    if (this.masterGain) {
      this.masterGain.gain.value = val ? 1.0 : 0;
    }
  }

  playKick(power = 1) {
    if (!this.enabled || !this.ctx) return;
    this._playNoiseBurst(power * 0.5, 0.06, 500);
  }

  playPass() {
    if (!this.enabled || !this.ctx) return;
    this._playNoiseBurst(0.3, 0.04, 800);
  }

  playTackle() {
    if (!this.enabled || !this.ctx) return;
    this._playNoiseBurst(0.4, 0.1, 100, 'lowpass');
  }

  playWhistle() {
    if (!this.enabled || !this.ctx) return;
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const lfo = this.ctx.createOscillator(); // vibrato
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = 2800;

    lfo.type = 'sine';
    lfo.frequency.value = 10;
    const lfoGain = this.ctx.createGain();
    lfoGain.gain.value = 50;

    lfo.connect(lfoGain);
    lfoGain.connect(osc.frequency);

    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.3, t + 0.05);
    gain.gain.setValueAtTime(0.3, t + 0.5);
    gain.gain.linearRampToValueAtTime(0, t + 0.6);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(t);
    lfo.start(t);
    osc.stop(t + 0.6);
    lfo.stop(t + 0.6);
  }

  playCheer() {
    if (!this.enabled || !this.ctx) return;
    const t = this.ctx.currentTime;
    const bufferSize = this.ctx.sampleRate * 3; // 3 seconds
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    const noise = this.ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 800;
    filter.Q.value = 1;

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.8, t + 0.2);
    gain.gain.setValueAtTime(0.8, t + 1.2);
    gain.gain.linearRampToValueAtTime(0, t + 3.0);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    noise.start(t);
  }

  playUI() {
    if (!this.enabled || !this.ctx) return;
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'sine';
    osc.frequency.value = 1200;
    
    gain.gain.setValueAtTime(0.15, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + 0.03);
    
    osc.connect(gain);
    gain.connect(this.masterGain);
    
    osc.start(t);
    osc.stop(t + 0.03);
  }

  _playNoiseBurst(vol, duration, freq, filterType = 'highpass') {
    const t = this.ctx.currentTime;
    const bufferSize = this.ctx.sampleRate * duration;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
    }

    const noise = this.ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = filterType;
    filter.frequency.value = freq;

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(vol, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + duration);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    noise.start(t);
    // Cleanup handled by audio context automatically when source finishes
  }
}

export const audio = new AudioEngine();