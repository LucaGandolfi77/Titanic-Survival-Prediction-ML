export class AudioManager {
  constructor() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.3;
    this.masterGain.connect(this.ctx.destination);

    this.enabled = true;
    this.spatialSources = [];
    this.listener = null;
  }

  resume() {
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  playFootstep() {
    if (!this.enabled) return;

    const noise = this.ctx.createBufferSource();
    const bufferSize = this.ctx.sampleRate * 0.1;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 150;

    const gain = this.ctx.createGain();
    gain.gain.value = 0.2;
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.1);

    noise.buffer = buffer;
    noise.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    noise.start(this.ctx.currentTime);
  }

  playSwoosh() {
    if (!this.enabled) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.type = 'sine';
    osc.frequency.setValueAtTime(800, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(200, this.ctx.currentTime + 0.3);

    gain.gain.setValueAtTime(0.3, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.3);

    osc.start(this.ctx.currentTime);
    osc.stop(this.ctx.currentTime + 0.3);
  }

  playPhoneRing() {
    if (!this.enabled) return;

    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(800, this.ctx.currentTime);
    osc.frequency.setValueAtTime(600, this.ctx.currentTime + 0.12);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.12, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.2);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.ctx.currentTime);
    osc.stop(this.ctx.currentTime + 0.2);
  }

  playCoffeePour() {
    if (!this.enabled) return;

    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(400, this.ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(600, this.ctx.currentTime + 0.15);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.3);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.ctx.currentTime);
    osc.stop(this.ctx.currentTime + 0.3);
  }

  playClick() {
    if (!this.enabled) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'square';
    osc.frequency.setValueAtTime(600, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(100, this.ctx.currentTime + 0.05);

    gain.gain.setValueAtTime(0.1, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.05);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.ctx.currentTime);
    osc.stop(this.ctx.currentTime + 0.05);
  }

  playPickup() {
    if (!this.enabled) return;

    const freqs = [523.25, 659.25, 783.99]; // C5, E5, G5

    for (let i = 0; i < freqs.length; i++) {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sine';
      osc.frequency.value = freqs[i];

      gain.gain.setValueAtTime(0, this.ctx.currentTime + i * 0.05);
      gain.gain.linearRampToValueAtTime(0.2, this.ctx.currentTime + i * 0.05 + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + i * 0.05 + 0.15);

      osc.connect(gain);
      gain.connect(this.masterGain);

      osc.start(this.ctx.currentTime + i * 0.05);
      osc.stop(this.ctx.currentTime + i * 0.05 + 0.15);
    }
  }

  playDialog() {
    if (!this.enabled) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(220, this.ctx.currentTime);
    osc.frequency.setValueAtTime(200, this.ctx.currentTime + 0.1);
    osc.frequency.setValueAtTime(250, this.ctx.currentTime + 0.2);

    gain.gain.setValueAtTime(0.1, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.4);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.ctx.currentTime);
    osc.stop(this.ctx.currentTime + 0.4);
  }

  playJump() {
    if (!this.enabled) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(300, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(800, this.ctx.currentTime + 0.1);

    gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.1);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.ctx.currentTime);
    osc.stop(this.ctx.currentTime + 0.1);
  }

  playAmbience(type) {
    if (!this.enabled) return;

    // Stop any existing ambience first
    this._stopAmbience();

    // Office hum (continuous)
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 400;

    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.value = 60;

    const gain = this.ctx.createGain();
    gain.gain.value = 0.05;

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.ctx.currentTime);
    this._ambienceOsc = osc;
    this._ambienceGain = gain;

    // Return reference for stopping
    return { osc, gain };
  }

  _stopAmbience() {
    if (this._ambienceOsc) {
      try {
        this._ambienceOsc.stop();
      } catch (e) {}
      this._ambienceOsc = null;
      this._ambienceGain = null;
    }
  }

  playVoidRumble(intensity = 0.5) {
    if (!this.enabled) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = 30;

    gain.gain.value = intensity * 0.1;

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(this.ctx.currentTime);
    this._voidOsc = osc;

    setTimeout(() => {
      if (this._voidOsc === osc) {
        try {
          osc.stop();
        } catch (e) {}
        this._voidOsc = null;
      }
    }, 500);

    return { osc, gain };
  }

  playLeverPull() {
    if (!this.enabled) return;

    // Mechanical clunk
    const noise = this.ctx.createBufferSource();
    const bufferSize = this.ctx.sampleRate * 0.15;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 200;

    const gain = this.ctx.createGain();
    gain.gain.value = 0.3;
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.15);

    noise.buffer = buffer;
    noise.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    noise.start(this.ctx.currentTime);

    // Ring
    const osc = this.ctx.createOscillator();
    const ringGain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.value = 1800;

    ringGain.gain.setValueAtTime(0.1, this.ctx.currentTime);
    ringGain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.04);

    osc.connect(ringGain);
    ringGain.connect(this.masterGain);

    osc.start(this.ctx.currentTime + 0.1);
    osc.stop(this.ctx.currentTime + 0.14);
  }

  setListenerPosition(x, y, z) {
    if (!this.listener) {
      this.listener = this.ctx.listener;
    }
    if (this.listener.positionX) {
      const t = this.ctx.currentTime;
      this.listener.positionX.setValueAtTime(x, t);
      this.listener.positionY.setValueAtTime(y, t);
      this.listener.positionZ.setValueAtTime(z, t);
    }
  }

  playSpatialPickup(position) {
    if (!this.enabled || !position) return;
    this.playSpatial('pickup', position, { duration: 0.25 });
  }

  playSpatialSwoosh(position) {
    if (!this.enabled || !position) return;
    this.playSpatial('swoosh', position, { duration: 0.35 });
  }

  playSpatialDialog(position) {
    if (!this.enabled || !position) return;
    this.playSpatial('dialog', position, { duration: 0.45 });
  }

  playSpatial(type, position, options = {}) {
    if (!this.enabled || !position) return null;

    const duration = options.duration || 0.3;
    const panner = this.ctx.createPanner();
    panner.panningModel = 'HRTF';
    panner.distanceModel = 'inverse';
    panner.refDistance = 1;
    panner.maxDistance = 50;
    panner.rolloffFactor = 1;
    panner.positionX.value = position.x;
    panner.positionY.value = position.y;
    panner.positionZ.value = position.z;
    panner.connect(this.masterGain);

    let source;
    if (type === 'pickup') {
      source = this.ctx.createOscillator();
      source.type = 'sine';
      const freqs = [523.25, 659.25, 783.99];
      source.frequency.value = freqs[Math.floor(Math.random() * freqs.length)];
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0, this.ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.2, this.ctx.currentTime + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.15);
      source.connect(gain);
      gain.connect(panner);
      source.start();
      source.stop(this.ctx.currentTime + 0.2);
    } else if (type === 'swoosh') {
      source = this.ctx.createOscillator();
      source.type = 'sine';
      source.frequency.setValueAtTime(800, this.ctx.currentTime);
      source.frequency.exponentialRampToValueAtTime(200, this.ctx.currentTime + 0.3);
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.3, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.3);
      source.connect(gain);
      gain.connect(panner);
      source.start();
      source.stop(this.ctx.currentTime + 0.35);
    } else if (type === 'dialog') {
      source = this.ctx.createOscillator();
      source.type = 'sawtooth';
      source.frequency.setValueAtTime(220, this.ctx.currentTime);
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.4);
      source.connect(gain);
      gain.connect(panner);
      source.start();
      source.stop(this.ctx.currentTime + 0.45);
    }

    if (source) {
      this.spatialSources.push({ panner, source, startTime: this.ctx.currentTime, duration });
    }
    return { panner };
  }

  updateSpatials() {
    const now = this.ctx.currentTime;
    this.spatialSources = this.spatialSources.filter((entry) => {
      const age = now - entry.startTime;
      if (age > entry.duration + 0.1) {
        try {
          entry.source.disconnect();
        } catch (e) {}
        entry.panner.disconnect();
        return false;
      }
      return true;
    });
  }

  setEnabled(enabled) {
    this.enabled = enabled;
    if (!enabled) {
      this.masterGain.gain.value = 0;
      this._stopAmbience();
      if (this._voidOsc) {
        try {
          this._voidOsc.stop();
        } catch (e) {}
        this._voidOsc = null;
      }
    } else {
      this.masterGain.gain.value = 0.3;
    }
  }
}
