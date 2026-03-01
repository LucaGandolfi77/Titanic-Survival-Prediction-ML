/* ── js/audio.js ── Web Audio API synthesized sounds ── */

/* ══════════════════════════════════════════════════════════
   AudioManager — all sounds are synthesized, no external files
   ══════════════════════════════════════════════════════════ */
export class AudioManager {
  constructor() {
    this.ctx    = null;   // AudioContext (created on first user gesture)
    this.master = null;   // GainNode
    this.muted  = false;

    // Engine oscillator (persistent)
    this._engineOsc  = null;
    this._engineGain = null;
  }

  /* ── Initialize audio context (call after user gesture) ── */
  init() {
    if (this.ctx) return;
    this.ctx    = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.4;
    this.master.connect(this.ctx.destination);

    // Start engine drone (always running, volume controlled)
    this._startEngine();
  }

  /* ── Engine hum (sawtooth oscillator modulated by throttle) ── */
  _startEngine() {
    const osc  = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.value = 80;
    gain.gain.value = 0;
    osc.connect(gain);

    // Low-pass filter for warmth
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 300;
    gain.connect(filter);
    filter.connect(this.master);

    osc.start();
    this._engineOsc  = osc;
    this._engineGain = gain;
    this._engineFilter = filter;
  }

  /* ── Update engine sound pitch/volume from throttle ── */
  updateEngine(throttle) {
    if (!this._engineOsc || this.muted) return;
    const t = this.ctx.currentTime;
    // Pitch ramps from 80 Hz (idle) to 220 Hz (full throttle)
    this._engineOsc.frequency.setTargetAtTime(80 + throttle * 140, t, 0.1);
    // Volume
    this._engineGain.gain.setTargetAtTime(throttle * 0.15, t, 0.05);
    // Filter opens with throttle
    this._engineFilter.frequency.setTargetAtTime(300 + throttle * 800, t, 0.1);
  }

  /* ── Gunshot (noise burst — very short) ── */
  playGunshot() {
    if (!this.ctx || this.muted) return;
    const t   = this.ctx.currentTime;
    const dur = 0.04;

    // White noise burst
    const bufSize = this.ctx.sampleRate * dur;
    const buf = this.ctx.createBuffer(1, bufSize, this.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) data[i] = (Math.random() * 2 - 1) * 0.6;

    const src  = this.ctx.createBufferSource();
    src.buffer = buf;

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.25, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

    const hp = this.ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 1000;

    src.connect(hp);
    hp.connect(gain);
    gain.connect(this.master);
    src.start(t);
    src.stop(t + dur);
  }

  /* ── Missile launch (sweep oscillator) ── */
  playMissileLaunch() {
    if (!this.ctx || this.muted) return;
    const t   = this.ctx.currentTime;
    const dur = 0.5;

    const osc  = this.ctx.createOscillator();
    osc.type   = 'sine';
    osc.frequency.setValueAtTime(200, t);
    osc.frequency.exponentialRampToValueAtTime(2000, t + dur);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.3, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

    osc.connect(gain);
    gain.connect(this.master);
    osc.start(t);
    osc.stop(t + dur);
  }

  /* ── Explosion (filtered noise, longer) ── */
  playExplosion() {
    if (!this.ctx || this.muted) return;
    const t   = this.ctx.currentTime;
    const dur = 0.8;

    const bufSize = this.ctx.sampleRate * dur;
    const buf = this.ctx.createBuffer(1, bufSize, this.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1;

    const src  = this.ctx.createBufferSource();
    src.buffer = buf;

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.5, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

    const lp = this.ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(800, t);
    lp.frequency.exponentialRampToValueAtTime(100, t + dur);

    src.connect(lp);
    lp.connect(gain);
    gain.connect(this.master);
    src.start(t);
    src.stop(t + dur);
  }

  /* ── Hit marker (sine blip) ── */
  playHit() {
    if (!this.ctx || this.muted) return;
    const t   = this.ctx.currentTime;
    const dur = 0.08;

    const osc  = this.ctx.createOscillator();
    osc.type   = 'sine';
    osc.frequency.setValueAtTime(1200, t);
    osc.frequency.setValueAtTime(800, t + dur * 0.5);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.2, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

    osc.connect(gain);
    gain.connect(this.master);
    osc.start(t);
    osc.stop(t + dur);
  }

  /* ── Damage taken sound (lower sine blip) ── */
  playDamage() {
    if (!this.ctx || this.muted) return;
    const t   = this.ctx.currentTime;
    const dur = 0.15;

    const osc  = this.ctx.createOscillator();
    osc.type   = 'sawtooth';
    osc.frequency.setValueAtTime(150, t);
    osc.frequency.exponentialRampToValueAtTime(60, t + dur);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.3, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

    osc.connect(gain);
    gain.connect(this.master);
    osc.start(t);
    osc.stop(t + dur);
  }

  /* ── Wave complete fanfare ── */
  playWaveComplete() {
    if (!this.ctx || this.muted) return;
    const t = this.ctx.currentTime;

    [523, 659, 784].forEach((freq, i) => {
      const osc  = this.ctx.createOscillator();
      osc.type   = 'sine';
      osc.frequency.value = freq;

      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0, t + i * 0.12);
      gain.gain.linearRampToValueAtTime(0.2, t + i * 0.12 + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.12 + 0.4);

      osc.connect(gain);
      gain.connect(this.master);
      osc.start(t + i * 0.12);
      osc.stop(t + i * 0.12 + 0.4);
    });
  }

  /* ── Toggle mute ── */
  toggleMute() {
    this.muted = !this.muted;
    if (this.master) {
      this.master.gain.value = this.muted ? 0 : 0.4;
    }
  }

  /* ── Resume context (after user gesture) ── */
  resume() {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }
}
