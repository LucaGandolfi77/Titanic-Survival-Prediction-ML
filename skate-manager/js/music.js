/* ===== Web Audio tempo music system ===== */

export class MusicEngine {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.enabled = true;
    this.sfxEnabled = true;
    this.currentTempo = 'slow';
    this.bpm = 120;
    this.beatInterval = null;
    this.oscillators = [];
    this.volume = 0.3;
  }

  init() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = this.volume;
    this.masterGain.connect(this.ctx.destination);
  }

  resume() {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  setVolume(v) {
    this.volume = v;
    if (this.masterGain) this.masterGain.gain.value = v;
  }

  setEnabled(on) {
    this.enabled = on;
    if (this.masterGain) this.masterGain.gain.value = on ? this.volume : 0;
    if (!on) this.stopMusic();
  }

  // ===== Background music =====
  startMusic(tempo = 'slow') {
    this.init();
    this.stopMusic();
    this.currentTempo = tempo;

    const tempoConfig = {
      slow:   { bpm: 120, freq: 220, detune: 0 },
      medium: { bpm: 150, freq: 330, detune: 200 },
      fast:   { bpm: 180, freq: 440, detune: 400 },
      max:    { bpm: 210, freq: 523, detune: 600 }
    };
    const cfg = tempoConfig[tempo] || tempoConfig.slow;
    this.bpm = cfg.bpm;

    // Base pad oscillator
    const pad = this.ctx.createOscillator();
    pad.type = 'sine';
    pad.frequency.value = cfg.freq * 0.5;
    const padGain = this.ctx.createGain();
    padGain.gain.value = 0.06;
    pad.connect(padGain);
    padGain.connect(this.masterGain);
    pad.start();
    this.oscillators.push({ osc: pad, gain: padGain });

    // Arpeggio sequence
    const notes = [cfg.freq, cfg.freq * 1.25, cfg.freq * 1.5, cfg.freq * 2];
    let noteIdx = 0;
    const beatMs = 60000 / this.bpm;
    this.beatInterval = setInterval(() => {
      if (!this.enabled) return;
      this._playBeatNote(notes[noteIdx % notes.length], beatMs * 0.6);
      noteIdx++;
      // Percussion on every other beat
      if (noteIdx % 2 === 0 && (tempo === 'medium' || tempo === 'fast' || tempo === 'max')) {
        this._playPercussion();
      }
    }, beatMs);
  }

  _playBeatNote(freq, dur) {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = 'triangle';
    osc.frequency.value = freq;
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.08, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur / 1000);
    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(t);
    osc.stop(t + dur / 1000);
  }

  _playPercussion() {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime;
    const bufSize = this.ctx.sampleRate * 0.04;
    const buf = this.ctx.createBuffer(1, bufSize, this.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufSize);
    }
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const filt = this.ctx.createBiquadFilter();
    filt.type = 'highpass';
    filt.frequency.value = 800;
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.07, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    src.connect(filt);
    filt.connect(gain);
    gain.connect(this.masterGain);
    src.start(t);
  }

  changeTempo(tempo) {
    if (tempo === this.currentTempo) return;
    this.startMusic(tempo);
  }

  stopMusic() {
    if (this.beatInterval) {
      clearInterval(this.beatInterval);
      this.beatInterval = null;
    }
    for (const { osc, gain } of this.oscillators) {
      try {
        osc.stop();
        osc.disconnect();
        gain.disconnect();
      } catch(e) {}
    }
    this.oscillators = [];
  }

  // ===== Sound effects =====
  playFormationChime() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    const notes = [261.63, 329.63, 392.0, 523.25]; // C4 E4 G4 C5
    notes.forEach((freq, i) => {
      const osc = this.ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = freq;
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.12, t + i * 0.08);
      gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.08 + 0.3);
      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(t + i * 0.08);
      osc.stop(t + i * 0.08 + 0.35);
    });
  }

  playWobbleStart() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, t);
    osc.frequency.exponentialRampToValueAtTime(660, t + 0.2);
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.15, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(t);
    osc.stop(t + 0.3);
  }

  playWobbleSave() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.value = 1047;
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.15, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(t);
    osc.stop(t + 0.12);
  }

  playWobbleFall() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    // Thud (noise burst)
    const bufSize = this.ctx.sampleRate * 0.1;
    const buf = this.ctx.createBuffer(1, bufSize, this.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < bufSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufSize);
    }
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const filt = this.ctx.createBiquadFilter();
    filt.type = 'lowpass';
    filt.frequency.value = 200;
    const gain = this.ctx.createGain();
    gain.gain.value = 0.2;
    src.connect(filt);
    filt.connect(gain);
    gain.connect(this.masterGain);
    src.start(t);

    // Sad trombone
    const osc = this.ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(330, t + 0.15);
    osc.frequency.exponentialRampToValueAtTime(200, t + 0.55);
    const g2 = this.ctx.createGain();
    g2.gain.setValueAtTime(0.08, t + 0.15);
    g2.gain.exponentialRampToValueAtTime(0.001, t + 0.6);
    osc.connect(g2);
    g2.connect(this.masterGain);
    osc.start(t + 0.15);
    osc.stop(t + 0.65);
  }

  playMilestone() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    const notes = [523.25, 659.25, 783.99];
    notes.forEach((freq, i) => {
      const osc = this.ctx.createOscillator();
      osc.type = 'square';
      osc.frequency.value = freq;
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.08, t + i * 0.06);
      gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.06 + 0.2);
      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(t + i * 0.06);
      osc.stop(t + i * 0.06 + 0.25);
    });
  }

  playWinFanfare() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    const notes = [261.63, 329.63, 392.0, 523.25, 783.99];
    notes.forEach((freq, i) => {
      const osc = this.ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = freq;
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.12, t + i * 0.12);
      gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.12 + 0.5);
      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(t + i * 0.12);
      osc.stop(t + i * 0.12 + 0.55);
    });
  }

  playLoseSad() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    const notes = [392, 349.23, 311.13, 261.63];
    notes.forEach((freq, i) => {
      const osc = this.ctx.createOscillator();
      osc.type = 'triangle';
      osc.frequency.value = freq;
      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.1, t + i * 0.2);
      gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.2 + 0.4);
      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(t + i * 0.2);
      osc.stop(t + i * 0.2 + 0.45);
    });
  }

  playClick() {
    if (!this.sfxEnabled) return;
    this.init();
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = 'square';
    osc.frequency.value = 800;
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.06, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(t);
    osc.stop(t + 0.05);
  }
}
