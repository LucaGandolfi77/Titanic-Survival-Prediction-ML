class AudioManager {
  constructor(){
    this.ctx = null;
    this._enabled = false;
    this._lastFootOdd = false;
    // lazy create on first user interaction
    const resume = async () => {
      if (!this.ctx){
        try{ this.ctx = new (window.AudioContext || window.webkitAudioContext)(); }catch(e){ this.ctx = null; }
      }
      if (this.ctx && this.ctx.state === 'suspended') await this.ctx.resume();
      this._enabled = !!this.ctx;
      window.removeEventListener('pointerdown', resume);
      window.removeEventListener('keydown', resume);
    };
    window.addEventListener('pointerdown', resume);
    window.addEventListener('keydown', resume);
  }

  play(name, opts = {}){
    if (!this._enabled) return;
    switch(name){
      case 'pop': return this._pop();
      case 'chime': return this._chime();
      case 'splash': return this._splash();
      case 'footstep': return this._footstep();
      case 'day': return this._dayJingle();
      case 'harvest': return this._harvest();
      case 'error': return this._error();
      default: return null;
    }
  }

  _createOscillator(type='sine', freq=440, duration=0.2, gain=0.1, freqEnd=null, attack=0.01){
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, now);
    if (freqEnd !== null){ osc.frequency.linearRampToValueAtTime(freqEnd, now + duration); }

    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(gain, now + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, now + duration + 0.02);

    osc.connect(g); g.connect(this.ctx.destination);
    osc.start(now); osc.stop(now + duration + 0.03);
    return { osc, g };
  }

  _pop(){
    // quick rising/then falling tone 440 -> 220
    this._createOscillator('sine', 440, 0.08, 0.12, 220, 0.005);
  }

  _chime(){
    // two partials with soft attack
    const now = this.ctx.currentTime;
    const dur = 0.4;
    const gMaster = this.ctx.createGain(); gMaster.gain.setValueAtTime(0.0001, now); gMaster.gain.linearRampToValueAtTime(0.14, now+0.06); gMaster.gain.exponentialRampToValueAtTime(0.0001, now+dur+0.05);
    gMaster.connect(this.ctx.destination);

    const o1 = this.ctx.createOscillator(); o1.type='sine'; o1.frequency.setValueAtTime(880, now); o1.connect(gMaster); o1.start(now); o1.stop(now+dur+0.05);
    const o2 = this.ctx.createOscillator(); o2.type='sine'; o2.frequency.setValueAtTime(1100, now); o2.connect(gMaster); o2.start(now+0.03); o2.stop(now+dur+0.05);
  }

  _splash(){
    // white noise burst through bandpass
    const now = this.ctx.currentTime; const dur = 0.3;
    const bufferSize = this.ctx.sampleRate * dur;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for(let i=0;i<bufferSize;i++) data[i] = (Math.random()*2-1) * (1 - i/bufferSize);
    const src = this.ctx.createBufferSource(); src.buffer = buffer;
    const filt = this.ctx.createBiquadFilter(); filt.type='bandpass'; filt.frequency.value = 800;
    const g = this.ctx.createGain(); g.gain.setValueAtTime(0.6, now); g.gain.exponentialRampToValueAtTime(0.0001, now+dur);
    src.connect(filt); filt.connect(g); g.connect(this.ctx.destination);
    src.start(now); src.stop(now+dur+0.02);
  }

  _footstep(){
    // alternating small click tone
    this._lastFootOdd = !this._lastFootOdd;
    const f = this._lastFootOdd ? 120 : 100; this._createOscillator('sine', f, 0.05, 0.06, null, 0.002);
  }

  _dayJingle(){
    const now = this.ctx.currentTime; const notes = [523,659,784]; const dur = 0.25;
    notes.forEach((n,i)=>{ const o = this.ctx.createOscillator(); const g = this.ctx.createGain(); const t = now + i*0.25; o.type='sine'; o.frequency.setValueAtTime(n, t); g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(0.12, t+0.03); g.gain.exponentialRampToValueAtTime(0.0001, t+dur); o.connect(g); g.connect(this.ctx.destination); o.start(t); o.stop(t+dur+0.02); });
  }

  _harvest(){
    // rising tone 660 -> 880
    this._createOscillator('sine', 660, 0.25, 0.12, 880, 0.01);
  }

  _error(){
    // square descending 220 -> 180
    this._createOscillator('square', 220, 0.15, 0.12, 180, 0.005);
  }
}

export default AudioManager;
