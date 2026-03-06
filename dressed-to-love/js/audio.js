// js/audio.js — Web Audio API procedural music + SFX
let ctx = null;
let masterGain = null;
let musicOscillators = [];
let _muted = false;
let _musicTheme = null;

function _getCtx(){
  if(!ctx){
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = ctx.createGain();
    masterGain.gain.value = 0.4;
    masterGain.connect(ctx.destination);
  }
  return ctx;
}

export function initAudio(){
  // Deferred: context created on first user interaction
  document.body.addEventListener('click', ()=>{ _getCtx(); }, {once:true});
}

export function setMuted(muted){
  _muted = muted;
  if(masterGain) masterGain.gain.value = muted ? 0 : 0.4;
}

export function isMuted(){ return _muted; }

// Themes: morning | evening | conflict | wedding | romance
export function playTheme(theme){
  if(_musicTheme === theme) return;
  stopMusic();
  _musicTheme = theme;
  const c = _getCtx();
  const sequences = {
    morning:  [261,294,329,349,392,440,494,523],
    evening:  [220,247,277,294,330,370,415,440],
    conflict: [196,208,220,247,196,208,233,196],
    wedding:  [349,392,440,392,523,494,440,523],
    romance:  [293,329,369,293,329,369,440,369]
  };
  const notes = sequences[theme] || sequences.morning;
  let step = 0;
  function playNote(){
    if(_muted) return;
    const osc = c.createOscillator();
    const gain= c.createGain();
    osc.type = theme==='conflict'?'sawtooth':'sine';
    osc.frequency.value = notes[step % notes.length];
    gain.gain.setValueAtTime(0.18, c.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime+0.45);
    osc.connect(gain);
    gain.connect(masterGain);
    osc.start(c.currentTime);
    osc.stop(c.currentTime + 0.45);
    musicOscillators.push(osc);
    step++;
  }
  const iv = setInterval(playNote, 480);
  _musicInterval = iv;
}

let _musicInterval = null;
export function stopMusic(){
  if(_musicInterval) clearInterval(_musicInterval);
  _musicInterval = null;
  _musicTheme = null;
  musicOscillators.forEach(o=>{ try{ o.stop(); }catch(_){} });
  musicOscillators = [];
}

// One-shot SFX
export function playSound(name){
  if(_muted) return;
  const c = _getCtx();
  const configs = {
    chime:    { type:'sine',     freq:880,  dur:0.3, vol:0.25 },
    fanfare:  { type:'triangle', freq:587,  dur:0.5, vol:0.30 },
    dramatic: { type:'sawtooth', freq:110,  dur:0.6, vol:0.20 },
    wedding:  { type:'sine',     freq:659,  dur:0.8, vol:0.28 },
    click:    { type:'square',   freq:440,  dur:0.08,vol:0.15 },
    heart:    { type:'sine',     freq:523,  dur:0.25,vol:0.20 },
    negative: { type:'sawtooth', freq:146,  dur:0.4, vol:0.18 },
    boost:    { type:'triangle', freq:784,  dur:0.3, vol:0.22 },
    toast:    { type:'sine',     freq:698,  dur:0.2, vol:0.18 },
  };
  const cfg = configs[name] || configs.click;
  const osc  = c.createOscillator();
  const gain = c.createGain();
  osc.type = cfg.type;
  osc.frequency.value = cfg.freq;
  gain.gain.setValueAtTime(cfg.vol, c.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + cfg.dur);
  osc.connect(gain);
  gain.connect(masterGain);
  osc.start(c.currentTime);
  osc.stop(c.currentTime + cfg.dur + 0.05);
}

export { ctx as audioCtx };

