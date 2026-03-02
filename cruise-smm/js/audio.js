/* ===== WEB AUDIO API — MUSIC + SFX ===== */

let audioCtx = null;
let masterGain = null;
let musicGain = null;
let sfxGain = null;
let currentMusic = null;
let sfxEnabled = true;
let musicVolume = 0.3;

export function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.5;
  masterGain.connect(audioCtx.destination);

  musicGain = audioCtx.createGain();
  musicGain.gain.value = musicVolume;
  musicGain.connect(masterGain);

  sfxGain = audioCtx.createGain();
  sfxGain.gain.value = 0.6;
  sfxGain.connect(masterGain);
}

export function resumeAudio() {
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

export function setMusicVolume(vol) {
  musicVolume = vol;
  if (musicGain) musicGain.gain.value = vol;
}

export function setSfxEnabled(enabled) {
  sfxEnabled = enabled;
  if (sfxGain) sfxGain.gain.value = enabled ? 0.6 : 0;
}

export function setMasterVolume(vol) {
  if (masterGain) masterGain.gain.value = vol;
}

function createOsc(freq, type, duration, gainNode, startTime = 0) {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0.3, audioCtx.currentTime + startTime);
  g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + startTime + duration);
  osc.connect(g);
  g.connect(gainNode);
  osc.start(audioCtx.currentTime + startTime);
  osc.stop(audioCtx.currentTime + startTime + duration);
}

function createNoise(duration, gainNode, startTime = 0) {
  if (!audioCtx) return;
  const bufferSize = audioCtx.sampleRate * duration;
  const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  const g = audioCtx.createGain();
  g.gain.setValueAtTime(0.15, audioCtx.currentTime + startTime);
  g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + startTime + duration);
  src.connect(g);
  g.connect(gainNode);
  src.start(audioCtx.currentTime + startTime);
}

export function playSound(id) {
  if (!audioCtx || !sfxEnabled) return;
  resumeAudio();

  switch (id) {
    case 'camera':
      createNoise(0.06, sfxGain);
      createOsc(800, 'square', 0.06, sfxGain);
      break;
    case 'photoPerf':
      createOsc(523, 'sine', 0.18, sfxGain, 0);
      createOsc(659, 'sine', 0.18, sfxGain, 0.12);
      createOsc(784, 'sine', 0.18, sfxGain, 0.24);
      break;
    case 'taskComplete':
      createOsc(523, 'sine', 0.15, sfxGain, 0);
      createOsc(659, 'sine', 0.15, sfxGain, 0.1);
      createOsc(784, 'sine', 0.15, sfxGain, 0.2);
      createOsc(1047, 'sine', 0.3, sfxGain, 0.3);
      break;
    case 'loveUp':
      createOsc(1047, 'sine', 0.15, sfxGain, 0);
      createOsc(1319, 'sine', 0.15, sfxGain, 0.1);
      createOsc(1568, 'sine', 0.2, sfxGain, 0.2);
      break;
    case 'loveDn':
      createOsc(523, 'sine', 0.2, sfxGain, 0);
      createOsc(440, 'sine', 0.2, sfxGain, 0.15);
      createOsc(349, 'sine', 0.3, sfxGain, 0.3);
      break;
    case 'teamHappy':
      createNoise(1, sfxGain);
      createOsc(600, 'sine', 0.2, sfxGain, 0);
      createOsc(800, 'sine', 0.2, sfxGain, 0.15);
      break;
    case 'money':
      createOsc(1200, 'sine', 0.08, sfxGain);
      createOsc(1500, 'sine', 0.08, sfxGain, 0.06);
      break;
    case 'fame':
      createOsc(880, 'sine', 0.1, sfxGain, 0);
      createOsc(1100, 'sine', 0.1, sfxGain, 0.08);
      createOsc(1320, 'sine', 0.15, sfxGain, 0.16);
      break;
    case 'milestone':
      for (let i = 0; i < 5; i++) {
        createOsc(523 * (1 + i * 0.25), 'sine', 0.12, sfxGain, i * 0.08);
      }
      break;
    case 'dateStart':
      createOsc(440, 'sine', 0.5, sfxGain, 0);
      createOsc(554, 'sine', 0.5, sfxGain, 0);
      createOsc(659, 'sine', 0.5, sfxGain, 0);
      break;
    case 'mgStart':
      createOsc(440, 'square', 0.06, sfxGain, 0);
      createOsc(550, 'square', 0.06, sfxGain, 0.08);
      createOsc(660, 'square', 0.06, sfxGain, 0.16);
      createOsc(880, 'square', 0.1, sfxGain, 0.24);
      break;
    case 'btn':
      createOsc(400, 'sine', 0.03, sfxGain);
      break;
    case 'error':
      createOsc(200, 'sawtooth', 0.2, sfxGain);
      break;
  }
}

export function startMusic(timeOfDay) {
  stopMusic();
  if (!audioCtx) return;
  resumeAudio();

  const bpm = timeOfDay === 'morning' ? 120 : timeOfDay === 'afternoon' ? 90 : timeOfDay === 'evening' ? 70 : 55;
  const beatLen = 60 / bpm;

  let notes;
  switch (timeOfDay) {
    case 'morning':
      notes = [261, 329, 392, 329, 261, 349, 440, 349]; // C E G E C F A F
      break;
    case 'afternoon':
      notes = [349, 440, 523, 440, 392, 349, 329, 261]; // F A C A G F E C
      break;
    case 'evening':
      notes = [330, 392, 494, 392, 330, 294, 349, 294]; // E G B G E D F D
      break;
    default:
      notes = [220, 261, 330, 261, 220, 196, 220, 196]; // A C E C A G A G
  }

  function playLoop() {
    if (!audioCtx || !musicGain) return;
    for (let i = 0; i < notes.length; i++) {
      const osc = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.value = notes[i];
      g.gain.setValueAtTime(0, audioCtx.currentTime + i * beatLen);
      g.gain.linearRampToValueAtTime(0.08, audioCtx.currentTime + i * beatLen + 0.05);
      g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + i * beatLen + beatLen * 0.9);
      osc.connect(g);
      g.connect(musicGain);
      osc.start(audioCtx.currentTime + i * beatLen);
      osc.stop(audioCtx.currentTime + (i + 1) * beatLen);
    }
    currentMusic = setTimeout(playLoop, notes.length * beatLen * 1000);
  }

  playLoop();
}

export function stopMusic() {
  if (currentMusic) {
    clearTimeout(currentMusic);
    currentMusic = null;
  }
}

export function getMusicTimeOfDay(slotIndex) {
  if (slotIndex <= 2) return 'morning';
  if (slotIndex <= 5) return 'afternoon';
  if (slotIndex <= 6) return 'evening';
  return 'night';
}
