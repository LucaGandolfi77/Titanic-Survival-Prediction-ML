export function createAudioEngine() {
  let ctx = null;
  let enabled = false;
  const noteFrequencies = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
  };

  function init() {
    if (ctx) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    enabled = true;
  }

  function playNote(layer, idx, activation = 0.5) {
    if (!enabled) init();
    if (!ctx) return;
    if (ctx.state === 'suspended') ctx.resume();

    const scaleNotes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C#', 'D#'];
    const noteIndex = (layer + idx) % scaleNotes.length;
    const noteName = scaleNotes[noteIndex];
    const baseFreq = noteFrequencies[noteName];
    const freq = baseFreq * Math.pow(2, layer * 0.5);
    const duration = 0.3 + activation * 0.5;
    const volume = 0.05 + activation * 0.15;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(2000 + activation * 4000, ctx.currentTime);

    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(volume, ctx.currentTime + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + duration);
  }

  function playChord(activations) {
    if (!enabled) init();
    activations.forEach((act, i) => {
      setTimeout(() => playNote(Math.floor(i / 4), i % 4, act), i * 50);
    });
  }

  function toggle() {
    if (!enabled) init();
    enabled = !enabled;
    if (enabled && ctx && ctx.state === 'suspended') ctx.resume();
    if (!enabled && ctx) ctx.suspend();
    return enabled;
  }

  function isEnabled() {
    return enabled;
  }

  return { init, playNote, playChord, toggle, isEnabled };
}
