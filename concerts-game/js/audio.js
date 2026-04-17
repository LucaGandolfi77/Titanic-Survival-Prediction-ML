/* audio.js — Web Audio API synth sounds */
window.G = window.G || {};

G.audioCtx = null;
G.muted = false;

G.initAudio = function() {
  try {
    G.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  } catch(e) {
    console.warn("Web Audio API not available — running silent");
  }
};

G.toggleMute = function() {
  G.muted = !G.muted;
  document.getElementById('mute-btn').textContent = G.muted ? '🔇' : '🔊';
};

/** Play a simple oscillator tone */
G.playTone = function(freq, duration, type) {
  if (G.muted || !G.audioCtx) return;
  try {
    if (G.audioCtx.state === 'suspended') G.audioCtx.resume();
    var osc = G.audioCtx.createOscillator();
    var gain = G.audioCtx.createGain();
    osc.type = type || 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.12, G.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, G.audioCtx.currentTime + (duration || 0.2));
    osc.connect(gain);
    gain.connect(G.audioCtx.destination);
    osc.start();
    osc.stop(G.audioCtx.currentTime + (duration || 0.2));
  } catch(e) { /* silent fallback */ }
};

/** Named sound effects */
G.sfx = {
  click: function() { G.playTone(800, 0.06, 'square'); },
  success: function() {
    G.playTone(523, 0.1);
    setTimeout(function(){ G.playTone(659, 0.1); }, 100);
    setTimeout(function(){ G.playTone(784, 0.15); }, 200);
  },
  fail: function() { G.playTone(200, 0.3, 'sawtooth'); },
  tick: function() { G.playTone(1000, 0.04, 'square'); },
  beat: function() { G.playTone(150, 0.12, 'sine'); },
  coin: function() {
    G.playTone(988, 0.06);
    setTimeout(function(){ G.playTone(1319, 0.1); }, 70);
  },
  reveal: function() {
    G.playTone(440, 0.08);
    setTimeout(function(){ G.playTone(554, 0.08); }, 100);
    setTimeout(function(){ G.playTone(659, 0.08); }, 200);
    setTimeout(function(){ G.playTone(880, 0.15); }, 300);
  },
  note: function(freq) { G.playTone(freq || 440, 0.1, 'triangle'); }
};
