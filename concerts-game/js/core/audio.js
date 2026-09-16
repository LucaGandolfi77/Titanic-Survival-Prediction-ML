/* AudioService — Clean interface wrapping raw Web Audio API */
G.AudioService = {
  /** Play a named sound effect */
  playSfx: function (name) {
    if (G.muted || !G.audioCtx) return
    try {
      if (G.sfx[name]) G.sfx[name]()
    } catch (e) {
      console.warn('AudioService sfx error:', name, e)
    }
  },

  /** Play background music for a screen */
  playMusic: function (screenId) {
    try {
      G.startMusic(screenId)
    } catch (e) {
      console.warn('AudioService music error:', e)
    }
  },

  /** Stop background music */
  stopMusic: function () {
    try {
      G.stopMusic()
    } catch (e) {
      console.warn('AudioService stopMusic error:', e)
    }
  },

  /** Set master volume (0-1) */
  setVolume: function (vol) {
    vol = Math.max(0, Math.min(1, vol))
    G._musicVolume = vol
    if (G.audioCtx && G.audioCtx.state === 'suspended') G.audioCtx.resume()
  },

  /** Toggle mute, returns new state */
  toggleMute: function () {
    G.toggleMute()
    return G.muted
  },

  /** Start procedural crowd noise */
  startCrowd: function () {
    try {
      G.startCrowdNoise()
    } catch (e) {
      console.warn('AudioService crowd error:', e)
    }
  },

  /** Set crowd volume (0-1) */
  setCrowdVolume: function (vol) {
    try {
      G.setCrowdVolume(vol)
    } catch (e) {
      console.warn('AudioService crowd vol error:', e)
    }
  },

  /** Check if audio is available */
  isAvailable: function () {
    return !!G.audioCtx
  }
}

/* audio.js — Web Audio API synth sounds */
window.G = window.G || {}

G.audioCtx = null
G.muted = false

G.initAudio = function () {
  try {
    G.audioCtx = new (window.AudioContext || window.webkitAudioContext)()
    G.analyser = G.audioCtx.createAnalyser()
    G.analyser.fftSize = 256
  } catch (e) {
    console.warn('Web Audio API not available — running silent')
  }
}

G.toggleMute = function () {
  G.muted = !G.muted
  document.getElementById('mute-btn').textContent = G.muted ? '🔇' : '🔊'
}

/** Play a simple oscillator tone */
G.playTone = function (freq, duration, type) {
  if (G.muted || !G.audioCtx) return
  try {
    if (G.audioCtx.state === 'suspended') G.audioCtx.resume()
    var osc = G.audioCtx.createOscillator()
    var gain = G.audioCtx.createGain()
    osc.type = type || 'sine'
    osc.frequency.value = freq
    gain.gain.setValueAtTime(0.12, G.audioCtx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, G.audioCtx.currentTime + (duration || 0.2))
    osc.connect(gain)
    gain.connect(G.analyser)
    G.analyser.connect(G.audioCtx.destination)
    osc.start()
    osc.stop(G.audioCtx.currentTime + (duration || 0.2))
  } catch (e) {
    /* silent fallback */
  }
}

/** Named sound effects */
G.sfx = {
  click: function () {
    G.playTone(800, 0.06, 'square')
  },
  success: function () {
    G.playTone(523, 0.1)
    setTimeout(function () {
      G.playTone(659, 0.1)
    }, 100)
    setTimeout(function () {
      G.playTone(784, 0.15)
    }, 200)
  },
  fail: function () {
    G.playTone(200, 0.3, 'sawtooth')
  },
  tick: function () {
    G.playTone(1000, 0.04, 'square')
  },
  beat: function () {
    G.playTone(150, 0.12, 'sine')
  },
  coin: function () {
    G.playTone(988, 0.06)
    setTimeout(function () {
      G.playTone(1319, 0.1)
    }, 70)
  },
  reveal: function () {
    G.playTone(440, 0.08)
    setTimeout(function () {
      G.playTone(554, 0.08)
    }, 100)
    setTimeout(function () {
      G.playTone(659, 0.08)
    }, 200)
    setTimeout(function () {
      G.playTone(880, 0.15)
    }, 300)
  },
  note: function (freq) {
    G.playTone(freq || 440, 0.1, 'triangle')
  },
  kick: function () {
    G.playTone(150, 0.1, 'sawtooth')
  },
  goal: function () {
    G.playTone(523, 0.1)
    setTimeout(function () {
      G.playTone(659, 0.1)
    }, 100)
    setTimeout(function () {
      G.playTone(784, 0.2)
    }, 200)
  },
  whistle: function () {
    G.playTone(800, 0.4, 'sine')
    setTimeout(function () {
      G.playTone(1000, 0.3, 'sine')
    }, 450)
  }
}

/* ── CROWD NOISE (Procedural) ── */
G.crowdSource = null
G.crowdGain = null
G.crowdFilter = null

/** Start procedural crowd noise. Volume scales with score (0-1). */
G.startCrowdNoise = function () {
  if (!G.audioCtx || G.crowdSource) return
  try {
    var bufferSize = G.audioCtx.sampleRate * 2
    var buffer = G.audioCtx.createBuffer(1, bufferSize, G.audioCtx.sampleRate)
    var data = buffer.getChannelData(0)
    for (var i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1
    }
    G.crowdSource = G.audioCtx.createBufferSource()
    G.crowdSource.buffer = buffer
    G.crowdSource.loop = true
    G.crowdGain = G.audioCtx.createGain()
    G.crowdGain.gain.value = 0
    G.crowdFilter = G.audioCtx.createBiquadFilter()
    G.crowdFilter.type = 'lowpass'
    G.crowdFilter.frequency.value = 2000
    G.crowdSource.connect(G.crowdFilter)
    G.crowdFilter.connect(G.crowdGain)
    G.crowdGain.connect(G.analyser)
    G.crowdSource.start()
  } catch (e) {
    /* silent */
  }
}

/** Set crowd volume (0-1), scaled by score */
G.setCrowdVolume = function (vol) {
  if (!G.crowdGain) return
  vol = Math.max(0, Math.min(1, vol))
  G.crowdGain.gain.linearRampToValueAtTime(vol * 0.08, G.audioCtx.currentTime + 0.3)
}

/** Stop crowd noise */
G.stopCrowdNoise = function () {
  if (G.crowdSource) {
    try {
      G.crowdSource.stop()
    } catch (e) {
      /* silent */
    }
    G.crowdSource = null
    G.crowdGain = null
    G.crowdFilter = null
  }
}

/* ── BACKGROUND MUSIC (Procedural) ── */
G._musicInterval = null
G._musicScreen = null
G._musicIndex = 0
G._musicPlaying = false

G.MUSIC_NOTES = {
  'screen-menu': [261.63, 329.63, 392.0, 329.63, 261.63, 196.0, 261.63, 329.63],
  'screen-map': [
    261.63, 329.63, 392.0, 523.25, 392.0, 329.63, 261.63, 196.0, 261.63, 329.63, 392.0, 523.25, 659.25, 523.25, 392.0,
    329.63
  ],
  'screen-calendar': [196.0, 261.63, 293.66, 261.63, 196.0, 164.81, 196.0, 220.0, 261.63, 220.0, 196.0, 164.81],
  'screen-discovery': [164.81, 146.83, 130.81, 146.83, 164.81, 196.0, 180.0, 160.0, 146.83, 130.81, 116.54, 130.81],
  'screen-booking': [220.0, 261.63, 293.66, 261.63, 220.0, 196.0, 220.0, 261.63, 329.63, 293.66, 261.63, 220.0],
  'screen-concert': [
    329.63, 392.0, 493.88, 659.25, 493.88, 392.0, 329.63, 440.0, 523.25, 659.25, 784.0, 659.25, 523.25, 440.0, 392.0,
    329.63
  ],
  'screen-result': [
    392.0, 493.88, 523.25, 659.25, 784.0, 659.25, 523.25, 493.88, 392.0, 440.0, 523.25, 659.25, 784.0, 880.0, 784.0,
    659.25
  ],
  'screen-organise': [
    293.66, 349.23, 392.0, 440.0, 392.0, 349.23, 293.66, 261.63, 293.66, 349.23, 440.0, 523.25, 440.0, 392.0, 349.23,
    293.66
  ],
  'screen-gameover': [
    293.66, 261.63, 220.0, 196.0, 164.81, 196.0, 220.0, 180.0, 146.83, 164.81, 130.81, 116.54, 130.81, 146.83, 164.81,
    130.81
  ]
}

G.startMusic = function (screenId) {
  G.stopMusic()
  if (G.muted || !G.audioCtx || !G.MUSIC_NOTES[screenId]) return
  G._musicScreen = screenId
  G._musicIndex = 0
  G._musicPlaying = true
  var notes = G.MUSIC_NOTES[screenId]
  G._musicInterval = setInterval(function () {
    if (!G._musicPlaying || G.muted) return
    var note = notes[G._musicIndex % notes.length]
    G.playTone(note, 0.45, 'triangle')
    G._musicIndex++
  }, 480)
}

/* ── AUDIO VISUALIZER ── */
G.visualizerActive = false
G._vizFrame = null

G.startVisualizer = function (canvasEl) {
  if (!G.analyser) return
  G.visualizerActive = true
  G._vizCanvas = canvasEl
  G._vizCtx = canvasEl.getContext('2d')
  G._vizLoop()
}

G._vizLoop = function () {
  if (!G.visualizerActive || !G._vizCtx) return
  var ctx = G._vizCtx
  var canvas = G._vizCanvas
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  var bufferLength = G.analyser.frequencyBinCount
  var dataArray = new Uint8Array(bufferLength)
  G.analyser.getByteFrequencyData(dataArray)
  var barCount = 32
  var barWidth = (canvas.width / barCount) * 0.7
  var gap = (canvas.width / barCount) * 0.3
  var centerY = canvas.height / 2
  for (var i = 0; i < barCount; i++) {
    var idx = Math.floor((i / barCount) * bufferLength)
    var value = dataArray[idx] / 255
    var barH = Math.max(2, value * canvas.height * 0.8)
    var x = i * (barWidth + gap) + gap / 2
    var hue = 280 + value * 60
    ctx.fillStyle = 'hsl(' + hue + ', 80%, ' + (40 + value * 30) + '%)'
    ctx.fillRect(x, centerY - barH / 2, barWidth, barH)
    ctx.fillRect(x, centerY + barH / 2, barWidth, barH)
  }
  G._vizFrame = requestAnimationFrame(G._vizLoop)
}

G.stopVisualizer = function () {
  G.visualizerActive = false
  if (G._vizFrame) {
    cancelAnimationFrame(G._vizFrame)
    G._vizFrame = null
  }
}
