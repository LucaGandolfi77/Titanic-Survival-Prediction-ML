/* utils.js — Shared utilities */
window.G = window.G || {}

G.paused = false
G.animationEnabled = true
G.volume = 0.8
G.achievements = []

/** Achievement definitions */
G.ACHIEVEMENTS = [
  {
    id: 'first-discovery',
    name: 'First Discovery',
    desc: 'Discover your first concert',
    icon: '🔍',
    check: function (s) {
      return s.discovered.size >= 1
    }
  },
  {
    id: 'hunter-5',
    name: 'Concert Hunter',
    desc: 'Discover 5 concerts',
    icon: '🎯',
    check: function (s) {
      return s.discovered.size >= 5
    }
  },
  {
    id: 'hunter-10',
    name: 'Star Gazer',
    desc: 'Discover 10 concerts',
    icon: '⭐',
    check: function (s) {
      return s.discovered.size >= 10
    }
  },
  {
    id: 'hunter-all',
    name: 'Full Map',
    desc: 'Discover all concerts',
    icon: '🗺️',
    check: function (s) {
      return s.discovered.size >= G.CONCERTS.length
    }
  },
  {
    id: 'first-attend',
    name: 'Showtime',
    desc: 'Attend your first concert',
    icon: '🎤',
    check: function (s) {
      return s.attended.size >= 1
    }
  },
  {
    id: 'attend-5',
    name: 'Fanatic',
    desc: 'Attend 5 concerts',
    icon: '🔥',
    check: function (s) {
      return s.attended.size >= 5
    }
  },
  {
    id: 'attend-10',
    name: 'Superfan',
    desc: 'Attend 10 concerts',
    icon: '💎',
    check: function (s) {
      return s.attended.size >= 10
    }
  },
  {
    id: 'attend-all',
    name: 'Tour Complete',
    desc: 'Attend all concerts',
    icon: '🏆',
    check: function (s) {
      return s.attended.size >= G.CONCERTS.length
    }
  },
  {
    id: 'festival-month',
    name: 'Festival Month',
    desc: 'Attend 3+ concerts in one month',
    icon: '🎉',
    check: function (s) {
      return G.getFestivalMultiplier(Object.keys(s.concertScores)[0] || 0) > 1
    }
  },
  {
    id: 'on-stage',
    name: 'On Stage',
    desc: 'Score 90%+ on any concert',
    icon: '🌟',
    check: function (s) {
      return Object.values(s.concertScores).some(function (sc) {
        return sc.score >= 90
      })
    }
  },
  {
    id: 'collector',
    name: 'Collector',
    desc: 'Earn 3 rare collectibles',
    icon: '🎁',
    check: function (s) {
      return s.collectibles.length >= 3
    }
  },
  {
    id: 'spender',
    name: 'Big Spender',
    desc: 'Spend €3000+ on bookings',
    icon: '💳',
    check: function (s) {
      return s.budget <= 2000
    }
  },
  {
    id: 'thrifty',
    name: 'Budget Master',
    desc: 'Have €3000+ remaining',
    icon: '💰',
    check: function (s) {
      return s.budget >= 3000
    }
  },
  {
    id: 'organiser',
    name: 'Promoter',
    desc: 'Organise 3 concerts',
    icon: '🎪',
    check: function (s) {
      return s.organisedCount >= 3
    }
  },
  {
    id: 'booker',
    name: 'Early Booker',
    desc: 'Book a concert 30+ days in advance',
    icon: '✈️',
    check: function (s) {
      var keys = Object.keys(s.booked)
      return keys.some(function (cid) {
        var cd = new Date(G.CONCERTS[cid].date + 'T00:00:00')
        return (cd - s.currentDate) / 86400000 > 30
      })
    }
  }
]

/** Unlock an achievement by ID */
G.unlockAchievement = function (id) {
  if (G.achievements.indexOf(id) !== -1) return
  G.achievements.push(id)
  var ach = G.ACHIEVEMENTS.find(function (a) {
    return a.id === id
  })
  if (ach) {
    G.toast('🏅 ' + ach.icon + ' ' + ach.name, 3000)
    G.sfx.success()
  }
  try {
    localStorage.setItem('concerts-achievements', JSON.stringify(G.achievements))
  } catch (e) {
    /* silent */
  }
}

/** Check and unlock achievements */
G.checkAchievements = function () {
  if (!G.state) return
  G.ACHIEVEMENTS.forEach(function (a) {
    try {
      if (a.check(G.state)) G.unlockAchievement(a.id)
    } catch (e) {
      /* silent */
    }
  })
}

/** Load achievements from localStorage */
G.loadAchievements = function () {
  try {
    return JSON.parse(localStorage.getItem('concerts-achievements') || '[]')
  } catch (e) {
    return []
  }
}

/** Toggle pause state for minigames */
G.togglePause = function () {
  G.paused = !G.paused
  var overlay = document.getElementById('pause-overlay')
  if (overlay) {
    overlay.style.display = G.paused ? 'flex' : 'none'
  }
  if (G.paused) {
    G.sfx.click()
  }
}
G.togglePause = function () {
  G.paused = !G.paused
  var overlay = document.getElementById('pause-overlay')
  if (overlay) {
    overlay.style.display = G.paused ? 'flex' : 'none'
  }
  if (G.paused) {
    G.sfx.click()
  }
}

/**
 * Set up a canvas for HiDPI rendering.
 * Scales the canvas buffer by devicePixelRatio while keeping CSS size the same.
 * Call this right after getting canvas.getContext('2d') and before drawing.
 * @param {HTMLCanvasElement} canvas
 * @param {number} W - CSS width in pixels
 * @param {number} H - CSS height in pixels
 * @returns {CanvasRenderingContext2D}
 */
G.setupHDCanvas = function (canvas, W, H) {
  var dpr = window.devicePixelRatio || 1
  canvas.width = W * dpr
  canvas.height = H * dpr
  canvas.style.width = W + 'px'
  canvas.style.height = H + 'px'
  var ctx = canvas.getContext('2d')
  ctx.scale(dpr, dpr)
  return ctx
}

/**
 * Flash a HUD element with a brief highlight animation
 * @param {string} id - HUD element id
 * @param {string} [color] - Flash color (CSS color)
 */
G.flashHUD = function (id, color) {
  var el = document.getElementById(id)
  if (!el) return
  var orig = el.style.textShadow || ''
  el.style.transition = 'none'
  el.style.textShadow = '0 0 10px ' + (color || 'var(--gold)')
  requestAnimationFrame(function () {
    el.style.transition = 'text-shadow 0.6s ease'
    el.style.textShadow = orig
  })
}

/**
 * Toggle sound mute
 */
G.toggleMute = function () {
  G.muted = !G.muted
  var btn = document.getElementById('mute-btn')
  if (btn) btn.textContent = G.muted ? '🔇' : '🔊'
}

/**
 * Set volume level
 * @param {number} vol - 0.0 to 1.0
 */
G.setVolume = function (vol) {
  G.volume = Math.max(0, Math.min(1, vol))
  if (G.audioCtx) {
    try {
      G.audioCtx.resume()
    } catch (e) {
      /* silent */
    }
  }
}

/**
 * Reset current game
 */
G.resetGame = function () {
  G.paused = false
  var overlay = document.getElementById('pause-overlay')
  if (overlay) overlay.style.display = 'none'
  var overlay2 = document.getElementById('settings-modal')
  if (overlay2) overlay2.classList.remove('active')
  G.newGame()
  G.updateHUD()
  G.showScreen('screen-menu')
}

/**
 * Persist high scores to localStorage and return updated stats
 * @param {object} state
 * @returns {object} leaderboard stats
 */
G.saveHighScore = function (state) {
  try {
    var key = 'concerts-challenge-scores'
    var scores = JSON.parse(localStorage.getItem(key) || '[]')
    scores.push({
      date: new Date().toISOString(),
      points: state.points,
      attended: state.concertsAttended,
      total: state.totalConcerts,
      budget: state.budget,
      organised: state.organisedCount
    })
    scores.sort(function (a, b) {
      return b.points - a.points
    })
    scores = scores.slice(0, 10)
    localStorage.setItem(key, JSON.stringify(scores))
    return scores
  } catch (e) {
    return []
  }
}

/**
 * Load high scores from localStorage
 * @returns {array}
 */
G.loadHighScores = function () {
  try {
    return JSON.parse(localStorage.getItem('concerts-challenge-scores') || '[]')
  } catch (e) {
    return []
  }
}

/**
 * Get best ever stats
 * @returns {object|null}
 */
G.getBestScore = function () {
  var scores = G.loadHighScores()
  return scores.length > 0 ? scores[0] : null
}

/**
 * Trigger screen shake animation
 * @param {number} [intensity=5] - Shake intensity in px
 * @param {number} [duration=300] - Duration in ms
 */
G.shake = function (intensity, duration) {
  if (!G.animationEnabled) return
  var shakeIntensity = typeof intensity === 'number' ? intensity : 5
  var shakeDuration = typeof duration === 'number' ? duration : 300
  var screen = document.getElementById(G.currentScreen)
  if (!screen) return
  screen.classList.remove('shake-active')
  void screen.offsetWidth
  screen.classList.add('shake-active')
  setTimeout(function () {
    screen.classList.remove('shake-active')
  }, shakeDuration)
}

/**
 * Build mini-map SVG for booking panel showing route to concert city
 * @param {object} concert - Concert data with mx, my, city properties
 * @returns {string} HTML string containing SVG
 */
G.buildMiniMap = function (concert) {
  var cx = 500
  var cy = 250
  var userX = cx + (Math.random() - 0.5) * 600
  var userY = cy + (Math.random() - 0.5) * 400
  var svg =
    '<svg viewBox="0 0 1000 500" style="width:100%;height:200px;border-radius:var(--radius);background:#080816;border:1px solid rgba(255,255,255,0.05)">'
  svg += '<rect width="1000" height="500" fill="#080816"/>'
  svg +=
    '<line x1="' +
    userX +
    '" y1="' +
    userY +
    '" x2="' +
    concert.mx +
    '" y2="' +
    concert.my +
    '" stroke="var(--teal)" stroke-width="2" stroke-dasharray="8,6" opacity="0.6"/>'
  svg += '<circle cx="' + userX + '" cy="' + userY + '" r="10" fill="var(--teal)" opacity="0.8"/>'
  svg += '<circle cx="' + concert.mx + '" cy="' + concert.my + '" r="10" fill="var(--pink)" opacity="0.8"/>'
  svg +=
    '<text x="' + userX + '" y="' + (userY - 16) + '" fill="var(--teal)" font-size="12" text-anchor="middle">You</text>'
  svg +=
    '<text x="' +
    concert.mx +
    '" y="' +
    (concert.my - 16) +
    '" fill="var(--pink)" font-size="12" text-anchor="middle">' +
    concert.city +
    '</text>'
  svg += '</svg>'
  return svg
}

/**
 * Spawn particle effects at a position
 * @param {string} type - 'sparkle', 'star', 'laser'
 * @param {number} x - X position (viewport)
 * @param {number} y - Y position (viewport)
 * @param {number} count - Number of particles
 */
G.spawnParticles = function (type, x, y, count) {
  if (!G.animationEnabled) return
  count = count || 20
  var colors = {
    sparkle: ['#ffd700', '#ff2d78', '#00f5d4', '#fff'],
    star: ['#ff2d78', '#ffd700', '#44ff88', '#ff8800'],
    laser: ['#00f5d4', '#44ff88', '#fff']
  }
  var particleColors = colors[type] || colors.sparkle
  for (var i = 0; i < count; i++) {
    var p = document.createElement('div')
    p.className = 'particle particle-' + type
    var angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.5
    var dist = 40 + Math.random() * 80
    var px = x + Math.cos(angle) * dist
    var py = y + Math.sin(angle) * dist
    p.style.left = px + 'px'
    p.style.top = py + 'px'
    p.style.backgroundColor = particleColors[Math.floor(Math.random() * particleColors.length)]
    p.style.animationDuration = 0.6 + Math.random() * 0.6 + 's'
    p.style.width = 4 + Math.random() * 4 + 'px'
    p.style.height = 4 + Math.random() * 4 + 'px'
    document.body.appendChild(p)
    ;(function (el) {
      setTimeout(function () {
        if (el.parentNode) el.parentNode.removeChild(el)
      }, 1200)
    })(p)
  }
}
