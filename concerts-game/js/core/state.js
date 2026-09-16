/* state.js — Game state management with observable store */
window.G = window.G || {}

G._subs = []

/** Subscribe to state changes */
G.subscribe = function (key, handler) {
  G._subs.push({ key: key, handler: handler })
}

/** Update state and notify subscribers
 *  @param {Object} updates - Key-value pairs to merge into state
 */
G.setState = function (updates) {
  if (!G.state) return
  Object.assign(G.state, updates)
  G._subs.forEach(function (sub) {
    if (updates[sub.key] !== undefined) sub.handler(G.state[sub.key])
  })
  G.emit('state:change', updates)
}

G.state = null

/** Difficulty presets */
G.DIFFICULTY = {
  casual: { budget: 8000, pointsMultiplier: 0.8, label: 'Casual 🌿' },
  balanced: { budget: 5000, pointsMultiplier: 1.0, label: 'Balanced ⚖️' },
  rockstar: { budget: 3000, pointsMultiplier: 1.5, label: 'Rockstar 🔥' }
}

/** Create a fresh game state */
G.newGame = function (difficulty) {
  difficulty = difficulty || 'balanced'
  var preset = G.DIFFICULTY[difficulty] || G.DIFFICULTY.balanced
  G.state = {
    points: 0,
    budget: preset.budget,
    concertsAttended: 0,
    currentDate: new Date(2026, 0, 1),
    discovered: new Set(),
    booked: {},
    attended: new Set(),
    expired: new Set(),
    collectibles: [],
    concertScores: {},
    organisedCount: 0,
    mgAssignment: {},
    totalConcerts: G.CONCERTS.length,
    calMonth: 0,
    merchPurchases: [],
    travelFlightTier: 0,
    travelHotelTier: 0,
    difficulty: difficulty,
    pointsMultiplier: preset.pointsMultiplier
  }
  G.CONCERTS.forEach(function (c) {
    G.state.mgAssignment[c.id] = G.MG_TYPES[Math.floor(Math.random() * G.MG_TYPES.length)]
  })
}

/** Advance in-game calendar by N days */
G.advanceDays = function (days) {
  var s = G.state
  s.currentDate = new Date(s.currentDate.getTime() + days * 86400000)
  // Mark expired concerts
  G.CONCERTS.forEach(function (c) {
    if (s.attended.has(c.id) || s.expired.has(c.id)) return
    var cd = new Date(c.date + 'T00:00:00')
    var expire = new Date(cd.getTime() + 3 * 86400000)
    if (s.currentDate > expire) {
      s.expired.add(c.id)
    }
  })
  G.updateHUD()
}

G.addPoints = function (pts) {
  G.state.points += Math.round(pts * (G.state.pointsMultiplier || 1))
  G.updateHUD()
  G.flashHUD('hud-points', 'var(--gold)')
  G.checkAchievements()
}

G.spendBudget = function (amount) {
  G.state.budget -= amount
  G.updateHUD()
  var hudBudget = document.getElementById('hud-budget')
  if (hudBudget) {
    if (G.state.budget < 500) {
      hudBudget.classList.add('low')
    } else {
      hudBudget.classList.remove('low')
    }
  }
  G.flashHUD('hud-budget', 'var(--teal)')
  G.checkAchievements()
  return G.state.budget >= 0
}

G.earnBudget = function (amount) {
  G.state.budget += Math.round(amount)
  G.updateHUD()
  var hudBudget = document.getElementById('hud-budget')
  if (hudBudget && G.state.budget >= 500) {
    hudBudget.classList.remove('low')
  }
  G.flashHUD('hud-budget', 'var(--success)')
  G.checkAchievements()
}

/** Format a Date to "Mon D, YYYY" */
G.formatDate = function (d) {
  var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return months[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear()
}

/** Is the concert playable right now? (booked + date window) */
G.isConcertAvailable = function (id) {
  var s = G.state
  if (!s.discovered.has(id) || s.attended.has(id) || s.expired.has(id)) return false
  if (!s.booked[id]) return false
  var cd = new Date(G.CONCERTS[id].date + 'T00:00:00')
  var expire = new Date(cd.getTime() + 3 * 86400000)
  return s.currentDate >= cd && s.currentDate <= expire
}

/** Can the player book this concert? */
G.isConcertBookable = function (id) {
  var s = G.state
  if (!s.discovered.has(id) || s.attended.has(id) || s.expired.has(id)) return false
  if (s.booked[id]) return false
  var cd = new Date(G.CONCERTS[id].date + 'T00:00:00')
  return s.currentDate < cd
}

/** Get next upcoming concert info (for HUD) */
G.getNextConcert = function () {
  var s = G.state
  var best = null,
    bestDiff = Infinity
  G.CONCERTS.forEach(function (c) {
    if (s.attended.has(c.id) || s.expired.has(c.id)) return
    var cd = new Date(c.date + 'T00:00:00')
    var diff = cd - s.currentDate
    if (diff > 0 && diff < bestDiff) {
      bestDiff = diff
      best = { concert: c, daysUntil: Math.ceil(diff / 86400000) }
    }
  })
  return best
}

/** Check if game should end */
G.checkGameEnd = function () {
  var s = G.state
  var allDone = G.CONCERTS.every(function (c) {
    return s.attended.has(c.id) || s.expired.has(c.id)
  })
  if (allDone) {
    setTimeout(function () {
      G.showGameOver()
    }, 600)
  }
}

/** Get the "festival month" multiplier for a concert */
G.getFestivalMultiplier = function (concertId) {
  var c = G.CONCERTS[concertId]
  var cd = new Date(c.date + 'T00:00:00')
  var monthKey = cd.getFullYear() + '-' + cd.getMonth()
  var count = 0
  G.state.attended.forEach(function (aid) {
    var ac = G.CONCERTS[aid]
    var acd = new Date(ac.date + 'T00:00:00')
    if (acd.getFullYear() + '-' + acd.getMonth() === monthKey) count++
  })
  return count >= 3 ? 1.5 : 1.0
}
