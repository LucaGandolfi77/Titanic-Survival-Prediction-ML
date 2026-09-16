/* challenges.js — Weekly challenge mode */
window.G = window.G || {}

G.CHALLENGES = [
  {
    id: 'week1',
    name: 'Budget Master',
    desc: 'Attend 5 concerts spending less than €2,000',
    target: 5,
    metric: 'concertsAttended',
    maxSpend: 2000
  },
  { id: 'week2', name: 'Score Hunter', desc: 'Get 3 On Stage placements (90%+)', target: 3, metric: 'onStage' },
  {
    id: 'week3',
    name: 'Calendar Punctual',
    desc: 'Attend 3 concerts in the same month',
    target: 3,
    metric: 'sameMonth'
  },
  { id: 'week4', name: 'Discovery Dash', desc: 'Discover 10 concerts in one game', target: 10, metric: 'discovered' },
  {
    id: 'week5',
    name: 'Marathon Runner',
    desc: 'Attend 8 concerts in a single game',
    target: 8,
    metric: 'concertsAttended'
  }
]

G._activeChallenge = null

/** Show challenge selection screen */
G.showChallenge = function () {
  G.sfx.click()
  G.showScreen('screen-challenge')
  var el = document.getElementById('challenge-content')
  var html = '<div class="fade-in">'
  html += '<h2 style="text-align:center;margin-bottom:16px">Weekly Challenges</h2>'
  G.CHALLENGES.forEach(function (ch) {
    html += '<div class="go-concert-item" style="cursor:pointer" onclick="G.startChallenge(\'' + ch.id + '\')">'
    html += '<span>' + ch.name + '</span>'
    html += '<span style="color:var(--gray-light);font-size:0.8rem;margin-left:8px">' + ch.desc + '</span>'
    html += '</div>'
  })
  html += '</div>'
  el.innerHTML = html
}

/** Start a challenge mode */
G.startChallenge = function (challengeId) {
  var ch = G.CHALLENGES.find(function (c) {
    return c.id === challengeId
  })
  if (!ch) return
  G._activeChallenge = ch
  G.sfx.click()
  G.toast('Challenge: ' + ch.name + ' — ' + ch.desc, 3000)
  G.startGame(G.state.difficulty || 'balanced')
}

/** Check challenge progress, returns { current, target, complete } */
G.checkChallenge = function () {
  if (!G._activeChallenge || !G.state) return null
  var ch = G._activeChallenge
  var current = 0
  if (ch.metric === 'concertsAttended') {
    current = G.state.concertsAttended
  } else if (ch.metric === 'discovered') {
    current = G.state.discovered.size
  } else if (ch.metric === 'onStage') {
    var scores = G.state.concertScores
    Object.keys(scores).forEach(function (id) {
      if (scores[id].score >= 90) current++
    })
  } else if (ch.metric === 'sameMonth') {
    var months = {}
    G.state.attended.forEach(function (id) {
      var c = G.CONCERTS[id]
      if (c) {
        var cd = new Date(c.date + 'T00:00:00')
        var mk = cd.getFullYear() + '-' + (cd.getMonth() + 1).toString().padStart(2, '0')
        months[mk] = (months[mk] || 0) + 1
      }
    })
    var max = 0
    Object.keys(months).forEach(function (m) {
      if (months[m] > max) max = months[m]
    })
    current = max
  }
  return { current: current, target: ch.target, complete: current >= ch.target, name: ch.name }
}

/** Get active challenge */
G.getActiveChallenge = function () {
  return G._activeChallenge
}
