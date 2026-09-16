/* friends.js — Fictional friends for ghost system */
window.G = window.G || {}

G.FRIENDS = [
  { id: 'alex', name: 'Alex', color: '#ff6b6b', nearCity: 'Tokyo', mx: 870, my: 140, scores: {} },
  { id: 'sam', name: 'Sam', color: '#4ecdc4', nearCity: 'Rio de Janeiro', mx: 310, my: 380, scores: {} },
  { id: 'riley', name: 'Riley', color: '#ffe66d', nearCity: 'London', mx: 460, my: 100, scores: {} },
  { id: 'jordan', name: 'Jordan', color: '#a8e6cf', nearCity: 'New York', mx: 180, my: 150, scores: {} }
]

G.FRIENDS.forEach(function (f) {
  G.CONCERTS.forEach(function (c) {
    var hash = (c.id * 7 + f.id.charCodeAt(0) * 13) % 100
    f.scores[c.id] = Math.max(20, Math.min(98, hash + Math.floor((c.id * 3) % 30)))
  })
})

G.getFriendScore = function (friendId, concertId) {
  var f = G.FRIENDS.find(function (fr) {
    return fr.id === friendId
  })
  if (!f || !f.scores[concertId]) return 0
  return f.scores[concertId]
}

G.isFriendAtConcert = function (concertId) {
  return G.FRIENDS.some(function (f) {
    return f.scores[concertId] !== undefined
  })
}
