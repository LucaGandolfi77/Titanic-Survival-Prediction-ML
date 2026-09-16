/* results.js — Post-concert result card + confetti + game over screen */
window.G = window.G || {}

/** Render result card HTML (pure function - no DOM side effects) */
G.renderResult = function (
  concertId,
  score,
  placement,
  badge,
  totalPoints,
  multiplier,
  artistBonus,
  budgetBonus,
  collectible,
  reviewKey
) {
  var c = G.CONCERTS[concertId]
  var artist = G.ARTISTS[c.artistKey]
  var placeClass = placement.toLowerCase().replace(/ /g, '-')
  var reviews = G.REVIEWS[reviewKey] || G.REVIEWS['seated']
  var review = reviews[Math.floor(Math.random() * reviews.length)]

  var html = '<div class="result-card fade-in">'
  html += '<h2>' + artist.emoji + ' ' + artist.name + '</h2>'
  html +=
    '<div style="color:var(--gray-light);font-size:0.85rem">' + c.city + ', ' + c.country + ' — ' + c.venue + '</div>'
  html += '<div class="result-badge">' + badge + '</div>'
  html += '<div class="result-placement ' + placeClass + '">' + placement + '</div>'
  html += '<div style="color:var(--gray);font-size:0.9rem;margin-bottom:8px">Score: ' + score + '%</div>'

  if (totalPoints > 0) {
    html += '<div class="result-points">+' + totalPoints + ' points</div>'
    var details = []
    if (artistBonus > 0) details.push('Artist bonus +' + artistBonus)
    if (multiplier > 1) details.push('Festival Month ×' + multiplier.toFixed(1))
    if (details.length > 0) html += '<div style="color:var(--gray);font-size:0.7rem">' + details.join(' · ') + '</div>'
  }

  if (budgetBonus > 0) {
    html += '<div style="color:var(--success);margin:8px 0">+€' + budgetBonus + ' budget bonus!</div>'
  }

  if (collectible) {
    html += '<div class="result-collectible">🏆 Rare: ' + collectible + '</div>'
  }

  html += '<div class="result-review">"' + review + '"</div>'
  html +=
    '<div id="result-score-animate" style="font-size:2rem;font-weight:800;color:var(--gold);opacity:0;animation:score-reveal 1s ease 0.3s forwards">🎯 ' +
    score +
    '%</div>'
  html += '<button class="btn btn-primary" id="result-done" style="margin-top:16px">Continue 🌍</button>'
  if (G._photoCanvas) {
    html += '<button class="btn btn-secondary" id="result-photo" style="margin-top:12px">📷 Photo Mode</button>'
  }
  html += '</div>'
  return html
}

/**
 * Show concert result card with animated confetti.
 * Called from concert.js G.processConcertResult
 */
G.showResult = function (
  concertId,
  score,
  placement,
  badge,
  totalPoints,
  multiplier,
  artistBonus,
  budgetBonus,
  collectible,
  reviewKey
) {
  G.showScreen('screen-result')
  var el = document.getElementById('result-content')
  el.innerHTML = G.renderResult(
    concertId,
    score,
    placement,
    badge,
    totalPoints,
    multiplier,
    artistBonus,
    budgetBonus,
    collectible,
    reviewKey
  )

  if (score >= 40) {
    G.showConfetti(score >= 90 ? 40 : score >= 70 ? 20 : 10)
  }

  document.getElementById('result-done').addEventListener('click', function () {
    G.sfx.click()
    G.showScreen('screen-map')
    G.checkGameEnd()
  })

  var photoBtn = document.getElementById('result-photo')
  if (photoBtn) {
    photoBtn.addEventListener('click', function () {
      G.photoMode(G._photoConcertId, G._photoCanvas, function () {
        G.showScreen('screen-result')
      })
    })
  }

  var shareBtn = document.createElement('button')
  shareBtn.className = 'btn btn-secondary'
  shareBtn.id = 'result-share'
  shareBtn.style.marginTop = '12px'
  shareBtn.textContent = '↗ Share Results'
  el.appendChild(shareBtn)
  shareBtn.addEventListener('click', function () {
    G.shareResults({
      artist: G.ARTISTS[G.CONCERTS[concertId].artistKey].name,
      city: G.CONCERTS[concertId].city,
      score: score,
      placement: placement,
      points: totalPoints || 0,
      budget: G.state ? G.state.budget : 0,
      date: G.CONCERTS[concertId].date
    })
  })

  var cardBtn = document.createElement('button')
  cardBtn.className = 'btn btn-secondary'
  cardBtn.id = 'result-card'
  cardBtn.style.marginTop = '12px'
  cardBtn.textContent = '📸 Save Card'
  el.appendChild(cardBtn)
  cardBtn.addEventListener('click', function () {
    G.downloadResultCard({
      artist: G.ARTISTS[G.CONCERTS[concertId].artistKey].name,
      emoji: G.ARTISTS[G.CONCERTS[concertId].artistKey].emoji,
      city: G.CONCERTS[concertId].city,
      country: G.CONCERTS[concertId].country,
      venue: G.CONCERTS[concertId].venue,
      score: score,
      placement: placement,
      points: totalPoints || 0,
      date: G.CONCERTS[concertId].date,
      budget: G.state ? G.state.budget : 0
    })
  })
}

/**
 * Spawn CSS confetti pieces
 */
G.showConfetti = function (count) {
  count = count || 25
  var colors = ['#ff2d78', '#00f5d4', '#ffd700', '#ff8800', '#44ff88', '#aa44ff', '#fff']

  for (var i = 0; i < count; i++) {
    var piece = document.createElement('div')
    piece.className = 'confetti-piece'
    piece.style.left = Math.random() * 100 + '%'
    piece.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)]
    piece.style.width = Math.random() * 8 + 5 + 'px'
    piece.style.height = Math.random() * 8 + 5 + 'px'
    piece.style.animationDelay = Math.random() * 1.5 + 's'
    piece.style.animationDuration = Math.random() * 1.5 + 2 + 's'
    document.body.appendChild(piece)

    // Remove after animation
    ;(function (p) {
      setTimeout(function () {
        if (p.parentNode) p.parentNode.removeChild(p)
      }, 4000)
    })(piece)
  }
}

/** Render game over HTML (pure function) */
G.renderGameOver = function () {
  var s = G.state
  var allAttended = s.concertsAttended === s.totalConcerts
  var title = allAttended ? 'Tour Complete!' : 'Tour Over!'
  var subtitle = allAttended
    ? 'You attended every single concert — legendary superfan!'
    : "Your concert chase has ended. Here's how you did:"
  var scores = G.saveHighScore(s)
  var best = scores[0] || null
  var isNewBest = best && best.points === s.points && scores.length > 0 && s.points > 0
  var html = '<div class="fade-in">'
  html += '<h1 class="gameover-title">' + title + '</h1>'
  html += '<p style="color:var(--gray-light);margin-bottom:20px">' + subtitle + '</p>'
  if (isNewBest) {
    html +=
      '<div style="color:var(--gold);font-size:1.2rem;font-weight:700;margin-bottom:16px">🔥 NEW HIGH SCORE!</div>'
  } else if (best) {
    html +=
      '<div style="color:var(--gray);font-size:0.85rem;margin-bottom:12px">Best: ' +
      best.points.toLocaleString() +
      ' pts · ' +
      best.attended +
      '/' +
      best.total +
      ' concerts</div>'
  }
  html += '<div class="gameover-stats">'
  html +=
    '<div class="go-stat"><div class="go-val">' +
    s.points.toLocaleString() +
    '</div><div class="go-label">Total Points</div></div>'
  html +=
    '<div class="go-stat"><div class="go-val">' +
    s.concertsAttended +
    '/' +
    s.totalConcerts +
    '</div><div class="go-label">Concerts Attended</div></div>'
  html +=
    '<div class="go-stat"><div class="go-val">€' +
    s.budget.toLocaleString() +
    '</div><div class="go-label">Remaining Budget</div></div>'
  html +=
    '<div class="go-stat"><div class="go-val">' +
    s.organisedCount +
    '</div><div class="go-label">Concerts Organised</div></div>'
  html += '</div>'
  if (scores.length > 0) {
    html += '<div class="gameover-list" style="margin-top:16px"><h3>🏆 High Scores</h3>'
    scores.slice(0, 5).forEach(function (sc, i) {
      var medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : '#' + (i + 1)
      html +=
        '<div class="go-concert-item"><span>' +
        medal +
        ' ' +
        sc.points.toLocaleString() +
        ' pts — ' +
        sc.attended +
        '/' +
        sc.total +
        ' concerts</span><span style="color:var(--gray);font-size:0.75rem">' +
        sc.date.split('T')[0] +
        '</span></div>'
    })
    html += '</div>'
  }
  var attendedList = []
  G.CONCERTS.forEach(function (c) {
    if (s.concertScores[c.id]) {
      var sc = s.concertScores[c.id]
      var artist = G.ARTISTS[c.artistKey]
      attendedList.push({
        label: artist.emoji + ' ' + artist.name + ' — ' + c.city,
        score: sc.score + '% · ' + sc.placement,
        points: sc.points + ' pts',
        badge: sc.badge
      })
    }
  })
  if (attendedList.length > 0) {
    html += '<div class="gameover-list"><h3>Concerts Attended</h3>'
    attendedList.forEach(function (item) {
      html +=
        '<div class="go-concert-item"><span>' +
        item.badge +
        ' ' +
        item.label +
        '</span><span style="color:var(--gold)">' +
        item.points +
        '</span></div>'
    })
    html += '</div>'
  }
  if (s.collectibles.length > 0) {
    html += '<div class="gameover-list"><h3>🏆 Rare Collectibles</h3>'
    s.collectibles.forEach(function (c) {
      html += '<div class="go-concert-item"><span>' + c + '</span></div>'
    })
    html += '</div>'
  }
  html +=
    '<button class="btn btn-primary" style="margin-top:24px;width:220px" id="gameover-replay">Play Again 🔄</button>'
  html += '</div>'
  return html
}

/**
 * Game Over / Score Summary screen
 */
G.showGameOver = function () {
  var s = G.state
  var el = document.getElementById('gameover-content')
  G.showScreen('screen-gameover')
  el.innerHTML = G.renderGameOver()
  if (window.lucide) lucide.createIcons()
  G.showConfetti(50)
  var shareBtn = document.createElement('button')
  shareBtn.className = 'btn btn-secondary'
  shareBtn.style.marginTop = '16px'
  shareBtn.textContent = '↗ Share Results'
  el.appendChild(shareBtn)
  shareBtn.addEventListener('click', function () {
    G.shareResults({ artist: '', city: '', score: 0, placement: '', points: s.points, budget: s.budget, date: '' })
  })
  var cardBtn = document.createElement('button')
  cardBtn.className = 'btn btn-secondary'
  cardBtn.style.marginTop = '12px'
  cardBtn.textContent = '📸 Save Card'
  el.appendChild(cardBtn)
  cardBtn.addEventListener('click', function () {
    G.downloadResultCard({
      artist: '',
      emoji: '',
      city: '',
      country: '',
      venue: '',
      score: 0,
      placement: '',
      points: s.points,
      date: '',
      budget: s.budget
    })
  })
  document.getElementById('gameover-replay').addEventListener('click', function () {
    G.sfx.click()
    G.startGame()
  })
}

/** Generate a shareable text summary */
G.shareResults = function (data) {
  var text = '🎵 Concerts Chase\n\n'
  if (data.artist) {
    text += data.emoji ? data.emoji + ' ' : ''
    text += data.artist + ' — ' + data.city + '\n'
    text += 'Score: ' + data.score + '% (' + data.placement + ')\n'
    text += 'Points: ' + (data.points || 0) + '\n'
  } else {
    text += 'Tour Complete!\n'
    text += 'Total Points: ' + (data.points || 0).toLocaleString() + '\n'
    text += 'Budget Remaining: €' + (data.budget || 0).toLocaleString() + '\n'
  }
  text += '\nPlay at Concerts Chase!'

  if (navigator.share) {
    navigator.share({ title: 'Concerts Chase', text: text }).catch(function () {})
  } else if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(function () {
      G.toast('📋 Results copied to clipboard!')
    })
  }
}

/** Draw results on canvas and download as PNG */
G.downloadResultCard = function (data) {
  var canvas = document.createElement('canvas')
  canvas.width = 360
  canvas.height = 460
  var ctx = canvas.getContext('2d')

  ctx.fillStyle = '#0a0a1a'
  ctx.fillRect(0, 0, 360, 460)

  ctx.strokeStyle = '#00f5d4'
  ctx.lineWidth = 3
  ctx.strokeRect(8, 8, 344, 444)

  ctx.fillStyle = '#ff2d78'
  ctx.font = 'bold 22px Outfit, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('Concerts Chase', 180, 45)

  if (data.emoji) ctx.font = '40px sans-serif'
  else ctx.font = '40px sans-serif'
  ctx.fillText(data.emoji || '🎵', 180, 90)

  ctx.fillStyle = '#fff'
  ctx.font = 'bold 18px Outfit, sans-serif'
  ctx.fillText(data.artist || 'Tour Summary', 180, 125)

  ctx.fillStyle = '#888'
  ctx.font = '13px Outfit, sans-serif'
  if (data.city) {
    ctx.fillText(data.city + ', ' + data.country + ' — ' + data.venue, 180, 148)
  }

  if (data.score) {
    ctx.fillStyle =
      data.score >= 90 ? '#ffd700' : data.score >= 70 ? '#00f5d4' : data.score >= 40 ? '#ff8800' : '#ff2d78'
    ctx.font = 'bold 36px Outfit, sans-serif'
    ctx.fillText(data.score + '%', 180, 200)
    ctx.fillStyle = '#aaa'
    ctx.font = '14px Outfit, sans-serif'
    ctx.fillText(data.placement, 180, 222)
  }

  ctx.fillStyle = '#666'
  ctx.font = '12px Outfit, sans-serif'
  if (data.date) {
    ctx.fillText(data.date, 180, 250)
  }

  var y = 270
  if (data.points) {
    ctx.fillStyle = '#ffd700'
    ctx.font = 'bold 16px Outfit, sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText('Points', 30, y)
    ctx.textAlign = 'right'
    ctx.fillText((data.points || 0).toLocaleString(), 330, y)
    y += 28
  }
  if (data.budget) {
    ctx.fillStyle = '#44ff88'
    ctx.font = 'bold 16px Outfit, sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText('Budget', 30, y)
    ctx.textAlign = 'right'
    ctx.fillText('€' + (data.budget || 0).toLocaleString(), 330, y)
    y += 28
  }

  ctx.fillStyle = '#444'
  ctx.font = '11px Outfit, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('Play at Concerts Chase', 180, y + 20)

  var link = document.createElement('a')
  link.download = 'concerts-chase-card.png'
  link.href = canvas.toDataURL('image/png')
  link.click()

  if (navigator.share && navigator.canShare && navigator.canShare({ type: 'image/png' })) {
    canvas.toBlob(function (blob) {
      var file = new File([blob], 'concerts-chase-card.png', { type: 'image/png' })
      navigator.share({ title: 'Concerts Chase Card', files: [file] }).catch(function () {})
    }, 'image/png')
  }
}
