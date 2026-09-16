/* cg-kpop.js — K-pop "Rhythm Sync" concert minigame
   Notes fall in 5 lanes (1-5). Press correct number key when note reaches line.
   Returns score 0-100. */
window.G = window.G || {}

G.cgKpop = function (concertId, onComplete) {
  var el = document.getElementById('concert-content')
  var W = 500,
    H = 350
  var LANES = 5
  var totalNotes = 30
  var notes = []
  var hits = 0
  var misses = 0
  var maxMisses = 5
  var currentRound = 0
  var gameOver = false
  var spawnInterval = null
  var animFrame = null
  var speed = 2
  var spawnRate = 700
  var gameTime = 0
  var lastTime = 0
  var spawnTimer = null

  var laneColors = ['#ff2d78', '#00f5d4', '#ffd700', '#ff8800', '#aa44ff']
  var laneKeys = ['1', '2', '3', '4', '5']

  var html = '<div class="cg-container fade-in">'
  html += '<div class="cg-title">💜 Rhythm Sync — BLACKPULSE</div>'
  html += '<div class="cg-instruction">Press <strong>1-5</strong> when notes reach the line! Miss 5 = game over.</div>'
  html += '<div class="cg-score" id="cg-score">Hits: 0 | Misses: 0/' + maxMisses + '</div>'
  html += '<div style="display:flex;justify-content:center;gap:6px;margin-top:6px">'
  laneKeys.forEach(function (k, i) {
    html +=
      '<div style="width:' +
      W / LANES +
      'px;text-align:center;color:' +
      laneColors[i] +
      ';font-weight:700;font-size:1.2rem">' +
      k +
      '</div>'
  })
  html += '</div>'
  html += '<div class="cg-canvas-wrap"><canvas id="cg-canvas" width="' + W + '" height="' + H + '"></canvas></div>'
  html += '</div>'
  el.innerHTML = html

  var canvas = document.getElementById('cg-canvas')
  var ctx = G.setupHDCanvas(canvas, W, H)
  var LANEW = W / LANES

  function spawnNote() {
    if (gameOver) return
    var lane = Math.floor(Math.random() * LANES)
    notes.push({ lane: lane, y: -20, hit: false, missed: false })
    currentRound++
  }

  spawnInterval = setInterval(function () {
    if (gameOver) return
    spawnNote()
    gameTime += spawnRate
    if (gameTime > 8000 && spawnRate > 500) {
      spawnRate = 550
      speed = 2.5
    }
    if (gameTime > 16000 && spawnRate > 400) {
      spawnRate = 380
      speed = 3
    }
    if (gameTime > 24000) {
      spawnRate = 300
      speed = 3.5
    }
  }, spawnRate)

  setTimeout(function () {
    if (!gameOver) endGame()
  }, 28000)

  lastTime = performance.now()
  animate()

  function animate() {
    if (gameOver) return
    var now = performance.now()
    var dt = (now - lastTime) / 16.67
    lastTime = now

    ctx.clearRect(0, 0, W, H)

    for (var i = 1; i < LANES; i++) {
      ctx.beginPath()
      ctx.moveTo(i * LANEW, 0)
      ctx.lineTo(i * LANEW, H)
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    var HIT_Y = H - 50
    var HIT_RANGE = 35

    ctx.beginPath()
    ctx.moveTo(0, HIT_Y)
    ctx.lineTo(W, HIT_Y)
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'
    ctx.lineWidth = 2
    ctx.stroke()

    notes.forEach(function (n) {
      if (n.hit || n.missed) return
      n.y += speed * dt
      var x = n.lane * LANEW + LANEW / 2
      ctx.beginPath()
      ctx.arc(x, n.y, 12, 0, Math.PI * 2)
      ctx.fillStyle = laneColors[n.lane]
      ctx.fill()
      ctx.beginPath()
      ctx.arc(x, n.y, 12, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(255,255,255,0.5)'
      ctx.lineWidth = 2
      ctx.stroke()

      if (n.y > HIT_Y + HIT_RANGE + 20) {
        n.missed = true
        misses++
        G.sfx.fail()
        updateScore()
        if (misses >= maxMisses) endGame()
      }
    })

    notes = notes.filter(function (n) {
      return !n.missed || n.y < H + 30
    })
    animFrame = requestAnimationFrame(animate)
  }

  function tryHit(lane) {
    if (gameOver) return
    var best = null
    var bestDist = Infinity
    notes.forEach(function (n) {
      if (n.hit || n.missed || n.lane !== lane) return
      var dist = Math.abs(n.y - (H - 50))
      if (dist < 40 && dist < bestDist) {
        bestDist = dist
        best = n
      }
    })
    if (best) {
      best.hit = true
      hits++
      G.sfx.note(330 + lane * 80)
      updateScore()
    }
  }

  function updateScore() {
    var se = document.getElementById('cg-score')
    if (se) se.textContent = 'Hits: ' + hits + ' | Misses: ' + misses + '/' + maxMisses
  }

  function onKey(e) {
    var idx = laneKeys.indexOf(e.key)
    if (idx >= 0) {
      e.preventDefault()
      tryHit(idx)
    }
  }
  document.addEventListener('keydown', onKey)

  canvas.addEventListener(
    'touchstart',
    function (e) {
      e.preventDefault()
      var touch = e.changedTouches[0]
      var rect = canvas.getBoundingClientRect()
      var x = ((touch.clientX - rect.left) / rect.width) * W
      var lane = Math.min(LANES - 1, Math.max(0, Math.floor(x / LANEW)))
      tryHit(lane)
    },
    { passive: false }
  )

  canvas.addEventListener('click', function (e) {
    var rect = canvas.getBoundingClientRect()
    var x = ((e.clientX - rect.left) / rect.width) * W
    var lane = Math.min(LANES - 1, Math.floor(x / LANEW))
    tryHit(lane)
  })

  function endGame() {
    if (gameOver) return
    gameOver = true
    clearInterval(spawnInterval)
    if (animFrame) cancelAnimationFrame(animFrame)
    document.removeEventListener('keydown', onKey)
    var score = totalNotes > 0 ? Math.round((hits / totalNotes) * 100) : 0
    onComplete(score)
  }
}
