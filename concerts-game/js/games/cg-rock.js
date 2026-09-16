/* cg-rock.js — Classic Rock "Drum Solo" concert minigame
   Notes fall in 4 lanes (D/F/J/K). Catch them like Maneskin but rock style.
   Speed increases faster. Returns score 0-100. */
window.G = window.G || {}

G.cgRock = function (concertId, onComplete) {
  var el = document.getElementById('concert-content')
  var W = 400,
    H = 400
  var LANES = 4
  var LANEW = W / LANES
  var laneKeys = ['d', 'f', 'j', 'k']
  var laneColors = ['#ff4444', '#ffaa00', '#44ff88', '#4488ff']
  var HIT_Y = H - 50
  var HIT_RANGE = 35
  var totalNotes = 40
  var notes = []
  var hits = 0
  var misses = 0
  var maxMisses = 5
  var gameOver = false
  var spawnInterval = null
  var animFrame = null
  var speed = 2
  var spawnRate = 600
  var gameTime = 0
  var lastTime = 0

  var html = '<div class="cg-container fade-in">'
  html += '<div class="cg-title">🥁 Drum Solo — The Thunder</div>'
  html +=
    '<div class="cg-instruction">Press <strong>D F J K</strong> when notes reach the line! Miss 5 = game over.</div>'
  html += '<div class="cg-score" id="cg-score">Hits: 0 | Misses: 0/' + maxMisses + '</div>'
  html += '<div style="display:flex;justify-content:center;gap:6px;margin-top:6px">'
  laneKeys.forEach(function (k, i) {
    html +=
      '<div style="width:' +
      LANEW +
      'px;text-align:center;color:' +
      laneColors[i] +
      ';font-weight:700;font-size:1.2rem">' +
      k.toUpperCase() +
      '</div>'
  })
  html += '</div>'
  html += '<div class="cg-canvas-wrap"><canvas id="cg-canvas" width="' + W + '" height="' + H + '"></canvas></div>'
  html += '</div>'
  el.innerHTML = html

  var canvas = document.getElementById('cg-canvas')
  var ctx = G.setupHDCanvas(canvas, W, H)

  spawnInterval = setInterval(function () {
    if (gameOver) return
    var lane = Math.floor(Math.random() * LANES)
    notes.push({ lane: lane, y: -20, hit: false, missed: false })
    gameTime += spawnRate
    if (gameTime > 6000) {
      spawnRate = 500
      speed = 2.5
    }
    if (gameTime > 12000) {
      spawnRate = 400
      speed = 3
    }
    if (gameTime > 18000) {
      spawnRate = 320
      speed = 3.5
    }
    if (gameTime > 24000) {
      spawnRate = 250
      speed = 4
    }
  }, spawnRate)

  setTimeout(function () {
    if (!gameOver) endGame()
  }, 30000)

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
      // Drum note shape (rectangle instead of circle)
      ctx.fillStyle = laneColors[n.lane]
      ctx.fillRect(x - 8, n.y - 6, 16, 12)
      ctx.strokeStyle = 'rgba(255,255,255,0.5)'
      ctx.lineWidth = 2
      ctx.strokeRect(x - 8, n.y - 6, 16, 12)

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
      var dist = Math.abs(n.y - HIT_Y)
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
    var idx = laneKeys.indexOf(e.key.toLowerCase())
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
