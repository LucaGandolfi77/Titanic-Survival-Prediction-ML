/* mg-pocketfootball.js — "Pocket Football" discovery minigame
   Quick 25-second football match. Score 2 goals against AI keeper. */
window.G = window.G || {}

G.mgPocketFootball = function (concertId, onComplete) {
  var c = G.CONCERTS[concertId]
  var artist = G.ARTISTS[c.artistKey]
  var el = document.getElementById('discovery-content')

  var timeLeft = 25
  var goals = 0
  var timer = null
  var animFrame = null
  var ball = { x: 150, y: 340, vx: 0, vy: 0, active: false }
  var gk = { x: 150 }
  var msg = 'Click the pitch to shoot!'

  render()
  startTimer()
  startAnimation()

  function render() {
    var html = '<div class="mg-container fade-in">'
    html += '<div class="mg-title">⚽ Pocket Football</div>'
    html += '<div class="mg-subtitle">Score 2 goals against the AI keeper!</div>'
    html += '<div class="mg-timer" id="mg-timer">' + timeLeft + 's</div>'
    html += '<div class="pf-scoreboard">Goals: <span id="pf-score" class="pf-goals">0</span> / 2</div>'
    html +=
      '<svg id="pf-pitch" viewBox="0 0 300 400" style="width:100%;max-width:320px;display:block;margin:8px auto;cursor:crosshair">'
    html += '<rect width="300" height="400" fill="#2d8a4e" rx="4"/>'
    html += '<line x1="150" y1="0" x2="150" y2="400" stroke="white" stroke-width="1" opacity="0.4"/>'
    html += '<circle cx="150" cy="200" r="40" stroke="white" stroke-width="1" fill="none" opacity="0.4"/>'
    html += '<line x1="100" y1="0" x2="200" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>'
    html += '<line x1="100" y1="400" x2="200" y2="400" stroke="white" stroke-width="1" opacity="0.4"/>'
    html += '<rect x="115" y="0" width="70" height="14" fill="white" opacity="0.85" rx="2"/>'
    html += '<rect x="115" y="386" width="70" height="14" fill="white" opacity="0.85" rx="2"/>'
    html += '<circle id="pf-gk" cx="' + gk.x + '" cy="25" r="12" fill="#ff4444" opacity="0.8"/>'
    html +=
      '<circle id="pf-ball" cx="' + ball.x + '" cy="' + ball.y + '" r="8" fill="white" stroke="#333" stroke-width="1"/>'
    html += '</svg>'
    html +=
      '<div id="pf-msg" style="color:var(--gray-light);font-size:0.8rem;margin-top:6px;min-height:20px">' +
      msg +
      '</div>'
    html += '</div>'
    el.innerHTML = html

    document.getElementById('pf-pitch').addEventListener('click', function (e) {
      if (ball.active) return
      var svg = this
      var rect = svg.getBoundingClientRect()
      var sx = 300 / rect.width
      var sy = 400 / rect.height
      var x = (e.clientX - rect.left) * sx
      var y = (e.clientY - rect.top) * sy
      shoot(x, y)
    })
  }

  function shoot(targetX, targetY) {
    var dx = targetX - ball.x
    var dy = targetY - ball.y
    var dist = Math.sqrt(dx * dx + dy * dy)
    if (dist < 1) return
    var speed = 4.5
    ball.vx = (dx / dist) * speed
    ball.vy = (dy / dist) * speed
    ball.active = true
    var goalMsgEl = document.getElementById('pf-msg')
    if (goalMsgEl) goalMsgEl.textContent = 'Shooting...'
    G.sfx.kick()
  }

  function startAnimation() {
    animFrame = requestAnimationFrame(frame)
  }

  function frame() {
    if (!ball.active) {
      animFrame = requestAnimationFrame(frame)
      return
    }

    ball.x += ball.vx
    ball.y += ball.vy

    // Goalkeeper tracks ball x in attacking half
    if (ball.y < 200 && ball.y > 14) {
      var diff = ball.x - gk.x
      gk.x += Math.sign(diff) * Math.min(Math.abs(diff), 2.5)
    }

    // Check goal
    if (ball.y <= 14 && ball.x > 115 && ball.x < 185) {
      if (Math.abs(ball.x - gk.x) < 20 && Math.random() < 0.3) {
        // Saved
        resetBall()
        var savedMsgEl = document.getElementById('pf-msg')
        if (savedMsgEl) savedMsgEl.textContent = 'Saved by the keeper!'
        G.sfx.fail()
      } else {
        // GOAL!
        goals++
        var scoreEl = document.getElementById('pf-score')
        if (scoreEl) scoreEl.textContent = goals
        var goalMsgEl = document.getElementById('pf-msg')
        if (goalMsgEl) goalMsgEl.textContent = 'GOAL! ' + goals + '/2'
        G.sfx.goal()
        G.spawnParticles('star', window.innerWidth / 2, window.innerHeight / 2, 15)

        if (goals >= 2) {
          clearInterval(timer)
          setTimeout(function () {
            finish(true)
          }, 600)
          return
        }
        resetBall()
      }
    }

    // Out of bounds
    if (ball.x < 0 || ball.x > 300 || ball.y > 400) {
      resetBall()
      var boundMsgEl = document.getElementById('pf-msg')
      if (boundMsgEl) boundMsgEl.textContent = 'Out of bounds! Click to shoot again.'
    }

    // Update ball/GK visuals
    var ballEl = document.getElementById('pf-ball')
    if (ballEl) {
      ballEl.setAttribute('cx', ball.x)
      ballEl.setAttribute('cy', ball.y)
    }
    var gkEl = document.getElementById('pf-gk')
    if (gkEl) {
      gkEl.setAttribute('cx', gk.x)
    }

    animFrame = requestAnimationFrame(frame)
  }

  function resetBall() {
    ball.x = 150
    ball.y = 340
    ball.vx = 0
    ball.vy = 0
    ball.active = false
    if (animFrame) cancelAnimationFrame(animFrame)
  }

  function startTimer() {
    timer = setInterval(function () {
      timeLeft--
      var te = document.getElementById('mg-timer')
      if (te) te.textContent = timeLeft + 's'
      if (timeLeft <= 0) {
        clearInterval(timer)
        if (animFrame) cancelAnimationFrame(animFrame)
        finish(false)
      }
    }, 1000)
  }

  function finish(success) {
    clearInterval(timer)
    if (animFrame) cancelAnimationFrame(animFrame)
    ball.active = false
    if (success) {
      G.sfx.reveal()
      el.innerHTML =
        '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--gold)">🎉 Match Won!</h2>' +
        '<p style="margin:12px 0">' +
        artist.emoji +
        ' ' +
        artist.name +
        ' — ' +
        artist.tour +
        '</p>' +
        '<p style="color:var(--teal)">' +
        c.city +
        ', ' +
        c.country +
        ' — ' +
        c.date +
        '</p>' +
        '<p style="color:var(--gray-light)">' +
        c.venue +
        '</p>' +
        '<button class="btn btn-primary" style="margin-top:16px" id="mg-done-btn">Continue</button>' +
        '</div>'
    } else {
      G.sfx.fail()
      el.innerHTML =
        '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--danger)">⏰ Time\'s Up!</h2>' +
        '<p style="color:var(--gray-light);margin:12px 0">You scored ' +
        goals +
        '/2 goals. Try again!</p>' +
        '<button class="btn btn-secondary" style="margin-top:16px" id="mg-done-btn">Back to Map</button>' +
        '</div>'
    }
    document.getElementById('mg-done-btn').addEventListener('click', function () {
      onComplete(success)
    })
  }
}
