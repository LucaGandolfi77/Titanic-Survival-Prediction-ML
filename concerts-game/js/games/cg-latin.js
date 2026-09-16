/* cg-latin.js — Latin "Dance Battle" concert minigame
   Arrow sequence appears, repeat it with correct timing.
   Sequence grows each round. Returns score 0-100. */
window.G = window.G || {}

G.cgLatin = function (concertId, onComplete) {
  var el = document.getElementById('concert-content')
  var directions = ['up', 'down', 'left', 'right']
  var arrows = { up: '↑', down: '↓', left: '←', right: '→' }
  var keyMap = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' }
  var colorMap = { up: '#ff2d78', down: '#00f5d4', left: '#ffd700', right: '#ff8800' }

  var totalRounds = 7
  var currentRound = 0
  var sequence = []
  var playerInput = []
  var isShowingSequence = false
  var isPlayerTurn = false
  var correctRounds = 0

  var html = '<div class="cg-container fade-in">'
  html += '<div class="cg-title">💃 Dance Battle — El Fuego</div>'
  html += '<div class="cg-instruction">Watch the sequence, then repeat it with arrow keys!</div>'
  html += '<div class="cg-score" id="cg-score">Round: 0/' + totalRounds + '</div>'
  html +=
    '<div id="cg-status" style="font-size:1.2rem;color:var(--gold);margin:12px 0;min-height:32px">Get ready…</div>'
  html += '<div class="simon-grid">'
  directions.forEach(function (d) {
    html +=
      '<button class="simon-btn" id="simon-latin-' +
      d +
      '" data-dir="' +
      d +
      '" style="border-color:' +
      colorMap[d] +
      '">' +
      arrows[d] +
      '</button>'
  })
  html += '</div>'
  html += '</div>'
  var statusEl, scoreEl
  el.innerHTML = html

  setTimeout(nextRound, 1000)

  function nextRound() {
    if (currentRound >= totalRounds) {
      finish()
      return
    }
    currentRound++
    playerInput = []
    isShowingSequence = true
    isPlayerTurn = false
    sequence.push(directions[Math.floor(Math.random() * 4)])

    scoreEl = document.getElementById('cg-score')
    if (scoreEl) scoreEl.textContent = 'Round: ' + currentRound + '/' + totalRounds + ' | Score: ' + correctRounds
    statusEl = document.getElementById('cg-status')
    if (statusEl) {
      statusEl.textContent = 'Watch the pattern!'
      statusEl.style.color = 'var(--gold)'
    }

    setSimonButtons(false)
    showSequence(0)
  }

  function showSequence(idx) {
    if (idx >= sequence.length) {
      isShowingSequence = false
      isPlayerTurn = true
      setSimonButtons(true)
      statusEl = document.getElementById('cg-status')
      if (statusEl) {
        statusEl.textContent = 'Your turn! ' + sequence.length + ' moves'
        statusEl.style.color = 'var(--teal)'
      }
      return
    }
    var dir = sequence[idx]
    var btn = document.getElementById('simon-latin-' + dir)
    btn.classList.add('lit')
    G.sfx.note(440 + directions.indexOf(dir) * 100)
    setTimeout(function () {
      btn.classList.remove('lit')
      setTimeout(function () {
        showSequence(idx + 1)
      }, 200)
    }, 500)
  }

  function playerPress(dir) {
    if (!isPlayerTurn) return
    var btn = document.getElementById('simon-latin-' + dir)
    btn.classList.add('lit')
    G.sfx.note(440 + directions.indexOf(dir) * 100)
    setTimeout(function () {
      btn.classList.remove('lit')
    }, 200)
    playerInput.push(dir)
    var idx = playerInput.length - 1
    if (playerInput[idx] !== sequence[idx]) {
      G.sfx.fail()
      statusEl = document.getElementById('cg-status')
      if (statusEl) {
        statusEl.textContent = '✗ Wrong move!'
        statusEl.style.color = 'var(--danger)'
      }
      isPlayerTurn = false
      setSimonButtons(false)
      setTimeout(nextRound, 1200)
      return
    }
    if (playerInput.length === sequence.length) {
      correctRounds++
      G.sfx.success()
      statusEl = document.getElementById('cg-status')
      if (statusEl) {
        statusEl.textContent = '✓ Perfect!'
        statusEl.style.color = 'var(--success)'
      }
      isPlayerTurn = false
      setSimonButtons(false)
      setTimeout(nextRound, 1200)
    }
  }

  function setSimonButtons(enabled) {
    document.querySelectorAll('[id^="simon-latin-"]').forEach(function (btn) {
      btn.disabled = !enabled
    })
  }

  document.querySelectorAll('[id^="simon-latin-"]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      playerPress(this.getAttribute('data-dir'))
    })
  })

  function onKey(e) {
    if (keyMap[e.key] && isPlayerTurn) {
      e.preventDefault()
      playerPress(keyMap[e.key])
    }
  }
  document.addEventListener('keydown', onKey)

  function finish() {
    document.removeEventListener('keydown', onKey)
    var score = Math.round((correctRounds / totalRounds) * 100)
    onComplete(score)
  }
}
