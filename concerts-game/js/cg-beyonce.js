/* cg-beyonce.js — Beyoncé "Formation Sync" concert minigame
   4×4 grid. Pattern flashes briefly, player must click same cells from memory.
   Harder each round. Returns score 0-100. */
window.G = window.G || {};

G.cgBeyonce = function(concertId, onComplete) {
  var el = document.getElementById('concert-content');

  var GRID = 4;
  var totalRounds = 5;
  var currentRound = 0;
  var correctRounds = 0;
  var pattern = [];
  var playerClicks = [];
  var isShowingPattern = false;
  var isPlayerTurn = false;

  render();
  setTimeout(nextRound, 800);

  function render() {
    var html = '<div class="cg-container fade-in">';
    html += '<div class="cg-title">👑 Formation Sync — Beyoncé</div>';
    html += '<div class="cg-instruction">Memorise the pattern, then click the same cells!</div>';
    html += '<div class="cg-score" id="cg-score">Round: 0/' + totalRounds + ' | Score: 0</div>';
    html += '<div id="cg-status" style="font-size:1.1rem;color:var(--gold);margin:10px 0;min-height:28px">Get ready…</div>';
    html += '<div class="formation-grid" id="formation-grid">';
    for (var i = 0; i < GRID * GRID; i++) {
      html += '<div class="formation-cell" data-idx="' + i + '"></div>';
    }
    html += '</div>';
    html += '</div>';
    el.innerHTML = html;

    // Attach click handlers
    document.querySelectorAll('.formation-cell').forEach(function(cell) {
      cell.addEventListener('click', function() {
        if (!isPlayerTurn) return;
        var idx = parseInt(this.getAttribute('data-idx'));
        handleCellClick(idx, this);
      });
    });
  }

  function nextRound() {
    if (currentRound >= totalRounds) {
      finish();
      return;
    }

    currentRound++;
    playerClicks = [];
    isPlayerTurn = false;
    isShowingPattern = true;

    // Generate pattern — more cells each round
    var cellCount = currentRound + 2; // 3, 4, 5, 6, 7 cells
    pattern = [];
    var available = [];
    for (var i = 0; i < GRID * GRID; i++) available.push(i);
    for (var j = 0; j < cellCount && available.length > 0; j++) {
      var pick = Math.floor(Math.random() * available.length);
      pattern.push(available[pick]);
      available.splice(pick, 1);
    }

    var scoreEl = document.getElementById('cg-score');
    if (scoreEl) scoreEl.textContent = 'Round: ' + currentRound + '/' + totalRounds + ' | Score: ' + correctRounds;

    var statusEl = document.getElementById('cg-status');
    if (statusEl) { statusEl.textContent = 'Watch the pattern!'; statusEl.style.color = 'var(--gold)'; }

    // Clear previous highlights
    clearGrid();

    // Show pattern
    setTimeout(function() {
      showPattern();
    }, 400);
  }

  function showPattern() {
    var cells = document.querySelectorAll('.formation-cell');
    pattern.forEach(function(idx) {
      cells[idx].classList.add('lit');
    });
    G.sfx.reveal();

    // Show time proportional to difficulty
    var showTime = Math.max(800, 2000 - (currentRound - 1) * 250);
    setTimeout(function() {
      clearGrid();
      isShowingPattern = false;
      isPlayerTurn = true;
      var statusEl = document.getElementById('cg-status');
      if (statusEl) {
        statusEl.textContent = 'Your turn! Click ' + pattern.length + ' cells';
        statusEl.style.color = 'var(--teal)';
      }
    }, showTime);
  }

  function clearGrid() {
    document.querySelectorAll('.formation-cell').forEach(function(cell) {
      cell.classList.remove('lit', 'selected', 'correct', 'wrong');
    });
  }

  function handleCellClick(idx, cellEl) {
    if (playerClicks.indexOf(idx) !== -1) return; // already clicked
    playerClicks.push(idx);
    cellEl.classList.add('selected');
    G.sfx.tick();

    // Check when player has clicked enough cells
    if (playerClicks.length >= pattern.length) {
      isPlayerTurn = false;
      setTimeout(evaluateRound, 300);
    }
  }

  function evaluateRound() {
    var cells = document.querySelectorAll('.formation-cell');
    var correct = 0;

    // Highlight correct/wrong
    playerClicks.forEach(function(idx) {
      if (pattern.indexOf(idx) !== -1) {
        cells[idx].classList.add('correct');
        correct++;
      } else {
        cells[idx].classList.add('wrong');
      }
    });

    // Show missed cells
    pattern.forEach(function(idx) {
      if (playerClicks.indexOf(idx) === -1) {
        cells[idx].classList.add('lit');
      }
    });

    var allCorrect = (correct === pattern.length && playerClicks.length === pattern.length);

    var statusEl = document.getElementById('cg-status');
    if (allCorrect) {
      correctRounds++;
      G.sfx.success();
      if (statusEl) { statusEl.textContent = '✓ Perfect formation!'; statusEl.style.color = 'var(--success)'; }
    } else {
      G.sfx.fail();
      if (statusEl) { statusEl.textContent = '✗ ' + correct + '/' + pattern.length + ' correct'; statusEl.style.color = 'var(--danger)'; }
    }

    setTimeout(nextRound, 1400);
  }

  function finish() {
    var score = Math.round((correctRounds / totalRounds) * 100);
    onComplete(score);
  }
};
