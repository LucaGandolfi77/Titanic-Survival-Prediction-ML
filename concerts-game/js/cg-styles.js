/* cg-styles.js — Harry Styles "Dance Move Mirror" concert minigame
   Simon Says with arrow keys. Replicate a growing sequence.
   Returns score 0-100. */
window.G = window.G || {};

G.cgStyles = function(concertId, onComplete) {
  var el = document.getElementById('concert-content');

  var directions = ['up', 'down', 'left', 'right'];
  var arrows = { up: '↑', down: '↓', left: '←', right: '→' };
  var keyMap = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };

  var totalRounds = 6;
  var currentRound = 0;
  var sequence = [];
  var playerInput = [];
  var isShowingSequence = false;
  var isPlayerTurn = false;
  var correctRounds = 0;

  var html = '<div class="cg-container fade-in">';
  html += '<div class="cg-title">🕺 Dance Move Mirror — Harry Styles</div>';
  html += '<div class="cg-instruction">Watch the sequence, then repeat it!</div>';
  html += '<div class="cg-score" id="cg-score">Round: 1/' + totalRounds + '</div>';
  html += '<div id="cg-status" style="font-size:1.2rem;color:var(--gold);margin:12px 0;min-height:32px">Watch closely…</div>';
  html += '<div class="simon-grid">';
  directions.forEach(function(d) {
    html += '<button class="simon-btn" id="simon-' + d + '" data-dir="' + d + '">' + arrows[d] + '</button>';
  });
  html += '</div>';
  html += '</div>';

  el.innerHTML = html;

  // Start
  setTimeout(nextRound, 1000);

  function nextRound() {
    if (currentRound >= totalRounds) {
      finish();
      return;
    }

    // Add a new direction to sequence
    sequence.push(directions[Math.floor(Math.random() * 4)]);
    currentRound++;
    playerInput = [];

    document.getElementById('cg-score').textContent = 'Round: ' + currentRound + '/' + totalRounds + ' | Score: ' + correctRounds;
    document.getElementById('cg-status').textContent = 'Watch closely…';
    document.getElementById('cg-status').style.color = 'var(--gold)';

    // Disable buttons during show
    setButtonsEnabled(false);
    isShowingSequence = true;
    isPlayerTurn = false;

    // Show sequence
    showSequence(0);
  }

  function showSequence(idx) {
    if (idx >= sequence.length) {
      // Done showing, player's turn
      isShowingSequence = false;
      isPlayerTurn = true;
      setButtonsEnabled(true);
      document.getElementById('cg-status').textContent = 'Your turn! (' + sequence.length + ' moves)';
      document.getElementById('cg-status').style.color = 'var(--teal)';
      return;
    }

    var dir = sequence[idx];
    var btn = document.getElementById('simon-' + dir);
    btn.classList.add('lit');
    G.sfx.note(440 + directions.indexOf(dir) * 100);

    setTimeout(function() {
      btn.classList.remove('lit');
      setTimeout(function() { showSequence(idx + 1); }, 200);
    }, 500);
  }

  function setButtonsEnabled(enabled) {
    directions.forEach(function(d) {
      document.getElementById('simon-' + d).disabled = !enabled;
    });
  }

  function playerPress(dir) {
    if (!isPlayerTurn) return;

    var btn = document.getElementById('simon-' + dir);
    btn.classList.add('lit');
    G.sfx.note(440 + directions.indexOf(dir) * 100);
    setTimeout(function() { btn.classList.remove('lit'); }, 200);

    playerInput.push(dir);
    var idx = playerInput.length - 1;

    if (playerInput[idx] !== sequence[idx]) {
      // Wrong!
      G.sfx.fail();
      document.getElementById('cg-status').textContent = '✗ Wrong move!';
      document.getElementById('cg-status').style.color = 'var(--danger)';
      isPlayerTurn = false;
      setButtonsEnabled(false);
      setTimeout(nextRound, 1200);
      return;
    }

    if (playerInput.length === sequence.length) {
      // Correct round!
      correctRounds++;
      G.sfx.success();
      document.getElementById('cg-status').textContent = '✓ Perfect!';
      document.getElementById('cg-status').style.color = 'var(--success)';
      isPlayerTurn = false;
      setButtonsEnabled(false);
      setTimeout(nextRound, 1200);
    }
  }

  // Click handlers
  directions.forEach(function(d) {
    document.getElementById('simon-' + d).addEventListener('click', function() {
      playerPress(d);
    });
  });

  // Keyboard handler
  function onKey(e) {
    if (keyMap[e.key] && isPlayerTurn) {
      e.preventDefault();
      playerPress(keyMap[e.key]);
    }
  }
  document.addEventListener('keydown', onKey);

  function finish() {
    document.removeEventListener('keydown', onKey);
    var score = Math.round((correctRounds / totalRounds) * 100);
    onComplete(score);
  }
};
