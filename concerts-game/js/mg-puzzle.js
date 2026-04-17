/* mg-puzzle.js — "Fan Photo Puzzle" discovery minigame
   3×3 sliding tile puzzle. Complete under 30 seconds for bonus €200. */
window.G = window.G || {};

G.mgPuzzle = function(concertId, onComplete) {
  var c = G.CONCERTS[concertId];
  var artist = G.ARTISTS[c.artistKey];
  var el = document.getElementById('discovery-content');

  // Generate solvable puzzle (tiles 1-8, 0 = empty)
  var tiles = [1,2,3,4,5,6,7,8,0];

  // Shuffle by making random valid moves (guarantees solvable)
  var emptyPos = 8;
  for (var s = 0; s < 200; s++) {
    var neighbors = getNeighbors(emptyPos);
    var pick = neighbors[Math.floor(Math.random() * neighbors.length)];
    tiles[emptyPos] = tiles[pick];
    tiles[pick] = 0;
    emptyPos = pick;
  }

  var timeLeft = 30;
  var timer = null;
  var moves = 0;

  render();

  function render() {
    var html = '<div class="mg-container fade-in">';
    html += '<div class="mg-title">🧩 Fan Photo Puzzle</div>';
    html += '<div class="mg-subtitle">Slide tiles to order 1-8. Complete for concert info + €200 bonus!</div>';
    html += '<div class="mg-timer" id="mg-timer">' + timeLeft + 's</div>';
    html += '<div style="color:var(--gray);font-size:0.8rem">Moves: <span id="puzzle-moves">' + moves + '</span></div>';
    html += '<div class="puzzle-grid" id="puzzle-grid">';
    tiles.forEach(function(t, i) {
      if (t === 0) {
        html += '<div class="puzzle-tile empty" data-pos="' + i + '"></div>';
      } else {
        html += '<div class="puzzle-tile" data-pos="' + i + '" data-val="' + t + '">' + t + '</div>';
      }
    });
    html += '</div>';
    html += '</div>';

    el.innerHTML = html;

    // Timer (only start once)
    if (!timer) {
      timer = setInterval(function() {
        timeLeft--;
        var te = document.getElementById('mg-timer');
        if (te) te.textContent = timeLeft + 's';
        if (timeLeft <= 0) { clearInterval(timer); finish(false); }
      }, 1000);
    } else {
      var te = document.getElementById('mg-timer');
      if (te) te.textContent = timeLeft + 's';
      var me = document.getElementById('puzzle-moves');
      if (me) me.textContent = moves;
    }

    // Tile click handlers
    document.querySelectorAll('.puzzle-tile:not(.empty)').forEach(function(tile) {
      tile.addEventListener('click', function() {
        var pos = parseInt(this.getAttribute('data-pos'));
        tryMove(pos);
      });
    });
  }

  function getNeighbors(pos) {
    var n = [];
    var row = Math.floor(pos / 3), col = pos % 3;
    if (row > 0) n.push(pos - 3);
    if (row < 2) n.push(pos + 3);
    if (col > 0) n.push(pos - 1);
    if (col < 2) n.push(pos + 1);
    return n;
  }

  function tryMove(pos) {
    var neighbors = getNeighbors(pos);
    if (neighbors.indexOf(emptyPos) === -1) return; // not adjacent to empty

    G.sfx.click();
    tiles[emptyPos] = tiles[pos];
    tiles[pos] = 0;
    emptyPos = pos;
    moves++;

    // Check win condition
    var solved = true;
    for (var i = 0; i < 8; i++) {
      if (tiles[i] !== i + 1) { solved = false; break; }
    }

    if (solved) {
      clearInterval(timer);
      setTimeout(function() { finish(true); }, 300);
    } else {
      render();
    }
  }

  function finish(success) {
    clearInterval(timer);
    if (success) {
      G.sfx.success();
      G.earnBudget(200);
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--gold)">🎉 Puzzle Solved!</h2>' +
        '<p style="margin:8px 0;color:var(--success)">+€200 bonus!</p>' +
        '<p style="margin:8px 0">' + artist.emoji + ' ' + artist.name + ' — ' + artist.tour + '</p>' +
        '<p style="color:var(--teal)">' + c.city + ', ' + c.country + ' — ' + c.date + '</p>' +
        '<p style="color:var(--gray-light)">' + c.venue + '</p>' +
        '<p style="color:var(--gray);font-size:0.8rem;margin-top:6px">' + moves + ' moves</p>' +
        '<button class="btn btn-primary" style="margin-top:16px" id="mg-done-btn">Continue</button>' +
        '</div>';
    } else {
      G.sfx.fail();
      el.innerHTML = '<div class="mg-container fade-in" style="justify-content:center;align-items:center;min-height:200px">' +
        '<h2 style="color:var(--danger)">⏰ Time\'s Up!</h2>' +
        '<p style="color:var(--gray-light);margin:12px 0">The puzzle beat you this time.</p>' +
        '<button class="btn btn-secondary" style="margin-top:16px" id="mg-done-btn">Back to Map</button>' +
        '</div>';
    }
    document.getElementById('mg-done-btn').addEventListener('click', function() {
      onComplete(success);
    });
  }
};
