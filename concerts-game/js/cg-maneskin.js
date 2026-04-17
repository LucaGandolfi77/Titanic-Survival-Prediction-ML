/* cg-maneskin.js — Måneskin "Guitar Hero Lite" concert minigame
   Notes fall from top in 4 lanes (A/S/D/F). Catch them. Miss 5 = over.
   Returns score 0-100. */
window.G = window.G || {};

G.cgManeskin = function(concertId, onComplete) {
  var el = document.getElementById('concert-content');

  var W = 400, H = 400;
  var LANES = 4;
  var LANE_W = W / LANES;
  var laneKeys = ['a', 's', 'd', 'f'];
  var laneColors = ['#ff2d78', '#00f5d4', '#ffd700', '#ff8800'];
  var HIT_Y = H - 50;
  var HIT_RANGE = 35;

  var notes = [];
  var totalNotes = 0;
  var hits = 0;
  var misses = 0;
  var maxMisses = 5;
  var spawnInterval = null;
  var animFrame = null;
  var speed = 2.5;
  var gameOver = false;
  var spawnRate = 800; // ms between notes
  var lastTime = 0;
  var gameTime = 0;

  var canvas, ctx;

  var html = '<div class="cg-container fade-in">';
  html += '<div class="cg-title">🎸 Guitar Hero Lite — Måneskin</div>';
  html += '<div class="cg-instruction">Press <strong>A S D F</strong> when notes reach the line! Miss 5 = game over.</div>';
  html += '<div class="cg-score" id="cg-score">Hits: 0 | Misses: 0/5</div>';
  html += '<div class="cg-canvas-wrap"><canvas id="cg-canvas" width="' + W + '" height="' + H + '"></canvas></div>';
  html += '<div style="display:flex;justify-content:center;gap:4px;margin-top:6px">';
  laneKeys.forEach(function(k, i) {
    html += '<div style="width:' + LANE_W + 'px;text-align:center;color:' + laneColors[i] + ';font-weight:700;font-size:1.2rem">' + k.toUpperCase() + '</div>';
  });
  html += '</div>';
  /* Mobile tap buttons — one big button per lane, visible on touch devices */
  html += '<div id="lane-btns" style="display:flex;gap:6px;justify-content:center;margin-top:10px">';
  laneKeys.forEach(function(k, i) {
    html += '<button class="lane-tap-btn" data-lane="' + i + '" style="' +
      'background:' + laneColors[i] + '22;border:2px solid ' + laneColors[i] + ';' +
      'color:' + laneColors[i] + ';font-weight:700;font-size:1.4rem;' +
      'width:68px;height:56px;border-radius:10px;cursor:pointer;' +
      'touch-action:manipulation;-webkit-tap-highlight-color:transparent;">' +
      k.toUpperCase() + '</button>';
  });
  html += '</div>';
  html += '</div>';

  el.innerHTML = html;
  canvas = document.getElementById('cg-canvas');
  ctx = canvas.getContext('2d');

  // Spawn notes
  spawnInterval = setInterval(function() {
    if (gameOver) return;
    var lane = Math.floor(Math.random() * LANES);
    notes.push({ lane: lane, y: -20, hit: false, missed: false });
    totalNotes++;

    // Increase difficulty over time
    gameTime += spawnRate;
    if (gameTime > 5000 && spawnRate > 400) {
      spawnRate = 600;
      speed = 3.0;
    }
    if (gameTime > 12000 && spawnRate > 300) {
      spawnRate = 400;
      speed = 3.5;
    }
    if (gameTime > 20000) {
      speed = 4.0;
    }
  }, spawnRate);

  // Auto-end after ~25 seconds
  setTimeout(function() {
    if (!gameOver) endGame();
  }, 25000);

  // Animation loop
  lastTime = performance.now();
  animate();

  function animate() {
    if (gameOver) return;
    var now = performance.now();
    var dt = (now - lastTime) / 16.67; // normalize to ~60fps
    lastTime = now;

    ctx.clearRect(0, 0, W, H);

    // Draw lane dividers
    for (var i = 1; i < LANES; i++) {
      ctx.beginPath();
      ctx.moveTo(i * LANE_W, 0);
      ctx.lineTo(i * LANE_W, H);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.stroke();
    }

    // Hit line
    ctx.beginPath();
    ctx.moveTo(0, HIT_Y);
    ctx.lineTo(W, HIT_Y);
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;

    // Target zones
    for (var j = 0; j < LANES; j++) {
      ctx.fillStyle = laneColors[j].replace(')', ',0.08)').replace('#', 'rgba(');
      // Simple approach: just draw subtle rect
      ctx.fillStyle = 'rgba(255,255,255,0.03)';
      ctx.fillRect(j * LANE_W, HIT_Y - HIT_RANGE, LANE_W, HIT_RANGE * 2);
    }

    // Update and draw notes
    notes.forEach(function(n) {
      if (n.hit || n.missed) return;
      n.y += speed * dt;

      // Note shape
      var x = n.lane * LANE_W + LANE_W / 2;
      ctx.beginPath();
      ctx.arc(x, n.y, 14, 0, Math.PI * 2);
      ctx.fillStyle = laneColors[n.lane];
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, n.y, 14, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.stroke();

      // Missed (past hit zone)
      if (n.y > HIT_Y + HIT_RANGE + 20) {
        n.missed = true;
        misses++;
        updateScore();
        if (misses >= maxMisses) endGame();
      }
    });

    // Clean up old notes
    notes = notes.filter(function(n) { return !n.missed || n.y < H + 30; });

    animFrame = requestAnimationFrame(animate);
  }

  function tryHit(lane) {
    if (gameOver) return;
    // Find closest note in this lane within hit range
    var best = null, bestDist = Infinity;
    notes.forEach(function(n) {
      if (n.hit || n.missed || n.lane !== lane) return;
      var dist = Math.abs(n.y - HIT_Y);
      if (dist < HIT_RANGE && dist < bestDist) {
        bestDist = dist;
        best = n;
      }
    });

    if (best) {
      best.hit = true;
      hits++;
      G.sfx.note(330 + lane * 80);
      updateScore();
    }
  }

  function updateScore() {
    var se = document.getElementById('cg-score');
    if (se) se.textContent = 'Hits: ' + hits + ' | Misses: ' + misses + '/' + maxMisses;
  }

  // Keyboard
  function onKey(e) {
    var idx = laneKeys.indexOf(e.key.toLowerCase());
    if (idx >= 0) {
      e.preventDefault();
      tryHit(idx);
    }
  }
  document.addEventListener('keydown', onKey);

  // Canvas touch — immediate response (touchstart, not click) for accurate timing
  canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    var touch = e.changedTouches[0];
    var rect = canvas.getBoundingClientRect();
    var x = (touch.clientX - rect.left) / rect.width * W;
    var lane = Math.min(LANES - 1, Math.max(0, Math.floor(x / LANE_W)));
    tryHit(lane);
  }, { passive: false });

  // Canvas click fallback for desktop
  canvas.addEventListener('click', function(e) {
    var rect = canvas.getBoundingClientRect();
    var x = (e.clientX - rect.left) / rect.width * W;
    var lane = Math.min(LANES - 1, Math.floor(x / LANE_W));
    tryHit(lane);
  });

  // Mobile lane tap buttons
  document.querySelectorAll('.lane-tap-btn').forEach(function(btn) {
    btn.addEventListener('touchstart', function(e) {
      e.preventDefault();
      tryHit(parseInt(this.getAttribute('data-lane')));
    }, { passive: false });
    btn.addEventListener('click', function() {
      tryHit(parseInt(this.getAttribute('data-lane')));
    });
  });

  function endGame() {
    if (gameOver) return;
    gameOver = true;
    clearInterval(spawnInterval);
    if (animFrame) cancelAnimationFrame(animFrame);
    document.removeEventListener('keydown', onKey);

    var score = totalNotes > 0 ? Math.round((hits / totalNotes) * 100) : 0;
    onComplete(score);
  }
};
