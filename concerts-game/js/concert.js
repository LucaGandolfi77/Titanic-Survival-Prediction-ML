/* concert.js — Discovery dispatcher & concert game dispatcher */
window.G = window.G || {};

/** Start a discovery minigame for a locked concert */
G.startDiscovery = function(concertId) {
  var mgType = G.state.mgAssignment[concertId];
  G.showScreen('screen-discovery');
  G.sfx.click();

  var onComplete = function(success) {
    if (success) {
      G.state.discovered.add(concertId);
      G.sfx.reveal();
    }
    G.advanceDays(Math.floor(Math.random() * 3) + 1); // 1-3 days
    G.showScreen('screen-map');
    G.checkGameEnd();
  };

  switch (mgType) {
    case 'gossip':  G.mgGossip(concertId, onComplete); break;
    case 'cipher':  G.mgCipher(concertId, onComplete); break;
    case 'auction': G.mgAuction(concertId, onComplete); break;
    case 'puzzle':  G.mgPuzzle(concertId, onComplete); break;
    default:        G.mgGossip(concertId, onComplete); break;
  }
};

/** Start the artist-specific concert minigame */
G.startConcertGame = function(concertId) {
  var c = G.CONCERTS[concertId];
  G.showScreen('screen-concert');
  G.sfx.click();

  var onComplete = function(score) {
    G.processConcertResult(concertId, score);
  };

  switch (c.artistKey) {
    case 'swift':    G.cgSwift(concertId, onComplete); break;
    case 'styles':   G.cgStyles(concertId, onComplete); break;
    case 'maneskin': G.cgManeskin(concertId, onComplete); break;
    case 'eilish':   G.cgEilish(concertId, onComplete); break;
    case 'beyonce':  G.cgBeyonce(concertId, onComplete); break;
    default:         G.cgBeatDrop(concertId, onComplete); break;
  }
};

/** "Beat Drop" fallback minigame — click when ball hits floor */
G.cgBeatDrop = function(concertId, onComplete) {
  var el = document.getElementById('concert-content');
  var W = 350, H = 250;
  var totalRounds = 10;
  var currentRound = 0;
  var hits = 0;
  var misses = 0;
  var maxMisses = 3;
  var ballY = 0, ballVY = 0, gravity = 0.3;
  var floorY = H - 30;
  var bouncing = false;
  var canClick = false;
  var gameOver = false;
  var animFrame = null;

  var html = '<div class="cg-container fade-in">';
  html += '<div class="cg-title">🎵 Beat Drop</div>';
  html += '<div class="cg-instruction">Click/tap exactly when the ball hits the floor!</div>';
  html += '<div class="cg-score" id="cg-score">Round: 0/' + totalRounds + ' | Hits: 0</div>';
  html += '<div class="cg-canvas-wrap"><canvas id="cg-canvas" width="' + W + '" height="' + H + '"></canvas></div>';
  html += '</div>';
  el.innerHTML = html;

  var canvas = document.getElementById('cg-canvas');
  var ctx = canvas.getContext('2d');

  startBounce();

  function startBounce() {
    if (currentRound >= totalRounds || misses >= maxMisses) { endGame(); return; }
    currentRound++;
    ballY = 30;
    ballVY = 0;
    canClick = true;
    bouncing = true;
    updateScore();
    animate();
  }

  function animate() {
    if (!bouncing || gameOver) return;
    ctx.clearRect(0, 0, W, H);

    // Floor
    ctx.fillStyle = '#333';
    ctx.fillRect(0, floorY, W, 4);

    // Ball
    ballVY += gravity;
    ballY += ballVY;

    // Hit floor
    if (ballY >= floorY - 12) {
      ballY = floorY - 12;
      // Bounce window — player should click now
      if (canClick) {
        // Brief window, then auto-miss after 300ms
        setTimeout(function() {
          if (canClick && bouncing) {
            canClick = false;
            misses++;
            G.sfx.fail();
            bouncing = false;
            setTimeout(startBounce, 600);
          }
        }, 350);
      }
    }

    ctx.beginPath();
    ctx.arc(W / 2, ballY, 12, 0, Math.PI * 2);
    ctx.fillStyle = '#ff2d78';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Target indicator
    if (ballY >= floorY - 40) {
      ctx.strokeStyle = 'rgba(255,215,0,0.4)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(W / 2, floorY - 12, 18, 0, Math.PI * 2);
      ctx.stroke();
    }

    animFrame = requestAnimationFrame(animate);
  }

  function handleClick() {
    if (!canClick || !bouncing || gameOver) return;
    canClick = false;
    bouncing = false;
    if (animFrame) cancelAnimationFrame(animFrame);

    var dist = Math.abs(ballY - (floorY - 12));
    if (dist < 20) {
      hits++;
      G.sfx.beat();
    } else {
      misses++;
      G.sfx.fail();
    }
    updateScore();
    setTimeout(startBounce, 500);
  }

  function updateScore() {
    var se = document.getElementById('cg-score');
    if (se) se.textContent = 'Round: ' + currentRound + '/' + totalRounds + ' | Hits: ' + hits + ' | Miss: ' + misses + '/' + maxMisses;
  }

  // Immediate touch response — no 300ms iOS delay
  canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    handleClick();
  }, { passive: false });
  canvas.addEventListener('click', function(e) {
    // Only fires on mouse; touch is handled above
    handleClick();
  });
  function onKey(e) { if (e.code === 'Space') { e.preventDefault(); handleClick(); } }
  document.addEventListener('keydown', onKey);

  function endGame() {
    gameOver = true;
    if (animFrame) cancelAnimationFrame(animFrame);
    document.removeEventListener('keydown', onKey);
    var score = totalRounds > 0 ? Math.round((hits / totalRounds) * 100) : 0;
    onComplete(score);
  }
};

/** Process the score from a concert minigame and show result */
G.processConcertResult = function(concertId, score) {
  var s = G.state;
  var c = G.CONCERTS[concertId];
  var artist = G.ARTISTS[c.artistKey];

  // Determine placement
  var placement, basePoints, badge, reviewKey;
  if (score >= 90) {
    placement = 'On Stage'; badge = '🌟'; basePoints = 600; reviewKey = 'on-stage';
  } else if (score >= 70) {
    placement = 'Parterre'; badge = '🎶'; basePoints = 250; reviewKey = 'parterre';
  } else if (score >= 40) {
    placement = 'Seated'; badge = '💺'; basePoints = 100; reviewKey = 'seated';
  } else {
    placement = 'Did Not Enter'; badge = '🚫'; basePoints = 0; reviewKey = 'no-entry';
  }

  // Mark attended
  s.attended.add(concertId);
  s.concertsAttended++;

  // Calculate points
  var artistBonus = (basePoints > 0) ? artist.bonus : 0;
  var multiplier = G.getFestivalMultiplier(concertId);
  var hotelMultiplier = 1.0;
  if (s.booked[concertId] && s.booked[concertId].hotelTier === 'VIP') {
    hotelMultiplier = 1.1;
  }
  var totalPoints = Math.round((basePoints + artistBonus) * multiplier * hotelMultiplier);

  // Budget bonus for on stage
  var budgetBonus = 0;
  if (score >= 90) budgetBonus = 500;

  // Rare collectible
  var collectible = null;
  if (score >= 90) {
    collectible = artist.emoji + ' ' + artist.name + ' — ' + c.city + ' VIP Pass';
    s.collectibles.push(collectible);
  }

  // Store score data
  s.concertScores[concertId] = {
    score: score,
    placement: placement,
    points: totalPoints,
    badge: badge
  };

  // Apply points and budget
  G.addPoints(totalPoints);
  if (budgetBonus > 0) G.earnBudget(budgetBonus);

  // Advance 1 day
  G.advanceDays(1);

  // Show result
  G.showResult(concertId, score, placement, badge, totalPoints, multiplier, artistBonus, budgetBonus, collectible, reviewKey);
};
