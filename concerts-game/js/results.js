/* results.js — Post-concert result card + confetti + game over screen */
window.G = window.G || {};

/**
 * Show concert result card with animated confetti.
 * Called from concert.js G.processConcertResult
 */
G.showResult = function(concertId, score, placement, badge, totalPoints, multiplier, artistBonus, budgetBonus, collectible, reviewKey) {
  var c = G.CONCERTS[concertId];
  var artist = G.ARTISTS[c.artistKey];
  var el = document.getElementById('result-content');

  G.showScreen('screen-result');

  // Placement CSS class
  var placeClass = placement.toLowerCase().replace(/ /g, '-');

  // Pick a random review
  var reviews = G.REVIEWS[reviewKey] || G.REVIEWS['seated'];
  var review = reviews[Math.floor(Math.random() * reviews.length)];

  var html = '<div class="result-card fade-in">';
  html += '<h2>' + artist.emoji + ' ' + artist.name + '</h2>';
  html += '<div style="color:var(--gray-light);font-size:0.85rem">' + c.city + ', ' + c.country + ' — ' + c.venue + '</div>';
  html += '<div class="result-badge">' + badge + '</div>';
  html += '<div class="result-placement ' + placeClass + '">' + placement + '</div>';
  html += '<div style="color:var(--gray);font-size:0.9rem;margin-bottom:8px">Score: ' + score + '%</div>';

  if (totalPoints > 0) {
    html += '<div class="result-points">+' + totalPoints + ' points</div>';
    // Breakdown
    var details = [];
    var basePoints = totalPoints;
    if (artistBonus > 0) details.push('Artist bonus +' + artistBonus);
    if (multiplier > 1) details.push('Festival Month ×' + multiplier.toFixed(1));
    if (details.length > 0) {
      html += '<div style="color:var(--gray);font-size:0.7rem">' + details.join(' · ') + '</div>';
    }
  }

  if (budgetBonus > 0) {
    html += '<div style="color:var(--success);margin:8px 0">+€' + budgetBonus + ' budget bonus!</div>';
  }

  if (collectible) {
    html += '<div class="result-collectible">🏆 Rare: ' + collectible + '</div>';
  }

  html += '<div class="result-review">"' + review + '"</div>';

  html += '<button class="btn btn-primary" id="result-done" style="margin-top:16px">Continue 🌍</button>';
  html += '</div>';

  el.innerHTML = html;

  // Confetti for good results
  if (score >= 40) {
    G.showConfetti(score >= 90 ? 40 : (score >= 70 ? 20 : 10));
  }

  document.getElementById('result-done').addEventListener('click', function() {
    G.sfx.click();
    G.showScreen('screen-map');
    G.checkGameEnd();
  });
};

/**
 * Spawn CSS confetti pieces
 */
G.showConfetti = function(count) {
  count = count || 25;
  var colors = ['#ff2d78', '#00f5d4', '#ffd700', '#ff8800', '#44ff88', '#aa44ff', '#fff'];

  for (var i = 0; i < count; i++) {
    var piece = document.createElement('div');
    piece.className = 'confetti-piece';
    piece.style.left = (Math.random() * 100) + '%';
    piece.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
    piece.style.width = (Math.random() * 8 + 5) + 'px';
    piece.style.height = (Math.random() * 8 + 5) + 'px';
    piece.style.animationDelay = (Math.random() * 1.5) + 's';
    piece.style.animationDuration = (Math.random() * 1.5 + 2) + 's';
    document.body.appendChild(piece);

    // Remove after animation
    (function(p) {
      setTimeout(function() {
        if (p.parentNode) p.parentNode.removeChild(p);
      }, 4000);
    })(piece);
  }
};

/**
 * Game Over / Score Summary screen
 */
G.showGameOver = function() {
  var s = G.state;
  var el = document.getElementById('gameover-content');
  G.showScreen('screen-gameover');

  // Determine if win or partial win
  var allAttended = s.concertsAttended === s.totalConcerts;
  var title = allAttended ? '🏆 Tour Complete!' : '🎵 Tour Over!';
  var subtitle = allAttended
    ? 'You attended every single concert — legendary superfan!'
    : 'Your concert chase has ended. Here\'s how you did:';

  var html = '<div class="fade-in">';
  html += '<h1 class="gameover-title">' + title + '</h1>';
  html += '<p style="color:var(--gray-light);margin-bottom:20px">' + subtitle + '</p>';

  // Stats grid
  html += '<div class="gameover-stats">';
  html += '<div class="go-stat"><div class="go-val">' + s.points.toLocaleString() + '</div><div class="go-label">Total Points</div></div>';
  html += '<div class="go-stat"><div class="go-val">' + s.concertsAttended + '/' + s.totalConcerts + '</div><div class="go-label">Concerts Attended</div></div>';
  html += '<div class="go-stat"><div class="go-val">€' + s.budget.toLocaleString() + '</div><div class="go-label">Remaining Budget</div></div>';
  html += '<div class="go-stat"><div class="go-val">' + s.organisedCount + '</div><div class="go-label">Concerts Organised</div></div>';
  html += '</div>';

  // Concert list
  var attendedList = [];
  G.CONCERTS.forEach(function(c) {
    if (s.concertScores[c.id]) {
      var sc = s.concertScores[c.id];
      var artist = G.ARTISTS[c.artistKey];
      attendedList.push({
        label: artist.emoji + ' ' + artist.name + ' — ' + c.city,
        score: sc.score + '% · ' + sc.placement,
        points: sc.points + ' pts',
        badge: sc.badge
      });
    }
  });

  if (attendedList.length > 0) {
    html += '<div class="gameover-list">';
    html += '<h3>Concerts Attended</h3>';
    attendedList.forEach(function(item) {
      html += '<div class="go-concert-item">';
      html += '<span>' + item.badge + ' ' + item.label + '</span>';
      html += '<span style="color:var(--gold)">' + item.points + '</span>';
      html += '</div>';
    });
    html += '</div>';
  }

  // Collectibles
  if (s.collectibles.length > 0) {
    html += '<div class="gameover-list">';
    html += '<h3>🏆 Rare Collectibles</h3>';
    s.collectibles.forEach(function(c) {
      html += '<div class="go-concert-item"><span>' + c + '</span></div>';
    });
    html += '</div>';
  }

  html += '<button class="btn btn-primary" style="margin-top:24px;width:220px" id="gameover-replay">Play Again 🔄</button>';
  html += '</div>';

  el.innerHTML = html;

  G.showConfetti(50);

  document.getElementById('gameover-replay').addEventListener('click', function() {
    G.sfx.click();
    G.startGame();
  });
};
