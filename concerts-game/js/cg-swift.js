/* cg-swift.js — Taylor Swift "Sing-Along Timing" concert minigame
   Lyrics appear word-by-word. Press SPACE when shrinking circle hits target.
   Returns score 0-100. */
window.G = window.G || {};

G.cgSwift = function(concertId, onComplete) {
  var el = document.getElementById('concert-content');

  var words = ["We", "are", "never", "ever", "getting", "back", "together",
               "like", "ever", "shake", "it", "off", "I", "knew", "you"];
  var totalRounds = Math.min(words.length, 12);
  var currentRound = 0;
  var hits = [];
  var waiting = false;
  var animFrame = null;
  var startTime = 0;

  var canvas, ctx;
  var W = 400, H = 300;

  var html = '<div class="cg-container fade-in">';
  html += '<div class="cg-title">🎤 Sing-Along Timing — Taylor Swift</div>';
  html += '<div class="cg-instruction">Press <strong>SPACE</strong> when the circle reaches the target ring!</div>';
  html += '<div class="cg-score" id="cg-score">Round: 1/' + totalRounds + ' | Hits: 0</div>';
  html += '<div class="cg-canvas-wrap"><canvas id="cg-canvas" width="' + W + '" height="' + H + '"></canvas></div>';
  html += '<div id="cg-word" style="font-size:1.8rem;font-weight:700;color:var(--pink);margin:8px 0;min-height:44px"></div>';
  html += '</div>';

  el.innerHTML = html;
  canvas = document.getElementById('cg-canvas');
  ctx = canvas.getContext('2d');

  // Start first round
  nextRound();

  function nextRound() {
    if (currentRound >= totalRounds) {
      finish();
      return;
    }
    document.getElementById('cg-word').textContent = words[currentRound];
    document.getElementById('cg-score').textContent =
      'Round: ' + (currentRound + 1) + '/' + totalRounds + ' | Hits: ' + hits.filter(function(h) { return h >= 0.7; }).length;

    waiting = true;
    startTime = performance.now();
    animate();
  }

  function animate() {
    if (!waiting) return;
    var elapsed = (performance.now() - startTime) / 1000;
    var duration = 2.0; // 2 seconds to shrink
    var progress = Math.min(elapsed / duration, 1);

    // Shrinking circle
    var maxR = 120;
    var targetR = 30;
    var currentR = maxR - (maxR - targetR) * progress;

    ctx.clearRect(0, 0, W, H);

    // Target ring
    ctx.beginPath();
    ctx.arc(W / 2, H / 2, targetR, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Inner target glow
    ctx.beginPath();
    ctx.arc(W / 2, H / 2, targetR - 5, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,215,0,0.3)';
    ctx.lineWidth = 8;
    ctx.stroke();

    // Shrinking circle
    ctx.beginPath();
    ctx.arc(W / 2, H / 2, currentR, 0, Math.PI * 2);
    ctx.strokeStyle = '#ff2d78';
    ctx.lineWidth = 4;
    ctx.stroke();

    if (progress >= 1) {
      // Missed — auto-fail this round
      recordHit(0);
      return;
    }

    animFrame = requestAnimationFrame(animate);
  }

  function recordHit(accuracy) {
    waiting = false;
    if (animFrame) cancelAnimationFrame(animFrame);
    hits.push(accuracy);

    // Flash feedback
    var word = document.getElementById('cg-word');
    if (accuracy >= 0.9) {
      word.style.color = '#ffd700';
      word.textContent = '★ PERFECT!';
      G.sfx.success();
    } else if (accuracy >= 0.7) {
      word.style.color = '#44ff88';
      word.textContent = '✓ Good!';
      G.sfx.coin();
    } else if (accuracy >= 0.4) {
      word.style.color = '#00f5d4';
      word.textContent = '~ OK';
      G.sfx.tick();
    } else {
      word.style.color = '#ff4444';
      word.textContent = '✗ Miss';
      G.sfx.fail();
    }

    currentRound++;
    setTimeout(nextRound, 800);
  }

  // Keyboard handler
  function onKey(e) {
    if (e.code === 'Space' && waiting) {
      e.preventDefault();
      var elapsed = (performance.now() - startTime) / 1000;
      var duration = 2.0;
      var progress = Math.min(elapsed / duration, 1);

      // Accuracy: how close to progress=1.0 (when circle matches target)
      var error = Math.abs(1.0 - progress);
      var accuracy = Math.max(0, 1 - error * 4); // 0.25 error = 0 accuracy
      recordHit(accuracy);
    }
  }

  document.addEventListener('keydown', onKey);

  // Mobile tap support — touchstart fires immediately (no 300ms delay)
  canvas.addEventListener('touchstart', function(e) {
    e.preventDefault(); // prevent 300ms click delay on iOS
    if (waiting) {
      var elapsed = (performance.now() - startTime) / 1000;
      var progress = Math.min(elapsed / 2.0, 1);
      var error = Math.abs(1.0 - progress);
      var accuracy = Math.max(0, 1 - error * 4);
      recordHit(accuracy);
    }
  }, { passive: false });

  // Desktop click fallback
  canvas.addEventListener('click', function() {
    if (waiting) {
      var elapsed = (performance.now() - startTime) / 1000;
      var progress = Math.min(elapsed / 2.0, 1);
      var error = Math.abs(1.0 - progress);
      var accuracy = Math.max(0, 1 - error * 4);
      recordHit(accuracy);
    }
  });

  function finish() {
    document.removeEventListener('keydown', onKey);
    var avgAccuracy = hits.reduce(function(s, v) { return s + v; }, 0) / hits.length;
    var score = Math.round(avgAccuracy * 100);
    onComplete(score);
  }
};
