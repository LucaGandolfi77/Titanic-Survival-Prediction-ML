/* cg-eilish.js — Billie Eilish "Whisper Echo" concert minigame
   Keep volume bar in a narrow "whisper" zone for 5 cumulative seconds.
   Hold SPACE to increase, release to decrease. Overshoot = strike. 3 strikes = fail.
   15 second time limit. Returns score 0-100. */
window.G = window.G || {};

G.cgEilish = function(concertId, onComplete) {
  var el = document.getElementById('concert-content');

  var W = 350, H = 300;
  var canvas, ctx;

  var volume = 20; // 0-100
  var targetMin = 40, targetMax = 55; // whisper zone
  var dangerZone = 80; // above this = strike
  var strikes = 0;
  var maxStrikes = 3;
  var inZoneTime = 0; // cumulative seconds in whisper zone
  var targetZoneTime = 5; // need 5s
  var timeLeft = 15;
  var holding = false;
  var gameOver = false;
  var animFrame = null;
  var timer = null;
  var lastTime = 0;
  var wasInDanger = false;

  var html = '<div class="cg-container fade-in">';
  html += '<div class="cg-title">🖤 Whisper Echo — Billie Eilish</div>';
  html += '<div class="cg-instruction">Hold <strong>SPACE</strong> to raise volume. Keep it in the green zone for 5 seconds!</div>';
  html += '<div class="cg-score" id="cg-score">In Zone: 0.0s / 5.0s | Strikes: 0/3 | Time: 15s</div>';
  html += '<div class="cg-canvas-wrap"><canvas id="cg-canvas" width="' + W + '" height="' + H + '"></canvas></div>';
  html += '</div>';

  el.innerHTML = html;
  canvas = document.getElementById('cg-canvas');
  ctx = canvas.getContext('2d');

  // Timer
  timer = setInterval(function() {
    if (gameOver) return;
    timeLeft--;
    if (timeLeft <= 0) endGame();
  }, 1000);

  // Animation loop
  lastTime = performance.now();
  animate();

  function animate() {
    if (gameOver) return;
    var now = performance.now();
    var dt = (now - lastTime) / 1000;
    lastTime = now;

    // Volume physics
    if (holding) {
      volume += 60 * dt; // rises when holding
    } else {
      volume -= 30 * dt; // falls when released
    }
    volume = Math.max(0, Math.min(100, volume));

    // Check zone
    var inZone = volume >= targetMin && volume <= targetMax;
    var inDanger = volume > dangerZone;

    if (inZone) {
      inZoneTime += dt;
      if (inZoneTime >= targetZoneTime) {
        endGame();
        return;
      }
    }

    if (inDanger && !wasInDanger) {
      strikes++;
      G.sfx.fail();
      wasInDanger = true;
      if (strikes >= maxStrikes) {
        endGame();
        return;
      }
    }
    if (!inDanger) wasInDanger = false;

    // Draw
    ctx.clearRect(0, 0, W, H);

    var barX = W / 2 - 30;
    var barW = 60;
    var barH = H - 40;
    var barY = 20;

    // Background bar
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(barX, barY, barW, barH);

    // Danger zone (above 80%)
    var dangerY = barY + barH * (1 - dangerZone / 100);
    ctx.fillStyle = 'rgba(255,68,68,0.15)';
    ctx.fillRect(barX, barY, barW, dangerY - barY);

    // Whisper zone
    var zoneTop = barY + barH * (1 - targetMax / 100);
    var zoneBot = barY + barH * (1 - targetMin / 100);
    ctx.fillStyle = 'rgba(68,255,136,0.2)';
    ctx.fillRect(barX, zoneTop, barW, zoneBot - zoneTop);
    ctx.strokeStyle = '#44ff88';
    ctx.lineWidth = 2;
    ctx.strokeRect(barX, zoneTop, barW, zoneBot - zoneTop);

    // Volume fill
    var fillH = barH * (volume / 100);
    var fillY = barY + barH - fillH;
    var fillColor = inDanger ? '#ff4444' : (inZone ? '#44ff88' : '#00f5d4');
    ctx.fillStyle = fillColor;
    ctx.fillRect(barX + 4, fillY, barW - 8, fillH);

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '11px Outfit, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('DANGER', barX - 6, dangerY + 4);
    ctx.fillStyle = '#44ff88';
    ctx.fillText('WHISPER', barX - 6, (zoneTop + zoneBot) / 2 + 4);

    // Zone progress bar at bottom
    var progW = (W - 40);
    var progH = 12;
    var progX = 20;
    var progY = H - 16;
    ctx.fillStyle = '#222';
    ctx.fillRect(progX, progY, progW, progH);
    ctx.fillStyle = '#44ff88';
    ctx.fillRect(progX, progY, progW * Math.min(1, inZoneTime / targetZoneTime), progH);

    // Strikes display
    ctx.fillStyle = '#ff4444';
    ctx.textAlign = 'left';
    ctx.font = '14px Outfit, sans-serif';
    for (var i = 0; i < maxStrikes; i++) {
      ctx.fillText(i < strikes ? '✗' : '○', 10 + i * 20, 35);
    }

    // Update score text
    var se = document.getElementById('cg-score');
    if (se) se.textContent = 'In Zone: ' + inZoneTime.toFixed(1) + 's / ' + targetZoneTime + 's | Strikes: ' + strikes + '/' + maxStrikes + ' | Time: ' + timeLeft + 's';

    animFrame = requestAnimationFrame(animate);
  }

  // Input handlers
  function onKeyDown(e) {
    if (e.code === 'Space') { e.preventDefault(); holding = true; }
  }
  function onKeyUp(e) {
    if (e.code === 'Space') { e.preventDefault(); holding = false; }
  }
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', onKeyUp);

  // Touch support — {passive: false} required for e.preventDefault() on iOS Safari
  canvas.addEventListener('touchstart', function(e) { e.preventDefault(); holding = true; }, { passive: false });
  canvas.addEventListener('touchend', function(e) { e.preventDefault(); holding = false; }, { passive: false });
  canvas.addEventListener('mousedown', function() { holding = true; });
  canvas.addEventListener('mouseup', function() { holding = false; });

  function endGame() {
    if (gameOver) return;
    gameOver = true;
    clearInterval(timer);
    if (animFrame) cancelAnimationFrame(animFrame);
    document.removeEventListener('keydown', onKeyDown);
    document.removeEventListener('keyup', onKeyUp);

    var score;
    if (strikes >= maxStrikes) {
      score = Math.round((inZoneTime / targetZoneTime) * 40); // max 40% if failed by strikes
    } else {
      score = Math.round(Math.min(1, inZoneTime / targetZoneTime) * 100);
    }
    onComplete(score);
  }
};
