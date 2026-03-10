// ── Suit config ───────────────────────────────────────────────────────────────
const SUIT_ICON  = { H: '♥', D: '♦', C: '♣', S: '♠' };
const SUIT_RED   = new Set(['H', 'D']);
const RANK_LABEL = { J: 'J', Q: 'Q', K: 'K', A: 'A' };

// Base URL for API calls. In some remote previews (github.dev / preview) POST
// requests are not proxied to the local Flask server — force localhost.
const API_BASE = (function(){
  const host = window.location.hostname || '';
  if (host.includes('github.dev') || host.includes('preview')) return 'http://localhost:5000';
  return '';
})();

let busy = false;
let peakMulti = 1;

// ── Audio (Web Audio API) ─────────────────────────────────────────────────────
const AC = new (window.AudioContext || window.webkitAudioContext)();

function beep(freq, dur = 0.12, type = 'sine', vol = 0.18) {
  const o = AC.createOscillator();
  const g = AC.createGain();
  o.connect(g); g.connect(AC.destination);
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(vol, AC.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, AC.currentTime + dur);
  o.start(); o.stop(AC.currentTime + dur);
}

function playCorrect(multi) {
  const base = 440 + multi * 60;
  [base, base * 1.25, base * 1.5].forEach((f, i) =>
    setTimeout(() => beep(f, .12, 'triangle', .14), i * 80)
  );
}
function playWrong() {
  beep(220, .25, 'sawtooth', .12);
  setTimeout(() => beep(180, .35, 'sawtooth', .1), 120);
}
function playCombo() {
  [523, 659, 784, 1047].forEach((f, i) =>
    setTimeout(() => beep(f, .15, 'sine', .15), i * 70)
  );
}

// ── Particle system ───────────────────────────────────────────────────────────
const cvs  = document.getElementById('particles');
const ctx2 = cvs.getContext('2d');
let particles = [];

function resizeCanvas() {
  cvs.width  = window.innerWidth;
  cvs.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function spawnParticles(x, y, color, n = 18) {
  for (let i = 0; i < n; i++) {
    const angle = Math.random() * Math.PI * 2;
    const speed = 2 + Math.random() * 5;
    particles.push({
      x, y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 2,
      alpha: 1,
      size: 4 + Math.random() * 6,
      color,
    });
  }
}

(function animParticles() {
  ctx2.clearRect(0, 0, cvs.width, cvs.height);
  particles = particles.filter(p => p.alpha > 0.02);
  particles.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    p.vy += 0.15; p.alpha -= 0.025;
    ctx2.globalAlpha = p.alpha;
    ctx2.fillStyle   = p.color;
    ctx2.beginPath();
    ctx2.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx2.fill();
  });
  ctx2.globalAlpha = 1;
  requestAnimationFrame(animParticles);
})();

// ── Floating +pts labels ──────────────────────────────────────────────────────
function spawnFloater(text, x, y, color = '#ffd700') {
  const el = document.createElement('div');
  el.className = 'floater';
  el.textContent = text;
  el.style.left  = `${x}px`;
  el.style.top   = `${y}px`;
  el.style.color = color;
  el.style.textShadow = `0 0 10px ${color}`;
  document.getElementById('floaters').appendChild(el);
  el.addEventListener('animationend', () => el.remove());
}

// ── Card rendering ────────────────────────────────────────────────────────────
function renderCard(el, card, animClass = 'deal-in') {
  const rank = card.rank;
  const suit = card.suit;
  const icon = SUIT_ICON[suit];
  const isRed = SUIT_RED.has(suit);

  el.className = `playing-card ${isRed ? 'red-suit' : 'black-suit'}`;
  el.innerHTML = `
    <div class="card-corner top">
      <span class="rank">${rank}</span>
      <span class="suit">${icon}</span>
    </div>
    <div class="card-center-suit">${icon}</div>
    <div class="card-corner bottom">
      <span class="rank">${rank}</span>
      <span class="suit">${icon}</span>
    </div>
  `;

  // Trigger animation
  void el.offsetWidth;
  el.classList.add(animClass);
}

// ── Show / hide screens ───────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── HUD update ────────────────────────────────────────────────────────────────
function updateHUD(score, streak, multi) {
  document.getElementById('hud-score').textContent  = score;
  document.getElementById('hud-streak').textContent = streak;
  document.getElementById('hud-multi').textContent  = multi;

  // Pip bar
  const pips = document.querySelectorAll('.pip');
  pips.forEach((p, i) => p.classList.toggle('active', i < multi));

  document.getElementById('multi-next-val').textContent = multi;

  // Streak progress bar (0–5 within current level)
  const pct = ((streak % 5) / 5) * 100;
  const bar = document.getElementById('streak-progress');
  bar.style.setProperty('--pct', pct + '%');
  document.getElementById('streak-val').textContent     = streak;
  document.getElementById('streak-to-next').textContent = 5 - (streak % 5);

  // Pulse pill on change
  ['pill-score','pill-streak','pill-multi'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('bump');
    void el.offsetWidth;
    el.classList.add('bump');
    setTimeout(() => el.classList.remove('bump'), 300);
  });
}

// ── Result badge ──────────────────────────────────────────────────────────────
function showBadge(emoji) {
  const b = document.getElementById('result-badge');
  b.className = '';
  b.textContent = emoji;
  void b.offsetWidth;
  b.addEventListener('animationend', () => { b.className = 'hidden'; }, { once: true });
}

// ── Disable / enable buttons ──────────────────────────────────────────────────
function setButtons(enabled) {
  document.getElementById('btn-higher').disabled = !enabled;
  document.getElementById('btn-lower').disabled  = !enabled;
}

// ── New game ──────────────────────────────────────────────────────────────────
async function startGame() {
  if (AC.state === 'suspended') AC.resume();
  peakMulti = 1;
  const data = await api('/api/new_game', 'POST');
  if (data.error) return;

  showScreen('screen-game');
  document.getElementById('save-msg').classList.add('hidden');

  // Show current card
  const curEl = document.getElementById('card-current');
  renderCard(curEl, data.current, 'deal-in');
  document.getElementById('card-next-wrap').classList.add('hidden');
  document.getElementById('card-prev-wrap').classList.add('hidden');
  document.getElementById('result-badge').className = 'hidden';
  document.getElementById('deck-count').textContent = data.deck_count;

  updateHUD(0, 0, 1);
  setButtons(true);
  busy = false;
}

// ── Guess ─────────────────────────────────────────────────────────────────────
async function guess(direction) {
  if (busy) return;
  busy = true;
  setButtons(false);

  const data = await api('/api/guess', 'POST', { direction });
  if (data.error) { busy = false; setButtons(true); return; }

  document.getElementById('deck-count').textContent = data.deck_count;
  peakMulti = Math.max(peakMulti, data.multiplier);

  // Reveal next card
  const prevEl  = document.getElementById('card-prev-wrap');
  const curEl   = document.getElementById('card-current-wrap');
  const nextWrap = document.getElementById('card-next-wrap');
  const nextEl  = document.getElementById('card-next');

  renderCard(nextEl, data.current, 'flip-in');
  nextWrap.classList.remove('hidden');

  await sleep(320);

  if (data.correct) {
    playCorrect(data.multiplier);
    showBadge('✅');

    // Particle burst from card
    const rect = curEl.getBoundingClientRect();
    spawnParticles(
      rect.left + rect.width / 2,
      rect.top  + rect.height / 2,
      '#06d6a0', 22
    );

    // Floating +pts
    spawnFloater(
      `+${data.pts_gained}`,
      rect.left + rect.width / 2 - 20,
      rect.top,
      data.multiplier >= 3 ? '#ff6b35' : '#ffd700'
    );

    // Combo milestone
    if (data.streak > 0 && data.streak % 5 === 0) {
      playCombo();
      spawnParticles(
        rect.left + rect.width / 2,
        rect.top  + rect.height / 2,
        '#ffd700', 40
      );
      spawnFloater(
        `🔥 COMBO ×${data.multiplier}!`,
        rect.left + rect.width / 2 - 60,
        rect.top - 40,
        '#ffd700'
      );
    }

    updateHUD(data.score, data.streak, data.multiplier);

    // Slide: next → current, hide prev
    await sleep(200);
    const curCard = document.getElementById('card-current');
    renderCard(curCard, data.current, 'deal-in');
    nextWrap.classList.add('hidden');

  } else {
    // Wrong
    playWrong();
    showBadge('❌');
    nextEl.classList.add('shake');
    document.getElementById('card-current').classList.add('shake');

    const rect = curEl.getBoundingClientRect();
    spawnParticles(
      rect.left + rect.width / 2,
      rect.top  + rect.height / 2,
      '#ef233c', 22
    );

    await sleep(600);
    showGameOver(data.score, data.streak);
    busy = false;
    return;
  }

  if (data.over || data.deck_empty) {
    await sleep(200);
    showGameOver(data.score, data.streak);
    busy = false;
    return;
  }

  await sleep(100);
  setButtons(true);
  busy = false;
}

// ── Game over ─────────────────────────────────────────────────────────────────
function showGameOver(score, streak) {
  document.getElementById('over-score').textContent  = score;
  document.getElementById('over-streak').textContent = streak;
  document.getElementById('over-multi').textContent  = `×${peakMulti}`;
  document.getElementById('over-result-icon').textContent =
    score >= 200 ? '🏆' : score >= 100 ? '⭐' : score >= 50 ? '😎' : '💀';

  const title = score >= 200 ? 'Legendary!' :
                score >= 100 ? 'Excellent!'  :
                score >= 50  ? 'Not Bad!'    : 'Game Over';
  document.getElementById('over-title').textContent = title;
  document.getElementById('save-msg').classList.add('hidden');

  showScreen('screen-over');
}

// ── Save score ────────────────────────────────────────────────────────────────
async function saveScore() {
  const name = document.getElementById('name-input').value.trim() || 'Anonymous';
  const res  = await api('/api/save_score', 'POST', { name });
  if (res.ok) {
    document.getElementById('save-form').classList.add('hidden');
    document.getElementById('save-msg').classList.remove('hidden');
  }
}

// ── Leaderboard ───────────────────────────────────────────────────────────────
async function showLeaderboard() {
  const data = await api('/api/leaderboard');
  const body = document.getElementById('lb-body');
  const medals = ['🥇','🥈','🥉'];
  body.innerHTML = data.map((r, i) => `
    <tr>
      <td>${medals[i] || i + 1}</td>
      <td>${r.name}</td>
      <td><strong>${r.score}</strong></td>
      <td>🔥 ${r.streak}</td>
    </tr>
  `).join('');
  document.getElementById('lb-modal').classList.remove('hidden');
}

function closeLeaderboard() {
  document.getElementById('lb-modal').classList.add('hidden');
}

// ── Utils ─────────────────────────────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const url = API_BASE + path;
  const res = await fetch(url, opts);

  // Read raw text first to avoid json() throwing on empty/non-JSON responses
  const text = await res.text();
  if (!text) {
    return { error: `Empty response (status ${res.status})` };
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    return { error: 'Invalid JSON response', status: res.status, body: text };
  }
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
window.addEventListener('keydown', e => {
  if (document.getElementById('screen-game').classList.contains('active')) {
    if (e.key === 'ArrowUp'   || e.key === 'h') guess('higher');
    if (e.key === 'ArrowDown' || e.key === 'l') guess('lower');
  }
});
