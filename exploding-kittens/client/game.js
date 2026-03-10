// ═══════════════════════════════════════════════════════
//  EXPLODING KITTENS — client-side game engine
// ═══════════════════════════════════════════════════════

// ── Card catalog ─────────────────────────────────────────────────────────────
const CARD_DEF = {
  BOMB:         { emoji: '💣', color: '#1a1a2e', border: '#ef233c', label: 'Bomb'         },
  DEFUSE:       { emoji: '🔧', color: '#1b4332', border: '#80b918', label: 'Defuse'       },
  SKIP:         { emoji: '⏭️', color: '#7b2811', border: '#f4a261', label: 'Skip'         },
  ATTACK:       { emoji: '⚔️', color: '#4a0010', border: '#ef233c', label: 'Attack'       },
  NOPE:         { emoji: '🚫', color: '#2d1b4e', border: '#9b5de5', label: 'Nope'         },
  SEE_FUTURE:   { emoji: '🔮', color: '#0d1b6e', border: '#4361ee', label: 'See Future'   },
  SHUFFLE:      { emoji: '🔀', color: '#0d3320', border: '#06d6a0', label: 'Shuffle'      },
  STEAL:        { emoji: '🎯', color: '#4a1040', border: '#f72585', label: 'Steal'        },
  FAVOR:        { emoji: '🙏', color: '#3a0a50', border: '#7b2d8b', label: 'Favor'        },
  REVERSE:      { emoji: '🔄', color: '#0d2a3a', border: '#457b9d', label: 'Reverse'      },
  TACOCAT:      { emoji: '🌮', color: '#3a2800', border: '#ffb703', label: 'Taco Cat'     },
  BEARD_CAT:    { emoji: '🧔', color: '#2d1b6e', border: '#8338ec', label: 'Beard Cat'    },
  RAINBOW_CAT:  { emoji: '🌈', color: '#0d3330', border: '#06d6a0', label: 'Rainbow Cat'  },
  CATTERMELON:  { emoji: '🍉', color: '#3a0010', border: '#ef233c', label: 'Cattermelon'  },
  HAIRY_POTATO: { emoji: '🥔', color: '#2a3a00', border: '#a7c957', label: 'Hairy Potato' },
};

const CAT_TYPES = new Set([
  'TACOCAT','BEARD_CAT','RAINBOW_CAT','CATTERMELON','HAIRY_POTATO'
]);

const TARGET_ACTIONS = new Set(['STEAL', 'FAVOR', 'CAT_PAIR']);

const CARD_DESCRIPTIONS = {
  BOMB:         'EXPLODE — unless you have a Defuse!',
  DEFUSE:       'Auto-used to counter a Bomb. Reinsert it secretly.',
  SKIP:         'End your turn without drawing a card.',
  ATTACK:       'Skip your draw — next player must take 2 turns.',
  NOPE:         'Cancel ANY action. Can be NOPEd back!',
  SEE_FUTURE:   'Peek at the top 3 cards of the deck.',
  SHUFFLE:      'Shuffle the deck randomly.',
  STEAL:        'Steal a random card from a target player.',
  FAVOR:        'Force a player to give you a card of their choice.',
  REVERSE:      'Reverse the turn order.',
  TACOCAT:      'Cat card. Play as a pair to steal a card!',
  BEARD_CAT:    'Cat card. Play as a pair to steal a card!',
  RAINBOW_CAT:  'Cat card. Play as a pair to steal a card!',
  CATTERMELON:  'Cat card. Play as a pair to steal a card!',
  HAIRY_POTATO: 'Cat card. Play as a pair to steal a card!',
};

// ── Session state ─────────────────────────────────────────────────────────────
let S = {
  roomId:   null,
  playerId: null,
  state:    null,     // last server state
  selected: [],       // selected card ids
  nopeTimer: null,
  pollTimer: null,
  prevLog:   [],
  prevLobbyLog: [],
  seeFutureSeen: false,
  serverLogTimer: null,
};

// ── Configurable backend host ─────────────────────────────────────────────────
// Priority: `window.EK_API_BASE` -> <meta name="api-base" content="..."> -> auto-detect
const BACKEND_BASE = (function() {
  if (window.EK_API_BASE) return window.EK_API_BASE;
  const m = document.querySelector('meta[name="api-base"]');
  if (m && m.content && m.content.trim()) return m.content.trim();
  // default behavior: when dev server serves static on port 5000 use relative paths
  return (location.port === '5000') ? '' : 'http://127.0.0.1:5000';
})();

// ── Particle system ───────────────────────────────────────────────────────────
const cvs = document.getElementById('particles');
const ctx = cvs.getContext('2d');
let pts   = [];

function resizeCvs() { cvs.width = innerWidth; cvs.height = innerHeight; }
resizeCvs();
window.addEventListener('resize', resizeCvs);

function spawnPts(x, y, color, n = 20, spread = 6) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const s = 1.5 + Math.random() * spread;
    pts.push({ x, y, vx: Math.cos(a)*s, vy: Math.sin(a)*s - 2,
               a: 1, size: 3 + Math.random()*5, color });
  }
}

(function animPts() {
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  pts = pts.filter(p => p.a > 0.02);
  pts.forEach(p => {
    p.x += p.vx; p.y += p.vy; p.vy += 0.12; p.a -= 0.022;
    ctx.globalAlpha = p.a;
    ctx.fillStyle   = p.color;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI*2); ctx.fill();
  });
  ctx.globalAlpha = 1;
  requestAnimationFrame(animPts);
})();

// ── Audio (Web Audio API) ─────────────────────────────────────────────────────
const AC = new (window.AudioContext || window.webkitAudioContext)();

// Global error handlers to surface issues in console
window.addEventListener('error', (e) => {
  console.error('Uncaught error:', e.error || e.message, e.filename + ':' + e.lineno);
});
window.addEventListener('unhandledrejection', (e) => {
  console.error('Unhandled promise rejection:', e.reason);
});
function tone(freq, dur = 0.12, type = 'sine', vol = 0.15, delay = 0) {
  const o = AC.createOscillator();
  const g = AC.createGain();
  o.connect(g); g.connect(AC.destination);
  o.type = type; o.frequency.value = freq;
  const t = AC.currentTime + delay;
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  o.start(t); o.stop(t + dur);
}

const SFX = {
  deal:    () => tone(600, .08, 'triangle', .10),
  play:    () => tone(880, .10, 'triangle', .12),
  draw:    () => { tone(440,.1,'sine',.1); tone(550,.1,'sine',.1,.1); },
  bomb:    () => {
    [200,160,120].forEach((f,i) => tone(f,.25,'sawtooth',.15,i*.1));
    setTimeout(() => tone(80,.5,'sawtooth',.2), 400);
  },
  defuse:  () => [523,659,784,1047].forEach((f,i) => tone(f,.12,'sine',.14,i*.07)),
  nope:    () => { tone(300,.15,'sawtooth',.15); tone(250,.2,'sawtooth',.12,.1); },
  win:     () => [523,659,784,1047,1318].forEach((f,i)=>tone(f,.15,'sine',.15,i*.09)),
  action:  () => tone(700,.08,'sine',.12),
  error:   () => tone(220,.25,'sawtooth',.1),
};

function resumeAC() { if (AC.state === 'suspended') AC.resume(); }

// ── Floaters ──────────────────────────────────────────────────────────────────
function floater(text, color = '#ffd700') {
  const el = document.createElement('div');
  el.className = 'floater';
  el.textContent = text;
  el.style.cssText = `
    position:fixed; left:${40+Math.random()*30}vw; top:${40+Math.random()*20}vh;
    color:${color}; font-size:1.6rem; font-weight:900; pointer-events:none;
    animation:floatUp .9s ease forwards; z-index:999;
    text-shadow:0 0 12px ${color};
  `;
  document.getElementById('floaters').appendChild(el);
  el.addEventListener('animationend', () => el.remove());
}

// ── API wrapper ───────────────────────────────────────────────────────────────
async function api(path, method = 'GET', body = null) {
  resumeAC();
  try {
    // Use configurable backend base for API calls
    const url = path.startsWith('/api') ? BACKEND_BASE + path : path;
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    const text = await res.text();
    try {
      const json = JSON.parse(text);
      console.log('API:', method, url, { body }, '→', json);
      return json;
    } catch (err) {
      console.error('API parse error: non-JSON response from', url, text.slice(0,200));
      return { error: 'Invalid JSON response' };
    }
  } catch(e) {
    console.error('API error:', method, path, e);
    return { error: 'Network error' };
  }
}

function showError(elId, msg) {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 4000);
}

// ── Screen routing ────────────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── Save / load session ───────────────────────────────────────────────────────
function saveSession() {
  localStorage.setItem('ek_session', JSON.stringify({
    roomId: S.roomId, playerId: S.playerId
  }));
}

function loadSession() {
  try {
    const d = JSON.parse(localStorage.getItem('ek_session') || '{}');
    S.roomId   = d.roomId   || null;
    S.playerId = d.playerId || null;
  } catch(_) {}
}

// ── Copy room code ────────────────────────────────────────────────────────────
function copyRoomCode() {
  navigator.clipboard?.writeText(S.roomId || '');
  floater('📋 Copied!', '#06d6a0');
}

// ── Create / join room ────────────────────────────────────────────────────────
async function createRoom() {
  const name = document.getElementById('player-name').value.trim() || 'Player';
  const data = await api('/api/create_room', 'POST', { name });
  if (data.error) { showError('home-error', data.error); return; }
  S.roomId   = data.room_id;
  S.playerId = data.player_id;
  saveSession();
  enterLobby();
}

async function joinRoom() {
  const name = document.getElementById('player-name').value.trim() || 'Player';
  const code = document.getElementById('room-code-input').value.trim().toUpperCase();
  if (!code) { showError('home-error', 'Enter a room code.'); return; }
  const data = await api('/api/join_room', 'POST', { name, room_id: code });
  if (data.error) { showError('home-error', data.error); return; }
  S.roomId   = data.room_id;
  S.playerId = data.player_id;
  saveSession();
  enterLobby();
}

function enterLobby() {
  document.getElementById('lobby-room-code').textContent = S.roomId;
  showScreen('screen-lobby');
  startPolling();
}

// ── Start game (host only) ────────────────────────────────────────────────────
async function startGame() {
  const data = await api('/api/start_game', 'POST',
    { room_id: S.roomId, player_id: S.playerId }
  );
  if (data.error) { showError('lobby-error', data.error); return; }
  S.selected = [];
  applyState(data);
}

// ── Polling ───────────────────────────────────────────────────────────────────
function startPolling() {
  stopPolling();
  S.pollTimer = setInterval(poll, 1500);
}

function stopPolling() {
  if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
}

async function poll() {
  if (!S.roomId || !S.playerId) return;
  console.log('poll: room=', S.roomId, 'pid=', S.playerId);
  const data = await api(`/api/state/${S.roomId}?pid=${S.playerId}`);
  if (data.error) return;
  applyState(data);
}

// ── Master state applier ──────────────────────────────────────────────────────
function applyState(data) {
  S.state = data;
  console.log('applyState:', data && data.state, data);

  if (data.state === 'lobby') {
    renderLobby(data);
    if (S.state?.state !== 'lobby') showScreen('screen-lobby');
    return;
  }

  if (data.state === 'game_over') {
    stopPolling();
    renderGameOver(data);
    return;
  }

  showScreen('screen-game');
  renderHUD(data);
  renderOpponents(data);
  renderHand(data);
  renderActionBar(data);
  renderLog(data);
  renderTableOverlays(data);
}

// ─────────────────────────────────────────────────────────────────────────────
//  RENDER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

// ── Lobby ─────────────────────────────────────────────────────────────────────
function renderLobby(data) {
  document.getElementById('lobby-room-code').textContent = S.roomId;

  const container = document.getElementById('lobby-players');
  container.innerHTML = '';
  (data.order || []).forEach(pid => {
    const p   = data.players[pid];
    const div = document.createElement('div');
    div.className = 'lobby-player-card' + (pid === data.host ? ' host' : '');
    div.textContent = p.name + (pid === S.playerId ? ' (you)' : '');
    container.appendChild(div);
  });

  const btnStart  = document.getElementById('btn-start');
  const waitingMsg = document.getElementById('waiting-msg');
  const isHost    = S.playerId === data.host;
  const enough    = Object.keys(data.players).length >= 2;

  if (isHost) {
    btnStart.classList.toggle('hidden', !enough);
    waitingMsg.classList.add('hidden');
    if (!enough) {
      waitingMsg.textContent = 'Need at least 2 players to start.';
      waitingMsg.classList.remove('hidden');
    }
  } else {
    btnStart.classList.add('hidden');
    waitingMsg.textContent = 'Waiting for host to start…';
    waitingMsg.classList.remove('hidden');
  }

  // Render recent lobby log (use separate prev buffer so we don't interfere with game log)
  const lobbyEntries = data.log || [];
  const lobbyContainer = document.getElementById('lobby-log');
  if (lobbyContainer) {
    lobbyContainer.innerHTML = (lobbyEntries || []).slice(0, 8).map(e =>
      `<div class="log-entry">${e.msg}</div>`
    ).join('');

    const firstNew = lobbyEntries[0];
    if (firstNew && S.prevLobbyLog[0]?.msg !== firstNew.msg) {
      const msg = firstNew.msg.toLowerCase();
      if (msg.includes('joined') || msg.includes('room')) {
        floater(firstNew.msg, '#06d6a0');
      }
    }
    S.prevLobbyLog = lobbyEntries;
  }
}

// ── Server logs polling ─────────────────────────────────────────────────────
async function fetchServerLogs() {
  try {
    const res = await api('/api/server_logs?n=60');
    if (!res || !res.logs) return;
    const el = document.getElementById('server-log');
    if (!el) return;
    el.innerHTML = (res.logs || []).map(l => `<div class="log-entry">${l}</div>`).join('');
  } catch (e) { console.error('fetchServerLogs error', e); }
}

function startServerLogPolling() {
  if (S.serverLogTimer) return;
  fetchServerLogs();
  S.serverLogTimer = setInterval(fetchServerLogs, 3000);
}

function stopServerLogPolling() {
  if (!S.serverLogTimer) return;
  clearInterval(S.serverLogTimer); S.serverLogTimer = null;
}

// ── HUD ───────────────────────────────────────────────────────────────────────
function renderHUD(data) {
  document.getElementById('hud-room').textContent = `💣 ${data.room_id}`;
  console.log('renderHUD:', data.room_id, data.state);

  const stateLabel = {
    playing:        '▶ Playing',
    nope_window:    '🚫 NOPE window',
    defuse_pending: '🔧 Defusing…',
    favor_pending:  '🙏 Favor…',
  };
  document.getElementById('hud-state-badge').textContent =
    stateLabel[data.state] || data.state;

  const curPid = data.current;
  const curName = curPid
    ? (data.players[curPid]?.name + (curPid === S.playerId ? ' (you)' : ''))
    : '—';
  document.getElementById('hud-turn').textContent =
    data.state === 'playing' ? `${curName}'s turn` :
    data.state === 'nope_window' ? '⏳ Action pending…' : '⏳ Waiting…';

  // Deck counter + discard top
  document.getElementById('deck-count-lbl').textContent = data.deck_count;
  const discardEl = document.getElementById('discard-top-card');
  if (data.discard_top) {
    const ct  = data.discard_top.split('_')[0];
    const def = CARD_DEF[ct] || {};
    discardEl.textContent = def.emoji || '?';
    discardEl.style.background  = def.color  || '#222';
    discardEl.style.borderColor = def.border || '#555';
  } else {
    discardEl.textContent = '';
    discardEl.style.background  = 'transparent';
    discardEl.style.borderColor = 'var(--border)';
  }
}

// ── Opponents ─────────────────────────────────────────────────────────────────
function renderOpponents(data) {
  const bar = document.getElementById('opponents-bar');
  bar.innerHTML = '';

  (data.order || []).forEach(pid => {
    if (pid === S.playerId) return;
    const p   = data.players[pid];
    const div = document.createElement('div');

    const isCurrentTurn = data.current === pid && data.state === 'playing';
    const isDead        = !p.alive;

    div.className = [
      'opponent-card',
      isCurrentTurn ? 'current-turn' : '',
      isDead        ? 'dead'         : '',
    ].filter(Boolean).join(' ');

    const backs = Array(p.count).fill('<div class="opp-card-back"></div>').join('');
    const turns = p.turns > 1 ? `<span class="opp-turns">×${p.turns} turns</span>` : '';
    const dead  = isDead ? '<span class="opp-dead">💀 OUT</span>' : '';

    div.innerHTML = `
      <div class="opp-name">${p.name} ${turns}${dead}</div>
      <div class="opp-hand">${backs}</div>
      <div class="opp-count">${p.count} card${p.count !== 1 ? 's' : ''}</div>
    `;
    bar.appendChild(div);
  });
}

// ── Hand ──────────────────────────────────────────────────────────────────────
function renderHand(data) {
  const hand   = data.players[S.playerId]?.hand || [];
  const isMyTurn = data.current === S.playerId && data.state === 'playing';
  const inNope   = data.state === 'nope_window';

  const container = document.getElementById('player-hand');

  // Keep selection only on valid cards
  S.selected = S.selected.filter(id => hand.includes(id));

  console.log('renderHand: handCount=', hand.length, 'selected=', S.selected.slice());

  container.innerHTML = '';
  hand.forEach(cardId => {
    const ct  = cardId.split('_')[0];
    const def = CARD_DEF[ct] || { emoji: '?', color: '#222', border: '#555', label: ct };
    const sel = S.selected.includes(cardId);

    const el = document.createElement('div');
    el.className = 'hand-card' + (sel ? ' selected' : '');
    el.dataset.id = cardId;
    el.style.cssText = `
      background:${def.color};
      border-color:${sel ? '#fff' : def.border};
      box-shadow: ${sel ? `0 0 16px ${def.border}, 0 -8px 0 #fff` : `0 6px 20px rgba(0,0,0,.5)`};
      transform: ${sel ? 'translateY(-16px) scale(1.08)' : ''};
    `;
    el.innerHTML = `
      <div class="hc-emoji">${def.emoji}</div>
      <div class="hc-label">${def.label}</div>
      <div class="hc-desc">${CARD_DESCRIPTIONS[ct] || ''}</div>
    `;
    el.addEventListener('click', () => toggleSelect(cardId, ct, hand));
    container.appendChild(el);
    SFX.deal();
  });

  document.getElementById('hand-title').textContent =
    `Your Hand (${hand.length} card${hand.length !== 1 ? 's' : ''})`;
}

function toggleSelect(cardId, ct, hand) {
  resumeAC();

  // NOPE is reactive — can't select manually
  if (ct === 'NOPE') {
    floater('🚫 Use NOPE reactively!', '#9b5de5');
    SFX.error(); return;
  }
  if (ct === 'DEFUSE') {
    floater('🔧 Defuse activates automatically!', '#80b918');
    SFX.error(); return;
  }

  if (S.selected.includes(cardId)) {
    S.selected = S.selected.filter(id => id !== cardId);
  } else {
    // Cat cards: only allow selecting same type
    if (CAT_TYPES.has(ct)) {
      const prevCats = S.selected.filter(id => CAT_TYPES.has(id.split('_')[0]));
      const prevType = prevCats.length ? prevCats[0].split('_')[0] : null;
      if (prevType && prevType !== ct) {
        floater('Pair must be same cat type!', '#f4a261');
        SFX.error(); return;
      }
      if (S.selected.length >= 3) {
        floater('Max 3 cat cards!', '#f4a261');
        SFX.error(); return;
      }
    } else {
      // Non-cat: clear any cats already selected, allow only 1
      S.selected = [];
    }
    S.selected.push(cardId);
  }

  renderHand(S.state);
  renderActionBar(S.state);
}

// ── Action bar ────────────────────────────────────────────────────────────────
function renderActionBar(data) {
  const isMyTurn   = data.current === S.playerId && data.state === 'playing';
  const inNope     = data.state === 'nope_window';
  const inDefuse   = data.state === 'defuse_pending' &&
                     data.defuse_me;
  const inFavor    = data.state === 'favor_pending' &&
                     data.favor_me;

  const btnDraw  = document.getElementById('btn-draw');
  const btnPlay  = document.getElementById('btn-play');
  const tgtSel   = document.getElementById('target-selector');
  const selInfo  = document.getElementById('selected-info');

  document.getElementById('play-count').textContent = S.selected.length;

  // Draw button
  btnDraw.disabled = !isMyTurn || S.selected.length > 0;
  btnDraw.style.opacity = btnDraw.disabled ? '.4' : '1';

  // Play button
  const canPlay = isMyTurn && S.selected.length > 0;
  btnPlay.disabled = !canPlay;
  btnPlay.style.opacity = canPlay ? '1' : '.4';

  // Target selector
  const needsTarget = S.selected.length > 0 && (() => {
    const ct = S.selected[0].split('_')[0];
    return TARGET_ACTIONS.has(ct) ||
      (CAT_TYPES.has(ct) && S.selected.length >= 2);
  })();

  if (needsTarget && isMyTurn) {
    tgtSel.classList.remove('hidden');
    const sel  = document.getElementById('target-select');
    const prev = sel.value;
    sel.innerHTML = '<option value="">Pick target…</option>' +
      (data.order || [])
        .filter(pid => pid !== S.playerId && data.players[pid]?.alive)
        .map(pid => `<option value="${pid}">${data.players[pid].name}</option>`)
        .join('');
    if (prev) sel.value = prev;
  } else {
    tgtSel.classList.add('hidden');
  }

  // Selected info
  if (S.selected.length > 0) {
    const ct  = S.selected[0].split('_')[0];
    const def = CARD_DEF[ct] || {};
    selInfo.textContent = S.selected.length === 1
      ? `Selected: ${def.label}`
      : `Selected: ${S.selected.length}× ${def.label}`;
  } else {
    selInfo.textContent = '';
  }
}

// ── Table overlays ────────────────────────────────────────────────────────────
function renderTableOverlays(data) {
  renderNope(data);
  renderSeeFuture(data);
  renderDefuse(data);
  renderFavor(data);
}

// NOPE window
function renderNope(data) {
  const panel = document.getElementById('nope-window');
  const canNope = data.can_nope;
  const inNope  = data.state === 'nope_window';

  if (!inNope) {
    panel.classList.add('hidden');
    clearNopeTimer();
    return;
  }

  panel.classList.remove('hidden');

  const p    = data.pending || {};
  const by   = p.by ? data.players[p.by]?.name : '?';
  const act  = p.action || '?';
  const tgt  = p.target ? ` → ${data.players[p.target]?.name}` : '';
  const nCt  = data.nope_count || 0;
  const netOk = nCt % 2 === 0;

  document.getElementById('nope-action-desc').innerHTML = `
    <strong>${by}</strong> plays
    <span class="action-badge">${CARD_DEF[act]?.emoji || ''} ${act}${tgt}</span>
    <br/><small style="color:${netOk ? '#06d6a0' : '#ef233c'}">
      ${nCt} NOPE${nCt !== 1 ? 's' : ''} — Action will ${netOk ? 'PROCEED ✅' : 'be CANCELLED 🚫'}
    </small>
  `;

  const btnNope = document.getElementById('btn-nope');
  btnNope.disabled = !canNope;

  document.getElementById('nope-status').textContent =
    canNope ? 'You have a NOPE card!' : 'No NOPE card in hand';

  // Countdown
  clearNopeTimer();
  const dl = data.nope_dl || 0;
  const tick = () => {
    const left = Math.max(0, Math.ceil(dl - Date.now() / 1000));
    document.getElementById('nope-countdown').textContent = left;
    if (left <= 0) clearNopeTimer();
    else S.nopeTimer = setTimeout(tick, 200);
  };
  tick();
}

function clearNopeTimer() {
  if (S.nopeTimer) { clearTimeout(S.nopeTimer); S.nopeTimer = null; }
}

// See Future
function renderSeeFuture(data) {
  const panel = document.getElementById('see-future-panel');
  if (!data.see_future || S.seeFutureSeen) {
    panel.classList.add('hidden');
    return;
  }
  panel.classList.remove('hidden');
  S.seeFutureSeen = false;

  const container = document.getElementById('see-future-cards');
  container.innerHTML = '';
  (data.see_future || []).forEach((cardId, i) => {
    const ct  = cardId.split('_')[0];
    const def = CARD_DEF[ct] || {};
    const el  = document.createElement('div');
    el.className = 'sf-card';
    el.style.cssText = `background:${def.color};border-color:${def.border}`;
    el.innerHTML = `
      <div class="sf-pos">#${i + 1} from top</div>
      <div class="sf-emoji">${def.emoji}</div>
      <div class="sf-label">${def.label}</div>
    `;
    container.appendChild(el);
  });
}

function dismissSeeFuture() {
  S.seeFutureSeen = true;
  document.getElementById('see-future-panel').classList.add('hidden');
}

// Defuse
function renderDefuse(data) {
  const panel = document.getElementById('defuse-panel');
  if (!data.defuse_me) { panel.classList.add('hidden'); return; }

  panel.classList.remove('hidden');
  const size   = data.defuse_me.deck_size || 0;
  const slider = document.getElementById('defuse-slider');
  const lbl    = document.getElementById('defuse-position-lbl');

  document.getElementById('defuse-deck-size').textContent = size;
  slider.max   = size;
  slider.value = Math.min(slider.value, size);

  const update = () => {
    const pos = parseInt(slider.value);
    lbl.textContent = pos === 0 ? 'Position: 0 (very top — next to draw!)'
      : pos >= size              ? `Position: ${pos} (very bottom — safest!)`
      : `Position: ${pos} of ${size}`;
  };
  slider.oninput = update;
  update();

  SFX.defuse();
}

async function insertBomb() {
  const pos  = parseInt(document.getElementById('defuse-slider').value);
  const data = await api('/api/insert_bomb', 'POST', {
    room_id: S.roomId, player_id: S.playerId, position: pos
  });
  if (data.error) { floater('❌ ' + data.error, '#ef233c'); SFX.error(); return; }
  S.selected = [];
  floater('🔧 Bomb reinserted!', '#80b918');
  applyState(data);
}

// Favor
function renderFavor(data) {
  const panel = document.getElementById('favor-panel');
  if (!data.favor_me) { panel.classList.add('hidden'); return; }

  panel.classList.remove('hidden');
  const by    = data.favor_me.by;
  const byName = data.players[by]?.name || '?';
  document.getElementById('favor-hint').textContent =
    `${byName} asked you for a favor — choose a card to give:`;

  const container = document.getElementById('favor-hand');
  container.innerHTML = '';

  const myHand = data.players[S.playerId]?.hand || [];
  myHand.forEach(cardId => {
    const ct  = cardId.split('_')[0];
    const def = CARD_DEF[ct] || {};
    const el  = document.createElement('div');
    el.className = 'favor-card';
    el.style.cssText = `background:${def.color};border-color:${def.border}`;
    el.innerHTML = `
      <div class="hc-emoji">${def.emoji}</div>
      <div class="hc-label">${def.label}</div>
    `;
    el.addEventListener('click', () => giveFavor(cardId));
    container.appendChild(el);
  });
}

async function giveFavor(cardId) {
  const data = await api('/api/give_favor', 'POST', {
    room_id: S.roomId, player_id: S.playerId, card_id: cardId
  });
  if (data.error) { floater('❌ ' + data.error, '#ef233c'); SFX.error(); return; }
  const ct  = cardId.split('_')[0];
  floater(`🙏 Gave ${CARD_DEF[ct]?.label || ct}`, '#7b2d8b');
  applyState(data);
}

// ── Log ───────────────────────────────────────────────────────────────────────
function renderLog(data) {
  const entries  = data.log || [];
  const container = document.getElementById('log-entries');
  const firstNew = entries[0];

  // Detect new log entry → visual feedback
  if (firstNew && S.prevLog[0]?.msg !== firstNew.msg) {
    const msg = firstNew.msg;
    if (msg.includes('💥')) { SFX.bomb(); spawnPts(innerWidth/2, innerHeight/2, '#ef233c', 40, 10); floater('💥 BOOM!', '#ef233c'); }
    else if (msg.includes('💀')) { floater('💀 Eliminated!', '#ef233c'); }
    else if (msg.includes('🔧')) { floater('🔧 Defused!', '#80b918'); }
    else if (msg.includes('🔮')) { SFX.action(); }
    else if (msg.includes('🚫')) { SFX.nope(); floater('🚫 NOPE!', '#9b5de5'); }
    else if (msg.includes('⏭️')) { SFX.action(); floater('⏭️ Skipped!', '#f4a261'); }
    else if (msg.includes('⚔️')) { SFX.action(); floater('⚔️ Attack!', '#ef233c'); }
    else if (msg.includes('🎯')) { SFX.action(); floater('🎯 Stolen!', '#f72585'); }
    else if (msg.includes('🔄')) { SFX.action(); floater('🔄 Reversed!', '#457b9d'); }
    else if (msg.includes('🏆')) { SFX.win(); spawnPts(innerWidth/2, innerHeight/2, '#ffd700', 60, 8); }
  }
  S.prevLog = entries;

  container.innerHTML = entries.slice(0, 15).map(e => `
    <div class="log-entry">${e.msg}</div>
  `).join('');
}

// ── Game Over ─────────────────────────────────────────────────────────────────
function renderGameOver(data) {
  const isWinner = data.winner === S.playerId;
  document.getElementById('gameover-icon').textContent  = isWinner ? '🏆' : '💀';
  document.getElementById('gameover-title').textContent =
    isWinner ? 'You Win!' :
    `${data.players[data.winner]?.name || '?'} Wins!`;

  const logEl = document.getElementById('gameover-log');
  logEl.innerHTML = (data.log || []).slice(0, 8).map(e =>
    `<div class="log-entry">${e.msg}</div>`
  ).join('');

  if (isWinner) {
    SFX.win();
    spawnPts(innerWidth/2, innerHeight/2, '#ffd700', 80, 10);
    setTimeout(() => spawnPts(innerWidth/3, innerHeight/3, '#f72585', 40, 8), 400);
    setTimeout(() => spawnPts(2*innerWidth/3, innerHeight/3, '#06d6a0', 40, 8), 700);
  } else {
    SFX.bomb();
  }
  showScreen('screen-gameover');
}

function goHome() {
  stopPolling();
  S.roomId = S.playerId = S.state = null;
  S.selected = []; S.prevLog = [];
  localStorage.removeItem('ek_session');
  showScreen('screen-home');
}

// ── Play selected cards ───────────────────────────────────────────────────────
async function playSelected() {
  if (!S.selected.length) return;
  if (S.state?.current !== S.playerId) {
    floater('Not your turn!', '#ef233c'); SFX.error(); return;
  }

  const ct       = S.selected[0].split('_')[0];
  const needsTgt = TARGET_ACTIONS.has(ct) ||
    (CAT_TYPES.has(ct) && S.selected.length >= 2);
  const target   = needsTgt
    ? document.getElementById('target-select').value : null;

  if (needsTgt && !target) {
    floater('Choose a target!', '#f4a261'); SFX.error(); return;
  }

  SFX.play();
  const data = await api('/api/play_card', 'POST', {
    room_id:   S.roomId,
    player_id: S.playerId,
    cards:     S.selected,
    target:    target || null,
  });

  if (data.error) { floater('❌ ' + data.error, '#ef233c'); SFX.error(); return; }
  S.selected = [];
  S.seeFutureSeen = false;
  applyState(data);
}

// ── Draw card ─────────────────────────────────────────────────────────────────
async function drawCard() {
  if (S.state?.current !== S.playerId) {
    floater('Not your turn!', '#ef233c'); SFX.error(); return;
  }
  SFX.draw();
  const data = await api('/api/draw_card', 'POST', {
    room_id:   S.roomId,
    player_id: S.playerId,
  });
  if (data.error) { floater('❌ ' + data.error, '#ef233c'); SFX.error(); return; }
  S.selected = [];
  applyState(data);
}

// ── NOPE ──────────────────────────────────────────────────────────────────────
async function playNope() {
  SFX.nope();
  const data = await api('/api/nope', 'POST', {
    room_id:   S.roomId,
    player_id: S.playerId,
  });
  if (data.error) { floater('❌ ' + data.error, '#ef233c'); SFX.error(); return; }
  floater('🚫 NOPE!', '#9b5de5');
  applyState(data);
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
window.addEventListener('keydown', e => {
  const inGame = document.getElementById('screen-game').classList.contains('active');
  if (!inGame) return;
  if (e.key === 'd' || e.key === 'D') drawCard();
  if (e.key === 'Enter')              playSelected();
  if (e.key === 'n' || e.key === 'N') {
    if (S.state?.state === 'nope_window' && S.state?.can_nope) playNope();
  }
  if (e.key === 'Escape') S.selected = [], renderHand(S.state), renderActionBar(S.state);
});

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  loadSession();

  // Try to resume a session
  if (S.roomId && S.playerId) {
    const data = await api(`/api/state/${S.roomId}?pid=${S.playerId}`);
    if (!data.error) {
      startPolling();
      startServerLogPolling();
      applyState(data);
      return;
    }
  }
  // start background server log polling even on home screen
  startServerLogPolling();
  showScreen('screen-home');
});
