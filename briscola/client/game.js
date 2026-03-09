// ── Suit display ─────────────────────────────────────────────────────────────
const SUIT_ICON  = { C: '🏆', D: '💰', B: '🪄', S: '⚔️' };
const SUIT_NAME  = { C: 'Cups', D: 'Coins', B: 'Clubs', S: 'Swords' };
const RANK_NAME  = {
  A: 'A', '3': '3', K: 'K', Q: 'Q', J: 'J',
  '7': '7', '6': '6', '5': '5', '4': '4', '2': '2'
};
const RANK_LABEL = {
  A: 'Ace', '3': 'Three', K: 'King', Q: 'Horse', J: 'Jack',
  '7': 'Seven', '6': 'Six', '5': 'Five', '4': 'Four', '2': 'Two'
};
const CARD_PTS = {
  A: 11, '3': 10, K: 4, Q: 3, J: 2,
  '7': 0, '6': 0, '5': 0, '4': 0, '2': 0
};

let state = null;

// ── API helpers ───────────────────────────────────────────────────────────────
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

async function newGame() {
  document.getElementById('overlay').classList.add('hidden');
  state = await api('/api/new_game', 'POST');
  render(state);
}

async function playCard(card) {
  if (!state || state.game_over) return;
  if (state.turn !== 'player') return;
  state = await api('/api/play_card', 'POST', { card });
  render(state);
}

// ── Card building ─────────────────────────────────────────────────────────────
function buildCard(cardId, opts = {}) {
  const { playable = false, isBack = false, briscolaSuit = null } = opts;

  const div = document.createElement('div');

  if (isBack) {
    div.className = 'card back';
    div.textContent = '🂠';
    return div;
  }

  const [rank, suit] = cardId.split('_');
  const pts   = CARD_PTS[rank];
  const icon  = SUIT_ICON[suit];
  const rLabel = RANK_NAME[rank];
  const isBris = suit === briscolaSuit;

  div.className = `card face suit-${suit}`;
  if (playable) div.classList.add('playable');
  else          div.classList.add('disabled');

  // Badge for briscola suit
  if (isBris) {
    const badge = document.createElement('span');
    badge.className = 'briscola-badge';
    badge.textContent = 'Trump';
    div.appendChild(badge);
  }

  div.innerHTML += `
    <div class="card-top">${rLabel}<br/>${icon}</div>
    <div class="card-center suit-${suit}">${icon}</div>
    <div class="card-bottom">${rLabel}<br/>${icon}</div>
  `;

  // Tooltip
  div.title = `${RANK_LABEL[rank]} of ${SUIT_NAME[suit]} — ${pts > 0 ? pts + ' pts' : '0 pts'}`;

  if (playable) {
    div.addEventListener('click', () => playCard(cardId));
  }

  return div;
}

// ── Render ────────────────────────────────────────────────────────────────────
function render(s) {
  if (!s) return;
  state = s;

  const canPlay = !s.game_over && s.turn === 'player';

  // ── Scores
  document.getElementById('score-player').textContent = s.scores.player;
  document.getElementById('score-ai').textContent     = s.scores.ai;
  document.getElementById('cap-player').textContent   = `${s.captured_counts.player} cards`;
  document.getElementById('cap-ai').textContent       = `${s.captured_counts.ai} cards`;

  // ── Deck + briscola
  const deckLabel = document.getElementById('deck-count-label');
  deckLabel.textContent = s.deck_count > 0 ? s.deck_count : '—';

  const brisDiv = document.getElementById('briscola-display');
  brisDiv.innerHTML = '';
  if (s.briscola_card) {
    const bc = buildCard(s.briscola_card, { briscolaSuit: s.briscola_suit });
    bc.style.fontSize = '.6rem';
    brisDiv.appendChild(bc);
  }

  // ── AI hand (card backs)
  const aiHandDiv = document.getElementById('ai-hand');
  aiHandDiv.innerHTML = '';
  for (let i = 0; i < s.ai_card_count; i++) {
    aiHandDiv.appendChild(buildCard(null, { isBack: true }));
  }

  // ── Table: AI card
  const tAI = document.getElementById('table-ai');
  tAI.innerHTML = '';
  if (s.table.ai) {
    tAI.appendChild(buildCard(s.table.ai, { briscolaSuit: s.briscola_suit }));
  }

  // ── Table: Player card
  const tPL = document.getElementById('table-player');
  tPL.innerHTML = '';
  if (s.table.player) {
    tPL.appendChild(buildCard(s.table.player, { briscolaSuit: s.briscola_suit }));
  }

  // ── Last trick banner
  const banner = document.getElementById('last-trick-banner');
  if (s.last_trick) {
    const lt = s.last_trick;
    const who = lt.winner === 'player' ? '🧑 You' : '🤖 AI';
    banner.textContent = lt.value > 0
      ? `Last trick → ${who} (+${lt.value} pts)`
      : `Last trick → ${who} (no points)`;
  } else {
    banner.textContent = '';
  }

  // ── Player hand
  const playerHandDiv = document.getElementById('player-hand');
  playerHandDiv.innerHTML = '';
  (s.player_hand || []).forEach(card => {
    playerHandDiv.appendChild(
      buildCard(card, { playable: canPlay, briscolaSuit: s.briscola_suit })
    );
  });

  // ── Message
  document.getElementById('message-text').textContent = s.message || '';

  // ── Game over
  if (s.game_over) {
    showOverlay(s);
  }
}

function showOverlay(s) {
  const overlay = document.getElementById('overlay');
  const icons   = { player: '🎉', ai: '🤖', draw: '🤝' };
  const titles  = { player: 'You Win!', ai: 'AI Wins!', draw: "It's a Draw!" };

  document.getElementById('overlay-icon').textContent  = icons[s.winner]  || '🃏';
  document.getElementById('overlay-title').textContent = titles[s.winner] || '';
  document.getElementById('overlay-score').textContent =
    `You: ${s.scores.player} pts  —  AI: ${s.scores.ai} pts`;

  overlay.classList.remove('hidden');
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  const s = await api('/api/game_state');
  if (s && !s.error) render(s);
});
