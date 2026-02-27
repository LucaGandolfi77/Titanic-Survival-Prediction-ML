/**
 * @fileoverview DOM manipulation, card rendering, and animations for Briscola P2P.
 *
 * Every visual update flows through this module — the game controller
 * never touches the DOM directly.
 *
 * @module ui
 */

import { RANK_LABELS, SUIT_SYMBOLS, cardId, cardToString, sanitizeText, formatTimestamp } from './utils.js';

/* ═══════════════════════════════════════════════════════════════
   Cached DOM references
   ═══════════════════════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* ═══════════════════════════════════════════════════════════════
   Card element factory
   ═══════════════════════════════════════════════════════════════ */

/**
 * Create an HTML element representing a card.
 * @param {{ suit: string, rank: number }|null} card  null = face-down
 * @param {boolean} [faceUp=true]
 * @returns {HTMLElement}
 */
export function createCardElement(card, faceUp = true) {
  const el = document.createElement('div');
  el.classList.add('card');

  if (!card || !faceUp) {
    el.classList.add('face-down');
    el.innerHTML = '<div class="card-back"><div class="card-back-pattern"></div></div>';
    return el;
  }

  el.classList.add(`suit-${card.suit}`);
  el.dataset.suit = card.suit;
  el.dataset.rank = String(card.rank);

  const rank = RANK_LABELS[card.rank] ?? String(card.rank);
  const sym  = SUIT_SYMBOLS[card.suit] ?? '?';

  el.innerHTML = `
    <div class="card-face">
      <div class="card-corner top-left">
        <span class="card-rank">${rank}</span>
        <span class="card-suit">${sym}</span>
      </div>
      <div class="card-center">
        <span class="card-suit">${sym}</span>
      </div>
      <div class="card-corner bottom-right">
        <span class="card-rank">${rank}</span>
        <span class="card-suit">${sym}</span>
      </div>
    </div>`;

  el.title = cardToString(card);
  return el;
}

/* ═══════════════════════════════════════════════════════════════
   Render functions
   ═══════════════════════════════════════════════════════════════ */

/**
 * Render the player's own hand.
 * @param {Array}   cards     Cards in hand (full objects).
 * @param {boolean} isMyTurn  Highlight playable cards.
 * @param {function} onClick  Called with the card object when clicked.
 */
export function renderHand(cards, isMyTurn, onClick) {
  const container = $('#my-hand');
  if (!container) return;
  container.innerHTML = '';

  cards.forEach((card) => {
    const el = createCardElement(card);
    if (isMyTurn) {
      el.classList.add('playable');
      el.addEventListener('click', () => onClick(card));
    } else {
      el.classList.add('disabled');
    }
    container.appendChild(el);
  });
}

/**
 * Render the opponent's hand as face-down cards.
 * @param {number} count  Number of cards in opponent's hand.
 */
export function renderOpponentHand(count) {
  const container = $('#opponent-hand');
  if (!container) return;
  container.innerHTML = '';

  for (let i = 0; i < count; i++) {
    container.appendChild(createCardElement(null, false));
  }
}

/**
 * Render the table area: played cards, briscola, deck.
 * @param {{ host: object|null, guest: object|null }} tableCards
 * @param {object}  briscola   The trump card.
 * @param {number}  deckCount  Remaining cards in deck.
 * @param {string}  myRole     "host" or "guest"
 */
export function renderTable(tableCards, briscola, deckCount, myRole) {
  /* ── Table cards ── */
  const opRole = myRole === 'host' ? 'guest' : 'host';

  const opSlot = $('#table-card-opponent');
  const mySlot = $('#table-card-mine');
  if (opSlot) {
    opSlot.innerHTML = '';
    if (tableCards[opRole]) opSlot.appendChild(createCardElement(tableCards[opRole]));
  }
  if (mySlot) {
    mySlot.innerHTML = '';
    if (tableCards[myRole]) mySlot.appendChild(createCardElement(tableCards[myRole]));
  }

  /* ── Briscola ── */
  const briscolaEl = $('#briscola-card');
  if (briscolaEl) {
    briscolaEl.innerHTML = '';
    if (briscola) {
      const cardEl = createCardElement(briscola);
      cardEl.classList.add('briscola-indicator');
      briscolaEl.appendChild(cardEl);
    }
  }

  /* ── Deck ── */
  const deckEl = $('#deck-area');
  if (deckEl) {
    if (deckCount > 0) {
      deckEl.innerHTML = `
        <div class="deck-stack">
          <div class="card face-down"><div class="card-back"><div class="card-back-pattern"></div></div></div>
          <span class="deck-count">${deckCount}</span>
        </div>`;
    } else {
      deckEl.innerHTML = '<div class="deck-empty"></div>';
    }
  }
}

/**
 * Update the score displays.
 * @param {number} myScore
 * @param {number} opponentScore
 */
export function renderScores(myScore, opponentScore) {
  const ms = $('#my-score');
  const os = $('#opponent-score');
  if (ms) ms.textContent = String(myScore);
  if (os) os.textContent = String(opponentScore);
}

/**
 * Show whose turn it is.
 * @param {"host"|"guest"} currentTurn
 * @param {"host"|"guest"} myRole
 */
export function showTurnIndicator(currentTurn, myRole) {
  const el = $('#turn-indicator');
  if (!el) return;

  const isMyTurn = currentTurn === myRole;
  el.textContent = isMyTurn ? 'Il tuo turno' : 'Turno avversario';
  el.className = 'turn-indicator ' + (isMyTurn ? 'my-turn' : 'opponent-turn');

  const myArea = $('#my-area');
  const opArea = $('#opponent-area');
  myArea?.classList.toggle('active-turn', isMyTurn);
  opArea?.classList.toggle('active-turn', !isMyTurn);
}

/**
 * Display a countdown timer.
 * @param {number} seconds  Remaining seconds (0 to hide).
 */
export function showCountdown(seconds) {
  const el = $('#countdown');
  if (!el) return;
  if (seconds <= 0) {
    el.textContent = '';
    el.classList.remove('visible');
  } else {
    el.textContent = `⏱ ${seconds}s`;
    el.classList.add('visible');
    if (seconds <= 10) el.classList.add('urgent');
    else el.classList.remove('urgent');
  }
}

/* ═══════════════════════════════════════════════════════════════
   Animations
   ═══════════════════════════════════════════════════════════════ */

/**
 * Animate a card being played from hand to the table.
 * @param {object} card
 * @param {"mine"|"opponent"} who
 * @returns {Promise<void>}
 */
export function animateCardPlay(card, who) {
  return new Promise((resolve) => {
    const slotId = who === 'mine' ? '#table-card-mine' : '#table-card-opponent';
    const slot = $(slotId);
    if (!slot) { resolve(); return; }

    const el = createCardElement(card);
    el.classList.add('card-animate-play');
    slot.innerHTML = '';
    slot.appendChild(el);

    el.addEventListener('animationend', () => {
      el.classList.remove('card-animate-play');
      resolve();
    }, { once: true });

    // Fallback if animation doesn't fire
    setTimeout(resolve, 400);
  });
}

/**
 * Animate trick collection: cards glow then slide to winner's pile.
 * @param {"mine"|"opponent"} winner
 * @returns {Promise<void>}
 */
export function animateTrickWin(winner) {
  return new Promise((resolve) => {
    const cards = $$('#table-cards .card');
    cards.forEach((c) => c.classList.add('trick-win'));

    setTimeout(() => {
      cards.forEach((c) => {
        c.classList.add(winner === 'mine' ? 'collect-mine' : 'collect-opponent');
      });
    }, 500);

    setTimeout(() => {
      $('#table-card-mine').innerHTML = '';
      $('#table-card-opponent').innerHTML = '';
      resolve();
    }, 1000);
  });
}

/**
 * Animate drawing a card from the deck.
 * @param {"mine"|"opponent"} who
 * @returns {Promise<void>}
 */
export function animateCardDraw(who) {
  return new Promise((resolve) => {
    const deck = $('#deck-area');
    if (!deck) { resolve(); return; }

    const ghost = document.createElement('div');
    ghost.className = 'card face-down card-draw-anim ' +
      (who === 'mine' ? 'draw-to-mine' : 'draw-to-opponent');
    ghost.innerHTML = '<div class="card-back"><div class="card-back-pattern"></div></div>';
    deck.appendChild(ghost);

    ghost.addEventListener('animationend', () => {
      ghost.remove();
      resolve();
    }, { once: true });

    setTimeout(() => { ghost.remove(); resolve(); }, 500);
  });
}

/* ═══════════════════════════════════════════════════════════════
   Modals & overlays
   ═══════════════════════════════════════════════════════════════ */

/**
 * Show the game-over modal.
 * @param {"host"|"guest"|"draw"} winner
 * @param {{ host: number, guest: number }} scores
 * @param {"host"|"guest"} myRole
 * @param {function} onRematch  Called when "Rematch" is clicked.
 * @param {function} onLeave    Called when "Leave" is clicked.
 */
export function showGameOver(winner, scores, myRole, onRematch, onLeave) {
  const opRole = myRole === 'host' ? 'guest' : 'host';
  let title, sub;
  if (winner === 'draw') {
    title = '🤝 Pareggio!';
    sub = 'Incredibile — 60 a 60!';
  } else if (winner === myRole) {
    title = '🏆 Hai vinto!';
    sub = 'Complimenti!';
  } else {
    title = '😞 Hai perso';
    sub = 'Ritenta, sarai più fortunato!';
  }

  const modal = document.createElement('div');
  modal.id = 'game-over-modal';
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content glass-panel">
      <h2>${title}</h2>
      <p>${sub}</p>
      <div class="final-scores">
        <div class="score-col">
          <span class="score-label">Tu</span>
          <span class="score-value">${scores[myRole]}</span>
        </div>
        <div class="score-divider">–</div>
        <div class="score-col">
          <span class="score-label">Avversario</span>
          <span class="score-value">${scores[opRole]}</span>
        </div>
      </div>
      <div class="modal-actions">
        <button id="btn-rematch" class="btn btn-primary">🔄 Rivincita</button>
        <button id="btn-leave" class="btn btn-secondary">🚪 Esci</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  modal.querySelector('#btn-rematch').addEventListener('click', () => {
    modal.remove();
    onRematch();
  });
  modal.querySelector('#btn-leave').addEventListener('click', () => {
    modal.remove();
    onLeave();
  });
}

/**
 * Show a rematch-request modal.
 * @param {function} onAccept
 * @param {function} onDecline
 */
export function showRematchRequest(onAccept, onDecline) {
  const modal = document.createElement('div');
  modal.id = 'rematch-modal';
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content glass-panel">
      <h3>🔄 Rivincita?</h3>
      <p>L'avversario vuole giocare ancora!</p>
      <div class="modal-actions">
        <button id="btn-accept" class="btn btn-primary">✅ Accetta</button>
        <button id="btn-decline" class="btn btn-secondary">❌ Rifiuta</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  modal.querySelector('#btn-accept').addEventListener('click', () => {
    modal.remove();
    onAccept();
  });
  modal.querySelector('#btn-decline').addEventListener('click', () => {
    modal.remove();
    onDecline();
  });
}

/**
 * Show a waiting/reconnect overlay.
 * @param {string} text
 * @param {number} [countdownSec]  Optional countdown.
 * @returns {{ update: function, remove: function }}
 */
export function showOverlay(text, countdownSec = 0) {
  let existing = $('#overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'overlay';
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `
    <div class="modal-content glass-panel">
      <p id="overlay-text">${sanitizeText(text)}</p>
      ${countdownSec > 0 ? `<p id="overlay-countdown">${countdownSec}s</p>` : ''}
    </div>`;
  document.body.appendChild(overlay);

  let remaining = countdownSec;
  let timer = null;

  if (countdownSec > 0) {
    timer = setInterval(() => {
      remaining--;
      const cd = overlay.querySelector('#overlay-countdown');
      if (cd) cd.textContent = `${remaining}s`;
      if (remaining <= 0) { clearInterval(timer); }
    }, 1000);
  }

  return {
    update(newText) {
      const t = overlay.querySelector('#overlay-text');
      if (t) t.textContent = newText;
    },
    remove() {
      if (timer) clearInterval(timer);
      overlay.remove();
    },
  };
}

/* ═══════════════════════════════════════════════════════════════
   Connection status badge
   ═══════════════════════════════════════════════════════════════ */

/**
 * Update the connection-status indicator in the header.
 * @param {string} state  "connected"|"connecting"|"disconnected"|"failed"
 */
export function showConnectionStatus(state) {
  const badge = $('#connection-status');
  if (!badge) return;

  const labels = {
    connected: '🟢 Connesso',
    connecting: '🟡 Connessione…',
    disconnected: '🔴 Disconnesso',
    failed: '🔴 Connessione persa',
    closed: '⚫ Chiuso',
  };

  badge.textContent = labels[state] ?? `⚪ ${state}`;
  badge.className = 'connection-badge ' + state;
}

/* ═══════════════════════════════════════════════════════════════
   Chat
   ═══════════════════════════════════════════════════════════════ */

/**
 * Append a chat message to the chat panel.
 * @param {string}  sender   Sender name.
 * @param {string}  text     Message text (will be sanitised).
 * @param {boolean} [isMine] Style as own message.
 */
export function showChatMessage(sender, text, isMine = false) {
  const list = $('#chat-messages');
  if (!list) return;

  const li = document.createElement('li');
  li.className = 'chat-msg' + (isMine ? ' mine' : '');
  li.innerHTML = `
    <span class="chat-sender">${sanitizeText(sender)}</span>
    <span class="chat-time">${formatTimestamp()}</span>
    <span class="chat-text">${sanitizeText(text)}</span>`;
  list.appendChild(li);
  list.scrollTop = list.scrollHeight;
}

/* ═══════════════════════════════════════════════════════════════
   Trick history (last 5 tricks)
   ═══════════════════════════════════════════════════════════════ */

/**
 * @typedef {{ winner: string, cards: object[], points: number }} TrickRecord
 */

/**
 * Render the collapsible trick-history panel.
 * @param {TrickRecord[]} tricks  Most-recent-first array.
 * @param {"host"|"guest"} myRole
 */
export function renderTrickHistory(tricks, myRole) {
  const container = $('#trick-history-list');
  if (!container) return;
  container.innerHTML = '';

  const last5 = tricks.slice(-5).reverse();
  last5.forEach((t, i) => {
    const who = t.winner === myRole ? 'Tu' : 'Avversario';
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="th-round">#${tricks.length - i}</span>
      <span class="th-cards">${t.cards.map(c => `${RANK_LABELS[c.rank]}${SUIT_SYMBOLS[c.suit]}`).join(' vs ')}</span>
      <span class="th-winner">${sanitizeText(who)}</span>
      <span class="th-points">+${t.points}</span>`;
    container.appendChild(li);
  });
}

/**
 * Show / hide briscola suit label.
 * @param {object} briscola  The trump card.
 */
export function showBriscolaLabel(briscola) {
  const el = $('#briscola-label');
  if (!el || !briscola) return;

  const suitNames = { coppe: 'Coppe', denari: 'Denari', bastoni: 'Bastoni', spade: 'Spade' };
  el.innerHTML = `Briscola: <strong class="suit-${briscola.suit}">${SUIT_SYMBOLS[briscola.suit]} ${suitNames[briscola.suit]}</strong>`;
}
