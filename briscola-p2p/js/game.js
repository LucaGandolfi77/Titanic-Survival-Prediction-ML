/**
 * @fileoverview Main game controller for Briscola P2P.
 *
 * Coordinates the WebRTC connection, game logic, and UI:
 * - **Host** is authoritative: computes game state, sends `sync_state`.
 * - **Guest** sends `play_card` requests; host validates and responds.
 *
 * @module game
 */

import { BriscolaWebRTC } from './webrtc.js';
import * as Briscola from './briscola.js';
import * as UI from './ui.js';
import { getSignalingUrl, sanitizeText, SoundEngine, cardToString, cardsEqual } from './utils.js';

/* ═══════════════════════════════════════════════════════════════
   Constants
   ═══════════════════════════════════════════════════════════════ */

const TURN_TIMEOUT  = 60;          // seconds per turn
const TRICK_DELAY   = 1400;        // ms to show both cards before resolving
const FORFEIT_DELAY = 30;          // seconds before forfeit on disconnect

/* ═══════════════════════════════════════════════════════════════
   State
   ═══════════════════════════════════════════════════════════════ */

/** @type {BriscolaWebRTC|null} */
let rtc = null;

/** @type {import('./briscola.js').GameState|null} Full state (host) or sanitised (guest). */
let gameState = null;

/** @type {"host"|"guest"} */
let myRole = 'guest';

/** @type {string} */
let myName = 'Giocatore';

/** @type {string} */
let opponentName = 'Avversario';

/** @type {SoundEngine} */
const sound = new SoundEngine();

/** @type {Array<{winner:string, cards:object[], points:number}>} */
let trickHistory = [];

/** @type {number|null} Turn timer interval id */
let turnTimerId = null;
let turnSecondsLeft = 0;

/** @type {{update:function, remove:function}|null} Overlay handle */
let disconnectOverlay = null;
let forfeitTimerId = null;

/* ═══════════════════════════════════════════════════════════════
   Initialisation
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  const roomCode = sessionStorage.getItem('briscola_room');
  myRole = /** @type {"host"|"guest"} */ (sessionStorage.getItem('briscola_role') ?? 'guest');
  myName = localStorage.getItem('briscola_name') || 'Giocatore';

  if (!roomCode) {
    location.href = 'index.html';
    return;
  }

  $('#room-code-display').textContent = roomCode;
  $('#my-name-display').textContent = myName;

  /* Chat setup */
  $('#chat-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    const input = $('#chat-input');
    const text = input?.value?.trim();
    if (!text) return;
    rtc?.sendMessage({ type: 'chat', text: text.slice(0, 200), sender: myName });
    UI.showChatMessage(myName, text, true);
    sound.notify();
    input.value = '';
  });

  /* Trick history toggle */
  $('#trick-history-toggle')?.addEventListener('click', () => {
    $('#trick-history-panel')?.classList.toggle('collapsed');
  });

  /* Connect */
  rtc = new BriscolaWebRTC(getSignalingUrl(), onPeerMessage, onConnectionChange);
  rtc.rejoinAndConnect(roomCode, myRole);

  UI.showConnectionStatus('connecting');
});

/* ═══════════════════════════════════════════════════════════════
   Connection state
   ═══════════════════════════════════════════════════════════════ */

/**
 * @param {string} state
 */
function onConnectionChange(state) {
  UI.showConnectionStatus(state);

  if (state === 'connected') {
    if (disconnectOverlay) { disconnectOverlay.remove(); disconnectOverlay = null; }
    if (forfeitTimerId) { clearTimeout(forfeitTimerId); forfeitTimerId = null; }

    if (myRole === 'host' && !gameState) {
      startNewGame();
    }

    /* Exchange names */
    rtc?.sendMessage({ type: 'name', name: myName });
  }

  if (state === 'disconnected' || state === 'failed') {
    clearTurnTimer();
    if (state === 'disconnected') {
      disconnectOverlay = UI.showOverlay('Avversario disconnesso — attendo riconnessione…', FORFEIT_DELAY);
      forfeitTimerId = setTimeout(() => {
        disconnectOverlay?.update('Avversario non si è riconnesso.');
        setTimeout(() => {
          disconnectOverlay?.remove();
          UI.showGameOver(myRole, { host: gameState?.players?.host?.score ?? 0, guest: gameState?.players?.guest?.score ?? 0 }, myRole,
            () => location.reload(),
            () => { location.href = 'index.html'; });
        }, 2000);
      }, FORFEIT_DELAY * 1000);
    }
  }
}

/* ═══════════════════════════════════════════════════════════════
   Peer message handler
   ═══════════════════════════════════════════════════════════════ */

/**
 * @param {Object} msg
 */
function onPeerMessage(msg) {
  switch (msg.type) {
    case 'game_start':
      handleGameStart(msg);
      break;

    case 'play_card':
      handlePlayCard(msg);
      break;

    case 'sync_state':
      handleSyncState(msg);
      break;

    case 'trick_result':
      handleTrickResult(msg);
      break;

    case 'chat':
      UI.showChatMessage(msg.sender ?? opponentName, msg.text);
      sound.notify();
      break;

    case 'rematch':
      handleRematch(msg);
      break;

    case 'name':
      opponentName = sanitizeText(msg.name ?? 'Avversario');
      $('#opponent-name-display').textContent = opponentName;
      break;

    case 'ping':
      rtc?.sendMessage({ type: 'pong' });
      break;

    case 'pong':
      break;

    default:
      console.warn('Unknown message type:', msg.type);
  }
}

/* ═══════════════════════════════════════════════════════════════
   Game flow
   ═══════════════════════════════════════════════════════════════ */

function startNewGame() {
  gameState = Briscola.createInitialState();
  trickHistory = [];

  /* Send sanitised state to guest */
  const guestView = Briscola.sanitizeStateForPeer(gameState, 'guest');
  rtc?.sendMessage({ type: 'game_start', state: guestView });

  renderFullState();
  startTurnTimer();
}

/**
 * Guest receives the initial game state.
 * @param {{ state: import('./briscola.js').GameState }} msg
 */
function handleGameStart(msg) {
  gameState = msg.state;
  trickHistory = [];
  renderFullState();
  startTurnTimer();
}

/**
 * A player clicks a card in their hand.
 * @param {import('./briscola.js').Card} card
 */
function onCardClick(card) {
  if (!gameState || gameState.phase !== 'playing') return;
  if (gameState.currentTurn !== myRole) return;

  if (myRole === 'host') {
    /* Host processes locally */
    const newState = Briscola.playCard(gameState, 'host', card);
    if (!newState) return;
    gameState = newState;
    sound.playCard();

    /* Send updated state to guest */
    rtc?.sendMessage({ type: 'sync_state', state: Briscola.sanitizeStateForPeer(gameState, 'guest') });
    renderFullState();

    if (gameState.phase === 'roundEnd') {
      resolveTrickAsHost();
    } else {
      startTurnTimer();
    }
  } else {
    /* Guest sends play_card request to host */
    rtc?.sendMessage({ type: 'play_card', card });
    sound.playCard();
  }
}

/**
 * Host receives a play_card from guest.
 * @param {{ card: import('./briscola.js').Card }} msg
 */
function handlePlayCard(msg) {
  if (myRole !== 'host') return;
  if (!gameState || gameState.currentTurn !== 'guest') return;

  const newState = Briscola.playCard(gameState, 'guest', msg.card);
  if (!newState) {
    rtc?.sendMessage({ type: 'error', message: 'Mossa non valida' });
    return;
  }

  gameState = newState;

  /* Send interim state so guest sees the card on the table */
  rtc?.sendMessage({ type: 'sync_state', state: Briscola.sanitizeStateForPeer(gameState, 'guest') });
  renderFullState();

  if (gameState.phase === 'roundEnd') {
    resolveTrickAsHost();
  } else {
    startTurnTimer();
  }
}

/**
 * Host resolves the trick after a brief delay (for animation).
 */
function resolveTrickAsHost() {
  clearTurnTimer();

  setTimeout(() => {
    if (!gameState || gameState.phase !== 'roundEnd') return;

    /* Determine trick winner before resolving (for history) */
    const firstCard  = gameState.tableCards[gameState.firstPlayer];
    const secondId   = gameState.firstPlayer === 'host' ? 'guest' : 'host';
    const secondCard = gameState.tableCards[secondId];
    const result     = Briscola.compareCards(firstCard, secondCard, gameState.briscola.suit);
    const winnerId   = result === 'first' ? gameState.firstPlayer : secondId;
    const pts        = firstCard.points + secondCard.points;

    trickHistory.push({
      winner: winnerId,
      cards: [firstCard, secondCard],
      points: pts,
    });

    gameState = Briscola.resolveTrick(gameState);
    sound.trickWin();

    /* Send resolved state + trick result to guest */
    rtc?.sendMessage({
      type: 'sync_state',
      state: Briscola.sanitizeStateForPeer(gameState, 'guest'),
    });
    rtc?.sendMessage({
      type: 'trick_result',
      winner: winnerId,
      cards: [firstCard, secondCard],
      points: pts,
    });

    renderFullState();

    if (gameState.phase === 'gameOver') {
      handleGameOver();
    } else {
      startTurnTimer();
    }
  }, TRICK_DELAY);
}

/**
 * Guest receives an updated game state.
 * @param {{ state: import('./briscola.js').GameState }} msg
 */
function handleSyncState(msg) {
  if (myRole !== 'guest') return;
  gameState = msg.state;
  renderFullState();

  if (gameState.phase === 'gameOver') {
    handleGameOver();
  } else if (gameState.phase === 'playing') {
    startTurnTimer();
  } else {
    clearTurnTimer();
  }
}

/**
 * Guest receives trick result (for history tracking).
 * @param {{ winner: string, cards: object[], points: number }} msg
 */
function handleTrickResult(msg) {
  trickHistory.push({
    winner: msg.winner,
    cards: msg.cards,
    points: msg.points,
  });
  sound.trickWin();
  UI.renderTrickHistory(trickHistory, myRole);
}

/* ═══════════════════════════════════════════════════════════════
   Game over & rematch
   ═══════════════════════════════════════════════════════════════ */

function handleGameOver() {
  clearTurnTimer();
  if (!gameState) return;

  const scores = {
    host: gameState.players.host.score,
    guest: gameState.players.guest.score,
  };

  UI.showGameOver(gameState.winner, scores, myRole,
    () => {
      rtc?.sendMessage({ type: 'rematch', accept: true });
    },
    () => {
      rtc?.sendMessage({ type: 'rematch', accept: false });
      location.href = 'index.html';
    },
  );
}

/**
 * @param {{ accept: boolean }} msg
 */
function handleRematch(msg) {
  if (msg.accept) {
    if (myRole === 'host') {
      UI.showRematchRequest(
        () => {
          rtc?.sendMessage({ type: 'rematch', accept: true });
          startNewGame();
        },
        () => {
          rtc?.sendMessage({ type: 'rematch', accept: false });
        },
      );
    } else {
      /* Guest requested rematch and host accepted → host will send game_start */
    }
  }
}

/* ═══════════════════════════════════════════════════════════════
   Turn timer
   ═══════════════════════════════════════════════════════════════ */

function startTurnTimer() {
  clearTurnTimer();
  turnSecondsLeft = TURN_TIMEOUT;
  UI.showCountdown(turnSecondsLeft);

  turnTimerId = setInterval(() => {
    turnSecondsLeft--;
    UI.showCountdown(turnSecondsLeft);
    if (turnSecondsLeft <= 0) {
      clearTurnTimer();
      /* Auto-play a random card on timeout (host enforces) */
      if (myRole === 'host' && gameState?.currentTurn && gameState.phase === 'playing') {
        const moves = Briscola.getValidMoves(gameState, gameState.currentTurn);
        if (moves.length > 0) {
          const autoCard = moves[Math.floor(Math.random() * moves.length)];
          if (gameState.currentTurn === 'host') {
            onCardClick(autoCard);
          } else {
            /* Simulate guest play */
            handlePlayCard({ card: autoCard });
          }
        }
      }
    }
  }, 1000);
}

function clearTurnTimer() {
  if (turnTimerId) { clearInterval(turnTimerId); turnTimerId = null; }
  UI.showCountdown(0);
}

/* ═══════════════════════════════════════════════════════════════
   Rendering
   ═══════════════════════════════════════════════════════════════ */

function renderFullState() {
  if (!gameState) return;

  const me = gameState.players[myRole];
  const op = gameState.players[myRole === 'host' ? 'guest' : 'host'];
  const isMyTurn = gameState.currentTurn === myRole && gameState.phase === 'playing';

  /* Hands */
  const myHand = me.hand.filter(Boolean);
  UI.renderHand(myHand, isMyTurn, onCardClick);

  const opCount = Array.isArray(op.hand) ? op.hand.length : 0;
  UI.renderOpponentHand(opCount);

  /* Table */
  const deckCount = Array.isArray(gameState.deck) ? gameState.deck.length : 0;
  UI.renderTable(gameState.tableCards, gameState.briscola, deckCount, myRole);

  /* Scores */
  UI.renderScores(me.score, op.score);

  /* Turn indicator */
  UI.showTurnIndicator(gameState.currentTurn, myRole);

  /* Briscola label */
  UI.showBriscolaLabel(gameState.briscola);

  /* Trick history */
  UI.renderTrickHistory(trickHistory, myRole);
}

/* ═══════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════ */

function $(sel) { return document.querySelector(sel); }
