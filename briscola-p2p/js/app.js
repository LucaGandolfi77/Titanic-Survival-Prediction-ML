/**
 * @fileoverview Landing page logic — create or join a room.
 *
 * Flow:
 * 1. User creates a room  →  generates a 6-char code, connects to
 *    signaling server, waits for a guest.
 * 2. User joins a room    →  enters a code, connects to signaling
 *    server, joins the room.
 * 3. When both players are in the room the signaling server notifies
 *    them.  Both pages store the session info in `sessionStorage` and
 *    redirect to `game.html`.
 *
 * @module app
 */

import { SignalingClient } from './signaling.js';
import { generateRoomCode, getSignalingUrl, sanitizeText } from './utils.js';

/* ═══════════════════════════════════════════════════════════════
   DOM references
   ═══════════════════════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);

/* ═══════════════════════════════════════════════════════════════
   State
   ═══════════════════════════════════════════════════════════════ */

let signaling = null;
let roomCode  = null;
let role      = null;

/* ═══════════════════════════════════════════════════════════════
   Initialisation (runs on page load)
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  /* Restore player name from localStorage */
  const savedName = localStorage.getItem('briscola_name') ?? '';
  const nameInput = $('#player-name');
  if (nameInput) nameInput.value = savedName;

  /* Auto-fill room code from URL ?room=XXXXXX */
  const params = new URLSearchParams(location.search);
  const urlRoom = params.get('room');
  if (urlRoom && /^[A-Z0-9]{6}$/i.test(urlRoom)) {
    const joinInput = $('#join-code');
    if (joinInput) joinInput.value = urlRoom.toUpperCase();
    showSection('join');
  }

  /* Button handlers */
  $('#btn-create')?.addEventListener('click', handleCreate);
  $('#btn-join')?.addEventListener('click', handleJoin);
  $('#btn-copy-link')?.addEventListener('click', handleCopyLink);
  $('#btn-copy-code')?.addEventListener('click', handleCopyCode);
  $('#btn-back')?.addEventListener('click', handleBack);
  $('#join-code')?.addEventListener('input', (e) => {
    e.target.value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 6);
  });
});

/* ═══════════════════════════════════════════════════════════════
   Handlers
   ═══════════════════════════════════════════════════════════════ */

async function handleCreate() {
  persistName();
  roomCode = generateRoomCode();
  role = 'host';

  showSection('waiting');
  $('#waiting-code').textContent = roomCode;

  /* Generate QR code (if the library is loaded) */
  const qrContainer = $('#qr-code');
  if (qrContainer && typeof QRCode !== 'undefined') {
    qrContainer.innerHTML = '';
    new QRCode(qrContainer, {
      text: getInviteLink(roomCode),
      width: 160,
      height: 160,
      colorDark: '#1a5c2a',
      colorLight: '#FFF8E7',
    });
  }

  try {
    signaling = new SignalingClient(getSignalingUrl());
    await signaling.connect();

    signaling.on('room_created', () => {
      showStatus('Stanza creata — in attesa dell\'avversario…');
    });

    signaling.on('guest_joined', () => {
      showStatus('Avversario connesso! Inizio partita…');
      goToGame();
    });

    signaling.on('error', (d) => {
      showStatus(`Errore: ${d.message}`, true);
    });

    signaling.send({ type: 'create_room', roomCode });

  } catch (err) {
    showStatus('Impossibile connettersi al server di segnalazione.', true);
    console.error(err);
  }
}

async function handleJoin() {
  persistName();
  const code = $('#join-code')?.value?.trim().toUpperCase();
  if (!code || code.length !== 6) {
    showStatus('Inserisci un codice di 6 caratteri.', true);
    return;
  }

  roomCode = code;
  role = 'guest';

  showSection('waiting');
  $('#waiting-code').textContent = roomCode;

  try {
    signaling = new SignalingClient(getSignalingUrl());
    await signaling.connect();

    signaling.on('room_joined', () => {
      showStatus('Connesso! Inizio partita…');
      goToGame();
    });

    signaling.on('error', (d) => {
      showStatus(`Errore: ${d.message}`, true);
      showSection('main');
    });

    signaling.send({ type: 'join_room', roomCode });

  } catch (err) {
    showStatus('Impossibile connettersi al server.', true);
    console.error(err);
  }
}

function handleCopyLink() {
  const link = getInviteLink(roomCode);
  navigator.clipboard.writeText(link).then(() => {
    const btn = $('#btn-copy-link');
    if (btn) { btn.textContent = '✅ Copiato!'; setTimeout(() => btn.textContent = '🔗 Copia Link', 2000); }
  });
}

function handleCopyCode() {
  navigator.clipboard.writeText(roomCode).then(() => {
    const btn = $('#btn-copy-code');
    if (btn) { btn.textContent = '✅ Copiato!'; setTimeout(() => btn.textContent = '📋 Copia Codice', 2000); }
  });
}

function handleBack() {
  signaling?.disconnect();
  showSection('main');
}

/* ═══════════════════════════════════════════════════════════════
   Navigation
   ═══════════════════════════════════════════════════════════════ */

function goToGame() {
  persistName();
  sessionStorage.setItem('briscola_room', roomCode);
  sessionStorage.setItem('briscola_role', role);
  signaling?.disconnect();

  setTimeout(() => {
    location.href = 'game.html';
  }, 800);
}

/* ═══════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════ */

function persistName() {
  const name = $('#player-name')?.value?.trim() || 'Giocatore';
  localStorage.setItem('briscola_name', name);
}

function getInviteLink(code) {
  return `${location.origin}${location.pathname}?room=${code}`;
}

function showSection(section) {
  $$('.lobby-section').forEach((el) => {
    el.classList.toggle('hidden', el.id !== `section-${section}`);
  });
}

function showStatus(msg, isError = false) {
  const el = $('#lobby-status');
  if (!el) return;
  el.textContent = msg;
  el.className = 'lobby-status ' + (isError ? 'error' : 'info');
}

function $$(sel) { return document.querySelectorAll(sel); }
