/**
 * @fileoverview Utility helpers for Briscola P2P.
 *
 * Pure functions shared across modules — room codes, shuffle, text
 * sanitisation, card formatting, and a tiny synthesised-sound engine.
 */

/* ═══════════════════════════════════════════════════════════════
   Room codes
   ═══════════════════════════════════════════════════════════════ */

/** Characters used for room codes (no ambiguous 0/O, 1/I/L). */
const CODE_CHARS = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';

/**
 * Generate a random 6-character alphanumeric room code.
 * @returns {string}
 */
export function generateRoomCode() {
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)];
  }
  return code;
}

/* ═══════════════════════════════════════════════════════════════
   Array helpers
   ═══════════════════════════════════════════════════════════════ */

/**
 * Fisher-Yates (Knuth) in-place shuffle.  Returns a **new** array.
 * @template T
 * @param {T[]} arr
 * @returns {T[]}
 */
export function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* ═══════════════════════════════════════════════════════════════
   Text helpers
   ═══════════════════════════════════════════════════════════════ */

/**
 * Sanitise user-supplied text to prevent XSS when inserting into the DOM.
 * @param {string} text
 * @returns {string} HTML-escaped text
 */
export function sanitizeText(text) {
  const div = document.createElement('div');
  div.textContent = String(text).slice(0, 200);
  return div.innerHTML;
}

/**
 * Format a timestamp as HH:MM (Italian locale).
 * @returns {string}
 */
export function formatTimestamp() {
  return new Date().toLocaleTimeString('it-IT', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

/* ═══════════════════════════════════════════════════════════════
   Card helpers
   ═══════════════════════════════════════════════════════════════ */

/** @type {Record<number, string>} */
const RANK_NAMES = {
  1: 'Asso', 2: 'Due', 3: 'Tre', 4: 'Quattro', 5: 'Cinque',
  6: 'Sei', 7: 'Sette', 11: 'Fante', 12: 'Cavallo', 13: 'Re',
};

/** @type {Record<string, string>} */
const SUIT_NAMES = {
  coppe: 'Coppe', denari: 'Denari', bastoni: 'Bastoni', spade: 'Spade',
};

/** @type {Record<number, string>} */
export const RANK_LABELS = {
  1: 'A', 2: '2', 3: '3', 4: '4', 5: '5',
  6: '6', 7: '7', 11: 'F', 12: 'C', 13: 'R',
};

/** @type {Record<string, string>} */
export const SUIT_SYMBOLS = {
  coppe:   '♥',
  denari:  '♦',
  bastoni: '♣',
  spade:   '♠',
};

/**
 * Human-readable card name, e.g. "Asso di Coppe".
 * @param {{ suit: string, rank: number }} card
 * @returns {string}
 */
export function cardToString(card) {
  return `${RANK_NAMES[card.rank] ?? card.rank} di ${SUIT_NAMES[card.suit] ?? card.suit}`;
}

/**
 * Unique deterministic id for a card, e.g. "coppe-1".
 * @param {{ suit: string, rank: number }} card
 * @returns {string}
 */
export function cardId(card) {
  return `${card.suit}-${card.rank}`;
}

/**
 * Structural equality for two cards.
 * @param {{ suit: string, rank: number }} a
 * @param {{ suit: string, rank: number }} b
 * @returns {boolean}
 */
export function cardsEqual(a, b) {
  return a.suit === b.suit && a.rank === b.rank;
}

/* ═══════════════════════════════════════════════════════════════
   Signaling URL
   ═══════════════════════════════════════════════════════════════ */

/**
 * Build the WebSocket URL for the signaling server.
 * Defaults to ws://localhost:8080 in development.
 * @returns {string}
 */
export function getSignalingUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
    return `ws://${location.hostname}:8080`;
  }
  return `${proto}://${location.hostname}:8080`;
}

/* ═══════════════════════════════════════════════════════════════
   Sound engine (AudioContext — synthesised, no audio files)
   ═══════════════════════════════════════════════════════════════ */

export class SoundEngine {
  constructor() {
    /** @type {AudioContext|null} */
    this._ctx = null;
    this.enabled = true;
  }

  /** @returns {AudioContext} */
  _getCtx() {
    if (!this._ctx) this._ctx = new AudioContext();
    return this._ctx;
  }

  /** Soft whoosh when a card is played. */
  playCard() {
    if (!this.enabled) return;
    try {
      const ctx = this._getCtx();
      const dur = 0.12;
      const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
      const data = buf.getChannelData(0);
      for (let i = 0; i < data.length; i++) {
        const t = i / data.length;
        data[i] = (Math.random() * 2 - 1) * (1 - t) * 0.15;
      }
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const flt = ctx.createBiquadFilter();
      flt.type = 'highpass';
      flt.frequency.value = 800;
      src.connect(flt).connect(ctx.destination);
      src.start();
    } catch { /* AudioContext unavailable */ }
  }

  /** Short rising tone when a trick is won. */
  trickWin() {
    if (!this.enabled) return;
    try {
      const ctx = this._getCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(660, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1100, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.18, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } catch { /* noop */ }
  }

  /** Notification blip (chat message). */
  notify() {
    if (!this.enabled) return;
    try {
      const ctx = this._getCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.15);
    } catch { /* noop */ }
  }
}
