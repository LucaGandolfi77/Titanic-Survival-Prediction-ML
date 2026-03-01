/**
 * Central reactive state — CardForge
 * @module state
 */

import { CARD_DATABASE } from "./database.js";
import { deepClone } from "./utils.js";

/* ── Storage key ───────────────────────────────────────── */
const STORAGE_KEY = "cardforge_deck";

/* ── Default state ─────────────────────────────────────── */
/** @returns {object} */
const defaultState = () => ({
  /* Deck builder */
  collection: [...CARD_DATABASE],
  currentDeck: [],          // { cardId:string, count:number }[]
  deckName: "My Deck",

  /* Game */
  drawPile: [],             // Card[]
  hand: [],                 // Card[]
  table: [],                // { card:Card, position:{x,y}, faceDown:boolean, uid:number }[]
  discardPile: [],          // Card[]
  gamePhase: "idle",        // idle | playing | ended
  cardsPlayed: 0,
  cardsDiscarded: 0,

  /* UI */
  currentSection: "deck",
  activeFilters: {},
  selectedCard: null,
  selectedHandIndex: -1,
  snapToGrid: false,
  /* Image overrides loaded from assets or user choice: { cardId: path } */
  imageOverrides: {},
});

/* ── State singleton ───────────────────────────────────── */
let _state = defaultState();

/** @type {Map<string, Set<Function>>} */
const _subscribers = new Map();

let _uidCounter = 0;
/** Generate a simple unique id for table card instances. */
export const nextUid = () => ++_uidCounter;

/* ── Public API ────────────────────────────────────────── */

/**
 * Read the full state (shallow copy at top level).
 * @returns {object}
 */
export const getState = () => ({ ..._state });

/**
 * Merge a partial update into state and notify subscribers.
 * @param {object} partial
 */
export const setState = (partial) => {
  const changedKeys = Object.keys(partial);
  Object.assign(_state, partial);

  changedKeys.forEach((key) => {
    const subs = _subscribers.get(key);
    if (subs) subs.forEach((fn) => fn(_state[key], _state));
  });

  /* Also notify catch‑all subscribers */
  const all = _subscribers.get("*");
  if (all) all.forEach((fn) => fn(_state));
};

/**
 * Subscribe to changes on a state key (or "*" for any change).
 * @param {string} key
 * @param {Function} callback
 * @returns {Function} unsubscribe
 */
export const subscribe = (key, callback) => {
  if (!_subscribers.has(key)) _subscribers.set(key, new Set());
  _subscribers.get(key).add(callback);
  return () => _subscribers.get(key)?.delete(callback);
};

/* ── Deck helpers ──────────────────────────────────────── */

/**
 * Count total cards currently in the deck.
 * @returns {number}
 */
export const deckTotalCount = () =>
  _state.currentDeck.reduce((s, e) => s + e.count, 0);

/**
 * Count copies of a specific card in the deck.
 * @param {string} cardId
 * @returns {number}
 */
export const deckCountOf = (cardId) =>
  _state.currentDeck.find((e) => e.cardId === cardId)?.count ?? 0;

/**
 * Add one copy of a card to the deck (max 4 copies, max 60 total).
 * @param {string} cardId
 * @returns {boolean} success
 */
export const addToDeck = (cardId) => {
  if (deckTotalCount() >= 60) return false;
  const entry = _state.currentDeck.find((e) => e.cardId === cardId);
  if (entry) {
    if (entry.count >= 4) return false;
    entry.count++;
  } else {
    _state.currentDeck.push({ cardId, count: 1 });
  }
  setState({ currentDeck: [..._state.currentDeck] });
  autoSave();
  return true;
};

/**
 * Remove one copy of a card from the deck.
 * @param {string} cardId
 * @returns {boolean}
 */
export const removeFromDeck = (cardId) => {
  const entry = _state.currentDeck.find((e) => e.cardId === cardId);
  if (!entry) return false;
  entry.count--;
  if (entry.count <= 0) {
    _state.currentDeck = _state.currentDeck.filter((e) => e.cardId !== cardId);
  }
  setState({ currentDeck: [..._state.currentDeck] });
  autoSave();
  return true;
};

/**
 * Clear the entire deck.
 */
export const clearDeck = () => {
  setState({ currentDeck: [] });
  autoSave();
};

/**
 * Set a persistent image override for a specific card id.
 * @param {string} cardId
 * @param {string|null} path  If null, the override is removed.
 */
export const setImageOverride = (cardId, path) => {
  const next = { ..._state.imageOverrides };
  if (!path) {
    delete next[cardId];
  } else {
    next[cardId] = path;
  }
  _state.imageOverrides = next;
  setState({ imageOverrides: next });
  autoSave();
};

/* ── LocalStorage persistence ──────────────────────────── */

/** Save the current deck + name to localStorage. */
export const saveToLocalStorage = () => {
  try {
    const data = {
      deckName: _state.deckName,
      currentDeck: _state.currentDeck,
      imageOverrides: _state.imageOverrides || {},
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch { /* quota exceeded, private mode, etc. */ }
};

const autoSave = saveToLocalStorage;

/**
 * Load saved deck from localStorage.
 * @returns {boolean} true if a deck was restored
 */
export const loadFromLocalStorage = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const data = JSON.parse(raw);
    if (data && Array.isArray(data.currentDeck)) {
      setState({
        currentDeck: data.currentDeck,
        deckName: data.deckName || "My Deck",
        imageOverrides: data.imageOverrides || {},
      });
      return true;
    }
  } catch { /* corrupted data */ }
  return false;
};

/**
 * Reset the full state back to defaults.
 */
export const resetState = () => {
  _state = defaultState();
  _uidCounter = 0;
  /* notify all */
  const all = _subscribers.get("*");
  if (all) all.forEach((fn) => fn(_state));
};
