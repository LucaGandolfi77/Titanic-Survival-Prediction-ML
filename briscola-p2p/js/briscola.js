/**
 * @fileoverview Pure Briscola game logic — no DOM, fully testable.
 *
 * Implements the two-player Italian card game "Briscola" with a 40-card
 * deck (suits: Coppe, Denari, Bastoni, Spade).
 *
 * All state-transforming functions are **pure** — they return a new
 * GameState and never mutate the input.
 *
 * @module briscola
 */

/* ═══════════════════════════════════════════════════════════════
   Constants
   ═══════════════════════════════════════════════════════════════ */

/** @type {readonly string[]} */
export const SUITS = Object.freeze(['coppe', 'denari', 'bastoni', 'spade']);

/** @type {readonly number[]} */
export const RANKS = Object.freeze([1, 2, 3, 4, 5, 6, 7, 11, 12, 13]);

/**
 * Card point values.
 * @type {Readonly<Record<number, number>>}
 */
export const POINTS_MAP = Object.freeze({
  1: 11,   // Asso
  3: 10,   // Tre
  13: 4,   // Re
  12: 3,   // Cavallo
  11: 2,   // Fante
});

/**
 * Rank ordering for trick comparison (higher = stronger).
 * @type {Readonly<Record<number, number>>}
 */
export const RANK_ORDER = Object.freeze({
  1: 10,   // Asso   (strongest)
  3: 9,    // Tre
  13: 8,   // Re
  12: 7,   // Cavallo
  11: 6,   // Fante
  7: 5,
  6: 4,
  5: 3,
  4: 2,
  2: 1,    // Due    (weakest)
});

/** Total points in a full deck. */
export const TOTAL_POINTS = 120;

/* ═══════════════════════════════════════════════════════════════
   Types (documented via JSDoc)
   ═══════════════════════════════════════════════════════════════

   @typedef {"coppe"|"denari"|"bastoni"|"spade"} Suit
   @typedef {1|2|3|4|5|6|7|11|12|13} Rank
   @typedef {{ suit: Suit, rank: Rank, points: number }} Card
   @typedef {{ id: "host"|"guest", hand: Card[], score: number, tricks: Card[] }} Player
   @typedef {{
     deck: Card[],
     briscola: Card,
     players: { host: Player, guest: Player },
     currentTurn: "host"|"guest",
     phase: "waiting"|"playing"|"roundEnd"|"gameOver",
     tableCards: { host: Card|null, guest: Card|null },
     firstPlayer: "host"|"guest",
     roundNumber: number,
     winner: "host"|"guest"|"draw"|null
   }} GameState
*/

/* ═══════════════════════════════════════════════════════════════
   Deck creation
   ═══════════════════════════════════════════════════════════════ */

/**
 * Create a standard 40-card Italian deck.
 * @returns {Card[]}
 */
export function createDeck() {
  /** @type {Card[]} */
  const deck = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({ suit, rank, points: POINTS_MAP[rank] ?? 0 });
    }
  }
  return deck;
}

/**
 * Shuffle a deck using the Fisher-Yates algorithm.
 * Returns a **new** array — the input is not mutated.
 * @param {Card[]} deck
 * @returns {Card[]}
 */
export function shuffleDeck(deck) {
  const a = [...deck];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* ═══════════════════════════════════════════════════════════════
   State initialisation
   ═══════════════════════════════════════════════════════════════ */

/**
 * Create the initial game state.
 *
 * 1. Shuffles a fresh 40-card deck.
 * 2. Deals 3 cards to each player.
 * 3. Turns up the next card as the *briscola* (trump).
 * 4. Places the briscola face-up at the **bottom** of the draw pile,
 *    so it is the last card drawn.
 *
 * The non-dealer ("guest") plays first.
 *
 * @returns {GameState}
 */
export function createInitialState() {
  const full = shuffleDeck(createDeck());

  const hostHand  = [full[0], full[1], full[2]];
  const guestHand = [full[3], full[4], full[5]];
  const briscola  = full[6];

  // Remaining cards: indices 7..39 + briscola at the end (drawn last)
  const deck = [...full.slice(7), briscola];

  return {
    deck,
    briscola,
    players: {
      host:  { id: 'host',  hand: hostHand,  score: 0, tricks: [] },
      guest: { id: 'guest', hand: guestHand, score: 0, tricks: [] },
    },
    currentTurn: 'guest',       // non-dealer plays first
    phase: 'playing',
    tableCards: { host: null, guest: null },
    firstPlayer: 'guest',
    roundNumber: 1,
    winner: null,
  };
}

/**
 * Deal 3 cards to each player from the deck and set the briscola.
 * Primarily useful for rematch — wraps `createInitialState`.
 * @param {GameState} _state  (ignored — a fresh state is returned)
 * @returns {GameState}
 */
export function dealCards(_state) {
  return createInitialState();
}

/* ═══════════════════════════════════════════════════════════════
   Ranking / comparison helpers
   ═══════════════════════════════════════════════════════════════ */

/**
 * Return the numeric strength of a rank (10 = Asso, 1 = Due).
 * @param {Rank} rank
 * @returns {number}
 */
export function getRankOrder(rank) {
  return RANK_ORDER[rank] ?? 0;
}

/**
 * Determine the winner of a two-card trick.
 *
 * Rules:
 * - If both cards share the same suit ⇒ higher rank wins.
 * - If exactly one is briscola ⇒ that one wins.
 * - If different non-briscola suits ⇒ the first card played wins.
 *
 * @param {Card}  firstCard      Card played first.
 * @param {Card}  secondCard     Card played second.
 * @param {Suit}  briscolaSuit   Trump suit.
 * @returns {"first"|"second"}
 */
export function compareCards(firstCard, secondCard, briscolaSuit) {
  const firstIsTrump  = firstCard.suit  === briscolaSuit;
  const secondIsTrump = secondCard.suit === briscolaSuit;

  if (firstIsTrump && secondIsTrump) {
    return getRankOrder(firstCard.rank) > getRankOrder(secondCard.rank)
      ? 'first' : 'second';
  }
  if (firstIsTrump)  return 'first';
  if (secondIsTrump) return 'second';

  if (firstCard.suit === secondCard.suit) {
    return getRankOrder(firstCard.rank) > getRankOrder(secondCard.rank)
      ? 'first' : 'second';
  }

  // Different non-trump suits ⇒ first card wins
  return 'first';
}

/* ═══════════════════════════════════════════════════════════════
   Core game actions
   ═══════════════════════════════════════════════════════════════ */

/**
 * Play a card from a player's hand onto the table.
 *
 * Returns the new state, or **null** if the move is invalid.
 *
 * @param {GameState}      state
 * @param {"host"|"guest"} player
 * @param {Card}           card
 * @returns {GameState|null}
 */
export function playCard(state, player, card) {
  if (state.phase !== 'playing')    return null;
  if (state.currentTurn !== player) return null;

  const hand = state.players[player].hand;
  const idx  = hand.findIndex(c => c.suit === card.suit && c.rank === card.rank);
  if (idx === -1) return null;

  const newState  = structuredClone(state);
  const newHand   = newState.players[player].hand;
  newHand.splice(idx, 1);
  newState.tableCards[player] = structuredClone(card);

  const other = player === 'host' ? 'guest' : 'host';

  if (newState.tableCards[other] !== null) {
    // Both cards on the table → time to resolve
    newState.phase = 'roundEnd';
  } else {
    newState.currentTurn = other;
  }

  return newState;
}

/**
 * Resolve the current trick.
 *
 * 1. Determine the winner.
 * 2. Award both cards (and their points) to the winner.
 * 3. Draw one card each from the deck (winner first).
 * 4. Set the turn to the trick winner.
 * 5. Check for game-over.
 *
 * @param {GameState} state  Must have `phase === "roundEnd"`.
 * @returns {GameState}
 */
export function resolveTrick(state) {
  if (state.phase !== 'roundEnd') return state;
  if (!state.tableCards.host || !state.tableCards.guest) return state;

  const s = structuredClone(state);

  const firstCard   = s.tableCards[s.firstPlayer];
  const secondId    = s.firstPlayer === 'host' ? 'guest' : 'host';
  const secondCard  = s.tableCards[secondId];

  const result      = compareCards(firstCard, secondCard, s.briscola.suit);
  const winnerId    = result === 'first' ? s.firstPlayer : secondId;
  const loserId     = winnerId === 'host' ? 'guest' : 'host';

  // Award trick
  const pts = firstCard.points + secondCard.points;
  s.players[winnerId].tricks.push(firstCard, secondCard);
  s.players[winnerId].score += pts;

  // Draw cards (winner first, loser second)
  if (s.deck.length > 0) s.players[winnerId].hand.push(s.deck.shift());
  if (s.deck.length > 0) s.players[loserId].hand.push(s.deck.shift());

  // Reset table
  s.tableCards = { host: null, guest: null };
  s.currentTurn  = winnerId;
  s.firstPlayer  = winnerId;
  s.roundNumber += 1;
  s.phase = 'playing';

  // Game over?
  if (s.deck.length === 0 &&
      s.players.host.hand.length === 0 &&
      s.players.guest.hand.length === 0) {
    s.phase = 'gameOver';
    if (s.players.host.score > s.players.guest.score) s.winner = 'host';
    else if (s.players.guest.score > s.players.host.score) s.winner = 'guest';
    else s.winner = 'draw';
  }

  return s;
}

/**
 * Check if the game is over and finalise the winner field if so.
 * @param {GameState} state
 * @returns {GameState}
 */
export function checkGameOver(state) {
  if (state.phase === 'gameOver') return state;
  if (state.deck.length === 0 &&
      state.players.host.hand.length === 0 &&
      state.players.guest.hand.length === 0) {
    const s = structuredClone(state);
    s.phase = 'gameOver';
    if (s.players.host.score > s.players.guest.score) s.winner = 'host';
    else if (s.players.guest.score > s.players.host.score) s.winner = 'guest';
    else s.winner = 'draw';
    return s;
  }
  return state;
}

/**
 * Return the list of playable cards for a player.
 * In standard Briscola, all cards in hand are always playable.
 * @param {GameState}      state
 * @param {"host"|"guest"} player
 * @returns {Card[]}
 */
export function getValidMoves(state, player) {
  if (state.phase !== 'playing' || state.currentTurn !== player) return [];
  return [...state.players[player].hand];
}

/* ═══════════════════════════════════════════════════════════════
   State sanitisation (for network transmission)
   ═══════════════════════════════════════════════════════════════ */

/**
 * Create a copy of the game state that hides the opponent's hand
 * and the deck order.  Used when sending state over the DataChannel
 * so neither player can inspect the other's cards.
 *
 * @param {GameState}      state
 * @param {"host"|"guest"} forPlayer  Whose perspective to keep visible.
 * @returns {GameState}
 */
export function sanitizeStateForPeer(state, forPlayer) {
  const s = structuredClone(state);
  const opponent = forPlayer === 'host' ? 'guest' : 'host';

  // Replace opponent hand with array of nulls (preserves count)
  s.players[opponent].hand = new Array(s.players[opponent].hand.length).fill(null);

  // Replace deck with nulls (preserves count)
  s.deck = new Array(s.deck.length).fill(null);

  return s;
}
