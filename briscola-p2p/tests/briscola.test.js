/**
 * @fileoverview Jest tests for the Briscola pure game-logic module.
 *
 * Run with:  npm test
 */

import { describe, test, expect, beforeEach } from '@jest/globals';
import {
  SUITS,
  RANKS,
  POINTS_MAP,
  RANK_ORDER,
  TOTAL_POINTS,
  createDeck,
  shuffleDeck,
  createInitialState,
  dealCards,
  getRankOrder,
  compareCards,
  playCard,
  resolveTrick,
  checkGameOver,
  getValidMoves,
  sanitizeStateForPeer,
} from '../js/briscola.js';

/* ═══════════════════════════════════════════════════════════════
   createDeck
   ═══════════════════════════════════════════════════════════════ */

describe('createDeck', () => {
  const deck = createDeck();

  test('produces exactly 40 cards', () => {
    expect(deck).toHaveLength(40);
  });

  test('contains 4 suits × 10 ranks', () => {
    for (const suit of SUITS) {
      const suited = deck.filter(c => c.suit === suit);
      expect(suited).toHaveLength(10);
      for (const rank of RANKS) {
        expect(suited.find(c => c.rank === rank)).toBeDefined();
      }
    }
  });

  test('total points sum to 120', () => {
    const total = deck.reduce((s, c) => s + c.points, 0);
    expect(total).toBe(TOTAL_POINTS);
  });

  test('each card has correct points', () => {
    for (const c of deck) {
      expect(c.points).toBe(POINTS_MAP[c.rank] ?? 0);
    }
  });

  test('is a new array each time', () => {
    expect(createDeck()).not.toBe(deck);
  });
});

/* ═══════════════════════════════════════════════════════════════
   shuffleDeck
   ═══════════════════════════════════════════════════════════════ */

describe('shuffleDeck', () => {
  test('returns a new array with the same cards', () => {
    const deck = createDeck();
    const shuffled = shuffleDeck(deck);
    expect(shuffled).toHaveLength(40);
    expect(shuffled).not.toBe(deck);
    // Same cards, possibly different order
    const sortFn = (a, b) => `${a.suit}${a.rank}`.localeCompare(`${b.suit}${b.rank}`);
    expect([...shuffled].sort(sortFn)).toEqual([...deck].sort(sortFn));
  });

  test('does not mutate the original deck', () => {
    const deck = createDeck();
    const copy = JSON.parse(JSON.stringify(deck));
    shuffleDeck(deck);
    expect(deck).toEqual(copy);
  });
});

/* ═══════════════════════════════════════════════════════════════
   createInitialState
   ═══════════════════════════════════════════════════════════════ */

describe('createInitialState', () => {
  let state;
  beforeEach(() => { state = createInitialState(); });

  test('deals 3 cards to each player', () => {
    expect(state.players.host.hand).toHaveLength(3);
    expect(state.players.guest.hand).toHaveLength(3);
  });

  test('remaining deck has 34 cards (including briscola at end)', () => {
    expect(state.deck).toHaveLength(34);
  });

  test('briscola card exists and is the last in the deck', () => {
    expect(state.briscola).toBeDefined();
    expect(state.briscola.suit).toBeDefined();
    expect(state.briscola.rank).toBeDefined();
    expect(state.deck[state.deck.length - 1]).toEqual(state.briscola);
  });

  test('guest plays first (non-dealer)', () => {
    expect(state.currentTurn).toBe('guest');
    expect(state.firstPlayer).toBe('guest');
  });

  test('phase is "playing"', () => {
    expect(state.phase).toBe('playing');
  });

  test('table cards are both null', () => {
    expect(state.tableCards.host).toBeNull();
    expect(state.tableCards.guest).toBeNull();
  });

  test('scores start at 0', () => {
    expect(state.players.host.score).toBe(0);
    expect(state.players.guest.score).toBe(0);
  });

  test('all 40 cards are accounted for', () => {
    const allCards = [
      ...state.players.host.hand,
      ...state.players.guest.hand,
      ...state.deck,
    ];
    // The briscola is both in deck (last position) and state.briscola
    // Deck includes briscola, so total = 3 + 3 + 34 = 40 ✓
    expect(allCards).toHaveLength(40);
    const total = allCards.reduce((s, c) => s + c.points, 0);
    expect(total).toBe(120);
  });
});

/* ═══════════════════════════════════════════════════════════════
   dealCards
   ═══════════════════════════════════════════════════════════════ */

describe('dealCards', () => {
  test('returns a fresh initial state', () => {
    const old = createInitialState();
    const fresh = dealCards(old);
    expect(fresh.players.host.hand).toHaveLength(3);
    expect(fresh.players.guest.hand).toHaveLength(3);
    expect(fresh.deck).toHaveLength(34);
    expect(fresh.phase).toBe('playing');
  });
});

/* ═══════════════════════════════════════════════════════════════
   getRankOrder
   ═══════════════════════════════════════════════════════════════ */

describe('getRankOrder', () => {
  test('Asso(1) is strongest → 10', () => { expect(getRankOrder(1)).toBe(10); });
  test('Tre(3) is second → 9', () => { expect(getRankOrder(3)).toBe(9); });
  test('Re(13) → 8', () => { expect(getRankOrder(13)).toBe(8); });
  test('Cavallo(12) → 7', () => { expect(getRankOrder(12)).toBe(7); });
  test('Fante(11) → 6', () => { expect(getRankOrder(11)).toBe(6); });
  test('Due(2) is weakest → 1', () => { expect(getRankOrder(2)).toBe(1); });
  test('unknown rank → 0', () => { expect(getRankOrder(99)).toBe(0); });
});

/* ═══════════════════════════════════════════════════════════════
   compareCards
   ═══════════════════════════════════════════════════════════════ */

describe('compareCards', () => {
  const briscola = 'coppe';

  test('same suit: higher rank wins', () => {
    const asso  = { suit: 'denari', rank: 1, points: 11 };
    const tre   = { suit: 'denari', rank: 3, points: 10 };
    expect(compareCards(asso, tre, briscola)).toBe('first');
    expect(compareCards(tre, asso, briscola)).toBe('second');
  });

  test('same suit: Re beats Cavallo', () => {
    const re  = { suit: 'bastoni', rank: 13, points: 4 };
    const cav = { suit: 'bastoni', rank: 12, points: 3 };
    expect(compareCards(re, cav, briscola)).toBe('first');
  });

  test('trump beats non-trump (first plays trump)', () => {
    const trump   = { suit: 'coppe', rank: 2, points: 0 };
    const regular = { suit: 'spade', rank: 1, points: 11 };
    expect(compareCards(trump, regular, briscola)).toBe('first');
  });

  test('trump beats non-trump (second plays trump)', () => {
    const regular = { suit: 'spade', rank: 1, points: 11 };
    const trump   = { suit: 'coppe', rank: 2, points: 0 };
    expect(compareCards(regular, trump, briscola)).toBe('second');
  });

  test('both trump: higher rank wins', () => {
    const asso = { suit: 'coppe', rank: 1, points: 11 };
    const due  = { suit: 'coppe', rank: 2, points: 0 };
    expect(compareCards(asso, due, briscola)).toBe('first');
    expect(compareCards(due, asso, briscola)).toBe('second');
  });

  test('different non-trump suits: first card wins', () => {
    const c1 = { suit: 'denari', rank: 2, points: 0 };
    const c2 = { suit: 'spade', rank: 1, points: 11 };
    expect(compareCards(c1, c2, briscola)).toBe('first');
  });

  test('different non-trump suits: first wins regardless of points', () => {
    const weak   = { suit: 'bastoni', rank: 2, points: 0 };
    const strong = { suit: 'denari',  rank: 1, points: 11 };
    expect(compareCards(weak, strong, briscola)).toBe('first');
  });
});

/* ═══════════════════════════════════════════════════════════════
   playCard
   ═══════════════════════════════════════════════════════════════ */

describe('playCard', () => {
  let state;
  beforeEach(() => { state = createInitialState(); });

  test('guest can play on their turn', () => {
    const card = state.players.guest.hand[0];
    const next = playCard(state, 'guest', card);
    expect(next).not.toBeNull();
    expect(next.tableCards.guest).toEqual(card);
    expect(next.players.guest.hand).toHaveLength(2);
    expect(next.currentTurn).toBe('host');
  });

  test('host cannot play on guest\'s turn', () => {
    const card = state.players.host.hand[0];
    expect(playCard(state, 'host', card)).toBeNull();
  });

  test('playing a card not in hand returns null', () => {
    const fake = { suit: 'coppe', rank: 99, points: 0 };
    expect(playCard(state, 'guest', fake)).toBeNull();
  });

  test('both cards played → phase becomes roundEnd', () => {
    const gCard = state.players.guest.hand[0];
    const s1 = playCard(state, 'guest', gCard);
    const hCard = s1.players.host.hand[0];
    const s2 = playCard(s1, 'host', hCard);
    expect(s2.phase).toBe('roundEnd');
    expect(s2.tableCards.host).toEqual(hCard);
    expect(s2.tableCards.guest).toEqual(gCard);
  });

  test('does not mutate original state', () => {
    const copy = JSON.parse(JSON.stringify(state));
    const card = state.players.guest.hand[0];
    playCard(state, 'guest', card);
    expect(state).toEqual(copy);
  });

  test('returns null if phase is not "playing"', () => {
    state.phase = 'gameOver';
    const card = state.players.guest.hand[0];
    expect(playCard(state, 'guest', card)).toBeNull();
  });
});

/* ═══════════════════════════════════════════════════════════════
   resolveTrick
   ═══════════════════════════════════════════════════════════════ */

describe('resolveTrick', () => {
  test('winner gets both cards and points', () => {
    let state = createInitialState();
    const gCard = state.players.guest.hand[0];
    state = playCard(state, 'guest', gCard);
    const hCard = state.players.host.hand[0];
    state = playCard(state, 'host', hCard);

    expect(state.phase).toBe('roundEnd');
    const resolved = resolveTrick(state);

    expect(resolved.phase).toBe('playing');
    expect(resolved.tableCards.host).toBeNull();
    expect(resolved.tableCards.guest).toBeNull();

    const totalScore = resolved.players.host.score + resolved.players.guest.score;
    expect(totalScore).toBe(gCard.points + hCard.points);
  });

  test('each player draws a card from deck (winner first)', () => {
    let state = createInitialState();
    const deckBefore = state.deck.length;
    const gCard = state.players.guest.hand[0];
    state = playCard(state, 'guest', gCard);
    const hCard = state.players.host.hand[0];
    state = playCard(state, 'host', hCard);
    const resolved = resolveTrick(state);

    // Each player drew 1 card → deck is 2 shorter
    expect(resolved.deck).toHaveLength(deckBefore - 2);
    // Each player has 3 cards again
    expect(resolved.players.host.hand).toHaveLength(3);
    expect(resolved.players.guest.hand).toHaveLength(3);
  });

  test('roundNumber increments', () => {
    let state = createInitialState();
    expect(state.roundNumber).toBe(1);
    const gCard = state.players.guest.hand[0];
    state = playCard(state, 'guest', gCard);
    const hCard = state.players.host.hand[0];
    state = playCard(state, 'host', hCard);
    const resolved = resolveTrick(state);
    expect(resolved.roundNumber).toBe(2);
  });

  test('does nothing if phase is not roundEnd', () => {
    const state = createInitialState();
    const result = resolveTrick(state);
    expect(result).toBe(state);
  });

  test('does not mutate original state', () => {
    let state = createInitialState();
    const gCard = state.players.guest.hand[0];
    state = playCard(state, 'guest', gCard);
    const hCard = state.players.host.hand[0];
    state = playCard(state, 'host', hCard);
    const copy = JSON.parse(JSON.stringify(state));
    resolveTrick(state);
    expect(state).toEqual(copy);
  });
});

/* ═══════════════════════════════════════════════════════════════
   checkGameOver
   ═══════════════════════════════════════════════════════════════ */

describe('checkGameOver', () => {
  test('returns same state if not game over', () => {
    const state = createInitialState();
    expect(checkGameOver(state)).toBe(state);
  });

  test('detects game over when deck and hands are empty', () => {
    const state = createInitialState();
    state.deck = [];
    state.players.host.hand = [];
    state.players.guest.hand = [];
    state.players.host.score = 70;
    state.players.guest.score = 50;

    const result = checkGameOver(state);
    expect(result.phase).toBe('gameOver');
    expect(result.winner).toBe('host');
  });

  test('detects draw', () => {
    const state = createInitialState();
    state.deck = [];
    state.players.host.hand = [];
    state.players.guest.hand = [];
    state.players.host.score = 60;
    state.players.guest.score = 60;

    const result = checkGameOver(state);
    expect(result.phase).toBe('gameOver');
    expect(result.winner).toBe('draw');
  });

  test('guest wins', () => {
    const state = createInitialState();
    state.deck = [];
    state.players.host.hand = [];
    state.players.guest.hand = [];
    state.players.host.score = 40;
    state.players.guest.score = 80;

    const result = checkGameOver(state);
    expect(result.winner).toBe('guest');
  });
});

/* ═══════════════════════════════════════════════════════════════
   getValidMoves
   ═══════════════════════════════════════════════════════════════ */

describe('getValidMoves', () => {
  test('returns all cards in hand on player\'s turn', () => {
    const state = createInitialState();
    const moves = getValidMoves(state, 'guest');
    expect(moves).toEqual(state.players.guest.hand);
  });

  test('returns empty if not player\'s turn', () => {
    const state = createInitialState();
    expect(getValidMoves(state, 'host')).toEqual([]);
  });

  test('returns empty if phase is not playing', () => {
    const state = createInitialState();
    state.phase = 'roundEnd';
    expect(getValidMoves(state, 'guest')).toEqual([]);
  });
});

/* ═══════════════════════════════════════════════════════════════
   sanitizeStateForPeer
   ═══════════════════════════════════════════════════════════════ */

describe('sanitizeStateForPeer', () => {
  test('hides opponent hand with nulls', () => {
    const state = createInitialState();
    const sanitised = sanitizeStateForPeer(state, 'guest');
    expect(sanitised.players.host.hand).toEqual([null, null, null]);
    expect(sanitised.players.guest.hand).toHaveLength(3);
    sanitised.players.guest.hand.forEach(c => expect(c).not.toBeNull());
  });

  test('hides deck order with nulls', () => {
    const state = createInitialState();
    const sanitised = sanitizeStateForPeer(state, 'host');
    expect(sanitised.deck).toHaveLength(34);
    sanitised.deck.forEach(c => expect(c).toBeNull());
  });

  test('preserves briscola and scores', () => {
    const state = createInitialState();
    state.players.host.score = 25;
    const sanitised = sanitizeStateForPeer(state, 'guest');
    expect(sanitised.briscola).toEqual(state.briscola);
    expect(sanitised.players.host.score).toBe(25);
  });

  test('does not mutate original state', () => {
    const state = createInitialState();
    const copy = JSON.parse(JSON.stringify(state));
    sanitizeStateForPeer(state, 'guest');
    expect(state).toEqual(copy);
  });
});

/* ═══════════════════════════════════════════════════════════════
   Full game simulation
   ═══════════════════════════════════════════════════════════════ */

describe('full game simulation', () => {
  test('completes 20 tricks with total score = 120', () => {
    let state = createInitialState();
    let trickCount = 0;

    while (state.phase !== 'gameOver') {
      // Current player plays their first valid card
      const currentPlayer = state.currentTurn;
      const moves = getValidMoves(state, currentPlayer);
      expect(moves.length).toBeGreaterThan(0);

      state = playCard(state, currentPlayer, moves[0]);
      expect(state).not.toBeNull();

      // If round ended, resolve the trick
      if (state.phase === 'roundEnd') {
        state = resolveTrick(state);
        trickCount++;
      }
    }

    expect(trickCount).toBe(20);
    expect(state.phase).toBe('gameOver');
    expect(state.players.host.score + state.players.guest.score).toBe(120);
    expect(state.winner).toBeDefined();
    expect(['host', 'guest', 'draw']).toContain(state.winner);
    expect(state.deck).toHaveLength(0);
    expect(state.players.host.hand).toHaveLength(0);
    expect(state.players.guest.hand).toHaveLength(0);
  });

  test('all 40 cards end up in trick piles', () => {
    let state = createInitialState();

    while (state.phase !== 'gameOver') {
      const moves = getValidMoves(state, state.currentTurn);
      state = playCard(state, state.currentTurn, moves[0]);
      if (state.phase === 'roundEnd') state = resolveTrick(state);
    }

    const allTricks = [
      ...state.players.host.tricks,
      ...state.players.guest.tricks,
    ];
    expect(allTricks).toHaveLength(40);
  });
});
