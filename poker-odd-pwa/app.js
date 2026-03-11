const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"];
const SUITS = [
  { key: "s", symbol: "♠", color: "black", name: "Spades" },
  { key: "h", symbol: "♥", color: "red", name: "Hearts" },
  { key: "d", symbol: "♦", color: "red", name: "Diamonds" },
  { key: "c", symbol: "♣", color: "black", name: "Clubs" }
];

const HAND_NAMES = [
  "High Card",
  "One Pair",
  "Two Pair",
  "Three of a Kind",
  "Straight",
  "Flush",
  "Full House",
  "Four of a Kind",
  "Straight Flush",
  "Royal Flush"
];

const HAND_CATEGORY_INDEX = {
  "High Card": 0,
  "One Pair": 1,
  "Two Pair": 2,
  "Three of a Kind": 3,
  "Straight": 4,
  "Flush": 5,
  "Full House": 6,
  "Four of a Kind": 7,
  "Straight Flush": 8,
  "Royal Flush": 9
};

const state = {
  hero: [null, null],
  board: [null, null, null, null, null],
  playerCount: 2,
  iterations: 50000,
  pickerMode: null,
  deferredPrompt: null
};

const els = {
  heroSlots: document.getElementById("heroSlots"),
  boardSlots: document.getElementById("boardSlots"),
  openHeroPicker: document.getElementById("openHeroPicker"),
  openBoardPicker: document.getElementById("openBoardPicker"),
  closePicker: document.getElementById("closePicker"),
  pickerModal: document.getElementById("pickerModal"),
  deckGrid: document.getElementById("deckGrid"),
  pickerTitle: document.getElementById("pickerTitle"),
  pickerHint: document.getElementById("pickerHint"),
  playerCount: document.getElementById("playerCount"),
  iterationCount: document.getElementById("iterationCount"),
  calculateBtn: document.getElementById("calculateBtn"),
  resetBtn: document.getElementById("resetBtn"),
  randomHeroBtn: document.getElementById("randomHeroBtn"),
  clearBoardBtn: document.getElementById("clearBoardBtn"),
  status: document.getElementById("status"),
  loading: document.getElementById("loading"),
  winPct: document.getElementById("winPct"),
  tiePct: document.getElementById("tiePct"),
  lossPct: document.getElementById("lossPct"),
  winBar: document.getElementById("winBar"),
  tieBar: document.getElementById("tieBar"),
  lossBar: document.getElementById("lossBar"),
  bestCurrentHand: document.getElementById("bestCurrentHand"),
  simCount: document.getElementById("simCount"),
  elapsedTime: document.getElementById("elapsedTime"),
  handProbTable: document.getElementById("handProbTable"),
  heroText: document.getElementById("heroText"),
  boardText: document.getElementById("boardText"),
  applyHeroText: document.getElementById("applyHeroText"),
  applyBoardText: document.getElementById("applyBoardText"),
  installBtn: document.getElementById("installBtn"),
  streetButtons: [...document.querySelectorAll(".street-btn")]
};

function buildDeck() {
  const deck = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({
        rank,
        suit: suit.key,
        suitSymbol: suit.symbol,
        suitColor: suit.color,
        code: `${rank}${suit.key}`,
        rankValue: rankToValue(rank)
      });
    }
  }
  return deck;
}

function rankToValue(rank) {
  if (rank === "A") return 14;
  if (rank === "K") return 13;
  if (rank === "Q") return 12;
  if (rank === "J") return 11;
  if (rank === "T") return 10;
  return Number(rank);
}

function normalizeToken(token) {
  let t = token.trim().toUpperCase();
  if (!t) return null;
  t = t.replace(/^10/, "T");
  const rank = t.slice(0, -1);
  const suit = t.slice(-1).toLowerCase();
  if (!RANKS.includes(rank)) return null;
  if (!SUITS.find(s => s.key === suit)) return null;
  return `${rank}${suit}`;
}

function parseCardsText(text) {
  if (!text.trim()) return [];
  const parts = text.split(/\s+/).filter(Boolean);
  const codes = parts.map(normalizeToken);
  if (codes.some(c => !c)) {
    throw new Error("Invalid card text. Use formats like As Kh or 10d 10c.");
  }
  const unique = new Set(codes);
  if (unique.size !== codes.length) {
    throw new Error("Duplicate cards found in text input.");
  }
  return codes;
}

function getDeckMap() {
  const map = new Map();
  for (const card of buildDeck()) map.set(card.code, card);
  return map;
}

const DECK_MAP = getDeckMap();

function cloneCard(code) {
  return code ? { ...DECK_MAP.get(code) } : null;
}

function getUsedCodes() {
  return [...state.hero, ...state.board].filter(Boolean);
}

function renderCardFace(code) {
  if (!code) return `<div class="card-slot empty">Empty</div>`;
  const card = DECK_MAP.get(code);
  return `
    <div class="card-slot ${card.suitColor}" aria-label="${card.rank} of ${SUITS.find(s => s.key === card.suit).name}">
      <div class="card-label-top">${displayRank(card.rank)}${card.suitSymbol}</div>
      <div class="card-suit-big">${card.suitSymbol}</div>
      <div class="card-label-bottom">${displayRank(card.rank)}${card.suitSymbol}</div>
    </div>
  `;
}

function displayRank(rank) {
  return rank === "T" ? "10" : rank;
}

function renderSlots() {
  els.heroSlots.innerHTML = state.hero
    .map(code => renderCardFace(code))
    .join("");

  els.boardSlots.innerHTML = state.board
    .map(code => renderCardFace(code))
    .join("");

  updateBestCurrentHand();
}

function openPicker(mode) {
  state.pickerMode = mode;
  els.pickerModal.classList.remove("hidden");
  els.pickerModal.setAttribute("aria-hidden", "false");

  if (mode === "hero") {
    els.pickerTitle.textContent = "Pick your 2 hole cards";
    els.pickerHint.textContent = "Select exactly two unique cards.";
  } else {
    els.pickerTitle.textContent = "Pick community cards";
    els.pickerHint.textContent = "Select flop, turn, and river cards as needed.";
  }

  renderDeckPicker();
}

function closePicker() {
  els.pickerModal.classList.add("hidden");
  els.pickerModal.setAttribute("aria-hidden", "true");
  state.pickerMode = null;
}

function renderDeckPicker() {
  const used = new Set(getUsedCodes());
  els.deckGrid.innerHTML = buildDeck().map(card => {
    const selected = state.hero.includes(card.code) || state.board.includes(card.code);
    const disabled = used.has(card.code) && !selected;
    return `
      <button
        class="card-tile ${card.suitColor} ${selected ? "selected" : ""} ${disabled ? "disabled" : ""}"
        data-code="${card.code}"
        ${disabled ? "disabled" : ""}
        aria-label="${displayRank(card.rank)} of ${SUITS.find(s => s.key === card.suit).name}"
      >
        <div class="card-label-top">${displayRank(card.rank)}${card.suitSymbol}</div>
        <div class="card-suit-big">${card.suitSymbol}</div>
        <div class="card-label-bottom">${displayRank(card.rank)}${card.suitSymbol}</div>
      </button>
    `;
  }).join("");
}

function assignCardToFirstEmpty(arr, code) {
  const idx = arr.findIndex(v => v === null);
  if (idx !== -1) arr[idx] = code;
}

function removeCardFromArray(arr, code) {
  const idx = arr.indexOf(code);
  if (idx !== -1) arr[idx] = null;
}

function compactBoard() {
  state.board = state.board.filter(Boolean);
  while (state.board.length < 5) state.board.push(null);
}

function handleDeckClick(e) {
  const btn = e.target.closest("[data-code]");
  if (!btn) return;
  const code = btn.dataset.code;

  if (state.pickerMode === "hero") {
    if (state.hero.includes(code)) {
      removeCardFromArray(state.hero, code);
    } else {
      if (state.hero.filter(Boolean).length >= 2) {
        setStatus("You can only select 2 hole cards.");
        return;
      }
      assignCardToFirstEmpty(state.hero, code);
    }
  }

  if (state.pickerMode === "board") {
    if (state.board.includes(code)) {
      removeCardFromArray(state.board, code);
      compactBoard();
    } else {
      if (state.board.filter(Boolean).length >= 5) {
        setStatus("Board can contain at most 5 cards.");
        return;
      }
      assignCardToFirstEmpty(state.board, code);
      compactBoard();
    }
  }

  renderDeckPicker();
  renderSlots();
}

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function resetResults() {
  setPct(els.winPct, els.winBar, 0);
  setPct(els.tiePct, els.tieBar, 0);
  setPct(els.lossPct, els.lossBar, 0);
  els.bestCurrentHand.textContent = "—";
  els.simCount.textContent = "0";
  els.elapsedTime.textContent = "0 ms";
  els.handProbTable.innerHTML = HAND_NAMES.map(name => `
    <tr>
      <td>${name}</td>
      <td>0.00%</td>
    </tr>
  `).join("");
}

function setPct(labelEl, barEl, value) {
  const pct = `${value.toFixed(2)}%`;
  labelEl.textContent = pct;
  barEl.style.width = pct;
}

function applyTextInput(target) {
  try {
    if (target === "hero") {
      const codes = parseCardsText(els.heroText.value);
      if (codes.length !== 2) throw new Error("Enter exactly 2 hero cards.");
      const combined = [...codes, ...state.board.filter(Boolean)];
      if (new Set(combined).size !== combined.length) {
        throw new Error("Hero cards conflict with board cards.");
      }
      state.hero = [codes[0], codes[1]];
      setStatus("Hero cards updated.");
    } else {
      const codes = parseCardsText(els.boardText.value);
      if (codes.length > 5) throw new Error("Board can contain at most 5 cards.");
      const combined = [...codes, ...state.hero.filter(Boolean)];
      if (new Set(combined).size !== combined.length) {
        throw new Error("Board cards conflict with hero cards.");
      }
      state.board = [...codes];
      while (state.board.length < 5) state.board.push(null);
      setStatus("Board updated.");
    }
    renderSlots();
  } catch (err) {
    setStatus(err.message, true);
  }
}

function sampleWithoutReplacement(arr, count) {
  const copy = arr.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, count);
}

function randomHeroHand() {
  const deck = buildDeck().map(c => c.code);
  const hand = sampleWithoutReplacement(deck, 2);
  state.hero = hand;
  state.board = [null, null, null, null, null];
  renderSlots();
  setStatus("Random hand selected.");
}

function setStreet(street) {
  if (street === "preflop") state.board = [null, null, null, null, null];
  if (street === "flop") state.board = [state.board[0], state.board[1], state.board[2], null, null];
  if (street === "turn") state.board = [state.board[0], state.board[1], state.board[2], state.board[3], null];
  if (street === "river") state.board = [state.board[0], state.board[1], state.board[2], state.board[3], state.board[4]];
  compactBoard();
  renderSlots();
  setStatus(`Street set to ${street}.`);
}

function cardCodesToObjects(codes) {
  return codes.map(code => cloneCard(code));
}

function countRanks(cards) {
  const map = new Map();
  for (const card of cards) {
    map.set(card.rankValue, (map.get(card.rankValue) || 0) + 1);
  }
  return map;
}

function countSuits(cards) {
  const map = new Map();
  for (const card of cards) {
    map.set(card.suit, (map.get(card.suit) || 0) + 1);
  }
  return map;
}

function getStraightHigh(rankValues) {
  const unique = [...new Set(rankValues)].sort((a, b) => b - a);
  if (unique.includes(14)) unique.push(1);
  let run = 1;
  for (let i = 0; i < unique.length - 1; i++) {
    if (unique[i] - 1 === unique[i + 1]) {
      run++;
      if (run >= 5) return unique[i - 3];
    } else {
      run = 1;
    }
  }
  return null;
}

function evaluateSeven(cards) {
  const rankMap = countRanks(cards);
  const suitMap = countSuits(cards);
  const allRanksDesc = [...rankMap.keys()].sort((a, b) => b - a);

  let flushSuit = null;
  for (const [suit, count] of suitMap.entries()) {
    if (count >= 5) {
      flushSuit = suit;
      break;
    }
  }

  if (flushSuit) {
    const flushCards = cards
      .filter(c => c.suit === flushSuit)
      .map(c => c.rankValue)
      .sort((a, b) => b - a);
    const sfHigh = getStraightHigh(flushCards);
    if (sfHigh) {
      if (sfHigh === 14) {
        return { category: 9, name: "Royal Flush", tiebreakers: [14] };
      }
      return { category: 8, name: "Straight Flush", tiebreakers: [sfHigh] };
    }
  }

  const groups = [...rankMap.entries()]
    .map(([rank, count]) => ({ rank, count }))
    .sort((a, b) => b.count - a.count || b.rank - a.rank);

  const quads = groups.filter(g => g.count === 4);
  if (quads.length) {
    const kicker = allRanksDesc.find(r => r !== quads[0].rank);
    return { category: 7, name: "Four of a Kind", tiebreakers: [quads[0].rank, kicker] };
  }

  const trips = groups.filter(g => g.count === 3).sort((a, b) => b.rank - a.rank);
  const pairs = groups.filter(g => g.count === 2).sort((a, b) => b.rank - a.rank);

  if (trips.length && (pairs.length || trips.length >= 2)) {
    const topTrip = trips[0].rank;
    const pairRank = trips.length >= 2 ? trips[1].rank : pairs[0].rank;
    return { category: 6, name: "Full House", tiebreakers: [topTrip, pairRank] };
  }

  if (flushSuit) {
    const flushRanks = cards
      .filter(c => c.suit === flushSuit)
      .map(c => c.rankValue)
      .sort((a, b) => b - a)
      .slice(0, 5);
    return { category: 5, name: "Flush", tiebreakers: flushRanks };
  }

  const straightHigh = getStraightHigh(cards.map(c => c.rankValue));
  if (straightHigh) {
    return { category: 4, name: "Straight", tiebreakers: [straightHigh] };
  }

  if (trips.length) {
    const kickers = allRanksDesc.filter(r => r !== trips[0].rank).slice(0, 2);
    return { category: 3, name: "Three of a Kind", tiebreakers: [trips[0].rank, ...kickers] };
  }

  if (pairs.length >= 2) {
    const highPair = pairs[0].rank;
    const lowPair = pairs[1].rank;
    const kicker = allRanksDesc.find(r => r !== highPair && r !== lowPair);
    return { category: 2, name: "Two Pair", tiebreakers: [highPair, lowPair, kicker] };
  }

  if (pairs.length === 1) {
    const pair = pairs[0].rank;
    const kickers = allRanksDesc.filter(r => r !== pair).slice(0, 3);
    return { category: 1, name: "One Pair", tiebreakers: [pair, ...kickers] };
  }

  return {
    category: 0,
    name: "High Card",
    tiebreakers: allRanksDesc.slice(0, 5)
  };
}

function compareHands(a, b) {
  if (a.category !== b.category) return a.category > b.category ? 1 : -1;
  const len = Math.max(a.tiebreakers.length, b.tiebreakers.length);
  for (let i = 0; i < len; i++) {
    const av = a.tiebreakers[i] || 0;
    const bv = b.tiebreakers[i] || 0;
    if (av !== bv) return av > bv ? 1 : -1;
  }
  return 0;
}

function updateBestCurrentHand() {
  const heroCodes = state.hero.filter(Boolean);
  const boardCodes = state.board.filter(Boolean);
  if (heroCodes.length !== 2) {
    els.bestCurrentHand.textContent = "—";
    return;
  }
  const cards = cardCodesToObjects([...heroCodes, ...boardCodes]);
  if (cards.length < 5) {
    els.bestCurrentHand.textContent = "Not enough cards yet";
    return;
  }
  const result = evaluateSeven(cards);
  els.bestCurrentHand.textContent = result.name;
}

function validateState() {
  const heroCodes = state.hero.filter(Boolean);
  const boardCodes = state.board.filter(Boolean);
  if (heroCodes.length !== 2) {
    throw new Error("Select exactly 2 hole cards.");
  }
  const all = [...heroCodes, ...boardCodes];
  if (new Set(all).size !== all.length) {
    throw new Error("Duplicate cards detected.");
  }
  if (boardCodes.length === 1 || boardCodes.length === 2) {
    throw new Error("Board must be 0, 3, 4, or 5 cards.");
  }
}

function availableDeckCodes() {
  const used = new Set(getUsedCodes());
  return buildDeck().map(c => c.code).filter(code => !used.has(code));
}

function formatPct(value, total) {
  return (value / total) * 100;
}

async function runSimulation() {
  try {
    validateState();
  } catch (err) {
    setStatus(err.message, true);
    return;
  }

  const iterations = Number(state.iterations);
  const playerCount = Number(state.playerCount);
  const opponents = playerCount - 1;
  const heroCodes = state.hero.filter(Boolean);
  const knownBoard = state.board.filter(Boolean);
  const neededBoard = 5 - knownBoard.length;
  const available = availableDeckCodes();

  els.loading.classList.remove("hidden");
  els.calculateBtn.disabled = true;
  setStatus("Simulation running…");
  await new Promise(r => setTimeout(r, 30));

  let wins = 0;
  let ties = 0;
  let losses = 0;
  const handCounts = new Array(HAND_NAMES.length).fill(0);

  const start = performance.now();

  for (let i = 0; i < iterations; i++) {
    const cardsNeeded = neededBoard + opponents * 2;
    const draw = sampleWithoutReplacement(available, cardsNeeded);

    const boardDraw = draw.slice(0, neededBoard);
    const finalBoard = [...knownBoard, ...boardDraw];

    const heroCards = cardCodesToObjects([...heroCodes, ...finalBoard]);
    const heroEval = evaluateSeven(heroCards);
    handCounts[HAND_CATEGORY_INDEX[heroEval.name]]++;

    let bestComparison = 1;
    let tiedPlayers = 0;
    let offset = neededBoard;

    for (let o = 0; o < opponents; o++) {
      const oppHole = draw.slice(offset, offset + 2);
      offset += 2;
      const oppCards = cardCodesToObjects([...oppHole, ...finalBoard]);
      const oppEval = evaluateSeven(oppCards);
      const cmp = compareHands(heroEval, oppEval);

      if (cmp < 0) {
        bestComparison = -1;
        break;
      } else if (cmp === 0) {
        tiedPlayers++;
        bestComparison = 0;
      }
    }

    if (bestComparison > 0) {
      wins++;
    } else if (bestComparison === 0) {
      ties++;
    } else {
      losses++;
    }

    if (i % 2000 === 0) {
      await new Promise(r => setTimeout(r, 0));
    }
  }

  const elapsed = performance.now() - start;
  const total = wins + ties + losses;

  setPct(els.winPct, els.winBar, formatPct(wins, total));
  setPct(els.tiePct, els.tieBar, formatPct(ties, total));
  setPct(els.lossPct, els.lossBar, formatPct(losses, total));
  els.simCount.textContent = iterations.toLocaleString();
  els.elapsedTime.textContent = `${elapsed.toFixed(0)} ms`;

  els.handProbTable.innerHTML = HAND_NAMES.map(name => {
    const idx = HAND_CATEGORY_INDEX[name];
    const pct = ((handCounts[idx] / iterations) * 100).toFixed(2);
    return `
      <tr>
        <td>${name}</td>
        <td>${pct}%</td>
      </tr>
    `;
  }).join("");

  setStatus("Simulation completed.");
  els.loading.classList.add("hidden");
  els.calculateBtn.disabled = false;
}

function resetAll() {
  state.hero = [null, null];
  state.board = [null, null, null, null, null];
  state.playerCount = 2;
  state.iterations = 50000;
  els.playerCount.value = "2";
  els.iterationCount.value = "50000";
  els.heroText.value = "";
  els.boardText.value = "";
  renderSlots();
  resetResults();
  setStatus("Reset complete.");
}

function attachEvents() {
  els.openHeroPicker.addEventListener("click", () => openPicker("hero"));
  els.openBoardPicker.addEventListener("click", () => openPicker("board"));
  els.closePicker.addEventListener("click", closePicker);
  els.pickerModal.addEventListener("click", e => {
    if (e.target === els.pickerModal) closePicker();
  });
  els.deckGrid.addEventListener("click", handleDeckClick);

  els.playerCount.addEventListener("change", e => {
    state.playerCount = Number(e.target.value);
  });

  els.iterationCount.addEventListener("change", e => {
    state.iterations = Number(e.target.value);
  });

  els.calculateBtn.addEventListener("click", runSimulation);
  els.resetBtn.addEventListener("click", resetAll);
  els.randomHeroBtn.addEventListener("click", randomHeroHand);
  els.clearBoardBtn.addEventListener("click", () => {
    state.board = [null, null, null, null, null];
    renderSlots();
    setStatus("Board cleared.");
  });

  els.applyHeroText.addEventListener("click", () => applyTextInput("hero"));
  els.applyBoardText.addEventListener("click", () => applyTextInput("board"));

  els.streetButtons.forEach(btn => {
    btn.addEventListener("click", () => setStreet(btn.dataset.street));
  });

  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !els.pickerModal.classList.contains("hidden")) {
      closePicker();
    }
  });

  window.addEventListener("beforeinstallprompt", e => {
    e.preventDefault();
    state.deferredPrompt = e;
    els.installBtn.classList.remove("hidden");
  });

  els.installBtn.addEventListener("click", async () => {
    if (!state.deferredPrompt) return;
    state.deferredPrompt.prompt();
    await state.deferredPrompt.userChoice;
    state.deferredPrompt = null;
    els.installBtn.classList.add("hidden");
  });
}

function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
}

function init() {
  renderSlots();
  resetResults();
  attachEvents();
  registerServiceWorker();
}

init();
