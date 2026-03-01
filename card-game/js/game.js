/**
 * Game Controller — CardForge
 * Manages the game table: draw, play, discard, shuffle, clear, end game.
 * @module game
 */

import { getState, setState, subscribe, nextUid, loadFromLocalStorage, resetState } from "./state.js";
import { getCardById } from "./database.js";
import { renderCard } from "./card-renderer.js";
import { makeDraggable, initDropZones } from "./drag-drop.js";
import { shuffle, showToast, openModal, closeModal, confirm as confirmDialog,
         showContextMenu, randomTilt } from "./utils.js";
import { navigate } from "./router.js";

/* ── DOM cache ─────────────────────────────────────────── */
let drawPileEl, discardPileEl, playZoneEl, handAreaEl,
    drawCountEl, discardCountEl, drawBtn, draw3Btn, shuffleBtn, clearTableBtn, newGameBtn, endGameBtn;

/* ── Init ──────────────────────────────────────────────── */

export const initGame = () => {
  cacheDOM();
  wireButtons();
  initDropZones();

  /* Listen for render events from drag-drop module (avoids circular import) */
  document.addEventListener("cardforge:render", () => {
    renderAll();
  });
};

/** Start (or restart) a game using the saved deck. */
export const startGame = () => {
  loadFromLocalStorage();
  const state = getState();
  const deck  = state.currentDeck;

  if (!deck || deck.length === 0) {
    showToast("No deck found — build one first!", "warning");
    navigate("deck");
    return;
  }

  /* Expand deck entries into individual draw-pile items */
  const drawPile = [];
  deck.forEach(({ cardId, count }) => {
    for (let i = 0; i < count; i++) {
      drawPile.push({ cardId, uid: nextUid() });
    }
  });

  shuffle(drawPile);

  setState({
    drawPile,
    hand: [],
    table: [],
    discardPile: [],
    gamePhase: "playing",
    cardsPlayed: 0,
    cardsDiscarded: 0,
  });

  renderAll();
  showToast("Game started — draw some cards!", "success");
};

/* ── Render helpers (exported for drag-drop module) ────── */

export const renderHandArea = () => {
  if (!handAreaEl) return;
  handAreaEl.innerHTML = "";
  const hand = getState().hand;

  hand.forEach((entry, idx) => {
    const card = getCardById(entry.cardId);
    if (!card) return;
    const el = renderCard(card, {
      size: "medium",
      inHand: true,
      draggable: true,
      tilt: randomTilt(idx, hand.length),
    });

    makeDraggable(el, { uid: entry.uid, cardId: entry.cardId, source: "hand" });

    /* Double-click to play directly */
    el.addEventListener("dblclick", () => {
      playFromHand(entry.uid);
    });

    /* Right-click context menu */
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      showContextMenu(e.clientX, e.clientY, [
        { label: "Play to table", action: () => playFromHand(entry.uid) },
        { label: "Discard", action: () => discardFromHand(entry.uid) },
        { label: "View details", action: () => showCardDetailModal(card) },
      ]);
    });

    handAreaEl.appendChild(el);
  });

  updateHandCount();
};

export const renderPlayZone = () => {
  if (!playZoneEl) return;
  playZoneEl.innerHTML = "";
  const table = getState().table;

  table.forEach((entry) => {
    const card = getCardById(entry.cardId);
    if (!card) return;
    const el = renderCard(card, {
      size: "medium",
      onTable: true,
      faceDown: entry.faceDown,
      position: entry.position,
      draggable: true,
    });

    makeDraggable(el, { uid: entry.uid, cardId: entry.cardId, source: "table" });

    /* Double-click to flip */
    el.addEventListener("dblclick", () => toggleFlip(entry.uid));

    /* Right-click context menu */
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      showContextMenu(e.clientX, e.clientY, [
        { label: entry.faceDown ? "Flip face-up" : "Flip face-down", action: () => toggleFlip(entry.uid) },
        { label: "Discard", action: () => discardFromTable(entry.uid) },
        { label: "Return to hand", action: () => returnToHand(entry.uid) },
        { label: "View details", action: () => showCardDetailModal(card) },
      ]);
    });

    playZoneEl.appendChild(el);
  });
};

const renderAll = () => {
  renderHandArea();
  renderPlayZone();
  updateDrawCount();
  updateDiscardCount();
  updateHandCount();
};

/* ── Pile counts ───────────────────────────────────────── */

const updateDrawCount = () => {
  if (drawCountEl) drawCountEl.textContent = getState().drawPile.length;
};
const updateDiscardCount = () => {
  if (discardCountEl) discardCountEl.textContent = getState().discardPile.length;
};
const updateHandCount = () => {
  const el = document.getElementById("hand-count");
  if (el) el.textContent = getState().hand.length;
};

/* ── Draw cards ────────────────────────────────────────── */

const drawCards = (n = 1) => {
  const state = getState();
  if (state.gamePhase !== "playing") return;

  let pile = [...state.drawPile];
  if (pile.length === 0) {
    showToast("Draw pile is empty — shuffle discard?", "warning");
    return;
  }

  const toDraw = Math.min(n, pile.length);
  const drawn  = pile.splice(0, toDraw);

  setState({
    drawPile: pile,
    hand: [...state.hand, ...drawn],
  });

  renderHandArea();
  updateDrawCount();

  if (toDraw === 1) {
    const card = getCardById(drawn[0].cardId);
    showToast(`Drew ${card?.name ?? "a card"}`, "info");
  } else {
    showToast(`Drew ${toDraw} cards`, "info");
  }
};

/* ── Play / Discard ────────────────────────────────────── */

const playFromHand = (uid) => {
  const state = getState();
  const idx   = state.hand.findIndex((c) => c.uid === uid);
  if (idx === -1) return;

  const entry   = state.hand[idx];
  const card    = getCardById(entry.cardId);
  const newHand = [...state.hand];
  newHand.splice(idx, 1);

  /* Place roughly centred in play zone */
  const pzRect = playZoneEl?.getBoundingClientRect();
  const x = pzRect ? pzRect.width / 2 - 60 + Math.random() * 40 - 20 : 100;
  const y = pzRect ? pzRect.height / 2 - 80 + Math.random() * 40 - 20 : 80;

  setState({
    hand: newHand,
    table: [...state.table, { ...entry, position: { x, y }, faceDown: false }],
    cardsPlayed: state.cardsPlayed + 1,
  });

  showToast(`Played ${card?.name ?? "card"}`, "info");
  renderHandArea();
  renderPlayZone();
};

const discardFromHand = (uid) => {
  const state = getState();
  const idx   = state.hand.findIndex((c) => c.uid === uid);
  if (idx === -1) return;

  const entry   = state.hand[idx];
  const card    = getCardById(entry.cardId);
  const newHand = [...state.hand];
  newHand.splice(idx, 1);

  setState({
    hand: newHand,
    discardPile: [...state.discardPile, { cardId: entry.cardId, uid: entry.uid }],
    cardsDiscarded: state.cardsDiscarded + 1,
  });

  showToast(`Discarded ${card?.name ?? "card"}`, "info");
  renderHandArea();
  updateDiscardCount();
};

const discardFromTable = (uid) => {
  const state = getState();
  const idx   = state.table.findIndex((c) => c.uid === uid);
  if (idx === -1) return;

  const entry    = state.table[idx];
  const card     = getCardById(entry.cardId);
  const newTable = [...state.table];
  newTable.splice(idx, 1);

  setState({
    table: newTable,
    discardPile: [...state.discardPile, { cardId: entry.cardId, uid: entry.uid }],
    cardsDiscarded: state.cardsDiscarded + 1,
  });

  showToast(`Discarded ${card?.name ?? "card"}`, "info");
  renderPlayZone();
  updateDiscardCount();
};

const returnToHand = (uid) => {
  const state = getState();
  const idx   = state.table.findIndex((c) => c.uid === uid);
  if (idx === -1) return;

  const entry    = state.table[idx];
  const card     = getCardById(entry.cardId);
  const newTable = [...state.table];
  newTable.splice(idx, 1);

  setState({
    table: newTable,
    hand: [...state.hand, { cardId: entry.cardId, uid: entry.uid }],
  });

  showToast(`Returned ${card?.name ?? "card"} to hand`, "info");
  renderPlayZone();
  renderHandArea();
};

/* ── Flip ──────────────────────────────────────────────── */

const toggleFlip = (uid) => {
  const state = getState();
  const newTable = state.table.map((c) =>
    c.uid === uid ? { ...c, faceDown: !c.faceDown } : c
  );
  setState({ table: newTable });
  renderPlayZone();
};

/* ── Shuffle discard → draw ────────────────────────────── */

const reshuffleDiscard = async () => {
  const state = getState();
  if (state.discardPile.length === 0) {
    showToast("Discard pile is empty.", "warning");
    return;
  }

  const yes = await confirmDialog("Shuffle discard pile back into draw pile?");
  if (!yes) return;

  const combined = [...state.drawPile, ...state.discardPile.map(({ cardId, uid }) => ({ cardId, uid }))];
  shuffle(combined);

  setState({
    drawPile: combined,
    discardPile: [],
  });

  showToast("Discard shuffled into draw pile.", "success");
  updateDrawCount();
  updateDiscardCount();

  /* Visual feedback */
  if (drawPileEl) {
    drawPileEl.classList.add("anim-shuffle");
    setTimeout(() => drawPileEl.classList.remove("anim-shuffle"), 600);
  }
};

/* ── Clear table ───────────────────────────────────────── */

const clearTable = async () => {
  const state = getState();
  if (state.table.length === 0) return;
  const yes = await confirmDialog("Discard all cards from the table?");
  if (!yes) return;

  const discarded = state.table.map((c) => ({ cardId: c.cardId, uid: c.uid }));

  setState({
    table: [],
    discardPile: [...state.discardPile, ...discarded],
    cardsDiscarded: state.cardsDiscarded + discarded.length,
  });

  showToast("Table cleared.", "info");
  renderPlayZone();
  updateDiscardCount();
};

/* ── New game / End game ───────────────────────────────── */

const newGame = async () => {
  const yes = await confirmDialog("Start a new game? Current progress will be lost.");
  if (!yes) return;
  startGame();
};

const endGame = async () => {
  const yes = await confirmDialog("End the current game?");
  if (!yes) return;

  const state = getState();
  setState({ gamePhase: "ended" });

  showGameSummary({
    cardsPlayed: state.cardsPlayed,
    cardsDiscarded: state.cardsDiscarded,
    cardsInHand: state.hand.length,
    cardsOnTable: state.table.length,
    cardsInDraw: state.drawPile.length,
    cardsInDiscard: state.discardPile.length,
  });
};

/* ── Game summary modal ────────────────────────────────── */

const showGameSummary = (stats) => {
  const body = document.getElementById("modal-body");
  if (!body) return;
  body.innerHTML = "";

  const wrap = document.createElement("div");
  wrap.className = "game-summary";
  wrap.innerHTML = `
    <h2>Game Over</h2>
    <ul class="summary-stats">
      <li><span>Cards Played</span><strong>${stats.cardsPlayed}</strong></li>
      <li><span>Cards Discarded</span><strong>${stats.cardsDiscarded}</strong></li>
      <li><span>Cards in Hand</span><strong>${stats.cardsInHand}</strong></li>
      <li><span>Cards on Table</span><strong>${stats.cardsOnTable}</strong></li>
      <li><span>Draw Pile</span><strong>${stats.cardsInDraw}</strong></li>
      <li><span>Discard Pile</span><strong>${stats.cardsInDiscard}</strong></li>
    </ul>
    <div class="summary-actions">
      <button class="btn btn-primary" id="summary-new-game">New Game</button>
      <button class="btn" id="summary-deck-builder">Back to Deck Builder</button>
    </div>
  `;

  body.appendChild(wrap);
  openModal();

  document.getElementById("summary-new-game")?.addEventListener("click", () => {
    closeModal();
    startGame();
  });

  document.getElementById("summary-deck-builder")?.addEventListener("click", () => {
    closeModal();
    navigate("deck");
  });
};

/* ── Card detail (from game context) ───────────────────── */

const showCardDetailModal = (card) => {
  const body = document.getElementById("modal-body");
  if (!body) return;
  body.innerHTML = "";

  const detail = document.createElement("div");
  detail.className = "card-detail";
  detail.innerHTML = `
    <div class="card-detail__image-wrap">
      <div class="card-detail__image" style="background:var(--bg-secondary);">
        <img src="${card.image}" alt="${card.name}" onerror="this.remove()">
      </div>
    </div>
    <div class="card-detail__info">
      <h2>${card.name}</h2>
      <p class="card-detail__desc">${card.description}</p>
      <div class="card-detail__meta">
        <span class="badge badge--${card.rarity.toLowerCase()}">${card.rarity}</span>
        <span class="badge badge--${card.element.toLowerCase()}">${card.element}</span>
        <span>Type: ${card.type}</span>
        <span>Cost: ${card.cost}</span>
        ${card.power != null ? `<span>Power: ${card.power}</span>` : ""}
        ${card.defense != null ? `<span>Defense: ${card.defense}</span>` : ""}
      </div>
      <div class="card-detail__tags">${card.tags.map((t) => `<span class="tag">#${t}</span>`).join(" ")}</div>
    </div>
  `;
  body.appendChild(detail);
  openModal();
};

/* ── DOM wiring ────────────────────────────────────────── */

const cacheDOM = () => {
  drawPileEl    = document.getElementById("draw-pile");
  discardPileEl = document.getElementById("discard-pile");
  playZoneEl    = document.getElementById("play-zone");
  handAreaEl    = document.getElementById("hand-area");
  drawCountEl   = document.querySelector("#draw-pile .pile__count");
  discardCountEl = document.querySelector("#discard-pile .pile__count");
  drawBtn       = document.getElementById("draw-btn");
  draw3Btn      = document.getElementById("draw3-btn");
  shuffleBtn    = document.getElementById("shuffle-btn");
  clearTableBtn = document.getElementById("clear-table-btn");
  newGameBtn    = document.getElementById("new-game-btn");
  endGameBtn    = document.getElementById("end-game-btn");
};

const wireButtons = () => {
  drawBtn?.addEventListener("click", () => drawCards(1));
  draw3Btn?.addEventListener("click", () => drawCards(3));
  shuffleBtn?.addEventListener("click", reshuffleDiscard);
  clearTableBtn?.addEventListener("click", clearTable);
  newGameBtn?.addEventListener("click", newGame);
  endGameBtn?.addEventListener("click", endGame);

  /* Click draw pile to draw */
  drawPileEl?.addEventListener("click", () => drawCards(1));

  /* Click discard pile to view it */
  discardPileEl?.addEventListener("click", showDiscardViewer);
};

/* ── Discard viewer ────────────────────────────────────── */

const showDiscardViewer = () => {
  const discard = getState().discardPile;
  if (discard.length === 0) {
    showToast("Discard pile is empty.", "info");
    return;
  }
  const body = document.getElementById("modal-body");
  if (!body) return;
  body.innerHTML = "";

  const wrap = document.createElement("div");
  wrap.className = "discard-viewer";
  wrap.innerHTML = `<h2>Discard Pile (${discard.length})</h2>`;
  const grid = document.createElement("div");
  grid.className = "discard-viewer__grid";

  discard.forEach(({ cardId }) => {
    const card = getCardById(cardId);
    if (!card) return;
    const el = renderCard(card, { size: "small" });
    el.addEventListener("click", () => showCardDetailModal(card));
    grid.appendChild(el);
  });

  wrap.appendChild(grid);
  body.appendChild(wrap);
  openModal();
};
