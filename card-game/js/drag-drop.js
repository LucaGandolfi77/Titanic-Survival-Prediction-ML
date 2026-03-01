/**
 * Drag & Drop — CardForge
 * HTML5 drag-and-drop for hand → play zone, within play zone, and table → discard.
 * @module drag-drop
 */

import { getState, setState } from "./state.js";
import { getCardById } from "./database.js";
import { showToast } from "./utils.js";

/* ── Constants ─────────────────────────────────────────── */
const DRAG_TYPE = "application/x-cardforge";

/* ── Make a card draggable ─────────────────────────────── */

/**
 * Attach drag listeners to a card element.
 * @param {HTMLElement} el   Card DOM element.
 * @param {object} payload   { uid, cardId, source: "hand" | "table" }
 */
export const makeDraggable = (el, payload) => {
  el.setAttribute("draggable", "true");

  el.addEventListener("dragstart", (e) => {
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData(DRAG_TYPE, JSON.stringify(payload));
    el.classList.add("card--dragging");

    /* Ghost image */
    const ghost = el.cloneNode(true);
    ghost.style.width = "100px";
    ghost.style.position = "absolute";
    ghost.style.top = "-9999px";
    document.body.appendChild(ghost);
    e.dataTransfer.setDragImage(ghost, 50, 70);
    requestAnimationFrame(() => ghost.remove());
  });

  el.addEventListener("dragend", () => {
    el.classList.remove("card--dragging");
    clearAllDropHighlights();
  });
};

/* ── Drop zones ────────────────────────────────────────── */

/**
 * Initialise all drop-zone listeners. Call once after DOM is ready.
 */
export const initDropZones = () => {
  const playZone    = document.getElementById("play-zone");
  const discardPile = document.getElementById("discard-pile");

  if (playZone) {
    playZone.addEventListener("dragover", handleDragOver);
    playZone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      playZone.classList.add("play-zone--dragover");
    });
    playZone.addEventListener("dragleave", (e) => {
      if (!playZone.contains(e.relatedTarget)) {
        playZone.classList.remove("play-zone--dragover");
      }
    });
    playZone.addEventListener("drop", handlePlayZoneDrop);
  }

  if (discardPile) {
    discardPile.addEventListener("dragover", handleDragOver);
    discardPile.addEventListener("dragenter", (e) => {
      e.preventDefault();
      discardPile.classList.add("pile--dragover");
    });
    discardPile.addEventListener("dragleave", (e) => {
      if (!discardPile.contains(e.relatedTarget)) {
        discardPile.classList.remove("pile--dragover");
      }
    });
    discardPile.addEventListener("drop", handleDiscardDrop);
  }
};

/* ── Event handlers ────────────────────────────────────── */

/** @param {DragEvent} e */
const handleDragOver = (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = "move";
};

/**
 * Drop onto play zone — play card from hand OR reposition card on table.
 * @param {DragEvent} e
 */
const handlePlayZoneDrop = (e) => {
  e.preventDefault();
  clearAllDropHighlights();

  const raw = e.dataTransfer.getData(DRAG_TYPE);
  if (!raw) return;
  const payload = JSON.parse(raw);

  const playZone = document.getElementById("play-zone");
  const rect     = playZone.getBoundingClientRect();
  const x        = e.clientX - rect.left;
  const y        = e.clientY - rect.top;

  if (payload.source === "hand") {
    playCardFromHand(payload.uid, x, y);
  } else if (payload.source === "table") {
    repositionCard(payload.uid, x, y);
  }
};

/**
 * Drop onto discard pile — discard from table.
 * @param {DragEvent} e
 */
const handleDiscardDrop = (e) => {
  e.preventDefault();
  clearAllDropHighlights();

  const raw = e.dataTransfer.getData(DRAG_TYPE);
  if (!raw) return;
  const payload = JSON.parse(raw);

  if (payload.source === "table") {
    discardFromTable(payload.uid);
  } else if (payload.source === "hand") {
    discardFromHand(payload.uid);
  }
};

/* ── State mutations ───────────────────────────────────── */

/**
 * Move card from hand to table at position { x, y }.
 */
const playCardFromHand = (uid, x, y) => {
  const state = getState();
  const idx   = state.hand.findIndex((c) => c.uid === uid);
  if (idx === -1) return;

  const entry = state.hand[idx];
  const card  = getCardById(entry.cardId);
  const newHand = [...state.hand];
  newHand.splice(idx, 1);

  const newTable = [
    ...state.table,
    { ...entry, position: { x, y }, faceDown: false },
  ];

  setState({
    hand: newHand,
    table: newTable,
    cardsPlayed: state.cardsPlayed + 1,
  });

  showToast(`Played ${card?.name ?? "card"}`, "info");
  document.dispatchEvent(new CustomEvent("cardforge:render"));
};

/**
 * Reposition an existing table card.
 */
const repositionCard = (uid, x, y) => {
  const state = getState();
  const newTable = state.table.map((c) =>
    c.uid === uid ? { ...c, position: { x, y } } : c
  );
  setState({ table: newTable });
  document.dispatchEvent(new CustomEvent("cardforge:render"));
};

/**
 * Discard a card that is currently on the table.
 */
const discardFromTable = (uid) => {
  const state = getState();
  const idx   = state.table.findIndex((c) => c.uid === uid);
  if (idx === -1) return;

  const entry   = state.table[idx];
  const card    = getCardById(entry.cardId);
  const newTable = [...state.table];
  newTable.splice(idx, 1);

  setState({
    table: newTable,
    discardPile: [...state.discardPile, { cardId: entry.cardId, uid: entry.uid }],
    cardsDiscarded: state.cardsDiscarded + 1,
  });

  showToast(`Discarded ${card?.name ?? "card"}`, "info");
  document.dispatchEvent(new CustomEvent("cardforge:render"));
};

/**
 * Discard a card directly from hand (drag to discard pile).
 */
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

  showToast(`Discarded ${card?.name ?? "card"} from hand`, "info");
  document.dispatchEvent(new CustomEvent("cardforge:render"));
};

/* ── Helpers ───────────────────────────────────────────── */

const clearAllDropHighlights = () => {
  document.querySelectorAll(".play-zone--dragover, .pile--dragover").forEach((el) =>
    el.classList.remove("play-zone--dragover", "pile--dragover")
  );
};
