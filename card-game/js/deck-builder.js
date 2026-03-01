/**
 * Deck Builder — CardForge
 * Orchestrates the deck-builder view: card grid, deck panel, filters, modals.
 * @module deck-builder
 */

import { CARD_DATABASE, getCardById } from "./database.js";
import { getState, setState, subscribe, addToDeck, removeFromDeck, clearDeck,
         deckTotalCount, deckCountOf, saveToLocalStorage, loadFromLocalStorage, setImageOverride } from "./state.js";
import { listAvailableImages } from "./assets.js";
import { applyFilters, buildFilterUI } from "./filters.js";
import { renderCard, renderMiniCard, renderCardDetail } from "./card-renderer.js";
import { showToast, openModal, closeModal, confirm as confirmDialog } from "./utils.js";
import { navigate } from "./router.js";

/* ── Cached DOM nodes ──────────────────────────────────── */
let cardGrid, deckList, deckNameInput, deckCounter, playBtn, clearBtn;

/* ── Initialise ────────────────────────────────────────── */

export const initDeckBuilder = () => {
  cardGrid      = document.getElementById("card-grid");
  deckList      = document.getElementById("deck-card-list");
  deckNameInput = document.getElementById("deck-name-input");
  deckCounter   = document.getElementById("deck-count");
  playBtn       = document.getElementById("play-btn");
  clearBtn      = document.getElementById("clear-deck-btn");

  /* Build filter UI (calls renderGrid on every change) */
  buildFilterUI(renderGrid);

  /* Deck‑name editing */
  if (deckNameInput) {
    deckNameInput.value = getState().deckName || "My Deck";
    deckNameInput.addEventListener("input", (e) => {
      setState({ deckName: e.target.value });
    });
  }

  /* Play button */
  if (playBtn) {
    playBtn.addEventListener("click", () => {
      if (deckTotalCount() < 5) {
        showToast("Add at least 5 cards to play.", "warning");
        return;
      }
      saveToLocalStorage();
      navigate("game");
    });
  }

  /* Clear deck */
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      if (deckTotalCount() === 0) return;
      const yes = await confirmDialog("Clear your entire deck?");
      if (yes) {
        clearDeck();
        renderDeckPanel();
        renderGrid();
        showToast("Deck cleared.", "info");
      }
    });
  }

  /* Save deck button */
  const saveBtn = document.getElementById("save-deck-btn");
  if (saveBtn) {
    saveBtn.addEventListener("click", () => {
      saveToLocalStorage();
      showToast("Deck saved!", "success");
    });
  }

  /* Subscribe to state changes */
  subscribe("currentDeck", () => {
    renderDeckPanel();
    updateDeckCounter();
    saveToLocalStorage();
  });

  /* Load saved state */
  loadFromLocalStorage();
  if (deckNameInput) deckNameInput.value = getState().deckName || "My Deck";

  /* Initial render */
  renderGrid();
  renderDeckPanel();
  updateDeckCounter();
};

/* ── Card grid ─────────────────────────────────────────── */

const renderGrid = () => {
  if (!cardGrid) return;
  const filtered = applyFilters();
  cardGrid.innerHTML = "";

  if (filtered.length === 0) {
    const msg = document.createElement("p");
    msg.className = "empty-msg";
    msg.textContent = "No cards match your filters.";
    cardGrid.appendChild(msg);
    updateResultCount(0);
    return;
  }

  filtered.forEach((card) => {
    const countInDeck = deckCountOf(card.id);
    const el = renderCard(card, {
      size: "medium",
      showOverlay: true,
      showCount: countInDeck > 0,
    });

    /* Make the whole card clearly clickable: pointer cursor, tooltip and keyboard activation */
    el.style.cursor = 'pointer';
    el.title = `Click to add ${card.name} to deck`;
    el.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        handleAddCard(card);
      }
    });

    /* Single click ⇒ add to deck */
    el.addEventListener("click", (e) => {
      if (e.target.closest("[data-action]")) return; /* ignore overlay buttons */
      handleAddCard(card);
    });

    /* Overlay "+1" button */
    const addBtn = el.querySelector("[data-action='add']");
    if (addBtn) addBtn.addEventListener("click", () => handleAddCard(card));

    /* Overlay "info" button */
    const infoBtn = el.querySelector("[data-action='info']");
    if (infoBtn) infoBtn.addEventListener("click", () => showCardDetail(card));

    /* Right‑click ⇒ card detail */
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      showCardDetail(card);
    });

    /* Update badge */
    if (countInDeck > 0) {
      const badge = el.querySelector(".card__count");
      if (badge) badge.textContent = `×${countInDeck}`;
    }

    cardGrid.appendChild(el);
  });

  updateResultCount(filtered.length);
};

const updateResultCount = (n) => {
  const el = document.getElementById("result-count");
  if (el) el.textContent = `${n} card${n !== 1 ? "s" : ""}`;
};

/* ── Deck panel ────────────────────────────────────────── */

const renderDeckPanel = () => {
  if (!deckList) return;
  const deck = getState().currentDeck;
  deckList.innerHTML = "";

  if (deck.length === 0) {
    const msg = document.createElement("p");
    msg.className = "empty-msg";
    msg.textContent = "Deck is empty.";
    deckList.appendChild(msg);
    return;
  }

  /* Sort by cost ascending then name */
  const sorted = [...deck].sort((a, b) => {
    const ca = getCardById(a.cardId);
    const cb = getCardById(b.cardId);
    return (ca.cost - cb.cost) || ca.name.localeCompare(cb.name);
  });

  sorted.forEach(({ cardId, count }) => {
    const card = getCardById(cardId);
    if (!card) return;
    const el = renderMiniCard(card, count);

    /* Click mini-card to remove one copy */
    el.addEventListener("click", () => {
      removeFromDeck(card.id);
      renderGrid();
    });

    /* Right‑click to open detail */
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      showCardDetail(card);
    });

    deckList.appendChild(el);
  });
};

const updateDeckCounter = () => {
  if (deckCounter) deckCounter.textContent = `${deckTotalCount()} / 60`;
  if (playBtn) playBtn.disabled = deckTotalCount() < 5;
};

/* ── Handlers ──────────────────────────────────────────── */

const handleAddCard = (card) => {
  const result = addToDeck(card.id);
  if (result === "max-copies") {
    showToast(`Max 4 copies of "${card.name}"`, "warning");
  } else if (result === "max-deck") {
    showToast("Deck is full (60 cards).", "warning");
  } else {
    showToast(`Added ${card.name}`, "success");
    renderGrid(); /* re-render to update count badges */
  }
};

/* ── Card detail modal ─────────────────────────────────── */

const showCardDetail = (card) => {
  const body = document.getElementById("modal-body");
  if (!body) return;

  const countInDeck = deckCountOf(card.id);
  body.innerHTML = "";
  const detail = renderCardDetail(card, { countInDeck, maxCopies: 4, maxDeck: 60, totalDeck: deckTotalCount() });

  /* Wire add/remove buttons inside detail */
  const addBtn = detail.querySelector("[data-detail-add]");
  const removeBtn = detail.querySelector("[data-detail-remove]");

  if (addBtn) addBtn.addEventListener("click", () => {
    handleAddCard(card);
    refreshDetailMeta(detail, card);
  });
  if (removeBtn) removeBtn.addEventListener("click", () => {
    removeFromDeck(card.id);
    refreshDetailMeta(detail, card);
    renderGrid();
  });

  body.appendChild(detail);
  openModal();

  /* Wire choose-image action (added in renderer) */
  const chooseBtn = detail.querySelector("[data-action='choose-image']");
  if (chooseBtn) chooseBtn.addEventListener("click", async () => {
    await openImagePicker(card, detail);
  });
};

/** Open an image picker modal showing files found in assets/cards and an upload control */
const openImagePicker = async (card, detailEl) => {
  const body = document.getElementById("modal-body");
  if (!body) return;
  body.innerHTML = "";

  const header = document.createElement("div");
  header.className = "image-picker__header";
  header.innerHTML = `<h2>Choose image for ${card.name}</h2>`;
  body.appendChild(header);

  const uploadWrap = document.createElement("div");
  uploadWrap.className = "image-picker__upload";
  const fileIn = document.createElement("input");
  fileIn.type = "file";
  fileIn.accept = "image/*";
  const uploadNote = document.createElement("div");
  uploadNote.textContent = "Or upload a local image (stored in browser)";
  uploadWrap.appendChild(uploadNote);
  uploadWrap.appendChild(fileIn);
  body.appendChild(uploadWrap);

  const grid = document.createElement("div");
  grid.className = "image-picker__grid";
  body.appendChild(grid);

  const removeBtn = document.createElement("button");
  removeBtn.className = "btn btn-small btn-danger";
  removeBtn.textContent = "Remove override";
  removeBtn.addEventListener("click", () => {
    setImageOverride(card.id, null);
    showToast("Image override removed.", "info");
    closeModal();
    renderGrid();
    renderDeckPanel();
  });
  body.appendChild(removeBtn);

  openModal();

  /* populate grid with available images */
  const imgs = await listAvailableImages();
  if (imgs.length === 0) {
    const p = document.createElement("p");
    p.textContent = "No images found in assets/cards.";
    grid.appendChild(p);
  } else {
    imgs.forEach((p) => {
      const t = document.createElement("div");
      t.className = "image-picker__tile";
      const im = document.createElement("img");
      im.src = p;
      im.alt = p;
      im.addEventListener("click", () => {
        setImageOverride(card.id, p);
        showToast("Image set for " + card.name, "success");
        closeModal();
        renderGrid();
        renderDeckPanel();
      });
      t.appendChild(im);
      grid.appendChild(t);
    });
  }

  /* handle file uploads (stored as data URL) */
  fileIn.addEventListener("change", (ev) => {
    const f = ev.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      setImageOverride(card.id, reader.result);
      showToast("Uploaded image set for " + card.name, "success");
      closeModal();
      renderGrid();
      renderDeckPanel();
    };
    reader.readAsDataURL(f);
  });
};

/** Update count display inside an already-open detail element */
const refreshDetailMeta = (detailEl, card) => {
  const countEl = detailEl.querySelector("[data-detail-count]");
  const removeBtn = detailEl.querySelector("[data-detail-remove]");
  const addBtn = detailEl.querySelector("[data-detail-add]");
  const n = deckCountOf(card.id);
  if (countEl) countEl.textContent = `In deck: ${n}`;
  if (removeBtn) removeBtn.disabled = n === 0;
  if (addBtn) addBtn.disabled = n >= 4 || deckTotalCount() >= 60;
};
