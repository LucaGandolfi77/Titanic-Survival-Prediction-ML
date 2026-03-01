/**
 * Main entry point — CardForge
 * Bootstraps all modules, wires global shortcuts & listeners.
 * @module main
 */

import { initRouter, navigate, getCurrentRoute } from "./router.js";
import { initDeckBuilder } from "./deck-builder.js";
import { initGame, startGame, renderHandArea } from "./game.js";
import { getState, subscribe, loadFromLocalStorage } from "./state.js";
import { closeModal, showToast, hideContextMenu } from "./utils.js";
import { getCardById } from "./database.js";

/* ── Bootstrap ─────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  loadFromLocalStorage();

  /* Modules */
  initRouter();
  initDeckBuilder();
  initGame();

  /* Thumbnails gallery */
  try {
    const { default: initThumbGallery } = await import('./thumbs.js');
    initThumbGallery();
  } catch (e) {
    // optional feature — ignore if module fails
    // console.warn('Thumb gallery failed to init', e);
  }

  /* If we land on #game and there's a deck, start automatically */
  if (getCurrentRoute() === "game") {
    startGame();
  }

  /* Global listeners */
  setupKeyboardShortcuts();
  setupGlobalClicks();
  setupCardZoom();

  console.info(
    "%c♠ CardForge loaded",
    "color:#d4af37;font-size:14px;font-weight:bold;"
  );
});

/* ── Keyboard shortcuts ────────────────────────────────── */

const setupKeyboardShortcuts = () => {
  document.addEventListener("keydown", (e) => {
    const tag = e.target.tagName;
    /* Ignore when typing in an input */
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    const key = e.key.toLowerCase();

    switch (key) {
      /* D = Draw one card (game view only) */
      case "d":
        if (getCurrentRoute() === "game" && getState().gamePhase === "playing") {
          document.getElementById("draw-btn")?.click();
        }
        break;

      /* Escape = close modal / hide context menu */
      case "escape":
        closeModal();
        hideContextMenu();
        closeZoom();
        break;

      /* Delete / Backspace = discard selected (future) */
      case "delete":
      case "backspace":
        /* reserved for future selection-based discard */
        break;

      /* 1‒9 navigate to specific deck/game */
      case "1":
        navigate("deck");
        break;
      case "2":
        if (getCurrentRoute() !== "game") {
          navigate("game");
          startGame();
        }
        break;

      /* N = new game */
      case "n":
        if (getCurrentRoute() === "game") {
          document.getElementById("new-game-btn")?.click();
        }
        break;

      /* S = save deck */
      case "s":
        if (getCurrentRoute() === "deck" && !e.metaKey && !e.ctrlKey) {
          document.getElementById("save-deck-btn")?.click();
        }
        break;

      /* Z = zoom hovered card */
      case "z":
        handleZoomKey();
        break;

      default:
        break;
    }
  });
};

/* ── Global click handlers ─────────────────────────────── */

const setupGlobalClicks = () => {
  /* Click outside modal → close */
  const overlay = document.getElementById("modal-overlay");
  if (overlay) {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closeModal();
    });
  }

  /* Modal close button */
  const closeBtn = document.getElementById("modal-close");
  if (closeBtn) closeBtn.addEventListener("click", closeModal);

  /* Hide context menu on any click */
  document.addEventListener("click", () => hideContextMenu());

  /* Navigation links */
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const route = link.dataset.nav;
      navigate(route);
      if (route === "game") {
        const s = getState();
        if (s.gamePhase !== "playing" || s.drawPile.length === 0 && s.hand.length === 0) {
          startGame();
        }
      }
    });
  });
};

/* ── Card zoom ─────────────────────────────────────────── */

let zoomOverlay = null;

const setupCardZoom = () => {
  /* Middle-click on any card to zoom */
  document.addEventListener("auxclick", (e) => {
    if (e.button !== 1) return; /* 1 = middle */
    const cardEl = e.target.closest(".card");
    if (!cardEl) return;
    e.preventDefault();
    const cardId = cardEl.dataset.cardId;
    if (cardId) zoomCard(cardId);
  });
};

const handleZoomKey = () => {
  /* Find the card the mouse is currently over */
  const hovered = document.querySelector(".card:hover");
  if (!hovered) return;
  const cardId = hovered.dataset.cardId;
  if (cardId) zoomCard(cardId);
};

const zoomCard = (cardId) => {
  const card = getCardById(cardId);
  if (!card) return;

  closeZoom(); /* close any previous zoom */

  zoomOverlay = document.createElement("div");
  zoomOverlay.className = "card-zoom-overlay";
  zoomOverlay.addEventListener("click", closeZoom);

  const img = document.createElement("div");
  img.className = "card-zoom";
  img.innerHTML = `
    <div class="card-zoom__inner">
      <img src="${card.image}" alt="${card.name}" onerror="this.style.display='none'">
      <h2>${card.name}</h2>
      <p>${card.description}</p>
      <div class="card-zoom__stats">
        <span>Cost: ${card.cost}</span>
        ${card.power != null ? `<span>Power: ${card.power}</span>` : ""}
        ${card.defense != null ? `<span>Defense: ${card.defense}</span>` : ""}
        <span>${card.rarity}</span>
        <span>${card.element}</span>
      </div>
    </div>
  `;

  zoomOverlay.appendChild(img);
  document.body.appendChild(zoomOverlay);
  requestAnimationFrame(() => zoomOverlay.classList.add("active"));
};

const closeZoom = () => {
  if (zoomOverlay) {
    zoomOverlay.remove();
    zoomOverlay = null;
  }
};
