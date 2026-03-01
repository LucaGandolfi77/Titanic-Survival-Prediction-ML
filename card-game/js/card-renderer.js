/**
 * Card renderer — CardForge
 * Builds and returns HTMLElements for a card data object.
 * Includes CSS‑only fallback faces when the JPEG image is missing.
 * @module card-renderer
 */

import { typeIcon, elementHex, randomTilt } from "./utils.js";
import { getState } from "./state.js";

/**
 * @typedef {Object} RenderOptions
 * @property {'small'|'medium'|'large'|'mini'} [size='medium']
 * @property {boolean}  [faceDown=false]
 * @property {number}   [showCount=0]
 * @property {boolean}  [inHand=false]
 * @property {boolean}  [onTable=false]
 * @property {{x:number,y:number}} [position]
 * @property {number}   [tilt=0]
 * @property {boolean}  [showOverlay=false]
 * @property {boolean}  [draggable=false]
 */

/**
 * Render a card HTMLElement from a card data object.
 * @param {import('./database.js').Card} card
 * @param {RenderOptions} opts
 * @returns {HTMLElement}
 */
export const renderCard = (card, opts = {}) => {
  const {
    size = "medium",
    faceDown = false,
    showCount = 0,
    inHand = false,
    onTable = false,
    position = null,
    tilt = 0,
    showOverlay = false,
    draggable = false,
  } = opts;

  const article = document.createElement("article");
  article.className = `card card--${size}`;
  article.dataset.cardId = card.id;
  article.dataset.cardType = card.type;
  article.setAttribute("data-rarity", card.rarity);
  article.setAttribute("role", "listitem");
  article.setAttribute("tabindex", "0");
  article.setAttribute("aria-label", card.name);

  if (inHand) article.classList.add("card--in-hand");
  if (onTable) article.classList.add("card--on-table");
  if (faceDown) article.classList.add("card--face-down");
  if (draggable) article.setAttribute("draggable", "true");

  if (position) {
    article.style.left = `${position.x}px`;
    article.style.top = `${position.y}px`;
  }
  if (tilt) {
    article.style.setProperty("--tilt", `${tilt}deg`);
    if (onTable) article.style.transform = `rotate(${tilt}deg)`;
  }

  /* ── Front face ────────────────────────────────────── */
  const front = document.createElement("div");
  front.className = "card__front";

  const img = document.createElement("img");
  img.className = "card__image";
  const override = getState().imageOverrides?.[card.id];
  img.src = override || card.image;
  img.alt = card.name;
  img.loading = "lazy";
  img.draggable = false;

  /* On image error → CSS fallback */
  img.onerror = () => {
    img.remove();
    front.appendChild(buildFallbackFace(card));
  };
  front.appendChild(img);

  /* Cost badge */
  const costBadge = document.createElement("span");
  costBadge.className = "card__cost";
  costBadge.textContent = card.cost;
  front.appendChild(costBadge);

  /* Rarity bar */
  const rarityBar = document.createElement("div");
  rarityBar.className = "card__rarity-bar";
  front.appendChild(rarityBar);

  article.appendChild(front);

  /* ── Back face ─────────────────────────────────────── */
  const back = document.createElement("div");
  back.className = "card__back";
  article.appendChild(back);

  /* ── Count badge ───────────────────────────────────── */
  if (showCount > 0) {
    const countEl = document.createElement("span");
    countEl.className = "card__count";
    countEl.textContent = `×${showCount}`;
    article.appendChild(countEl);
  }

  /* ── Hover overlay (for grid) ──────────────────────── */
  if (showOverlay) {
    const overlay = document.createElement("div");
    overlay.className = "card__overlay";

    const oName = document.createElement("span");
    oName.className = "card__overlay-name";
    oName.textContent = card.name;
    overlay.appendChild(oName);

    const oBtn = document.createElement("button");
    oBtn.className = "card__overlay-btn";
    oBtn.textContent = "Add to Deck";
    oBtn.dataset.action = "add";
    overlay.appendChild(oBtn);

    article.appendChild(overlay);
  }

  return article;
};

/* ── CSS‑only fallback face ────────────────────────────── */

/**
 * Build a CSS‑gradient card face when the real image is missing.
 * @param {import('./database.js').Card} card
 * @returns {HTMLElement}
 */
const buildFallbackFace = (card) => {
  const hex = elementHex(card.element);
  const wrap = document.createElement("div");
  wrap.className = "card__fallback";
  wrap.style.background = `linear-gradient(135deg, ${hex}22 0%, ${hex}88 50%, ${hex}44 100%)`;

  /* Type icon */
  const icon = document.createElement("span");
  icon.className = "card__fallback-icon";
  icon.textContent = typeIcon(card.type);
  wrap.appendChild(icon);

  /* Card name */
  const name = document.createElement("span");
  name.className = "card__fallback-name";
  name.textContent = card.name;
  wrap.appendChild(name);

  /* Stats (if applicable) */
  if (card.power !== null || card.defense !== null) {
    const stats = document.createElement("div");
    stats.className = "card__fallback-stats";
    if (card.power !== null) {
      const pw = document.createElement("span");
      pw.textContent = `⚔${card.power}`;
      stats.appendChild(pw);
    }
    if (card.defense !== null) {
      const df = document.createElement("span");
      df.textContent = `🛡${card.defense}`;
      stats.appendChild(df);
    }
    wrap.appendChild(stats);
  }

  return wrap;
};

/**
 * Render a "mini" thumbnail for the deck panel.
 * @param {import('./database.js').Card} card
 * @param {number} count
 * @returns {HTMLElement}
 */
export const renderMiniCard = (card, count) => {
  const wrap = document.createElement("div");
  wrap.className = "deck-mini-card";
  wrap.dataset.cardId = card.id;
  wrap.title = `${card.name} (×${count}) — click to remove`;

  const img = document.createElement("img");
  const override = getState().imageOverrides?.[card.id];
  img.src = override || card.image;
  img.alt = card.name;
  img.draggable = false;
  img.onerror = () => {
    img.remove();
    const fb = buildFallbackFace(card);
    fb.style.fontSize = "0.45rem";
    wrap.appendChild(fb);
  };
  wrap.appendChild(img);

  if (count > 1) {
    const badge = document.createElement("span");
    badge.className = "deck-mini-card__count";
    badge.textContent = count;
    wrap.appendChild(badge);
  }

  return wrap;
};

/**
 * Build the card‑detail modal body for a given card.
 * @param {import('./database.js').Card} card
 * @param {{inDeckCount:number, deckFull:boolean}} meta
 * @returns {HTMLElement}
 */
export const renderCardDetail = (card, meta) => {
  const { inDeckCount = 0, deckFull = false } = meta;

  const el = document.createElement("div");
  el.className = "card-detail";

  /* Image / fallback */
  const imgWrap = document.createElement("div");
  imgWrap.className = "card-detail__image-wrap";
  const img = document.createElement("img");
  const override = getState().imageOverrides?.[card.id];
  img.src = override || card.image;
  img.alt = card.name;
  img.onerror = () => {
    img.remove();
    imgWrap.appendChild(buildFallbackFace(card));
  };
  imgWrap.appendChild(img);
  el.appendChild(imgWrap);

  /* Info column */
  const info = document.createElement("div");
  info.className = "card-detail__info";

  info.innerHTML = `
    <h3 class="card-detail__name">${card.name}</h3>
    <div class="card-detail__badges">
      <span class="card-detail__badge card-detail__badge--type">${typeIcon(card.type)} ${card.type}</span>
      <span class="card-detail__badge card-detail__badge--rarity" data-val="${card.rarity}">${card.rarity}</span>
      <span class="card-detail__badge card-detail__badge--element" data-val="${card.element}">${card.element}</span>
    </div>
    <div class="card-detail__stats">
      <div class="card-detail__stat">
        <span class="card-detail__stat-value">${card.cost}</span>
        <span class="card-detail__stat-label">Cost</span>
      </div>
      ${card.power !== null ? `<div class="card-detail__stat"><span class="card-detail__stat-value">${card.power}</span><span class="card-detail__stat-label">Power</span></div>` : ""}
      ${card.defense !== null ? `<div class="card-detail__stat"><span class="card-detail__stat-value">${card.defense}</span><span class="card-detail__stat-label">Defense</span></div>` : ""}
    </div>
    <p class="card-detail__desc">"${card.description}"</p>
    <div class="card-detail__tags">
      ${card.tags.map((t) => `<span class="card-detail__tag">#${t}</span>`).join("")}
    </div>
    <div class="card-detail__actions">
      <button class="btn-primary" data-action="add-detail" ${inDeckCount >= 4 || deckFull ? "disabled" : ""}>
        Add to Deck (${inDeckCount}/4)
      </button>
      <button class="btn-danger" data-action="remove-detail" ${inDeckCount <= 0 ? "disabled" : ""}>
        Remove Copy
      </button>
      <button class="btn" data-action="choose-image">Choose Image</button>
    </div>
  `;
  el.appendChild(info);

  /* Close button */
  const closeBtn = document.createElement("button");
  closeBtn.className = "modal-close";
  closeBtn.textContent = "×";
  closeBtn.dataset.action = "close-modal";
  el.appendChild(closeBtn);

  return el;
};
