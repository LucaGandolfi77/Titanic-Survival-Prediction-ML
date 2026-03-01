/**
 * Utility helpers — CardForge
 * @module utils
 */

/**
 * Fisher‑Yates (Knuth) shuffle — returns a new shuffled array.
 * @param {Array} arr
 * @returns {Array}
 */
export const shuffle = (arr) => {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.trunc(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

/**
 * Deep‑clone a JSON‑serialisable value (structuredClone wrapper).
 * @param {*} value
 * @returns {*}
 */
export const deepClone = (value) => {
  try {
    return structuredClone(value);
  } catch {
    return JSON.parse(JSON.stringify(value));
  }
};

/**
 * Debounce a function by `ms` milliseconds.
 * @param {Function} fn
 * @param {number} ms
 * @returns {Function}
 */
export const debounce = (fn, ms = 200) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

/**
 * Clamp a numeric value between min and max.
 * @param {number} val
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export const clamp = (val, min, max) => Math.min(Math.max(val, min), max);

/**
 * Generate a random integer between min and max (inclusive).
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export const randomInt = (min, max) =>
  Math.trunc(Math.random() * (max - min + 1)) + min;

/**
 * Generate a random float tilt between -range and +range degrees.
 * @param {number} range
 * @returns {number}
 */
export const randomTilt = (range = 5) =>
  (Math.random() * range * 2 - range);

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'info'|'success'|'warning'|'error'} type
 * @param {number} duration  Auto‑dismiss in ms (default 2500)
 */
export const showToast = (message, type = "info", duration = 2500) => {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const el = document.createElement("div");
  el.className = `toast toast--${type}`;
  el.textContent = message;
  container.appendChild(el);

  setTimeout(() => {
    el.classList.add("toast--leaving");
    el.addEventListener("animationend", () => el.remove());
  }, duration);
};

/**
 * Open a generic modal.
 * @param {string|HTMLElement} content  HTML string or element
 */
export const openModal = (content) => {
  const overlay = document.getElementById("modal-overlay");
  const container = document.getElementById("modal-content");
  if (!overlay || !container) return;

  if (typeof content === "string") {
    container.innerHTML = content;
  } else {
    container.innerHTML = "";
    container.appendChild(content);
  }
  overlay.classList.remove("hidden");
};

/**
 * Close the currently open modal.
 */
export const closeModal = () => {
  const overlay = document.getElementById("modal-overlay");
  if (overlay) overlay.classList.add("hidden");
};

/**
 * Show a confirmation dialog inside the modal.
 * @param {string} title
 * @param {string} text
 * @returns {Promise<boolean>}
 */
export const confirm = (title, text) =>
  new Promise((resolve) => {
    const wrap = document.createElement("div");
    wrap.className = "confirm-dialog";
    wrap.innerHTML = `
      <h3 class="confirm-dialog__title">${title}</h3>
      <p class="confirm-dialog__text">${text}</p>
      <div class="confirm-dialog__actions">
        <button class="btn-danger" data-action="cancel">Cancel</button>
        <button class="btn-primary" data-action="confirm">Confirm</button>
      </div>
    `;
    openModal(wrap);
    wrap.addEventListener("click", (e) => {
      const action = e.target.dataset?.action;
      if (action === "confirm") { closeModal(); resolve(true); }
      if (action === "cancel")  { closeModal(); resolve(false); }
    });
  });

/**
 * Show a context menu at the given screen position.
 * @param {number} x
 * @param {number} y
 * @param {{label:string, action:Function}[]} items
 */
export const showContextMenu = (x, y, items) => {
  const menu = document.getElementById("context-menu");
  if (!menu) return;
  menu.innerHTML = "";
  items.forEach(({ label, action }) => {
    const btn = document.createElement("button");
    btn.className = "context-menu__item";
    btn.textContent = label;
    btn.addEventListener("click", () => {
      action();
      hideContextMenu();
    });
    menu.appendChild(btn);
  });
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  menu.classList.remove("hidden");
};

/** Hide the context‑menu. */
export const hideContextMenu = () => {
  const menu = document.getElementById("context-menu");
  if (menu) menu.classList.add("hidden");
};

/**
 * Get the element‑specific colour CSS variable name.
 * @param {string} element
 * @returns {string}  CSS colour value
 */
export const getElementColor = (element) => {
  const map = {
    Fire: "var(--element-fire)",
    Water: "var(--element-water)",
    Earth: "var(--element-earth)",
    Air: "var(--element-air)",
    Light: "var(--element-light)",
    Dark: "var(--element-dark)",
    Neutral: "var(--element-neutral)",
  };
  return map[element] ?? map.Neutral;
};

/**
 * Get rarity colour CSS variable name.
 * @param {string} rarity
 * @returns {string}
 */
export const getRarityColor = (rarity) => {
  const map = {
    Common: "var(--rarity-common)",
    Uncommon: "var(--rarity-uncommon)",
    Rare: "var(--rarity-rare)",
    Legendary: "var(--rarity-legendary)",
  };
  return map[rarity] ?? map.Common;
};

/**
 * Map card type to an emoji icon.
 * @param {string} type
 * @returns {string}
 */
export const typeIcon = (type) => {
  const map = {
    Attack: "⚔️",
    Defense: "🛡️",
    Magic: "✨",
    Creature: "🐉",
    Artifact: "🏺",
  };
  return map[type] ?? "🃏";
};

/**
 * Map element to a raw hex colour (for canvas / gradients).
 * @param {string} element
 * @returns {string}
 */
export const elementHex = (element) => {
  const map = {
    Fire: "#f97316",
    Water: "#38bdf8",
    Earth: "#84cc16",
    Air: "#e2e8f0",
    Light: "#fde68a",
    Dark: "#a855f7",
    Neutral: "#6b7280",
  };
  return map[element] ?? map.Neutral;
};
