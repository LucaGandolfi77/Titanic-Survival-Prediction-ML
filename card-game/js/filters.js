/**
 * Search & filter logic — CardForge
 * @module filters
 */

import { CARD_DATABASE } from "./database.js";
import { getState, setState } from "./state.js";
import { debounce } from "./utils.js";

/* ── Filter definitions ────────────────────────────────── */
const TYPES    = ["Attack", "Defense", "Magic", "Creature", "Artifact"];
const ELEMENTS = ["Fire", "Water", "Earth", "Air", "Light", "Dark", "Neutral"];
const RARITIES = ["Common", "Uncommon", "Rare", "Legendary"];

/* ── Apply filters to the database ─────────────────────── */

/**
 * Filter the card database using the active filter state.
 * @returns {import('./database.js').Card[]}
 */
export const applyFilters = () => {
  const f = getState().activeFilters;

  return CARD_DATABASE.filter((card) => {
    /* Text search */
    if (f.search) {
      const q = f.search.toLowerCase();
      const haystack = `${card.name} ${card.description} ${card.tags.join(" ")}`.toLowerCase();
      if (!haystack.includes(q)) return false;
    }

    /* Type */
    if (f.types && f.types.length > 0 && !f.types.includes(card.type)) return false;

    /* Element */
    if (f.elements && f.elements.length > 0 && !f.elements.includes(card.element)) return false;

    /* Rarity */
    if (f.rarities && f.rarities.length > 0 && !f.rarities.includes(card.rarity)) return false;

    /* Cost range */
    if (f.costMin != null && card.cost < f.costMin) return false;
    if (f.costMax != null && card.cost > f.costMax) return false;

    /* Power range */
    if (f.powerMin != null) {
      if (card.power === null || card.power < f.powerMin) return false;
    }
    if (f.powerMax != null) {
      if (card.power === null || card.power > f.powerMax) return false;
    }

    return true;
  });
};

/* ── Build filter UI ───────────────────────────────────── */

/**
 * Build all filter groups inside #filter-groups.
 * @param {Function} onFilterChange  callback after any filter change
 */
export const buildFilterUI = (onFilterChange) => {
  const container = document.getElementById("filter-groups");
  if (!container) return;
  container.innerHTML = "";

  /* Search input is separate in HTML — wire it up */
  const searchInput = document.getElementById("card-search");
  if (searchInput) {
    const onSearch = debounce((e) => {
      const filters = getState().activeFilters;
      filters.search = e.target.value.trim();
      setState({ activeFilters: { ...filters } });
      onFilterChange();
    }, 200);
    searchInput.addEventListener("input", onSearch);

    /* Add clear button if not already present */
    if (!searchInput.parentElement.querySelector(".search-clear")) {
      const clearBtn = document.createElement("button");
      clearBtn.className = "search-clear";
      clearBtn.textContent = "×";
      clearBtn.type = "button";
      clearBtn.addEventListener("click", () => {
        searchInput.value = "";
        const filters = getState().activeFilters;
        filters.search = "";
        setState({ activeFilters: { ...filters } });
        onFilterChange();
      });
      searchInput.parentElement.appendChild(clearBtn);
    }
  }

  /* Chip‑based filter groups */
  const chipGroup = (title, values, filterKey) =>
    buildChipGroup(container, title, values, filterKey, onFilterChange);

  chipGroup("Type", TYPES, "types");
  chipGroup("Element", ELEMENTS, "elements");
  chipGroup("Rarity", RARITIES, "rarities");

  /* Range sliders */
  buildRangeSlider(container, "Cost", 1, 10, "costMin", "costMax", onFilterChange);
  buildRangeSlider(container, "Power", 0, 10, "powerMin", "powerMax", onFilterChange);

  /* Reset button */
  const resetBtn = document.getElementById("reset-filters");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      setState({ activeFilters: {} });
      if (searchInput) searchInput.value = "";
      rebuildChipStates(container);
      resetSliders(container);
      onFilterChange();
    });
  }
};

/* ── Chip group builder ────────────────────────────────── */

/**
 * @param {HTMLElement} parent
 * @param {string} title
 * @param {string[]} values
 * @param {string} filterKey
 * @param {Function} onChange
 */
const buildChipGroup = (parent, title, values, filterKey, onChange) => {
  const group = document.createElement("div");
  group.className = "filter-group";
  group.dataset.filterKey = filterKey;

  const header = document.createElement("button");
  header.className = "filter-group__header";
  header.innerHTML = `
    <span>${title}</span>
    <span class="filter-group__chevron">▾</span>
  `;
  header.addEventListener("click", () => group.classList.toggle("collapsed"));
  group.appendChild(header);

  const body = document.createElement("div");
  body.className = "filter-group__body";

  values.forEach((val) => {
    const chip = document.createElement("label");
    chip.className = "filter-chip";
    chip.textContent = val;
    chip.addEventListener("click", () => {
      chip.classList.toggle("active");
      const filters = getState().activeFilters;
      const active = [...body.querySelectorAll(".filter-chip.active")].map((c) => c.textContent);
      filters[filterKey] = active.length ? active : [];
      setState({ activeFilters: { ...filters } });
      onChange();
    });
    body.appendChild(chip);
  });

  group.appendChild(body);
  parent.appendChild(group);
};

/* ── Range slider builder ──────────────────────────────── */

/**
 * @param {HTMLElement} parent
 * @param {string} label
 * @param {number} min
 * @param {number} max
 * @param {string} minKey
 * @param {string} maxKey
 * @param {Function} onChange
 */
const buildRangeSlider = (parent, label, min, max, minKey, maxKey, onChange) => {
  const group = document.createElement("div");
  group.className = "filter-group";

  const header = document.createElement("button");
  header.className = "filter-group__header";
  header.innerHTML = `<span>${label} Range</span><span class="filter-group__chevron">▾</span>`;
  header.addEventListener("click", () => group.classList.toggle("collapsed"));
  group.appendChild(header);

  const body = document.createElement("div");
  body.className = "filter-group__body";
  body.style.flexDirection = "column";

  /* Min slider */
  const rangeMin = document.createElement("div");
  rangeMin.className = "filter-range";
  const minLabel = document.createElement("div");
  minLabel.className = "filter-range__label";
  minLabel.innerHTML = `<span>Min</span><span class="range-val" data-key="${minKey}">${min}</span>`;
  const minInput = document.createElement("input");
  minInput.type = "range";
  minInput.min = min;
  minInput.max = max;
  minInput.value = min;
  minInput.dataset.key = minKey;
  rangeMin.appendChild(minLabel);
  rangeMin.appendChild(minInput);
  body.appendChild(rangeMin);

  /* Max slider */
  const rangeMax = document.createElement("div");
  rangeMax.className = "filter-range";
  const maxLabel = document.createElement("div");
  maxLabel.className = "filter-range__label";
  maxLabel.innerHTML = `<span>Max</span><span class="range-val" data-key="${maxKey}">${max}</span>`;
  const maxInput = document.createElement("input");
  maxInput.type = "range";
  maxInput.min = min;
  maxInput.max = max;
  maxInput.value = max;
  maxInput.dataset.key = maxKey;
  rangeMax.appendChild(maxLabel);
  rangeMax.appendChild(maxInput);
  body.appendChild(rangeMax);

  /* Listeners */
  const update = () => {
    const lo = parseInt(minInput.value, 10);
    const hi = parseInt(maxInput.value, 10);
    minLabel.querySelector(".range-val").textContent = lo;
    maxLabel.querySelector(".range-val").textContent = hi;
    const filters = getState().activeFilters;
    filters[minKey] = lo;
    filters[maxKey] = hi;
    setState({ activeFilters: { ...filters } });
    onChange();
  };
  minInput.addEventListener("input", update);
  maxInput.addEventListener("input", update);

  group.appendChild(body);
  parent.appendChild(group);
};

/* ── Helpers for reset ─────────────────────────────────── */

const rebuildChipStates = (container) => {
  container.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
};

const resetSliders = (container) => {
  container.querySelectorAll("input[type=range]").forEach((inp) => {
    inp.value = inp.dataset.key?.includes("Min") || inp.dataset.key?.includes("cost") && inp.dataset.key?.includes("Min")
      ? inp.min
      : inp.max;
    const valSpan = container.querySelector(`.range-val[data-key="${inp.dataset.key}"]`);
    if (valSpan) valSpan.textContent = inp.value;
  });
};
