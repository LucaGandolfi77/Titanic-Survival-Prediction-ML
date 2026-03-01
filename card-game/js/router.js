/**
 * Hash‑based client‑side router — CardForge
 * @module router
 */

import { setState, getState } from "./state.js";

const ROUTES = ["deck", "game"];

/**
 * Navigate to a route (adds # prefix if missing).
 * @param {string} route  "deck" | "game"
 */
export const navigate = (route) => {
  const clean = route.replace(/^#/, "");
  if (!ROUTES.includes(clean)) return;
  window.location.hash = `#${clean}`;
};

/**
 * Get the current route name.
 * @returns {string}
 */
export const getCurrentRoute = () => {
  const hash = window.location.hash.replace(/^#/, "");
  return ROUTES.includes(hash) ? hash : "deck";
};

/* ── Internal: apply route ─────────────────────────────── */
const applyRoute = () => {
  const route = getCurrentRoute();
  setState({ currentSection: route });

  /* Toggle sections */
  document.querySelectorAll(".app-section").forEach((sec) => {
    const isTarget = sec.id === `section-${route}`;
    if (isTarget) {
      sec.classList.add("active");
    } else {
      sec.classList.remove("active");
    }
  });

  /* Toggle nav links */
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle("active", link.dataset.route === route);
  });
};

/**
 * Initialise the router — call once on app start.
 */
export const initRouter = () => {
  window.addEventListener("hashchange", applyRoute);
  applyRoute();
};
