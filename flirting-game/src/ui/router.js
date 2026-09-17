// Screen manager with View Transitions API (progressive enhancement):
// falls back to instant swap + CSS entrance animation when unsupported.

import { els } from './dom.js';

const PANELS = { setup: 'setupScreen', game: 'gameScreen', end: 'endScreen' };
let current = null;

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function showScreen(name) {
  if (current === name || !PANELS[name]) return;
  current = name;

  const swap = () => {
    for (const key of Object.keys(PANELS)) {
      els[PANELS[key]].classList.toggle('active', key === name);
    }
  };

  if (typeof document.startViewTransition === 'function' && !prefersReducedMotion()) {
    document.startViewTransition(swap);
  } else {
    swap();
  }
}

export function currentScreen() {
  return current;
}
