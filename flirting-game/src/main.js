// Entry point: bootstraps features, registers the Service Worker with an
// update flow (toast + SKIP_WAITING), and preloads dialogues during idle.

import { initSetup } from './features/setup/setup.js';
import { initGame } from './features/game/game.js';
import { startGame } from './features/game/game.js';
import { loadDialogues } from './data/dialogues.js';
import { showToast } from './ui/toast.js';

function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;

  let hasController = Boolean(navigator.serviceWorker.controller);
  let refreshing = false;

  navigator.serviceWorker.addEventListener('controllerchange', () => {
    // First install claims the page: no reload needed.
    if (!hasController) {
      hasController = true;
      return;
    }
    if (refreshing) return;
    refreshing = true;
    window.location.reload();
  });

  navigator.serviceWorker
    .register('./sw.js')
    .then((reg) => {
      reg.addEventListener('updatefound', () => {
        const installing = reg.installing;
        if (!installing) return;
        installing.addEventListener('statechange', () => {
          if (installing.state === 'installed' && navigator.serviceWorker.controller) {
            showToast('New version available', {
              actionLabel: 'Reload',
              onAction: () => installing.postMessage({ type: 'SKIP_WAITING' }),
              duration: 0 // sticky until tapped
            });
          }
        });
      });
    })
    .catch((err) => console.error('Service Worker registration failed', err));
}

function preloadDialogues() {
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      loadDialogues().catch(() => {});
    });
  } else {
    setTimeout(() => {
      loadDialogues().catch(() => {});
    }, 1500);
  }
}

function boot() {
  initSetup();
  initGame();
  registerServiceWorker();
  preloadDialogues();

  // Manifest shortcut deep-link: Speed Crush/?action=play starts instantly.
  if (new URLSearchParams(window.location.search).get('action') === 'play') {
    startGame();
  }
}

if (document.readyState === 'complete') {
  boot();
} else {
  window.addEventListener('load', boot);
}
