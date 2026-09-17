// Entry point: bootstraps features, registers the Service Worker (module
// worker, with the update flow + Background Sync message channel), and
// preloads dialogues + the optional neural model during idle.

import { initSetup } from './features/setup/setup.js';
import { initGame } from './features/game/game.js';
import { startGame } from './features/game/game.js';
import { loadDialogues } from './data/dialogues.js';
import { showToast } from './ui/toast.js';
import { storage } from './core/storage.js';
import { detectCapabilities } from './ai/capabilities.js';
import { ensureNeural } from './ai/sentiment.js';

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

  // Background Sync completion → notify the player.
  navigator.serviceWorker.addEventListener('message', (event) => {
    if (event.data?.type === 'milestones-synced') {
      showToast(`Milestones synced ✓ (${event.data.count})`, { duration: 3000 });
    }
  });

  navigator.serviceWorker
    .register('./sw.js', { type: 'module' })
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

function preloadAssets() {
  const warmUp = async () => {
    loadDialogues().catch(() => {});
    // Neural model already downloaded in a previous session → warm it up
    // from the browser cache so custom-line sentiment is neural instantly.
    if (storage.load().aiModelDownloaded && navigator.onLine) {
      const caps = await detectCapabilities();
      ensureNeural(caps).catch(() => {});
    }
  };
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => warmUp());
  } else {
    setTimeout(() => warmUp(), 1500);
  }
}

function boot() {
  initSetup();
  initGame();
  registerServiceWorker();
  preloadAssets();

  // Share Target (native share) + manifest shortcut deep-link.
  const params = new URLSearchParams(window.location.search);
  const sharedText = (params.get('text') || '').trim().slice(0, 300);
  if (params.get('action') === 'play' || sharedText) {
    if (sharedText) setState({ sharedPrompt: sharedText });
    startGame();
  }
}

// SW notificationclick action ('play') → start a new story.
navigator.serviceWorker?.addEventListener?.('message', (event) => {
  if (event.data?.type === 'open-game') {
    startGame();
  }
});

if (document.readyState === 'complete') {
  boot();
} else {
  window.addEventListener('load', boot);
}
