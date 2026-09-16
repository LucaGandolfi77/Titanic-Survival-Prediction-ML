/* ===== Skate Manager service worker — offline support ===== */
const CACHE = 'skate-manager-v1';
const ASSETS = [
  './index.html',
  './manifest.webmanifest',
  './icon.svg',
  './css/variables.css',
  './css/reset.css',
  './css/layout.css',
  './css/roster.css',
  './css/rink.css',
  './css/ui.css',
  './css/animations.css',
  './js/main.js',
  './js/config.js',
  './js/i18n.js',
  './js/types.js',
  './js/state.js',
  './js/utils.js',
  './js/skaters.js',
  './js/formations.js',
  './js/squad.js',
  './js/competitions.js',
  './js/sponsors.js',
  './js/music.js',
  './js/rink-renderer.js',
  './js/minigame.js',
  './js/minigame/engine.js',
  './js/minigame/scoring.js',
  './js/ui.js',
  './js/ui/screens.js',
  './js/ui/feedback.js',
  './js/ui/modals.js',
  './js/ui/skater-card.js',
  './js/ui/refresh.js',
  './js/ui/panels/overview.js',
  './js/ui/panels/squad.js',
  './js/ui/panels/market.js',
  './js/ui/panels/calendar.js',
  './js/ui/panels/sponsors.js',
  './js/ui/panels/standings.js',
  './js/ui/panels/results.js',
  './js/ui/panels/stats.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((hit) => {
      if (hit) return hit;
      return fetch(e.request).then((res) => {
        if (res.ok && e.request.method === 'GET') {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return res;
      });
    })
  );
});
