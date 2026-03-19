/* ============================================================
   Daisy Field — Service Worker
   Cache-first strategy for full offline capability.
   ============================================================ */

const CACHE_NAME = 'daisy-field-v1';

const ASSETS = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './daisy.js',
  './confetti.js',
  './manifest.json'
];

/* ---- Install: pre-cache all shell assets ---- */
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

/* ---- Activate: purge old caches ---- */
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_NAME)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

/* ---- Fetch: cache-first, fall back to network ---- */
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
