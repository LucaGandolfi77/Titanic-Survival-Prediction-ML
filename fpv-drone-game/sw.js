const CACHE_NAME = 'drone-strike-v1';

const STATIC_ASSETS = [
  './',
  './index.html',
  './css/variables.css',
  './css/reset.css',
  './css/ui.css',
  './css/hud.css',
  './css/controls.css',
  './js/utils.js',
  './js/main.js',
  './js/drone.js',
  './js/world.js',
  './js/scene.js',
  './js/collision.js',
  './js/enemies.js',
  './js/weapons.js',
  './js/effects.js',
  './js/audio.js',
  './js/controls.js',
  './js/hud.js',
  './js/ui.js',
  './manifest.json',
];

const CDN_CACHE = 'drone-strike-cdn-v1';
const CDN_URLS = [
  'https://unpkg.com/three@0.158.0/build/three.module.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS)).then(() => {
      return caches.open(CDN_CACHE).then((cache) => {
        return Promise.allSettled(
          CDN_URLS.map((url) => fetch(url).then((r) => cache.put(url, r)).catch(() => {}))
        );
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((n) => n !== CACHE_NAME && n !== CDN_CACHE)
          .map((n) => caches.delete(n))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (url.hostname.includes('unpkg.com') || url.hostname.includes('cdn.jsdelivr.net')) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request)
          .then((response) => {
            const clone = response.clone();
            caches.open(CDN_CACHE).then((cache) => cache.put(request, clone));
            return response;
          })
          .catch(() => caches.match(request));
      })
    );
    return;
  }

  if (request.method !== 'GET') return;

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => cached || caches.match('./index.html'));
    })
  );
});

self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-scores') {
    event.waitUntil(Promise.resolve());
  }
});
