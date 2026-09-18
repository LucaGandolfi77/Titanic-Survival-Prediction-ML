const CACHE_NAME = 'subway-runner-v1';
const CDN_CACHE = 'subway-runner-cdn-v1';

const STATIC_ASSETS = [
  './',
  './index.html',
  './css/variables.css',
  './css/reset.css',
  './css/hud.css',
  './css/ui.css',
  './css/controls.css',
  './js/main.js',
  './js/utils.js',
  './js/scene.js',
  './js/track.js',
  './js/player.js',
  './js/controls.js',
  './js/obstacles.js',
  './js/collectibles.js',
  './js/collision.js',
  './js/effects.js',
  './js/audio.js',
  './js/hud.js',
  './js/ui.js',
  './js/characters.js',
  './manifest.webmanifest'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME && k !== CDN_CACHE)
            .map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);

  if (url.hostname.includes('unpkg.com') || url.hostname.includes('cdn.jsdelivr.net') || url.hostname.includes('cdnjs.cloudflare.com') || url.hostname.includes('fonts.googleapis.com') || url.hostname.includes('fonts.gstatic.com')) {
    event.respondWith(
      caches.open(CDN_CACHE).then(cache =>
        cache.match(request).then(cached => {
          const fetchPromise = cache.put(request, fetch(request).then(resp => {
            if (resp.ok) return resp.clone();
            return resp;
          })).catch(() => cached);

          return cached || fetchPromise;
        })
      )
    );
    return;
  }

  event.respondWith(
    caches.match(request).then(cached => {
      if (cached) return cached;
      return fetch(request).then(response => {
        if (response.ok && url.origin === self.location.origin) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
        }
        return response;
      }).catch(() => {
        if (request.mode === 'navigate') {
          return caches.match('./index.html');
        }
        return new Response('Offline', { status: 503 });
      });
    })
  );
});
