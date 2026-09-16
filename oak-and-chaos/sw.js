const CACHE_VERSION = 'v2';
const STATIC_CACHE = `oakchaos-static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `oakchaos-dynamic-${CACHE_VERSION}`;
const MAX_DYNAMIC_AGE = 7 * 24 * 60 * 60 * 1000;

const STATIC_ASSETS = [
  '/', '/index.html',
  '/css/variables.css', '/css/reset.css', '/css/layout.css',
  '/css/ui.css', '/css/animations.css',
  '/css/oak.css', '/css/casino.css', '/css/breeding.css',
  '/js/main.js', '/js/oak.js', '/js/casino.js', '/js/breeding.js',
  '/js/population.js', '/js/events.js', '/js/renderer.js',
  '/js/ui.js', '/js/utils.js',
  '/manifest.json', '/sw.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== 'GET') return;

  const isStatic = STATIC_ASSETS.some(
    (asset) => url.pathname === asset || url.pathname.endsWith(asset)
  );

  event.respondWith(
    isStatic
      ? caches.match(request).then((cached) => {
          if (cached) return cached;
          return fetch(request).then((response) => {
            const clone = response.clone();
            caches.open(STATIC_CACHE).then((cache) => cache.put(request, clone));
            return response;
          });
        })
      : caches.match(request).then((cached) => {
          const fetched = fetch(request)
            .then((response) => {
              const clone = response.clone();
              caches.open(DYNAMIC_CACHE).then(async (cache) => {
                const existing = await cache.match(request);
                if (!existing) {
                  cache.put(request, clone);
                }
              });
              return response;
            })
            .catch(() => cached);
          return cached || fetched;
        })
  );
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
