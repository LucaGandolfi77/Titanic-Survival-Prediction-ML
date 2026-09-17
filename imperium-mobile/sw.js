const CACHE_NAME = 'imperium-v4';
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/css/variables.css',
  '/css/reset.css',
  '/css/ui.css',
  '/css/hud.css',
  '/css/controls.css',
  '/css/animations.css',
  '/js/utils/helpers.js',
  '/js/utils/hex-math.js',
  '/js/utils/storage.js',
  '/js/data/factions.js',
  '/js/data/buildings.js',
  '/js/data/units.js',
  '/js/data/tech-tree.js',
  '/js/data/map-templates.js',
  '/js/map.js',
  '/js/citizen.js',
  '/js/building.js',
  '/js/resource.js',
  '/js/manpower.js',
  '/js/diplomacy.js',
  '/js/mercenary.js',
  '/js/unit.js',
  '/js/combat.js',
  '/js/time-ripple.js',
  '/js/weather.js',
  '/js/dna-mutation.js',
  '/js/hero.js',
  '/js/territory.js',
  '/js/campaign.js',
  '/js/ai.js',
  '/js/fog.js',
  '/js/render/camera.js',
  '/js/render/renderer.js',
  '/js/render/particles.js',
  '/js/main.js',
  '/js/ui/screens.js',
  '/js/ui/hud.js',
  '/js/ui/notifications.js',
  '/js/assets.js',
  '/manifest.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) {
        const fetched = fetch(event.request).then((res) => {
          if (res && res.status === 200) {
            const clone = res.clone();
            caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
          }
          return res;
        }).catch(() => cached);
        return fetched;
      }
      return fetch(event.request).then((res) => {
        if (res && res.status === 200) {
          const clone = res.clone();
          caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
        }
        return res;
      }).catch(() => new Response('Offline', { status: 503, statusText: 'Service Unavailable' }));
    })
  );
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
