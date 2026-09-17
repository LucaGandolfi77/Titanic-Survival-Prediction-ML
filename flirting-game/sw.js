// Service Worker for Speed Crush PWA — versioned, with a layered strategy:
// - Navigations: Network-first with offline fallbacks
// - Same-origin assets: Stale-While-Revalidate
// - Cross-origin (Google Fonts): Cache-First
// Plus a message channel so the UI can trigger SKIP_WAITING on updates.

const VERSION = 'v4';
const STATIC_CACHE = `speed-crush-static-${VERSION}`;
const RUNTIME_CACHE = `speed-crush-runtime-${VERSION}`;

const PRECACHE_URLS = [
  './',
  './index.html',
  './offline.html',
  './style.css',
  './manifest.json',
  './dialogues.json',
  './icons/icon-192.svg',
  './icons/icon-512.svg',
  './icons/maskable-192.svg',
  './icons/maskable-512.svg',
  './src/main.js',
  './src/core/state.js',
  './src/core/timer.js',
  './src/core/storage.js',
  './src/core/haptics.js',
  './src/core/ambient.js',
  './src/core/difficulty.js',
  './src/core/webauthn.js',
  './src/core/queue.js',
  './src/core/i18n.js',
  './src/core/motion.js',
  './src/data/dialogues.js',
  './src/data/validate.js',
  './src/ai/capabilities.js',
  './src/ai/sentiment.js',
  './src/ai/generator.js',
  './src/ui/dom.js',
  './src/ui/router.js',
  './src/ui/toast.js',
  './src/features/setup/setup.js',
  './src/features/game/game.js',
  './src/features/ending/ending.js',
  './src/features/voice/voice.js',
  './src/features/stats/stats.js',
  './src/features/achievements/achievements.js',
  './src/features/notifications/notifications.js',
  './src/features/duet/duet.js',
  './src/features/story/story.js',
  './src/features/story/trailer.js'
];

// Install: precache the core resources, take over immediately.
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(STATIC_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

// Activate: clean up any cache from a previous version.
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== STATIC_CACHE && key !== RUNTIME_CACHE)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Only handle GET requests: cache.put() throws on non-GET.
  if (request.method !== 'GET') return;

  // Navigations (HTML) → Network-first with offline fallbacks.
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(STATIC_CACHE).then((cache) => cache.put('./index.html', copy));
          return response;
        })
        .catch(async () => {
          const cached = await caches.match('./index.html', { ignoreSearch: true });
          return cached || caches.match('./offline.html');
        })
    );
    return;
  }

  let requestUrl;
  try {
    requestUrl = new URL(request.url);
  } catch {
    return;
  }

  // Same-origin assets → Stale-While-Revalidate.
  if (requestUrl.origin === self.location.origin) {
    event.respondWith(
      caches.match(request).then((cached) => {
        const network = fetch(request)
          .then((response) => {
            if (response && response.ok) {
              const copy = response.clone();
              caches.open(RUNTIME_CACHE).then((cache) => cache.put(request, copy));
            }
            return response;
          })
          .catch(() => cached);
        return cached || network;
      })
    );
    return;
  }

  // Cross-origin (Google Fonts) → Cache-First (immutable content).
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          if (response && (response.ok || response.type === 'opaque')) {
            const copy = response.clone();
            caches.open(RUNTIME_CACHE).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => cached);
    })
  );
});

// Update flow: the UI toast posts SKIP_WAITING to activate the new version.
self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') self.skipWaiting();
});

// Background Sync: drain the offline milestone queue when connectivity
// returns, then notify the listening clients. The SW is registered as a
// module worker, so dynamic import() is available here.
self.addEventListener('sync', (event) => {
  if (event.tag !== 'sync-milestones') return;
  event.waitUntil(
    import('./src/core/queue.js')
      .then((queue) => queue.drainMilestones())
      .then((items) =>
        self.clients
          .matchAll({ includeUncontrolled: true })
          .then((clients) => {
            clients.forEach((client) =>
              client.postMessage({ type: 'milestones-synced', count: items.length })
            );
          })
      )
  );
});

// Push Notifications: rich notifications with quick-reply actions.
// The contextual reminder is scheduled by the page via Notification
// Triggers (experimental); 'play' focuses/opens the app and starts a story.
self.addEventListener('notificationclick', (event) => {
  const action = event.action;
  event.notification.close();
  if (action === 'dismiss') return;
  event.waitUntil(
    self.clients
      .matchAll({ type: 'window', includeUncontrolled: true })
      .then((clients) => {
        for (const client of clients) {
          if ('focus' in client) {
            client.focus();
            client.postMessage({ type: 'open-game' });
            return;
          }
        }
        return self.clients.openWindow('./index.html?action=play');
      })
  );
});

self.addEventListener('notificationclose', (event) => {
  // Analytics hook for the future cloud backend.
  console.log('Notification closed', event.notification.tag);
});
