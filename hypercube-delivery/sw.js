const CACHE_STATIC = 'hds-static-v1';
const CACHE_RUNTIME = 'hds-runtime-v1';
const CACHE_FALLBACK = 'hds-fallback-v1';

const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/css/variables.css',
    '/css/reset.css',
    '/css/ui.css',
    '/css/hud.css',
    '/css/animations.css',
    '/manifest.json',
    '/js/utils.js',
    '/js/hypercube.js',
];

const THREE_CDN_URLS = [
    'https://unpkg.com/three@0.158.0/build/three.module.js',
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_STATIC).then((cache) => {
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys
                    .filter((key) => key !== CACHE_STATIC && key !== CACHE_RUNTIME && key !== CACHE_FALLBACK)
                    .map((key) => caches.delete(key))
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    if (event.request.method !== 'GET') return;

    if (THREE_CDN_URLS.some((url) => event.request.url.includes(url))) {
        event.respondWith(cacheFirst(event.request));
        return;
    }

    event.respondWith(
        staleWhileRevalidate(event.request)
    );
});

async function staleWhileRevalidate(request) {
    const cache = await caches.open(CACHE_STATIC);
    const cached = await cache.match(request);

    const fetchPromise = fetch(request)
        .then((response) => {
            if (response && response.ok && response.status !== 0) {
                cache.put(request, response.clone());
            }
            return response;
        })
        .catch(() => {
            return caches.match('/index.html');
        });

    return cached || fetchPromise;
}

async function cacheFirst(request) {
    const cache = await caches.open(CACHE_FALLBACK);
    const cached = await cache.match(request);
    if (cached) return cached;

    try {
        const response = await fetch(request);
        if (response && response.ok) {
            cache.put(request, response.clone());
        }
        return response;
    } catch {
        return new Response('Offline', { status: 503, statusText: 'Offline' });
    }
}

self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-scores') {
        event.waitUntil(syncScores());
    }
});

async function syncScores() {
    const clients = await self.clients.matchAll({ type: 'window' });
    for (const client of clients) {
        client.postMessage({ type: 'SCORES_SYNC' });
    }
}

self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});
