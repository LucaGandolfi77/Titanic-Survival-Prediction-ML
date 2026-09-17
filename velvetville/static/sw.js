const CACHE_NAME = 'velvetville-v1';
const STATIC_ASSETS = [
  '/',
  '/static/css/main.css',
  '/static/css/login.css',
  '/static/css/game.css',
  '/static/js/app.js',
  '/static/js/game-logic.js',
  '/static/js/chat.js',
  '/static/js/sw-register.js',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (request.url.includes('/api/')) {
    event.respondWith(networkFirst(request));
  } else if (request.url.includes('/static/js/') || request.url.includes('/static/css/')) {
    event.respondWith(staleWhileRevalidate(request));
  } else {
    event.respondWith(cacheFirst(request));
  }
});

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  const fetchPromise = fetch(request)
    .then((response) => {
      cache.put(request, response.clone());
      return response;
    })
    .catch(() => cached);
  return cached || fetchPromise;
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    cache.put(request, response.clone());
    return response;
  } catch (e) {
    return new Response('Offline', { status: 503, statusText: 'Offline' });
  }
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    cache.put(request, response.clone());
    return response;
  } catch (e) {
    const cached = await cache.match(request);
    if (cached) return cached;
    return new Response(
      JSON.stringify({ offline: true, message: 'You are offline' }),
      { headers: { 'Content-Type': 'application/json' } }
    );
  }
}

self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-votes') {
    event.waitUntil(syncPendingVotes());
  }
});

async function syncPendingVotes() {
  const votes = await getPendingVotes();
  for (const vote of votes) {
    try {
      await fetch('/api/vote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_username: vote.target, score: vote.score }),
      });
      await removePendingVote(vote.id);
    } catch (e) {
      console.warn('Sync failed for vote', vote.id);
    }
  }
}

async function getPendingVotes() {
  const data = await idbGet('pending-votes');
  return data || [];
}

async function idbGet(store) {
  return new Promise((resolve) => {
    const request = indexedDB.open('velvetville', 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('pending-votes')) {
        db.createObjectStore('pending-votes', { keyPath: 'id', autoIncrement: true });
      }
    };
    request.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction('pending-votes', 'readonly');
      const store = tx.objectStore('pending-votes');
      const getAll = store.getAll();
      getAll.onsuccess = () => resolve(getAll.result);
      getAll.onerror = () => resolve([]);
    };
    request.onerror = () => resolve([]);
  });
}

async function idbAdd(store, value) {
  return new Promise((resolve) => {
    const request = indexedDB.open('velvetville', 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(store)) {
        db.createObjectStore(store, { keyPath: 'id', autoIncrement: true });
      }
    };
    request.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction(store, 'readwrite');
      const st = tx.objectStore(store);
      st.add(value);
      tx.oncomplete = () => resolve();
    };
  });
}

async function idbDeleteByKey(store, key) {
  return new Promise((resolve) => {
    const request = indexedDB.open('velvetville', 1);
    request.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction(store, 'readwrite');
      const st = tx.objectStore(store);
      st.delete(key);
      tx.oncomplete = () => resolve();
    };
  });
}

async function removePendingVote(id) {
  await idbDeleteByKey('pending-votes', id);
}
