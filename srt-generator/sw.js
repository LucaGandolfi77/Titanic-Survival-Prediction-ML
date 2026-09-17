const SHELL_CACHE = 'srt-shell-v2';
const CDN_CACHE = 'srt-cdn-v2';
const APP_CACHE = 'srt-app-v2';

const SHELL_FILES = [
  '/',
  '/index.html',
  '/srt_editor.html',
  '/audio_to_srt_free.html',
  '/video_editor.html',
  '/video_to_audio.html',
  '/video_with_subs.html',
  '/css/shared.css',
  '/js/utils.js',
  '/manifest.webmanifest'
];

const CDN_FILES = [
  'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2',
  'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.8/dist/ffmpeg.min.js'
];

// Install — precache app shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => {
      return cache.addAll(SHELL_FILES);
    }).then(() => self.skipWaiting())
  );
});

// Activate — clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((k) => k !== SHELL_CACHE && k !== CDN_CACHE && k !== APP_CACHE)
          .map((k) => caches.delete(k))
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch — Stale-While-Revalidate for shell, Cache-First for CDN
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const request = event.request;

  // Skip non-GET requests
  if (request.method !== 'GET') return;

  // CDN resources — Cache-First
  if (url.hostname.includes('cdn.jsdelivr.net')) {
    event.respondWith(
      caches.open(CDN_CACHE).then(async (cache) => {
        const cached = await cache.match(request);
        if (cached) return cached;
        try {
          const response = await fetch(request);
          if (response.ok) {
            cache.put(request, response.clone());
          }
          return response;
        } catch {
          return cached;
        }
      })
    );
    return;
  }

  // App shell — Stale-While-Revalidate
  if (url.origin === self.location.origin || url.origin === location.origin) {
    event.respondWith(
      caches.open(SHELL_CACHE).then(async (cache) => {
        const cached = await cache.match(request);
        const fetchPromise = fetch(request)
          .then((response) => {
            if (response.ok) {
              cache.put(request, response.clone());
            }
            return response;
          })
          .catch(() => cached);
        return cached || fetchPromise;
      })
    );
    return;
  }
});
