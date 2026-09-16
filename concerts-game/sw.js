/* sw.js — Service Worker for offline mode */
var CACHE_NAME = 'concerts-chase-v1'
var APP_SHELL = [
  './',
  './index.html',
  './css/variables.css',
  './css/base.css',
  './css/hud.css',
  './css/screens.css',
  './css/minigames.css',
  './css/animations.css',
  './js/utils.js',
  './js/data/data.js',
  './js/core/state.js',
  './js/core/audio.js',
  './js/core/eventBus.js',
  './js/screens/screens.js',
  './js/screens/hud.js',
  './js/screens/map.js',
  './js/screens/calendar.js',
  './js/screens/booking.js',
  './js/screens/results.js',
  './js/screens/main.js',
  './js/games/mg-gossip.js',
  './js/games/mg-cipher.js',
  './js/games/mg-auction.js',
  './js/games/mg-puzzle.js',
  './js/games/mg-streetteam.js',
  './js/games/cg-swift.js',
  './js/games/cg-styles.js',
  './js/games/cg-maneskin.js',
  './js/games/cg-eilish.js',
  './js/games/cg-beyonce.js',
  './js/games/cg-kpop.js',
  './js/games/cg-latin.js',
  './js/games/cg-rock.js',
  './js/games/concert.js',
  './js/games/merch.js',
  './js/games/photo.js',
  './js/games/travel.js',
  './js/games/organise.js',
  './js/data/friends.js'
]

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function (cache) {
      return cache.addAll(APP_SHELL)
    })
  )
  self.skipWaiting()
})

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (names) {
      return Promise.all(
        names.filter(function (n) { return n !== CACHE_NAME }).map(function (n) { return caches.delete(n) })
      )
    })
  )
  self.clients.claim()
})

self.addEventListener('fetch', function (event) {
  if (event.request.method !== 'GET') return
  event.respondWith(
    caches.match(event.request).then(function (cached) {
      return (
        cached ||
        fetch(event.request).catch(function () {
          return caches.match('./')
        })
      )
    })
  )
})
