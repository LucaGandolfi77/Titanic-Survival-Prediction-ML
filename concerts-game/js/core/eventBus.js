/* eventBus.js — G.on / G.emit / G.off for inter-module communication */
window.G = window.G || {}

G._listeners = {}

/** Subscribe to an event */
G.on = function (event, handler) {
  if (!G._listeners[event]) G._listeners[event] = []
  G._listeners[event].push(handler)
  return function () {
    G.off(event, handler)
  }
}

/** Emit an event with optional data */
G.emit = function (event, data) {
  if (!G._listeners[event]) return
  G._listeners[event].forEach(function (handler) {
    try {
      handler(data)
    } catch (e) {
      console.error('EventBus error on "' + event + '":', e)
    }
  })
}

/** Unsubscribe a specific handler from an event */
G.off = function (event, handler) {
  if (!G._listeners[event]) return
  G._listeners[event] = G._listeners[event].filter(function (h) {
    return h !== handler
  })
  if (G._listeners[event].length === 0) delete G._listeners[event]
}

/** Emit event once, auto-unsubscribing after first trigger */
G.once = function (event, handler) {
  return G.on(event, function (data) {
    handler(data)
    G.off(event, handler)
  })
}
