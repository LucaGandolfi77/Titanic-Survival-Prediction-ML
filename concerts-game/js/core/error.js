/* error.js — Error handling layer (S-8) */
window.G = window.G || {}

G.errorLog = []

/** Log and handle errors gracefully */
G.handleError = function (context, error) {
  var entry = { context: context, message: error ? error.message : 'Unknown error', time: Date.now() }
  G.errorLog.push(entry)
  console.error('[Concerts Chase]', context, error)
  G.emit('error', entry)
}

/** Safe execution wrapper - catches and logs errors */
G.safeExecute = function (fn, context, fallback) {
  try {
    return fn()
  } catch (e) {
    G.handleError(context || 'safeExecute', e)
    return fallback !== undefined ? fallback : undefined
  }
}

/** Error boundary wrapper for async operations */
G.errorBoundary = function (fn) {
  return function () {
    return G.safeExecute(fn, fn.name || 'anonymous', undefined)
  }
}
