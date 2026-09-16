/* visualizer.js — Audio visualizer for concert screen */
window.G = window.G || {}

G.VIS_BARS = 32

/** Initialize visualizer canvas on concert screen */
G.initVisualizer = function () {
  var el = document.getElementById('concert-content')
  if (!el) return
  var canvas = document.createElement('canvas')
  canvas.id = 'viz-canvas'
  canvas.width = 400
  canvas.height = 60
  canvas.style.cssText = 'width:100%;max-width:400px;margin:8px auto;display:block;border-radius:8px'
  var label = document.createElement('div')
  label.style.cssText = 'text-align:center;font-size:0.7rem;color:var(--gray-light);margin-bottom:4px'
  label.textContent = '♪ Live Audio'
  el.insertBefore(label, el.firstChild)
  el.insertBefore(canvas, el.firstChild)
  if (G.analyser) {
    G.startVisualizer(canvas)
  }
}

/** Remove visualizer canvas */
G.destroyVisualizer = function () {
  G.stopVisualizer()
  var canvas = document.getElementById('viz-canvas')
  if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas)
}
