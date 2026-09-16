/* canvasGame.js — Base class for canvas minigames (S-7) */
window.G = window.G || {}

/**
 * Base class for all canvas-based minigames.
 * Subclass and override init/update/render/destroy as needed.
 */
G.CanvasGame = function (canvasElement, onComplete) {
  /** @type {HTMLCanvasElement} */
  this.canvas = canvasElement
  /** @type {CanvasRenderingContext2D} */
  this.ctx = canvasElement.getContext('2d')
  /** @type {Function} */
  this.onComplete = onComplete
  /** @type {boolean} */
  this.active = false
  /** @type {number} */
  this.lastTime = 0
  /** @type {number} */
  this.animFrame = null
  /** @type {Object} */
  this.state = {}
}

/** Initialize the game - override in subclass */
G.CanvasGame.prototype.init = function () {
  this.active = true
  this.lastTime = performance.now()
  this._loop = this._loop.bind(this)
  this._loop()
}

/** Main loop - override update/render in subclass */
G.CanvasGame.prototype._loop = function () {
  if (!this.active) return
  var now = performance.now()
  var dt = Math.min((now - this.lastTime) / 1000, 0.1)
  this.lastTime = now
  this.update(dt)
  this.render(this.ctx)
  this.animFrame = requestAnimationFrame(this._loop)
}

/** Update game logic per frame - override */
G.CanvasGame.prototype.update = function (dt) {}

/** Render game per frame - override */
G.CanvasGame.prototype.render = function (ctx) {}

/** Handle input events - override */
G.CanvasGame.prototype.handleInput = function (event) {}

/** End the game and call onComplete */
G.CanvasGame.prototype.end = function (result) {
  this.active = false
  if (this.animFrame) cancelAnimationFrame(this.animFrame)
  if (this.onComplete) this.onComplete(result)
}

/** Destroy and clean up */
G.CanvasGame.prototype.destroy = function () {
  this.active = false
  if (this.animFrame) cancelAnimationFrame(this.animFrame)
  this.canvas = null
  this.ctx = null
  this.onComplete = null
  this.state = {}
}
