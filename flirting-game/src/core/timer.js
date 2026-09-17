// rAF-based timer engine, pause-safe by design:
// - the countdown only accumulates time while the tab is visible
//   (long frames are clamped, hidden time is never counted);
// - one-shot delays pause while hidden and resume on return;
// - onFinish can only fire once per run (no double-timeout race).

export class TimerEngine {
  constructor({ onTick, onFinish } = {}) {
    this.onTick = onTick || (() => {});
    this.onFinish = onFinish || (() => {});
    this._raf = 0;
    this._last = 0;
    this._remaining = 0;
    this._duration = 0;
    this._running = false;
    this._delayTimer = 0;
    this._delayRemaining = 0;
    this._delayCb = null;
    this._onVisibility = () => this._handleVisibility();
    document.addEventListener('visibilitychange', this._onVisibility);
  }

  /** Start a countdown of `durationSeconds` seconds. */
  start(durationSeconds) {
    this.cancel();
    this._duration = Math.max(0, durationSeconds * 1000);
    this._remaining = this._duration;
    this._running = true;
    this._last = performance.now();
    this._scheduleFrame();
  }

  /** Stop the countdown. Safe to call multiple times. */
  cancel() {
    this._running = false;
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = 0;
    }
  }

  /** Schedule `cb` after `ms` ms; pauses while the tab is hidden. */
  delay(ms, cb) {
    this.cancelDelay();
    this._delayCb = cb;
    this._delayRemaining = Math.max(0, ms);
    this._armDelay();
  }

  /** Run the pending delay callback immediately. Returns true if one was pending. */
  skipDelay() {
    if (!this._delayCb) return false;
    clearTimeout(this._delayTimer);
    const cb = this._delayCb;
    this._delayCb = null;
    this._delayRemaining = 0;
    this._delayTimer = 0;
    if (cb) cb();
    return true;
  }

  cancelDelay() {
    clearTimeout(this._delayTimer);
    this._delayCb = null;
    this._delayRemaining = 0;
    this._delayTimer = 0;
  }

  destroy() {
    this.cancel();
    this.cancelDelay();
    document.removeEventListener('visibilitychange', this._onVisibility);
  }

  _scheduleFrame() {
    this._raf = requestAnimationFrame(() => this._tick());
  }

  _tick() {
    if (!this._running) return;
    const now = performance.now();
    let delta = now - this._last;
    this._last = now;
    // Long main-thread jank or a tab switch should never punish the player.
    if (delta > 100) delta = 16;
    this._remaining = Math.max(0, this._remaining - delta);
    this.onTick(
      this._remaining / 1000,
      this._duration > 0 ? this._remaining / this._duration : 0
    );
    if (this._remaining <= 0) {
      // Guard: onFinish fires exactly once per run.
      this._running = false;
      this.onFinish();
      return;
    }
    this._scheduleFrame();
  }

  _armDelay() {
    if (!this._delayCb) return;
    clearTimeout(this._delayTimer);
    this._delayTimer = setTimeout(() => {
      this._delayTimer = 0;
      const cb = this._delayCb;
      this._delayCb = null;
      this._delayRemaining = 0;
      if (cb) cb();
    }, this._delayRemaining);
  }

  _handleVisibility() {
    if (document.hidden) {
      // Pause the pending response delay (the countdown rAF stops on its own).
      if (this._delayCb) {
        clearTimeout(this._delayTimer);
        this._delayTimer = 0;
      }
    } else {
      // Reset the countdown baseline so hidden time is never counted.
      this._last = performance.now();
      if (this._delayCb && this._delayRemaining > 0 && !this._delayTimer) {
        this._armDelay();
      }
    }
  }
}
