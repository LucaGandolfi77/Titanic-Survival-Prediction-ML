export class InputController {
  constructor() {
    this.keys = {};
    this.swipe = { left: false, right: false, up: false, down: false };
    this._touchStartX = 0;
    this._touchStartY = 0;
    this._touchStartTime = 0;
    this._swipeThreshold = 30;
    this._swipeTimeLimit = 300;
    this._swipeHandled = false;

    window.addEventListener('keydown', (e) => this._onKeyDown(e));
    window.addEventListener('keyup', (e) => this._onKeyUp(e));
    window.addEventListener('touchstart', (e) => this._onTouchStart(e), { passive: false });
    window.addEventListener('touchmove', (e) => this._onTouchMove(e), { passive: false });
    window.addEventListener('touchend', (e) => this._onTouchEnd(e), { passive: false });

    this._setupMobileButtons();
  }

  _onKeyDown(e) {
    this.keys[e.code] = true;
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Space'].includes(e.code)) {
      e.preventDefault();
    }
  }

  _onKeyUp(e) {
    this.keys[e.code] = false;
  }

  _onTouchStart(e) {
    if (e.target.closest('#hud-mobile-controls') || e.target.closest('.btn') || e.target.closest('.character-card')) return;
    const touch = e.touches[0];
    this._touchStartX = touch.clientX;
    this._touchStartY = touch.clientY;
    this._touchStartTime = Date.now();
    this._swipeHandled = false;
  }

  _onTouchMove(e) {
    if (this._swipeHandled) return;
    const touch = e.touches[0];
    const dx = touch.clientX - this._touchStartX;
    const dy = touch.clientY - this._touchStartY;
    const elapsed = Date.now() - this._touchStartTime;

    if (elapsed > this._swipeTimeLimit) return;

    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    if (absDx > this._swipeThreshold || absDy > this._swipeThreshold) {
      this._swipeHandled = true;
      if (absDx > absDy) {
        this.swipe[dx > 0 ? 'right' : 'left'] = true;
      } else {
        this.swipe[dy > 0 ? 'down' : 'up'] = true;
      }
    }
  }

  _onTouchEnd() {
    this._swipeHandled = false;
  }

  _setupMobileButtons() {
    const bind = (id, action) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('touchstart', (e) => { e.stopPropagation(); action('down'); }, { passive: false });
      el.addEventListener('touchend', (e) => { e.stopPropagation(); action('up'); }, { passive: false });
    };

    bind('touch-left', (state) => { this.keys['ArrowLeft'] = state === 'down'; });
    bind('touch-right', (state) => { this.keys['ArrowRight'] = state === 'down'; });
    bind('touch-jump', (state) => { this.keys['ArrowUp'] = state === 'down'; });
    bind('touch-slide', (state) => { this.keys['ArrowDown'] = state === 'down'; });
  }

  getInput() {
    const input = {
      left: false,
      right: false,
      jump: false,
      slide: false,
      pause: false
    };

    if (this.keys['ArrowLeft'] || this.keys['KeyA']) input.left = true;
    if (this.keys['ArrowRight'] || this.keys['KeyD']) input.right = true;
    if (this.keys['ArrowUp'] || this.keys['KeyW'] || this.keys['Space']) input.jump = true;
    if (this.keys['ArrowDown'] || this.keys['KeyS']) input.slide = true;
    if (this.keys['Escape'] || this.keys['KeyP']) input.pause = true;

    if (this.swipe.left) { input.left = true; this.swipe.left = false; }
    if (this.swipe.right) { input.right = true; this.swipe.right = false; }
    if (this.swipe.up) { input.jump = true; this.swipe.up = false; }
    if (this.swipe.down) { input.slide = true; this.swipe.down = false; }

    return input;
  }

  consumePause() {
    const was = this.keys['Escape'] || this.keys['KeyP'];
    this.keys['Escape'] = false;
    this.keys['KeyP'] = false;
    return was;
  }
}
