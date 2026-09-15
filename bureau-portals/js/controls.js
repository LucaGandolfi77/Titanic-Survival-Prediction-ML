import { bus } from './event-bus.js';
import { MathUtils } from './utils.js';

export class Controls {
  constructor(player, bus) {
    this.player = player;
    this.bus = bus;
    this.enabled = false;

    // Store handler references for cleanup
    this._onKeyDown = null;
    this._onKeyUp = null;
    this._onPointerLockChange = null;
    this._onClick = null;

    this.setupEventListeners();
  }

  setupEventListeners() {
    // Keyboard
    this._onKeyDown = (e) => {
      if (!this.enabled) return;
      this.handleKeyDown(e);
    };

    this._onKeyUp = (e) => {
      if (!this.enabled) return;
      this.handleKeyUp(e);
    };

    // Pointer lock
    this._onPointerLockChange = () => {
      this.enabled = document.pointerLockElement === document.body;
    };

    // Click to request pointer lock
    this._onClick = () => {
      if (!document.pointerLockElement) {
        document.body.requestPointerLock();
      }
    };

    document.addEventListener('keydown', this._onKeyDown);
    document.addEventListener('keyup', this._onKeyUp);
    document.addEventListener('pointerlockchange', this._onPointerLockChange);
    document.documentElement.addEventListener('click', this._onClick);

    // Mobile controls
    this.setupMobileControls();
  }

  destroy() {
    if (this._onKeyDown) document.removeEventListener('keydown', this._onKeyDown);
    if (this._onKeyUp) document.removeEventListener('keyup', this._onKeyUp);
    if (this._onPointerLockChange)
      document.removeEventListener('pointerlockchange', this._onPointerLockChange);
    if (this._onClick) document.removeEventListener('click', this._onClick);
  }

  handleKeyDown(event) {
    const key = event.key.toLowerCase();
    this.player.keys[key] = true;

    // Special actions
    if (key === 'e') {
      this.tryInteract();
    }
    if (key === 'f') {
      this.toggleFlashlight();
    }
    if (key === 'i') {
      this.toggleInventory();
    }
    if (key === 'p') {
      this.togglePause();
    }
    if (key === 'escape') {
      this.togglePause();
    }
  }

  handleKeyUp(event) {
    const key = event.key.toLowerCase();
    this.player.keys[key] = false;
  }

  setupMobileControls() {
    // Mobile joystick (simplified)
    const canvas = document.getElementById('game-canvas');
    if (!canvas) return;

    const touchStart = { x: 0, y: 0 };
    let isTouching = false;

    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length > 0) {
        touchStart.x = e.touches[0].clientX;
        touchStart.y = e.touches[0].clientY;
        isTouching = true;
      }
    });

    canvas.addEventListener('touchmove', (e) => {
      if (isTouching && e.touches.length > 0) {
        const dx = e.touches[0].clientX - touchStart.x;
        const dy = e.touches[0].clientY - touchStart.y;

        const magnitude = Math.sqrt(dx * dx + dy * dy);
        if (magnitude > 50) {
          // Movement detected
          if (Math.abs(dx) > Math.abs(dy)) {
            if (dx > 0) {
              this.player.keys['d'] = true;
              this.player.keys['a'] = false;
            } else {
              this.player.keys['a'] = true;
              this.player.keys['d'] = false;
            }
          } else {
            if (dy > 0) {
              this.player.keys['s'] = true;
              this.player.keys['w'] = false;
            } else {
              this.player.keys['w'] = true;
              this.player.keys['s'] = false;
            }
          }
        }
      }
    });

    canvas.addEventListener('touchend', () => {
      isTouching = false;
      this.player.keys['w'] = false;
      this.player.keys['a'] = false;
      this.player.keys['s'] = false;
      this.player.keys['d'] = false;
    });
  }

  tryInteract() {
    this.bus.emit('input:interact', { position: this.player.position });
  }

  toggleFlashlight() {
    if (this.player.hasFlashlight) {
      this.player.flashlightIntensity = this.player.flashlightIntensity > 0 ? 0 : 1;
    }
  }

  toggleInventory() {
    this.bus.emit('ui:toggle-inventory');
  }

  togglePause() {
    this.bus.emit('input:pause');
  }

  setEnabled(enabled) {
    this.enabled = enabled;
  }
}
