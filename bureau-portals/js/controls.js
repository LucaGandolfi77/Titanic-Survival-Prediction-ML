export class Controls {
  constructor(player) {
    this.player = player;
    this.enabled = false;
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Keyboard
    document.addEventListener('keydown', (e) => {
      if (!this.enabled) return;
      this.handleKeyDown(e);
    });

    document.addEventListener('keyup', (e) => {
      if (!this.enabled) return;
      this.handleKeyUp(e);
    });

    // Mouse move handled in Player class via pointer lock
    
    // Pointer lock
    document.addEventListener('pointerlockchange', () => {
      this.enabled = document.pointerLockElement === document.documentElement;
    });

    // Click to request pointer lock
    document.documentElement.addEventListener('click', () => {
      if (!document.pointerLockElement) {
        document.documentElement.requestPointerLock();
      }
    });

    // Mobile controls
    this.setupMobileControls();
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

    let touchStart = { x: 0, y: 0 };
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
    if (window.game && window.game.puzzle) {
      const item = window.game.itemManager.checkInteraction(
        this.player.position,
        2.0
      );
      
      if (item) {
        window.game.puzzle.addItemToInventory(item);
        window.game.itemManager.collectItem(item);
        if (window.game.audio) {
          window.game.audio.playPickup();
        }
      }
    }
  }

  toggleFlashlight() {
    if (this.player.hasFlashlight) {
      this.player.flashlightIntensity = this.player.flashlightIntensity > 0 ? 0 : 1;
    }
  }

  toggleInventory() {
    if (window.game && window.game.hud) {
      window.game.hud.toggleInventoryView();
    }
  }

  togglePause() {
    if (window.game) {
      window.game.togglePause();
    }
  }

  setEnabled(enabled) {
    this.enabled = enabled;
  }
}