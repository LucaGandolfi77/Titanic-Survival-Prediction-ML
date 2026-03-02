import { Vector2 } from './utils.js';

export class Controls {
  constructor() {
    this.joyVec = new Vector2(0, 0); // Normalized -1 to 1
    this.buttons = {
      pass: false,
      shoot: false,
      tackle: false,
      switch: false
    };
    this.shootChargeStart = 0;
    this.isCharging = false;
    
    // Touch tracking
    this.touchIdJoy = null;
    this.joyOrigin = new Vector2(0, 0);
    this.joyCurrent = new Vector2(0, 0);
    
    // Desktop keys
    this.keys = {};

    this.setupListeners();
  }

  setupListeners() {
    // Touch
    const zone = document.getElementById('joystick-zone');
    zone.addEventListener('touchstart', (e) => this.handleJoyStart(e), { passive: false });
    zone.addEventListener('touchmove', (e) => this.handleJoyMove(e), { passive: false });
    zone.addEventListener('touchend', (e) => this.handleJoyEnd(e), { passive: false });

    // Buttons
    this.bindButton('btn-pass', 'pass');
    this.bindButton('btn-tackle', 'tackle');
    this.bindButton('btn-switch', 'switch');
    
    // Shoot (special handling for hold)
    const btnShoot = document.getElementById('btn-shoot');
    btnShoot.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.buttons.shoot = true;
      this.isCharging = true;
      this.shootChargeStart = performance.now();
      btnShoot.classList.add('charging');
    }, { passive: false });

    btnShoot.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.buttons.shoot = false;
      this.isCharging = false;
      btnShoot.classList.remove('charging');
      // Trigger shoot logic in main loop via checking 'wasReleased' flag or similar
      // For now, main loop will check isCharging transition
    }, { passive: false });

    // Keyboard
    window.addEventListener('keydown', (e) => {
      this.keys[e.code] = true;
      if (e.code === 'Space') this.buttons.pass = true;
      if (e.code === 'KeyF') {
        if (!this.isCharging) {
          this.isCharging = true;
          this.shootChargeStart = performance.now();
          // Simulate button UI
          document.getElementById('btn-shoot').classList.add('charging');
        }
        this.buttons.shoot = true;
      }
      if (e.code === 'KeyT') this.buttons.tackle = true;
      if (e.code === 'Tab') {
        e.preventDefault();
        this.buttons.switch = true;
      }
      this.updateKeyboardVec();
    });

    window.addEventListener('keyup', (e) => {
      this.keys[e.code] = false;
      if (e.code === 'Space') this.buttons.pass = false;
      if (e.code === 'KeyF') {
        this.buttons.shoot = false;
        this.isCharging = false;
        document.getElementById('btn-shoot').classList.remove('charging');
      }
      if (e.code === 'KeyT') this.buttons.tackle = false;
      if (e.code === 'Tab') this.buttons.switch = false;
      this.updateKeyboardVec();
    });
  }

  bindButton(id, key) {
    const btn = document.getElementById(id);
    btn.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.buttons[key] = true;
      btn.classList.add('active');
    }, { passive: false });
    
    btn.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.buttons[key] = false;
      btn.classList.remove('active');
    }, { passive: false });
  }

  handleJoyStart(e) {
    e.preventDefault();
    const touch = e.changedTouches[0];
    this.touchIdJoy = touch.identifier;
    this.joyOrigin.set(touch.clientX, touch.clientY);
    this.joyCurrent.set(touch.clientX, touch.clientY);
    
    // Move visual joystick to touch point
    const outer = document.getElementById('joystick-outer');
    outer.style.left = touch.clientX + 'px';
    outer.style.top = touch.clientY + 'px'; // Actually bottom is fixed, need absolute
    // Using simple fixed inline styles overrides CSS bottom/left
    outer.style.bottom = 'auto'; // Reset CSS
    outer.classList.add('active');
  }

  handleJoyMove(e) {
    e.preventDefault();
    for (let i = 0; i < e.changedTouches.length; i++) {
      if (e.changedTouches[i].identifier === this.touchIdJoy) {
        const touch = e.changedTouches[i];
        this.joyCurrent.set(touch.clientX, touch.clientY);
        
        // Vector math
        const delta = new Vector2(this.joyCurrent.x - this.joyOrigin.x, this.joyCurrent.y - this.joyOrigin.y);
        const dist = delta.mag();
        const maxDist = 40; // Joystick radius
        
        if (dist > maxDist) {
          delta.normalize().multiplyScalar(maxDist);
        }
        
        // Update knob visual
        const knob = document.getElementById('joystick-knob');
        knob.style.transform = `translate(${delta.x}px, ${delta.y}px)`;
        
        // Normalize output (-1 to 1)
        this.joyVec.set(delta.x / maxDist, delta.y / maxDist);
      }
    }
  }

  handleJoyEnd(e) {
    e.preventDefault();
    for (let i = 0; i < e.changedTouches.length; i++) {
      if (e.changedTouches[i].identifier === this.touchIdJoy) {
        this.touchIdJoy = null;
        this.joyVec.set(0, 0);
        document.getElementById('joystick-outer').classList.remove('active');
        document.getElementById('joystick-knob').style.transform = `translate(0px, 0px)`;
      }
    }
  }

  updateKeyboardVec() {
    // Only use if no touch active
    if (this.touchIdJoy !== null) return;
    
    let x = 0;
    let y = 0;
    if (this.keys['ArrowUp'] || this.keys['KeyW']) y -= 1;
    if (this.keys['ArrowDown'] || this.keys['KeyS']) y += 1;
    if (this.keys['ArrowLeft'] || this.keys['KeyA']) x -= 1;
    if (this.keys['ArrowRight'] || this.keys['KeyD']) x += 1;
    
    this.joyVec.set(x, y);
    if (this.joyVec.mag() > 0) this.joyVec.normalize();
  }

  getOutput() {
    return this.joyVec.clone();
  }
}