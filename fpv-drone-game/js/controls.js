/* ── js/controls.js ── Keyboard, mouse (pointer-lock) & touch joystick input ── */

import { MathUtils } from './utils.js';

/* ╔═══════════════════════════════════════════════════════════╗
   ║  VirtualJoystick — touch-based analog stick               ║
   ╚═══════════════════════════════════════════════════════════╝ */
class VirtualJoystick {
  _boundTouchStart = this._onTouchStart.bind(this);
  _boundTouchMove = this._onTouchMove.bind(this);
  _boundTouchEnd = this._onTouchEnd.bind(this);

  constructor(outerEl, knobEl) {
    this.outer  = outerEl;
    this.knob   = knobEl;
    this.x      = 0;
    this.y      = 0;
    this.active = false;
    this._touchId = null;

    this.outer.addEventListener('touchstart', this._boundTouchStart, { passive: false });
    document.addEventListener('touchmove',    this._boundTouchMove,  { passive: false });
    document.addEventListener('touchend',     this._boundTouchEnd);
    document.addEventListener('touchcancel',  this._boundTouchEnd);
  }

  _onTouchStart(e) {
    if (this._touchId !== null) return;
    e.preventDefault();
    const t = e.changedTouches[0];
    this._touchId = t.identifier;
    this.active = true;
    this._update(t);
  }

  _onTouchMove(e) {
    for (const t of e.changedTouches) {
      if (t.identifier === this._touchId) {
        e.preventDefault();
        this._update(t);
        return;
      }
    }
  }

  _onTouchEnd(e) {
    for (const t of e.changedTouches) {
      if (t.identifier === this._touchId) {
        this._touchId = null;
        this.active = false;
        this.x = 0;
        this.y = 0;
        this.knob.style.transform = 'translate(-50%, -50%)';
        return;
      }
    }
  }

  _update(touch) {
    const rect = this.outer.getBoundingClientRect();
    const cx   = rect.left + rect.width  / 2;
    const cy   = rect.top  + rect.height / 2;
    let dx = touch.clientX - cx;
    let dy = touch.clientY - cy;
    const maxR = rect.width / 2;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > maxR) {
      dx *= maxR / dist;
      dy *= maxR / dist;
    }
    this.x =  dx / maxR;
    this.y = -dy / maxR;
    this.knob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
  }

  dispose() {
    this.outer.removeEventListener('touchstart', this._boundTouchStart);
    document.removeEventListener('touchmove',    this._boundTouchMove);
    document.removeEventListener('touchend',     this._boundTouchEnd);
    document.removeEventListener('touchcancel',  this._boundTouchEnd);
  }
}

/* ╔═══════════════════════════════════════════════════════════╗
   ║  InputController — unified input from all sources          ║
   ╚═══════════════════════════════════════════════════════════╝ */
export class InputController {
  constructor(canvas) {
    this.canvas = canvas;

    this._keys = {};

    this._mouseDX = 0;
    this._mouseDY = 0;
    this._mouseLeft  = false;
    this._mouseRight = false;
    this._pointerLocked = false;

    this._smoothPitch = 0;
    this._smoothRoll  = 0;

    this._leftJoystick  = null;
    this._rightJoystick = null;

    this._touchFire    = false;
    this._touchMissile = false;
    this._plasmaPressed = false;

    this._pausePressed   = false;
    this._missilePressed = false;

    this._setupKeyboard();
    this._setupMouse();
    this._setupTouch();
  }

  _setupKeyboard() {
    window.addEventListener('keydown', (e) => {
      this._keys[e.code] = true;
      if (e.code === 'Escape') this._pausePressed = true;
      if (e.code === 'KeyE')   this._missilePressed = true;
      if (e.code === 'KeyQ')   this._plasmaPressed = true;
    });
    window.addEventListener('keyup', (e) => {
      this._keys[e.code] = false;
    });
  }

  _setupMouse() {
    this.canvas.addEventListener('click', () => {
      if (!this._pointerLocked) {
        this.canvas.requestPointerLock();
      }
    });

    document.addEventListener('pointerlockchange', () => {
      this._pointerLocked = document.pointerLockElement === this.canvas;
    });

    document.addEventListener('mousemove', (e) => {
      if (!this._pointerLocked) return;
      this._mouseDX += e.movementX;
      this._mouseDY += e.movementY;
    });

    document.addEventListener('mousedown', (e) => {
      if (!this._pointerLocked) return;
      if (e.button === 0) this._mouseLeft  = true;
      if (e.button === 2) { this._mouseRight = true; this._missilePressed = true; }
    });
    document.addEventListener('mouseup', (e) => {
      if (e.button === 0) this._mouseLeft  = false;
      if (e.button === 2) this._mouseRight = false;
    });

    this.canvas.addEventListener('contextmenu', e => e.preventDefault());
  }

  _setupTouch() {
    const ljOuter = document.getElementById('joystick-left');
    const ljKnob  = document.getElementById('joystick-left-knob');
    const rjOuter = document.getElementById('joystick-right');
    const rjKnob  = document.getElementById('joystick-right-knob');

    if (ljOuter && ljKnob) {
      this._leftJoystick = new VirtualJoystick(ljOuter, ljKnob);
    }
    if (rjOuter && rjKnob) {
      this._rightJoystick = new VirtualJoystick(rjOuter, rjKnob);
    }

    const fireBtn    = document.getElementById('btn-fire');
    const missileBtn = document.getElementById('btn-missile');

    if (fireBtn) {
      fireBtn.addEventListener('touchstart', (e) => { e.preventDefault(); this._touchFire = true; });
      fireBtn.addEventListener('touchend',   ()  => { this._touchFire = false; });
      fireBtn.addEventListener('touchcancel', () => { this._touchFire = false; });
    }
    if (missileBtn) {
      missileBtn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        this._touchMissile = true;
        this._missilePressed = true;
      });
      missileBtn.addEventListener('touchend',   ()  => { this._touchMissile = false; });
      missileBtn.addEventListener('touchcancel', () => { this._touchMissile = false; });
    }
  }

  requestPointerLock() {
    this.canvas.requestPointerLock();
  }

  getInput() {
    const kb = this._keys;
    const lj = this._leftJoystick;
    const rj = this._rightJoystick;

    let throttle = 0;
    if (kb['KeyW']) throttle += 1;
    if (kb['KeyS']) throttle -= 1;
    if (lj) throttle = MathUtils.clamp(throttle + lj.y, -1, 1);

    let yaw = 0;
    if (kb['KeyA']) yaw -= 1;
    if (kb['KeyD']) yaw += 1;
    if (lj) yaw = MathUtils.clamp(yaw + lj.x, -1, 1);

    const sensitivity = 0.05;
    if (this._pointerLocked) {
      this._smoothPitch += this._mouseDY * sensitivity;
      this._smoothRoll  += this._mouseDX * sensitivity;
    }

    this._smoothPitch *= 0.90;
    this._smoothRoll  *= 0.90;

    let pitch = MathUtils.clamp(this._smoothPitch, -1, 1);
    let roll  = MathUtils.clamp(this._smoothRoll,  -1, 1);

    if (kb['ArrowUp'])   pitch = -1;
    if (kb['ArrowDown']) pitch =  1;
    if (kb['ArrowLeft'])  roll  = -1;
    if (kb['ArrowRight']) roll  =  1;

    if (rj) {
      pitch = MathUtils.clamp(pitch + rj.y, -1, 1);
      roll  = MathUtils.clamp(roll  + rj.x, -1, 1);
    }

    const firePrimary  = this._mouseLeft || this._touchFire || !!kb['Space'];
    const fireMissile  = this._missilePressed;
    const firePlasma   = this._plasmaPressed;
    const boost        = !!kb['ShiftLeft'] || !!kb['ShiftRight'];
    const pause        = this._pausePressed;

    this._mouseDX = 0;
    this._mouseDY = 0;
    this._pausePressed   = false;
    this._missilePressed = false;
    this._plasmaPressed  = false;

    return { throttle, pitch, roll, yaw, firePrimary, fireMissile, firePlasma, boost, pause };
  }
}
