/* ── js/hud.js ── HUD DOM updates ── */

import { MathUtils } from './utils.js';

/* ══════════════════════════════════════════════════════════
   HUD — reads game state, writes to DOM elements
   ══════════════════════════════════════════════════════════ */
export class HUD {
  constructor() {
    // Cache DOM refs
    this.el = {
      hud:           document.getElementById('hud'),
      healthFill:    document.getElementById('health-fill'),
      healthText:    document.getElementById('health-text'),
      score:         document.getElementById('score-value'),
      wave:          document.getElementById('wave-value'),
      enemyCount:    document.getElementById('enemy-count'),
      crosshair:     document.getElementById('crosshair'),
      ammoValue:     document.getElementById('ammo-value'),
      missileValue:  document.getElementById('missile-value'),
      altitudeFill:  document.getElementById('altitude-fill'),
      altitudeValue: document.getElementById('altitude-value'),
      speedFill:     document.getElementById('speed-fill'),
      speedValue:    document.getElementById('speed-value'),
      compassStrip:  document.getElementById('compass-strip'),
      damageVignette: document.getElementById('damage-vignette'),
      waveBanner:    document.getElementById('wave-banner'),
    };

    this._damageTimer = 0;
  }

  show() {
    if (this.el.hud) this.el.hud.classList.remove('hidden');
  }

  hide() {
    if (this.el.hud) this.el.hud.classList.add('hidden');
  }

  /* ── Main update (call every frame) ── */
  update(drone, weapons, enemies, score, wave) {
    // Health bar
    const hpPct = (drone.health / drone.maxHealth) * 100;
    if (this.el.healthFill) {
      this.el.healthFill.style.width = hpPct + '%';
      // Color: green → yellow → red
      if (hpPct > 60) {
        this.el.healthFill.style.background = 'linear-gradient(90deg, #00ff41, #00cc33)';
      } else if (hpPct > 30) {
        this.el.healthFill.style.background = 'linear-gradient(90deg, #ffaa00, #ff8800)';
      } else {
        this.el.healthFill.style.background = 'linear-gradient(90deg, #ff3333, #cc0000)';
      }
    }
    if (this.el.healthText) {
      this.el.healthText.textContent = Math.ceil(drone.health);
    }

    // Score
    if (this.el.score) this.el.score.textContent = score;

    // Wave & enemies
    if (this.el.wave)       this.el.wave.textContent = wave;
    if (this.el.enemyCount) this.el.enemyCount.textContent = enemies.aliveCount;

    // Ammo
    if (this.el.ammoValue)    this.el.ammoValue.textContent = '∞';
    if (this.el.missileValue) this.el.missileValue.textContent = weapons.missileCount;

    // Crosshair firing state
    if (this.el.crosshair) {
      if (weapons.isFiring) {
        this.el.crosshair.classList.add('firing');
      } else {
        this.el.crosshair.classList.remove('firing');
      }
    }

    // Altitude meter (0–300 m)
    const altPct = MathUtils.clamp(drone.altitude / 300, 0, 1) * 100;
    if (this.el.altitudeFill)  this.el.altitudeFill.style.height = altPct + '%';
    if (this.el.altitudeValue) this.el.altitudeValue.textContent = Math.round(drone.altitude) + 'm';

    // Speed meter (0–60 → normalized)
    const spdPct = MathUtils.clamp(drone.speed / 60, 0, 1) * 100;
    if (this.el.speedFill)  this.el.speedFill.style.height = spdPct + '%';
    if (this.el.speedValue) this.el.speedValue.textContent = Math.round(drone.speed);

    // Compass (horizontal strip)
    if (this.el.compassStrip) {
      // yaw → degrees
      const yawDeg = ((drone.yaw * 180 / Math.PI) % 360 + 360) % 360;
      // Offset the strip: 1 full rotation = strip width / 2
      const offset = -(yawDeg / 360) * 200; // 200px = one rotation of compass
      this.el.compassStrip.style.transform = `translateX(${offset}px)`;
    }
  }

  /* ── Flash damage vignette ── */
  flashDamage() {
    if (!this.el.damageVignette) return;
    this.el.damageVignette.classList.remove('hidden');
    this.el.damageVignette.classList.add('flash');

    // Remove after animation
    clearTimeout(this._damageTimeout);
    this._damageTimeout = setTimeout(() => {
      this.el.damageVignette.classList.remove('flash');
      this.el.damageVignette.classList.add('hidden');
    }, 300);
  }

  /* ── Show wave banner ── */
  showWaveBanner(waveNum) {
    if (!this.el.waveBanner) return;
    this.el.waveBanner.textContent = `WAVE ${waveNum}`;
    this.el.waveBanner.classList.remove('hidden');
    this.el.waveBanner.classList.add('animate');

    setTimeout(() => {
      this.el.waveBanner.classList.remove('animate');
      this.el.waveBanner.classList.add('hidden');
    }, 2500);
  }
}
