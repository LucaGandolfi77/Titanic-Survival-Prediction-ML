export class HUD {
  constructor() {
    this.scoreText = document.getElementById('score-text');
    this.coinText = document.getElementById('coin-text');
    this.distText = document.getElementById('dist-text');
    this.multiplierEl = document.getElementById('hud-multiplier');
    this.comboEl = document.getElementById('hud-combo');
    this.powerupEl = document.getElementById('hud-powerup');
    this.damageFlash = document.getElementById('damage-flash');
    this.hudEl = document.getElementById('hud');
  }

  show() {
    this.hudEl.classList.remove('hidden');
  }

  hide() {
    this.hudEl.classList.add('hidden');
  }

  updateScore(score) {
    this.scoreText.textContent = Math.floor(score).toLocaleString();
  }

  updateCoins(coins) {
    this.coinText.textContent = coins.toLocaleString();
  }

  updateDistance(meters) {
    this.distText.textContent = Math.floor(meters) + 'm';
  }

  setMultiplier(value) {
    if (value > 1) {
      this.multiplierEl.textContent = 'x' + value;
      this.multiplierEl.classList.add('active');
    } else {
      this.multiplierEl.classList.remove('active');
    }
  }

  showCombo(combo) {
    if (combo >= 5) {
      this.comboEl.textContent = combo + ' COMBO!';
      this.comboEl.classList.add('active');
      clearTimeout(this._comboTimeout);
      this._comboTimeout = setTimeout(() => {
        this.comboEl.classList.remove('active');
      }, 1500);
    }
  }

  updatePowerups(activePowerups) {
    this.powerupEl.innerHTML = '';
    for (const [type, data] of Object.entries(activePowerups)) {
      const timeLeft = Math.max(0, (data.endTime - performance.now()) / 1000);
      if (timeLeft <= 0) continue;
      const div = document.createElement('div');
      div.className = 'powerup-indicator';
      const icons = { magnet: 'MAG', multiplier: 'x2', shield: 'SHD', jetpack: 'JET' };
      div.innerHTML = `<span class="powerup-icon">${icons[type] || '?'}</span><span class="powerup-timer">${timeLeft.toFixed(1)}s</span>`;
      this.powerupEl.appendChild(div);
    }
  }

  flashDamage() {
    this.damageFlash.classList.add('active');
    clearTimeout(this._damageTimeout);
    this._damageTimeout = setTimeout(() => {
      this.damageFlash.classList.remove('active');
    }, 400);
  }

  hideAll() {
    this.hide();
    this.powerupEl.innerHTML = '';
    this.comboEl.classList.remove('active');
    this.multiplierEl.classList.remove('active');
    this.damageFlash.classList.remove('active');
  }
}
