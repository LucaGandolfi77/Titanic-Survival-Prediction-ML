import { AGES } from '../data/tech-tree.js';

export class HUDController {
  constructor() {
    this.resources = { food: 500, wood: 400, gold: 300, stone: 200 };
    this.population = { current: 0, max: 20 };
    this.age = 0;
    this.speed = 1;
    this.manpower = { current: 50, max: 100, overdraft: false };
  }

  updateResources(res) {
    if (res) this.resources = res;
    this._render();
  }

  updatePopulation(current, max) {
    this.population = { current, max };
    this._render();
  }

  updateAge(ageIndex) {
    this.age = ageIndex;
    this._render();
  }

  setSpeed(speed) {
    this.speed = speed;
    this._render();
  }

  setManpower(current, max, overdraft) {
    this.manpower = { current, max, overdraft };
    this._render();
  }

  setWeather(icon, effects) {
    this.weatherIcon = icon || '☀️';
    this.weatherEffects = effects || {};
    this._render();
  }

  setTimeRipple(active, state, cooldownPercent) {
    this.timeRippleActive = active;
    this.timeRippleState = state;
    this.timeRippleCooldown = cooldownPercent || 0;
    this._render();
  }

  setActivePlayer(player) {
    this.activePlayer = player;
    this._render();
  }

  _render() {
    const r = this.resources;
    const resEl = document.getElementById('hud-resources');
    if (resEl) {
      resEl.innerHTML = `
        <div class="res"><span class="res-icon">🌾</span>${Math.floor(r.food)}</div>
        <div class="res"><span class="res-icon">🪵</span>${Math.floor(r.wood)}</div>
        <div class="res"><span class="res-icon">🪙</span>${Math.floor(r.gold)}</div>
        <div class="res"><span class="res-icon">🪨</span>${Math.floor(r.stone)}</div>
      `;
    }

    const ageEl = document.getElementById('hud-age');
    if (ageEl) {
      const ages = ['Dark', 'Feudal', 'Castle', 'Imperial'];
      ageEl.textContent = `${ages[this.age] || 'Dark'} Age`;
    }

    const popEl = document.getElementById('hud-population');
    if (popEl) {
      popEl.textContent = `👥 ${this.population.current}/${this.population.max}`;
    }

    const mpEl = document.getElementById('hud-manpower');
    if (mpEl) {
      const mp = this.manpower;
      const overdraftClass = mp.overdraft ? ' style="color:#9b2226"' : '';
      mpEl.innerHTML = `⚔️ ${Math.floor(mp.current)}/${mp.max}${mp.overdraft ? ' ⚠️' : ''}`;
      mpEl.setAttribute('style', overdraftClass);
    }

    const wEl = document.getElementById('hud-weather');
    if (wEl) {
      wEl.textContent = `${this.weatherIcon || '☀️'}`;
    }

    const trEl = document.getElementById('hud-timeripple');
    if (trEl) {
      if (this.timeRippleActive) {
        const icons = { accelerate: '⚡', slow: '🕰️', freeze: '❄️' };
        trEl.innerHTML = `${icons[this.timeRippleState] || ''} TIME`;
        trEl.style.display = 'inline';
      } else if (this.timeRippleCooldown > 0) {
        trEl.innerHTML = '⏱️';
        trEl.title = `Cooldown: ${Math.round(this.timeRippleCooldown * 100)}%`;
        trEl.style.display = 'inline';
      } else {
        trEl.style.display = 'none';
      }
    }

    const apEl = document.getElementById('hud-active-player');
    if (apEl) {
      if (this.activePlayer) {
        const colors = { 1: '#357a38', 2: '#c43030' };
        apEl.textContent = `👤 P${this.activePlayer}`;
        apEl.style.color = colors[this.activePlayer] || '#fff';
        apEl.style.display = 'inline';
      } else {
        apEl.style.display = 'none';
      }
    }
  }
}

export const hud = new HUDController();
