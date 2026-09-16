/* ── js/main.js ── Entry point: game loop, state management ── */

import { OakTree }          from './oak.js';
import { Casino }           from './casino.js';
import { BreedingLab }      from './breeding.js';
import { PopulationManager } from './population.js';
import { EventSystem }      from './events.js';
import { UiManager }        from './ui.js';
import { SeededRNG }        from './utils.js';
import {
  renderOak, renderHUD, renderMachines, renderPartners,
  renderFamily, renderUpgrades, renderRevenue, renderStaff,
  renderEventToast, addEventLogEntry, spawnCoinBurst,
} from './renderer.js';

/* ══════════════════════════════════════════════════════════
   SAVE KEY
   ══════════════════════════════════════════════════════════ */
const SAVE_KEY = 'oak_and_chaos_save_v1';

/* ══════════════════════════════════════════════════════════
   Game — central controller
   ══════════════════════════════════════════════════════════ */
class Game {
  constructor() {
    this.oak        = new OakTree();
    this.casino     = new Casino();
    this.breeding   = new BreedingLab();
    this.population = new PopulationManager();
    this.events     = new EventSystem();
    this.ui         = new UiManager();

    this.gameDay    = 1;
    this.dayPhase   = 'day';     // day | night
    this._dayTimer  = 0;         // tracks day/night cycle
    this._dayLen    = 60;        // seconds per day phase
    this._nightLen  = 30;

    this.speed      = 1;         // 1×, 2×, 5×
    this.paused     = false;
    this.won        = false;
    this._seededRNG = false;     // enable for reproducible testing

    this._prevTs    = 0;
    this._autoSaveTimer = 0;
    this._renderDirty  = true;
    this._rafId = null;

    // Seeded RNG — expose for testing
    this.rng = {
      useSeeded: false,
      _native: Math.random.bind(Math.random),
      _seeded: null,
      enable(seed) { this.useSeeded = true; this._seeded = new SeededRNG(seed || Date.now()); },
      disable() { this.useSeeded = false; this._seeded = null; },
      random() { return this.useSeeded ? this._seeded.next() : this._native(); },
    };

    // Machine spin callback → coin bursts
    this.casino.onSpinResult = (machine, result) => {
      if (result.win)     spawnCoinBurst(machine.id);
      if (result.jackpot) this.ui.showToast('weird', '🎰 JACKPOT!', `${machine.name} hit the jackpot!`);
    };

    // Event system listener
    this.events.onEvent(evt => {
      renderEventToast(evt);
      addEventLogEntry(evt);
    });
  }

  /* ═══════════ boot ═══════════ */
  start() {
    this.load();
    this.ui.init(this);

    this._renderAll();

    this._prevTs = performance.now();
    this._rafId = requestAnimationFrame(ts => this._loop(ts));
  }

  /* ═══════════ game loop ═══════════ */
  _loop(ts) {
    const rawDelta = (ts - this._prevTs) / 1000;
    this._prevTs = ts;

    if (!this.paused && !this.won) {
      const delta = Math.min(rawDelta, 0.25) * this.speed;
      this._tick(delta);
    }

    this._autoSaveTimer += rawDelta;
    if (this._autoSaveTimer >= 30) {
      this._autoSaveTimer = 0;
      this.save();
    }

    if (this._renderDirty || rawDelta > 0.06) {
      this._renderAll();
      this._renderDirty = false;
    }

    this._rafId = requestAnimationFrame(t => this._loop(t));
  }

  stop() {
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  /* ═══════════ tick ═══════════ */
  _tick(delta) {
    // Day-night cycle
    this._dayTimer += delta;
    const phaseLen = this.dayPhase === 'day' ? this._dayLen : this._nightLen;
    if (this._dayTimer >= phaseLen) {
      this._dayTimer -= phaseLen;
      if (this.dayPhase === 'day') {
        this.dayPhase = 'night';
      } else {
        this.dayPhase = 'day';
        this.gameDay++;
      }
    }

    // Sunlight factor: day = 1.0, night = 0.2
    const sunlight = this.dayPhase === 'day' ? 1.0 : 0.2;

    // Update systems
    const milestones = this.oak.update(delta, sunlight);
    if (milestones && milestones.length > 0) {
      for (const m of milestones) {
        this.ui.showAchievement(`${m.name} — ${m.desc}`);
      }
    }
    this.casino.update(delta, this.oak.height);
    this.breeding.update(delta);
    this.population.updateAges(delta);
    this.events.update(delta, this._makeGameState());

    // Win condition
    if (this.oak.height >= 100 && !this.won) {
      this.won = true;
      this.ui.showWinScreen(this.oak, this.casino, this.population);
    }

    this._renderDirty = true;
  }

  /* ═══════════ gameState object for events ═══════════ */
  _makeGameState() {
    return {
      oak: this.oak,
      casino: this.casino,
      breeding: this.breeding,
      population: this.population,
      _oakQuote: '',   // filled by zarghun_speaks event
    };
  }

  /* ═══════════ render all panels ═══════════ */
  _renderAll() {
    try {
      renderOak(this.oak, this.dayPhase);
      renderHUD(this.oak, this.casino, this.population, this.gameDay, this.dayPhase);
      renderMachines(this.casino);
      renderPartners(this.population.partners, this.oak);
      renderFamily(this.population.offspring);
      renderUpgrades(this.oak.upgrades, this.oak.dnaPoints);
      renderRevenue(this.casino);
      renderStaff(this.casino, this.population.offspring);
    } catch (e) {
      console.error('Render error:', e);
      this._renderDirty = true;
    }
  }

  /* ═══════════ public API used by UiManager ═══════════ */
  requestRender() { this._renderDirty = true; }

  setSpeed(s) {
    this.speed = s;
    this.paused = false;
  }

  togglePause() {
    this.paused = !this.paused;
  }

  /* ═══════════ save / load ═══════════ */
  save() {
    try {
      const data = {
        version: 1,
        timestamp: Date.now(),
        gameDay: this.gameDay,
        dayPhase: this.dayPhase,
        dayTimer: this._dayTimer,
        oak: this.oak.toJSON(),
        casino: this.casino.toJSON(),
        population: this.population.toJSON(),
        events: this.events.toJSON(),
      };
      localStorage.setItem(SAVE_KEY, JSON.stringify(data));
    } catch (e) {
      console.warn('Save failed:', e);
    }
  }

  load() {
    try {
      const raw = localStorage.getItem(SAVE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (!data) {
        console.warn('Load: empty data');
        return;
      }
      if (data.version !== 1) {
        console.warn(`Load: version mismatch (expected 1, got ${data.version}). Save may be incompatible.`);
        return;
      }

      this.gameDay    = data.gameDay || 1;
      this.dayPhase   = data.dayPhase || 'day';
      this._dayTimer  = data.dayTimer || 0;

      this.oak.loadJSON(data.oak);
      this.casino.loadJSON(data.casino);
      this.population.loadJSON(data.population);
      this.events.loadJSON(data.events);

      const offlineMs = Date.now() - (data.timestamp || Date.now());
      const offlineSec = Math.min(offlineMs / 1000, 7200);
      if (offlineSec > 5) {
        this._applyOfflineProgress(offlineSec);
        this.ui.showToast('success', '⏩ Offline Progress',
          `Simulated ${Math.floor(offlineSec / 60)} minutes while you were away.`);
      }
    } catch (e) {
      console.error('Load failed:', e);
      this.ui.showToast('error', '💾 Load Error', 'Failed to restore save data.');
    }
  }

  _applyOfflineProgress(seconds) {
    const steps = Math.floor(seconds / 2);
    for (let i = 0; i < steps; i++) {
      this.oak.update(2, 0.6);
      this.casino.update(2, this.oak.height);
      this.breeding.update(2);
      this.population.updateAges(2);
    }
  }

  /* ═══════════ reset / play again ═══════════ */
  resetGame() {
    this.save();
    this.stop();
    localStorage.removeItem(SAVE_KEY);
    location.reload();
  }
}

/* ══════════════════════════════════════════════════════════
   BOOT
   ══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  const game = new Game();
  game.start();

  // Expose for debug console
  window.__game = game;
});
