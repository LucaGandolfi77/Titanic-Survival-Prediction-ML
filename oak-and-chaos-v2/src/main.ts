import { OakTree } from './oak';
import { Casino } from './casino';
import { BreedingLab } from './breeding';
import { PopulationManager } from './population';
import { EventSystem } from './events';
import { UiManager } from './ui';
import { SeededRNG } from './utils';
import type { GameState } from '../types/game';
import {
  renderOak, renderHUD, renderMachines, renderPartners,
  renderFamily, renderUpgrades, renderRevenue, renderStaff,
  renderEventToast, addEventLogEntry, spawnCoinBurst,
} from './renderer';

const SAVE_KEY = 'oak_and_chaos_save_v2';

export class Game {
  oak: OakTree = new OakTree();
  casino: Casino = new Casino();
  breeding: BreedingLab = new BreedingLab();
  population: PopulationManager = new PopulationManager();
  events: EventSystem = new EventSystem();
  ui: UiManager = new UiManager();

  gameDay: number = 1;
  dayPhase: string = 'day';
  _dayTimer: number = 0;
  _dayLen: number = 60;
  _nightLen: number = 30;

  speed: number = 1;
  paused: boolean = false;
  won: boolean = false;
  _seededRNG: boolean = false;

  private _prevTs: number = 0;
  private _autoSaveTimer: number = 0;
  private _renderDirty: boolean = true;
  private _rafId: number | null = null;

  rng: {
    useSeeded: boolean;
    _native: () => number;
    _seeded: SeededRNG | null;
    enable(seed?: number): void;
    disable(): void;
    random(): number;
  } = {
    useSeeded: false,
    _native: Math.random,
    _seeded: null,
    enable(seed?: number) { this.useSeeded = true; this._seeded = new SeededRNG(seed || Date.now()); },
    disable() { this.useSeeded = false; this._seeded = null; },
    random() { return this.useSeeded && this._seeded ? this._seeded.next() : this._native(); },
  };

  constructor() {
    this.casino.onSpinResult = (machine, result, revenue): void => {
      if (result.win) spawnCoinBurst(machine.id);
      if (result.jackpot) this.ui.showToast('weird', '🎰 JACKPOT!', `${machine.name} hit the jackpot!`);
    };
    this.events.onEvent(evt => { renderEventToast(evt); addEventLogEntry(evt); });
  }

  start(): void {
    this.load();
    this.ui.init(this);
    this._renderAll();
    this._prevTs = performance.now();
    this._rafId = requestAnimationFrame(ts => this._loop(ts));
  }

  private _loop(ts: number): void {
    const rawDelta = (ts - this._prevTs) / 1000;
    this._prevTs = ts;
    if (!this.paused && !this.won) {
      const delta = Math.min(rawDelta, 0.25) * this.speed;
      this._tick(delta);
    }
    this._autoSaveTimer += rawDelta;
    if (this._autoSaveTimer >= 30) { this._autoSaveTimer = 0; this.save(); }
    if (this._renderDirty || rawDelta > 0.06) { this._renderAll(); this._renderDirty = false; }
    this._rafId = requestAnimationFrame(t => this._loop(t));
  }

  stop(): void {
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
  }

  private _tick(delta: number): void {
    this._dayTimer += delta;
    const phaseLen = this.dayPhase === 'day' ? this._dayLen : this._nightLen;
    if (this._dayTimer >= phaseLen) {
      this._dayTimer -= phaseLen;
      if (this.dayPhase === 'day') { this.dayPhase = 'night'; }
      else { this.dayPhase = 'day'; this.gameDay++; }
    }
    const sunlight = this.dayPhase === 'day' ? 1.0 : 0.2;
    const milestones = this.oak.update(delta, sunlight);
    if (milestones && milestones.length > 0) {
      for (const m of milestones) { this.ui.showAchievement(`${m.name} — ${m.desc}`); }
    }
    this.casino.update(delta, this.oak.height);
    this.breeding.update(delta);
    this.population.updateAges(delta);
    this.events.update(delta, this._makeGameState());
    if (this.oak.height >= 100 && !this.won) { this.won = true; this.ui.showWinScreen(this.oak, this.casino, this.population); }
    this._renderDirty = true;
  }

  private _makeGameState(): GameState {
    return { oak: this.oak, casino: this.casino, breeding: this.breeding, population: this.population, _oakQuote: '' };
  }

  private _renderAll(): void {
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

  requestRender(): void { this._renderDirty = true; }

  setSpeed(s: number): void { this.speed = s; this.paused = false; }

  togglePause(): void { this.paused = !this.paused; }

  save(): void {
    try {
      const data = {
        version: 2, timestamp: Date.now(), gameDay: this.gameDay, dayPhase: this.dayPhase, dayTimer: this._dayTimer,
        oak: this.oak.toJSON(), casino: this.casino.toJSON(), population: this.population.toJSON(), events: this.events.toJSON(),
      };
      localStorage.setItem(SAVE_KEY, JSON.stringify(data));
    } catch (e) { console.warn('Save failed:', e); }
  }

  load(): void {
    try {
      const raw = localStorage.getItem(SAVE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (!data) return;
      if (data.version !== 2) { console.warn(`Load: version mismatch (expected 2, got ${data.version})`); return; }
      this.gameDay = data.gameDay || 1; this.dayPhase = data.dayPhase || 'day'; this._dayTimer = data.dayTimer || 0;
      this.oak.loadJSON(data.oak); this.casino.loadJSON(data.casino); this.population.loadJSON(data.population); this.events.loadJSON(data.events);
      const offlineMs = Date.now() - (data.timestamp || Date.now());
      const offlineSec = Math.min(offlineMs / 1000, 7200);
      if (offlineSec > 5) { this._applyOfflineProgress(offlineSec); this.ui.showToast('success', '⏩ Offline Progress', `Simulated ${Math.floor(offlineSec / 60)} minutes while you were away.`); }
    } catch (e) { console.error('Load failed:', e); this.ui.showToast('error', '💾 Load Error', 'Failed to restore save data.'); }
  }

  private _applyOfflineProgress(seconds: number): void {
    const steps = Math.floor(seconds / 2);
    for (let i = 0; i < steps; i++) { this.oak.update(2, 0.6); this.casino.update(2, this.oak.height); this.breeding.update(2); this.population.updateAges(2); }
  }

  resetGame(): void { this.save(); this.stop(); localStorage.removeItem(SAVE_KEY); location.reload(); }
}

document.addEventListener('DOMContentLoaded', () => {
  const game = new Game();
  game.start();
  (window as any).__game = game;
});
