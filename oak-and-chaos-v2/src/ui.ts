import { ROLES } from './breeding';
import type { OakTree } from './oak';
import type { Casino } from './casino';
import type { PopulationManager } from './population';
import type { BreedingLab } from './breeding';
import type { EventSystem } from './events';
import type { Offspring, MachineState, SpinResult } from '../types/game';

interface GameRef {
  oak: OakTree;
  breeding: BreedingLab;
  population: PopulationManager;
  events: EventSystem;
  ui: UiManager;
  gameDay: number;
  dayPhase: string;
  speed: number;
  paused: boolean;
  won: boolean;
  requestRender(): void;
  setSpeed(s: number): void;
  togglePause(): void;
  save(): void;
  stop(): void;
  resetGame(): void;
  casino: Casino;
}

export class UiManager {
  private gameRef?: GameRef;
  private _konamiTriggered: boolean = false;

  init(game: GameRef): void {
    this.gameRef = game;
    this._konamiTriggered = false;
    this._bindTabs();
    this._bindSpeedButtons();
    this._bindOakActions();
    this._bindBreeding();
    this._bindSpend();
    this._bindModal();
    this._delegateClicks();
    this._bindKeyboard();
    this._bindKonami();
  }

  private _bindTabs(): void {
    const leftTabs = document.getElementById('left-tabs');
    if (leftTabs) {
      leftTabs.addEventListener('click', e => {
        const btn = (e.target as HTMLElement).closest('.tab');
        if (!btn) return;
        this._switchTab('left-panel', btn.getAttribute('data-tab') || '');
        btn.parentElement?.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
      });
    }
    const rightTabs = document.getElementById('right-tabs');
    if (rightTabs) {
      rightTabs.addEventListener('click', e => {
        const btn = (e.target as HTMLElement).closest('.tab');
        if (!btn) return;
        this._switchTab('right-panel', btn.getAttribute('data-tab') || '');
        btn.parentElement?.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
      });
    }
  }

  private _switchTab(panelId: string, tabId: string): void {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    const target = document.getElementById(tabId);
    if (target) target.classList.add('active');
  }

  private _bindSpeedButtons(): void {
    const btns = document.querySelectorAll<HTMLButtonElement>('.speed-btn[data-speed]');
    btns.forEach(b => {
      b.addEventListener('click', () => {
        const spd = parseInt(b.dataset.speed || '1', 10);
        this.gameRef?.setSpeed(spd);
        btns.forEach(bb => bb.classList.remove('active'));
        b.classList.add('active');
        const pp = document.getElementById('btn-pause');
        if (pp) pp.classList.remove('active');
      });
    });
    const pauseBtn = document.getElementById('btn-pause');
    if (pauseBtn) {
      pauseBtn.addEventListener('click', () => {
        this.gameRef?.togglePause();
        const pb = document.getElementById('btn-pause');
        if (pb) { pb.classList.toggle('active'); pb.textContent = this.gameRef?.paused ? '▶️' : '⏸'; }
      });
    }
  }

  private _bindOakActions(): void {
    const growBtn = document.getElementById('btn-grow');
    if (growBtn) {
      growBtn.addEventListener('click', () => {
        if (!this.gameRef) return;
        const result = this.gameRef.oak.grow();
        if (result) { this._flashButton('btn-grow', 'success'); this._showSpeech('Growing... 🌱', 1500); }
        else { this._flashButton('btn-grow', 'fail'); }
      });
    }
    const acornBtn = document.getElementById('btn-acorn');
    if (acornBtn) {
      acornBtn.addEventListener('click', () => {
        if (!this.gameRef) return;
        const ok = this.gameRef.oak.produceAcorn();
        if (ok) { this._flashButton('btn-acorn', 'success'); this._showSpeech('🌰 Acorn produced!', 1200); }
        else { this._flashButton('btn-acorn', 'fail'); }
      });
    }
    const medBtn = document.getElementById('btn-meditate');
    if (medBtn) {
      medBtn.addEventListener('click', () => {
        if (!this.gameRef) return;
        this.gameRef.oak.meditate();
        const btn = document.getElementById('btn-meditate');
        if (btn) {
          if (this.gameRef.oak.isMeditating) { btn.classList.add('active'); btn.innerHTML = '🧘 Stop <span class="cost">active</span>'; this._showSpeech('Entering deep meditation... 🧬', 2000); }
          else { btn.classList.remove('active'); btn.innerHTML = '🧘 Meditate'; }
        }
      });
    }
  }

  private _bindBreeding(): void {
    const filters = document.querySelector('.breed-filters');
    if (!filters) return;
    filters.addEventListener('click', e => {
      const btn = (e.target as HTMLElement).closest('.filter-btn');
      if (!btn) return;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.getAttribute('data-filter') || 'all';
      document.querySelectorAll<HTMLDivElement>('.partner-card').forEach(card => {
        card.style.display = filter === 'all' ? '' : (card.classList.contains(`type-${filter}`) ? '' : 'none');
      });
    });
  }

  private _bindSpend(): void {
    const nutrients = document.getElementById('btn-nutrients');
    if (nutrients) nutrients.addEventListener('click', () => {
      if (!this.gameRef) return;
      if (this.gameRef.casino.totalCoins >= 50) { this.gameRef.casino.totalCoins -= 50; this.gameRef.oak.addBuff('energyGen', 60, 2); this.showToast('success', '🧪 Nutrients', 'Energy regen ×2 for 60s!'); }
      else { this.showToast('error', '💰 Not enough', 'Need 50 coins'); }
    });
    const fertilizer = document.getElementById('btn-fertilizer');
    if (fertilizer) fertilizer.addEventListener('click', () => {
      if (!this.gameRef) return;
      if (this.gameRef.casino.totalCoins >= 200) { this.gameRef.casino.totalCoins -= 200; this.gameRef.oak.addBuff('growthSpeed', 120, 1.5); this.showToast('success', '🌿 Fertilizer', 'Growth ×1.5 for 120s!'); }
      else { this.showToast('error', '💰 Not enough', 'Need 200 coins'); }
    });
    const bribe = document.getElementById('btn-bribe');
    if (bribe) bribe.addEventListener('click', () => {
      if (!this.gameRef) return;
      if (this.gameRef.casino.totalCoins >= 300) { this.gameRef.casino.totalCoins -= 300; this.gameRef.events.temporarilyBlock('tax_inspector', 300); this.showToast('success', '🤫 Inspector Bribed', 'No inspectors for 5 minutes!'); }
      else { this.showToast('error', '💰 Not enough', 'Need 300 coins'); }
    });
    const unl5 = document.getElementById('btn-unlock-m5');
    if (unl5) unl5.addEventListener('click', () => {
      if (!this.gameRef) return;
      if (this.gameRef.casino.totalCoins >= 500) { const m = this.gameRef.casino.machines.find(m => m.id === 5); if (m && m.isLocked) { this.gameRef.casino.totalCoins -= 500; m.isLocked = false; this.showToast('success', '🎰 Unlocked!', 'Machine 5 is now active!'); } }
      else { this.showToast('error', '💰 Not enough', 'Need 500 coins'); }
    });
    const unl6 = document.getElementById('btn-unlock-m6');
    if (unl6) unl6.addEventListener('click', () => {
      if (!this.gameRef) return;
      if (this.gameRef.casino.totalCoins >= 500) { const m = this.gameRef.casino.machines.find(m => m.id === 6); if (m && m.isLocked) { this.gameRef.casino.totalCoins -= 500; m.isLocked = false; this.showToast('success', '🎰 Unlocked!', 'Machine 6 is now active!'); } }
      else { this.showToast('error', '💰 Not enough', 'Need 500 coins'); }
    });
  }

  private _bindModal(): void {
    const mc = document.getElementById('modal-close');
    if (mc) mc.addEventListener('click', () => this.closeModal());
    const mo = document.getElementById('modal-overlay');
    if (mo) mo.addEventListener('click', e => { if (e.target === mo) this.closeModal(); });
    const pa = document.getElementById('btn-play-again');
    if (pa) pa.addEventListener('click', () => this.gameRef?.resetGame());
  }

  openModal(html: string): void {
    const mb = document.getElementById('modal-body'); if (mb) mb.innerHTML = html;
    const mo = document.getElementById('modal-overlay'); if (mo) mo.classList.remove('hidden');
  }

  closeModal(): void {
    const mo = document.getElementById('modal-overlay'); if (mo) mo.classList.add('hidden');
  }

  private _bindKeyboard(): void {
    document.addEventListener('keydown', (e) => {
      if (!this.gameRef) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === 'Space' || e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        this.gameRef.togglePause();
        const pauseBtn = document.getElementById('btn-pause');
        if (pauseBtn) { pauseBtn.classList.toggle('active'); pauseBtn.textContent = this.gameRef.paused ? '▶️' : '⏸'; }
      }
      if (e.key === '1') {
        this.gameRef.setSpeed(1);
        document.querySelectorAll('.speed-btn[data-speed]').forEach(b => b.classList.remove('active'));
        const b1 = document.getElementById('btn-speed-1'); if (b1) b1.classList.add('active');
        const pp = document.getElementById('btn-pause'); if (pp) pp.classList.remove('active');
      }
      if (e.key === '2') {
        this.gameRef.setSpeed(2);
        document.querySelectorAll('.speed-btn[data-speed]').forEach(b => b.classList.remove('active'));
        const b2 = document.getElementById('btn-speed-2'); if (b2) b2.classList.add('active');
        const pp = document.getElementById('btn-pause'); if (pp) pp.classList.remove('active');
      }
      if (e.key === '5') {
        this.gameRef.setSpeed(5);
        document.querySelectorAll('.speed-btn[data-speed]').forEach(b => b.classList.remove('active'));
        const b5 = document.getElementById('btn-speed-5'); if (b5) b5.classList.add('active');
        const pp = document.getElementById('btn-pause'); if (pp) pp.classList.remove('active');
      }
    });
  }

  private _bindKonami(): void {
    const KONAMI_SEQUENCE = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'KeyB', 'KeyA'];
    let konamiIndex = 0;
    let konamiTimer: ReturnType<typeof setTimeout> | null = null;
    document.addEventListener('keydown', (e) => {
      if (this._konamiTriggered || !this.gameRef) return;
      if (konamiTimer) clearTimeout(konamiTimer);
      konamiTimer = setTimeout(() => { konamiIndex = 0; }, 2000);
      if (e.code === KONAMI_SEQUENCE[konamiIndex]) {
        konamiIndex++;
        if (konamiIndex >= KONAMI_SEQUENCE.length) {
          this._konamiTriggered = true;
          this._triggerChaosMode();
        }
      } else {
        konamiIndex = (e.code === KONAMI_SEQUENCE[0]) ? 1 : 0;
      }
    });
  }

  private _triggerChaosMode(): void {
    const g = this.gameRef; if (!g) return;
    g.oak.acorns = 9999; g.oak.dnaPoints = 9999; g.oak.energy = g.oak.maxEnergy; g.casino.totalCoins = 999999;
    if (navigator.vibrate) navigator.vibrate([100, 50, 100, 50, 200]);
    this.showToast('chaos', '🔓 CHAOS MODE', 'Tutto sbloccato! Konami Code attivato.');
    const oakEl = document.getElementById('oak-visual');
    if (oakEl) { oakEl.classList.add('chaos-mode'); setTimeout(() => oakEl.classList.remove('chaos-mode'), 5000); }
    try { localStorage.setItem('oak_and_chaos_chaos', Date.now().toString()); } catch (e) { /* ignore */ }
    g.requestRender();
  }

  private _delegateClicks(): void {
    document.addEventListener('click', e => {
      const target = e.target as HTMLElement;
      const breedBtn = target.closest<HTMLButtonElement>('.breed-btn');
      if (breedBtn) {
        const partnerId = breedBtn.dataset.partner || '';
        this._handleBreed(partnerId);
        return;
      }
      const repairBtn = target.closest<HTMLButtonElement>('.slot-repair-btn');
      if (repairBtn) {
        const machineId = parseInt(repairBtn.dataset.machine || '0', 10);
        this._handleRepair(machineId);
        return;
      }
      const assignBtn = target.closest<HTMLButtonElement>('.assign-role-btn');
      if (assignBtn) { this._showRoleModal(assignBtn.dataset.offspring || ''); return; }
      const fireBtn = target.closest<HTMLButtonElement>('.fire-role-btn');
      if (fireBtn) {
        this.gameRef?.population.removeRole(fireBtn.dataset.offspring || '');
        this.showToast('info', '❌ Role Removed', 'Offspring is now unassigned.');
        this.gameRef?.requestRender();
        return;
      }
      const upgradeBtn = target.closest<HTMLButtonElement>('.upgrade-buy');
      if (upgradeBtn && !upgradeBtn.disabled) {
        const upgradeId = upgradeBtn.dataset.upgrade || '';
        const ok = this.gameRef?.oak.purchaseUpgrade(upgradeId);
        if (ok) { this.showToast('success', '🧬 Upgraded!', `${upgradeId} purchased!`); this.gameRef?.requestRender(); }
        return;
      }
      const roleBtn = target.closest<HTMLButtonElement>('.role-select-btn');
      if (roleBtn) {
        this.gameRef?.population.assignRole(roleBtn.dataset.offspringId || '', roleBtn.dataset.roleId || '');
        this.closeModal();
        this.showToast('success', '📋 Role Assigned', `Role: ${roleBtn.dataset.roleId || ''}`);
        this.gameRef?.requestRender();
        return;
      }
    });
  }

  private _handleBreed(partnerId: string): void {
    const partner = this.gameRef?.population.getPartner(partnerId);
    const oak = this.gameRef?.oak;
    if (!partner || !oak || !this.gameRef) return;
    const result = this.gameRef.breeding.attemptBreed(oak, partner);
    if (result.success && result.offspring) {
      this.gameRef.population.addOffspring(result.offspring);
      this._showBreedResult(result.offspring);
    }
  }

  private _showBreedResult(offspring: Offspring): void {
    const html = `<div class="breed-result-modal"><div class="breed-offspring-emoji">${offspring.emoji}</div><h2>${offspring.name}</h2><p class="breed-type">${offspring.type} — Gen ${offspring.generation}</p><div class="breed-stats"><span>❤️ ${offspring.stats.health}</span><span>⚡ ${offspring.stats.energy}</span><span>💪 ${offspring.stats.strength}</span><span>💎 ${offspring.stats.charisma}</span><span>⚡ ${offspring.stats.speed}</span><span>🍀 ${offspring.stats.luck}</span></div><p class="breed-traits">🧬 ${offspring.traits.join(', ')}</p><p class="breed-desc">${offspring.description}</p></div>`;
    this.openModal(html);
  }

  resetGameViaModal(): void {
    if (this.gameRef) { this.gameRef.save(); this.gameRef.stop(); }
  }

  private _handleRepair(machineId: number): void {
    if (!this.gameRef) return;
    if (this.gameRef.casino.totalCoins >= 150) {
      const m = this.gameRef.casino.machines.find(m => m.id === machineId);
      if (m && m.isBroken) { this.gameRef.casino.totalCoins -= 150; m.isBroken = false; this.showToast('success', '🔧 Repaired!', `Machine ${machineId} is back online.`); this.gameRef.requestRender(); }
    } else { this.showToast('error', '💰 Not enough', 'Need 150 coins to repair.'); }
  }

  private _showRoleModal(offspringId: string): void {
    const roleBtns = ROLES.map(r => `<button class="role-select-btn" data-offspring-id="${offspringId}" data-role-id="${r.id}">${r.icon} ${r.name}<span class="role-desc">${r.bonus}</span></button>`).join('');
    this.openModal(`<h2>📋 Assign Role</h2><div class="role-grid">${roleBtns}</div>`);
  }

  showToast(type: 'success' | 'error' | 'weird' | 'info' | 'chaos', title: string, message: string): void {
    const area = document.getElementById('toast-area');
    if (!area) return;
    const toast = document.createElement('div');
    const cls = type === 'success' ? 'toast-positive' : type === 'error' ? 'toast-negative' : type === 'weird' ? 'toast-weird' : 'toast-positive';
    toast.className = `toast ${cls}`;
    toast.innerHTML = `<span class="toast-icon">${title.split(' ')[0]}</span><span class="toast-text"><strong>${title}</strong><br>${message}</span>`;
    area.appendChild(toast);
    setTimeout(() => toast.remove(), 4200);
    if (navigator.vibrate) navigator.vibrate(30);
  }

  showAchievement(text: string): void {
    const popup = document.getElementById('achievement-popup');
    const name = document.getElementById('achievement-name');
    if (name) name.textContent = text;
    if (popup) { popup.classList.remove('hidden'); setTimeout(() => popup.classList.add('hidden'), 4000); }
    if (navigator.vibrate) navigator.vibrate([50, 30, 50]);
  }

  showWinScreen(oak: OakTree, casino: Casino, population: PopulationManager): void {
    const ws = document.getElementById('win-stats');
    if (ws) ws.innerHTML = `<p>Height: ${oak.height.toFixed(1)}m</p><p>Age: ${Math.floor(oak.age)} years</p><p>Offspring: ${population.offspring.length}</p><p>Total Revenue: ${casino.totalCoins.toLocaleString()} 🪙</p>`;
    const win = document.getElementById('win-screen');
    if (win) win.classList.remove('hidden');
  }

  private _showSpeech(text: string, ms: number = 2000): void {
    const el = document.getElementById('oak-speech');
    if (!el) return;
    el.textContent = text;
    el.classList.remove('hidden');
    el.style.animation = 'none';
    void el.offsetHeight;
    el.style.animation = 'oakTalk 0.4s ease-out';
    setTimeout(() => el.classList.add('hidden'), ms);
  }

  private _flashButton(id: string, type: string): void {
    const btn = document.getElementById(id);
    if (btn) { btn.classList.add(`flash-${type}`); setTimeout(() => btn.classList.remove(`flash-${type}`), 400); }
  }
}
