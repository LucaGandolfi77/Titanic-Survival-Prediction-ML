/* ── js/ui.js ── Panel switching, button handlers, modals, notifications ── */

import { ROLES } from './breeding.js';

/**
 * UiManager — owns all DOM bindings, tabs, modals, notifications, keyboard shortcuts.
 * Receives references to game systems via init().
 */
export class UiManager {
  /**
   * Wire all DOM event listeners and initialize sub-managers.
   * @param {Game} game - Central game controller reference
   */
  init(game) {
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

  /* ═══════════ Tab switching ═══════════ */
  _bindTabs() {
    document.getElementById('left-tabs').addEventListener('click', e => {
      const btn = e.target.closest('.tab');
      if (!btn) return;
      this._switchTab('left-panel', btn.dataset.tab);
      btn.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
    });

    document.getElementById('right-tabs').addEventListener('click', e => {
      const btn = e.target.closest('.tab');
      if (!btn) return;
      this._switchTab('right-panel', btn.dataset.tab);
      btn.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
    });
  }

  /**
   * Switch active tab within a panel.
   * @param {string} panelId - 'left-panel' or 'right-panel'
   * @param {string} tabId - target tab content id
   */
  _switchTab(panelId, tabId) {
    panel.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    const target = document.getElementById(tabId);
    if (target) target.classList.add('active');
  }

  /* ═══════════ Speed controls ═══════════ */
  _bindSpeedButtons() {
    const btns = document.querySelectorAll('.speed-btn[data-speed]');
    btns.forEach(b => {
      b.addEventListener('click', () => {
        const spd = parseInt(b.dataset.speed, 10);
        this.gameRef.setSpeed(spd);
        btns.forEach(bb => bb.classList.remove('active'));
        b.classList.add('active');
        document.getElementById('btn-pause').classList.remove('active');
      });
    });

    document.getElementById('btn-pause').addEventListener('click', () => {
      this.gameRef.togglePause();
      const pauseBtn = document.getElementById('btn-pause');
      pauseBtn.classList.toggle('active');
      pauseBtn.textContent = this.gameRef.paused ? '▶️' : '⏸';
    });
  }

  /* ═══════════ Oak action buttons ═══════════ */
  _bindOakActions() {
    document.getElementById('btn-grow').addEventListener('click', () => {
      const result = this.gameRef.oak.grow();
      if (result) {
        this._flashButton('btn-grow', 'success');
        this._showSpeech('Growing... 🌱', 1500);
      } else {
        this._flashButton('btn-grow', 'fail');
      }
    });

    document.getElementById('btn-acorn').addEventListener('click', () => {
      const ok = this.gameRef.oak.produceAcorn();
      if (ok) {
        this._flashButton('btn-acorn', 'success');
        this._showSpeech('🌰 Acorn produced!', 1200);
      } else {
        this._flashButton('btn-acorn', 'fail');
      }
    });

    document.getElementById('btn-meditate').addEventListener('click', () => {
      this.gameRef.oak.meditate();
      const btn = document.getElementById('btn-meditate');
      if (this.gameRef.oak.isMeditating) {
        btn.classList.add('active');
        btn.innerHTML = '🧘 Stop <span class="cost">active</span>';
        this._showSpeech('Entering deep meditation... 🧬', 2000);
      } else {
        btn.classList.remove('active');
        btn.innerHTML = '🧘 Meditate';
      }
    });
  }

  /* ═══════════ Breeding ═══════════ */
  _bindBreeding() {
    // Filter buttons
    document.querySelector('.breed-filters')?.addEventListener('click', e => {
      const btn = e.target.closest('.filter-btn');
      if (!btn) return;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.dataset.filter;
      document.querySelectorAll('.partner-card').forEach(card => {
        if (filter === 'all') {
          card.style.display = '';
        } else {
          card.style.display = card.classList.contains(`type-${filter}`) ? '' : 'none';
        }
      });
    });
  }

  /* ═══════════ Spend buttons (Revenue tab) ═══════════ */
  _bindSpend() {
    document.getElementById('btn-nutrients')?.addEventListener('click', () => {
      if (this.gameRef.casino.totalCoins >= 50) {
        this.gameRef.casino.totalCoins -= 50;
        this.gameRef.oak.addBuff('energyGen', 60, 2);
        this.showToast('success', '🧪 Nutrients', 'Energy regen ×2 for 60s!');
      } else {
        this.showToast('error', '💰 Not enough', 'Need 50 coins');
      }
    });

    document.getElementById('btn-fertilizer')?.addEventListener('click', () => {
      if (this.gameRef.casino.totalCoins >= 200) {
        this.gameRef.casino.totalCoins -= 200;
        this.gameRef.oak.addBuff('growthSpeed', 120, 1.5);
        this.showToast('success', '🌿 Fertilizer', 'Growth ×1.5 for 120s!');
      } else {
        this.showToast('error', '💰 Not enough', 'Need 200 coins');
      }
    });

    document.getElementById('btn-bribe')?.addEventListener('click', () => {
      if (this.gameRef.casino.totalCoins >= 300) {
        this.gameRef.casino.totalCoins -= 300;
        this.gameRef.events.temporarilyBlock('tax_inspector', 300);
        this.showToast('success', '🤫 Inspector Bribed', 'No inspectors for 5 minutes!');
      } else {
        this.showToast('error', '💰 Not enough', 'Need 300 coins');
      }
    });

    document.getElementById('btn-unlock-m5')?.addEventListener('click', () => {
      if (this.gameRef.casino.totalCoins >= 500) {
        const m = this.gameRef.casino.machines.find(m => m.id === 5);
        if (m && m.isLocked) {
          this.gameRef.casino.totalCoins -= 500;
          m.isLocked = false;
          this.showToast('success', '🎰 Unlocked!', 'Machine 5 is now active!');
        }
      } else {
        this.showToast('error', '💰 Not enough', 'Need 500 coins');
      }
    });

    document.getElementById('btn-unlock-m6')?.addEventListener('click', () => {
      if (this.gameRef.casino.totalCoins >= 500) {
        const m = this.gameRef.casino.machines.find(m => m.id === 6);
        if (m && m.isLocked) {
          this.gameRef.casino.totalCoins -= 500;
          m.isLocked = false;
          this.showToast('success', '🎰 Unlocked!', 'Machine 6 is now active!');
        }
      } else {
        this.showToast('error', '💰 Not enough', 'Need 500 coins');
      }
    });
  }

  /* ═══════════ Modal ═══════════ */
  _bindModal() {
    document.getElementById('modal-close').addEventListener('click', () => this.closeModal());
    document.getElementById('modal-overlay').addEventListener('click', e => {
      if (e.target === document.getElementById('modal-overlay')) this.closeModal();
    });
    document.getElementById('btn-play-again')?.addEventListener('click', () => {
      this.gameRef.resetGame();
    });
  }

  /**
   * Open a modal with HTML content.
   * @param {string} html - Modal body HTML
   */
  openModal(html) {
    document.getElementById('modal-body').innerHTML = html;
    document.getElementById('modal-overlay').classList.remove('hidden');
    this._modalOpen = true;
  }

  /**
   * Close the active modal.
   */
  closeModal() {
    document.getElementById('modal-overlay').classList.add('hidden');
    this._modalOpen = false;
  }

  /* ═══════════ Keyboard shortcuts ═══════════ */
  _bindKeyboard() {
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.code === 'Space' || e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        this.gameRef.togglePause();
        const pauseBtn = document.getElementById('btn-pause');
        if (pauseBtn) {
          pauseBtn.classList.toggle('active');
          pauseBtn.textContent = this.gameRef.paused ? '▶️' : '⏸';
        }
      }
      if (e.key === '1') {
        this.gameRef.setSpeed(1);
        document.querySelectorAll('.speed-btn[data-speed]').forEach(b => b.classList.remove('active'));
        const b1 = document.getElementById('btn-speed-1');
        if (b1) b1.classList.add('active');
        document.getElementById('btn-pause')?.classList.remove('active');
      }
      if (e.key === '2') {
        this.gameRef.setSpeed(2);
        document.querySelectorAll('.speed-btn[data-speed]').forEach(b => b.classList.remove('active'));
        const b2 = document.getElementById('btn-speed-2');
        if (b2) b2.classList.add('active');
        document.getElementById('btn-pause')?.classList.remove('active');
      }
      if (e.key === '5') {
        this.gameRef.setSpeed(5);
        document.querySelectorAll('.speed-btn[data-speed]').forEach(b => b.classList.remove('active'));
        const b5 = document.getElementById('btn-speed-5');
        if (b5) b5.classList.add('active');
        document.getElementById('btn-pause')?.classList.remove('active');
      }
    });
  }

  /* ═══════════ Konami Code Easter Egg ═══════════ */
  _bindKonami() {
    const KONAMI_SEQUENCE = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'KeyB', 'KeyA'];
    let konamiIndex = 0;
    let konamiTimer = null;

    document.addEventListener('keydown', (e) => {
      if (this._konamiTriggered) return;

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

  _triggerChaosMode() {
    const game = this.gameRef;
    game.oak.acorns = 9999;
    game.oak.dnaPoints = 9999;
    game.oak.energy = game.oak.maxEnergy;
    game.casino.totalCoins = 999999;

    if (navigator.vibrate) navigator.vibrate([100, 50, 100, 50, 200]);
    this.showToast('chaos', '🔓 CHAOS MODE', 'Tutto sbloccato! Konami Code attivato.');

    const oakEl = document.getElementById('oak-visual');
    if (oakEl) {
      oakEl.classList.add('chaos-mode');
      setTimeout(() => oakEl.classList.remove('chaos-mode'), 5000);
    }

    try {
      localStorage.setItem('oak_and_chaos_chaos', Date.now().toString());
    } catch (e) { /* ignore */ }

    game.requestRender();
  }
  _delegateClicks() {
    document.addEventListener('click', e => {
      // Breed buttons
      const breedBtn = e.target.closest('.breed-btn');
      if (breedBtn) {
        const partnerId = breedBtn.dataset.partner;
        this._handleBreed(partnerId);
        return;
      }

      // Repair buttons
      const repairBtn = e.target.closest('.slot-repair-btn');
      if (repairBtn) {
        const machineId = parseInt(repairBtn.dataset.machine, 10);
        this._handleRepair(machineId);
        return;
      }

      // Assign role
      const assignBtn = e.target.closest('.assign-role-btn');
      if (assignBtn) {
        const offspringId = assignBtn.dataset.offspring;
        this._showRoleModal(offspringId);
        return;
      }

      // Fire role
      const fireBtn = e.target.closest('.fire-role-btn');
      if (fireBtn) {
        const offspringId = fireBtn.dataset.offspring;
        this.gameRef.population.removeRole(offspringId);
        this.showToast('info', '❌ Role Removed', 'Offspring is now unassigned.');
        this.gameRef.requestRender();
        return;
      }

      // Upgrade buy
      const upgradeBtn = e.target.closest('.upgrade-buy');
      if (upgradeBtn && !upgradeBtn.disabled) {
        const upgradeId = upgradeBtn.dataset.upgrade;
        const ok = this.gameRef.oak.purchaseUpgrade(upgradeId);
        if (ok) {
          this.showToast('success', '🧬 Upgraded!', `${upgradeId} purchased!`);
          this.gameRef.requestRender();
        }
        return;
      }

      // Role selection in modal
      const roleBtn = e.target.closest('.role-select-btn');
      if (roleBtn) {
        const offspringId = roleBtn.dataset.offspringId;
        const roleId = roleBtn.dataset.roleId;
        this.gameRef.population.assignRole(offspringId, roleId);
        this.closeModal();
        this.showToast('success', '📋 Role Assigned', `Role: ${roleId}`);
        this.gameRef.requestRender();
        return;
      }
    });
  }

  /* ═══════════ breed handler ═══════════ */
  _showBreedResult(offspring) {
    const html = `
      <div class="breed-result-modal">
        <div class="breed-offspring-emoji">${offspring.emoji}</div>
        <h2>${offspring.name}</h2>
        <p class="breed-type">${offspring.type} — Gen ${offspring.generation}</p>
        <div class="breed-stats">
          <span>❤️ ${offspring.stats.health}</span>
          <span>⚡ ${offspring.stats.energy}</span>
          <span>💪 ${offspring.stats.strength}</span>
          <span>💎 ${offspring.stats.charisma}</span>
          <span>⚡ ${offspring.stats.speed}</span>
          <span>🍀 ${offspring.stats.luck}</span>
        </div>
        <p class="breed-traits">🧬 ${offspring.traits.join(', ')}</p>
        <p class="breed-desc">${offspring.description}</p>
      </div>
    `;
    this.openModal(html);
  }

  /**
   * Reset the game: save state, stop loop, clear storage, reload.
   */
  resetGameViaModal() {
    if (this.gameRef) {
      this.gameRef.save();
      this.gameRef.stop();
    }
  }

  /* ═══════════ repair handler ═══════════ */
  _handleRepair(machineId) {
    if (this.gameRef.casino.totalCoins >= 150) {
      const m = this.gameRef.casino.machines.find(m => m.id === machineId);
      if (m && m.isBroken) {
        this.gameRef.casino.totalCoins -= 150;
        m.isBroken = false;
        this.showToast('success', '🔧 Repaired!', `Machine ${machineId} is back online.`);
        this.gameRef.requestRender();
      }
    } else {
      this.showToast('error', '💰 Not enough', 'Need 150 coins to repair.');
    }
  }

  /* ═══════════ role assignment modal ═══════════ */
  _showRoleModal(offspringId) {
    const roleBtns = ROLES.map(r => `
      <button class="role-select-btn" data-offspring-id="${offspringId}" data-role-id="${r.id}">
        ${r.icon} ${r.name}
        <span class="role-desc">${r.bonus}</span>
      </button>
    `).join('');
    this.openModal(`
      <h2>📋 Assign Role</h2>
      <div class="role-grid">${roleBtns}</div>
    `);
  }

  /* ═══════════ Toasts ═══════════ */
  /**
   * Show a toast notification.
   * @param {'success'|'error'|'weird'|'info'} type - Toast style variant
   * @param {string} title - Short headline
   * @param {string} message - Body text
   */
  showToast(type, title, message) {
    const area = document.getElementById('toast-area');
    const toast = document.createElement('div');
    const cls = type === 'success' ? 'toast-positive' :
                type === 'error' ? 'toast-negative' :
                type === 'weird' ? 'toast-weird' : 'toast-positive';
    toast.className = `toast ${cls}`;
    toast.innerHTML = `<span class="toast-icon">${title.split(' ')[0]}</span><span class="toast-text"><strong>${title}</strong><br>${message}</span>`;
    area.appendChild(toast);
    setTimeout(() => toast.remove(), 4200);
    if (navigator.vibrate) navigator.vibrate(30);
  }

  /* ═══════════ Achievement popup ═══════════ */
  /**
   * Show achievement popup.
   * @param {string} text - Achievement description
   */
  showAchievement(text) {
    const popup = document.getElementById('achievement-popup');
    document.getElementById('achievement-name').textContent = text;
    popup.classList.remove('hidden');
    setTimeout(() => popup.classList.add('hidden'), 4000);
    if (navigator.vibrate) navigator.vibrate([50, 30, 50]);
  }

  /* ═══════════ Win screen ═══════════ */
  /**
   * Display the win screen with final stats.
   * @param {OakTree} oak - Oak tree instance
   * @param {Casino} casino - Casino instance
   * @param {PopulationManager} population - Population manager
   */
  showWinScreen(oak, casino, population) {
    document.getElementById('win-stats').innerHTML = `
      <p>Height: ${oak.height.toFixed(1)}m</p>
      <p>Age: ${Math.floor(oak.age)} years</p>
      <p>Offspring: ${population.offspring.length}</p>
      <p>Total Revenue: ${casino.totalCoins.toLocaleString()} 🪙</p>
    `;
    document.getElementById('win-screen').classList.remove('hidden');
  }

  /* ═══════════ Oak speech bubble ═══════════ */
  _showSpeech(text, ms = 2000) {
    const el = document.getElementById('oak-speech');
    el.textContent = text;
    el.classList.remove('hidden');
    el.style.animation = 'none';
    void el.offsetHeight;
    el.style.animation = 'oakTalk 0.4s ease-out';
    setTimeout(() => el.classList.add('hidden'), ms);
  }

  /* ═══════════ button flash util ═══════════ */
  _flashButton(id, type) {
    const btn = document.getElementById(id);
    btn.classList.add(`flash-${type}`);
    setTimeout(() => btn.classList.remove(`flash-${type}`), 400);
  }
}
