/* ── js/ui.js ── Panel switching, button handlers, modals, notifications ── */

import { ROLES } from './breeding.js';

/**
 * UiManager — owns all DOM bindings.
 * Receives references to game systems via init().
 */
export class UiManager {
  constructor() {
    this.gameRef = null;   // set by init()
    this._modalOpen = false;
  }

  /* ═══════════ init — wire everything up ═══════════ */
  init(game) {
    this.gameRef = game;
    this._bindTabs();
    this._bindSpeedButtons();
    this._bindOakActions();
    this._bindBreeding();
    this._bindSpend();
    this._bindModal();
    this._delegateClicks();
    this._initDebugPanel();
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

  _switchTab(panelId, tabId) {
    const panel = document.getElementById(panelId);
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

  openModal(html) {
    document.getElementById('modal-body').innerHTML = html;
    document.getElementById('modal-overlay').classList.remove('hidden');
    this._modalOpen = true;
  }

  closeModal() {
    document.getElementById('modal-overlay').classList.add('hidden');
    this._modalOpen = false;
  }

  /* ═══════════ Delegated click handlers ═══════════ */
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
  _handleBreed(partnerId) {
    const partner = this.gameRef.population.getPartner(partnerId);
    console.log('[ui] _handleBreed called', { partnerId });
    if (!partner) return;
    console.log('[ui] partner info', { name: partner.name, category: partner.category, compatibility: partner.compatibility });
    const result = this.gameRef.breeding.attemptBreed(this.gameRef.oak, partner);
    console.log('[ui] breeding result', result);
    if (result.success) {
      this.gameRef.population.addOffspring(result.offspring);
      this.showToast('success', '🎉 New Offspring!', `${result.offspring.name} was born!`);
      this._showBreedResult(result.offspring);
    } else {
      this.showToast('error', '❌ Breed Failed', result.reason || 'Incompatible or insufficient resources.');
    }
    this.gameRef.requestRender();
  }

  /* ═══════════ Debug panel for testing (grant resources) ═══════════ */
  _initDebugPanel() {
    try {
      const panel = document.createElement('div');
      panel.id = 'debug-panel';
      panel.style.position = 'fixed';
      panel.style.right = '12px';
      panel.style.bottom = '12px';
      panel.style.zIndex = '9999';
      panel.style.background = 'rgba(0,0,0,0.6)';
      panel.style.color = '#fff';
      panel.style.padding = '8px';
      panel.style.borderRadius = '8px';
      panel.style.fontSize = '12px';
      panel.style.display = 'flex';
      panel.style.gap = '6px';
      panel.innerHTML = `
        <button id="dbg-acorn" style="padding:6px">+5 🌰</button>
        <button id="dbg-dna" style="padding:6px">+5 🧬</button>
        <button id="dbg-energy" style="padding:6px">Fill ⚡</button>
        <button id="dbg-coins" style="padding:6px">+1000 🪙</button>
      `;
      document.body.appendChild(panel);

      document.getElementById('dbg-acorn').addEventListener('click', () => {
        this.gameRef.oak.acorns = (this.gameRef.oak.acorns || 0) + 5;
        console.log('[debug] granted 5 acorns', { acorns: this.gameRef.oak.acorns });
        this.showToast('success', 'Debug', 'Granted 5 acorns');
        this.gameRef.requestRender();
      });
      document.getElementById('dbg-dna').addEventListener('click', () => {
        this.gameRef.oak.dnaPoints = (this.gameRef.oak.dnaPoints || 0) + 5;
        console.log('[debug] granted 5 dna', { dna: this.gameRef.oak.dnaPoints });
        this.showToast('success', 'Debug', 'Granted 5 DNA');
        this.gameRef.requestRender();
      });
      document.getElementById('dbg-energy').addEventListener('click', () => {
        this.gameRef.oak.energy = this.gameRef.oak.maxEnergy;
        console.log('[debug] filled energy', { energy: this.gameRef.oak.energy });
        this.showToast('success', 'Debug', 'Energy filled');
        this.gameRef.requestRender();
      });
      document.getElementById('dbg-coins').addEventListener('click', () => {
        this.gameRef.casino.totalCoins = (this.gameRef.casino.totalCoins || 0) + 1000;
        console.log('[debug] granted coins', { coins: this.gameRef.casino.totalCoins });
        this.showToast('success', 'Debug', 'Granted 1000 coins');
        this.gameRef.requestRender();
      });
    } catch (e) {
      console.warn('Debug panel init failed:', e);
    }
  }

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
  }

  /* ═══════════ Achievement popup ═══════════ */
  showAchievement(text) {
    const popup = document.getElementById('achievement-popup');
    document.getElementById('achievement-name').textContent = text;
    popup.classList.remove('hidden');
    setTimeout(() => popup.classList.add('hidden'), 4000);
  }

  /* ═══════════ Win screen ═══════════ */
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
