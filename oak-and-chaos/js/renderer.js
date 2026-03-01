/* ── js/renderer.js ── DOM rendering of all game entities ── */

import { formatNumber } from './utils.js';
import { ROLES } from './breeding.js';

/* ══════════════════════════════════════════════════════════
   renderOak — update oak visual CSS + stats in the DOM
   ══════════════════════════════════════════════════════════ */
export function renderOak(oak, dayPhase) {
  // Stage class
  const oakEl = document.getElementById('oak-visual');
  const stageClass = oak.getStageClass();
  if (!oakEl.classList.contains(stageClass)) {
    oakEl.className = stageClass;
    if (oak.isMeditating) oakEl.classList.add('meditating');
    // Trigger grow animation
    oakEl.style.animation = 'none';
    void oakEl.offsetHeight;
    oakEl.style.animation = 'oakGrow 1s ease';
  }

  // Meditating
  if (oak.isMeditating && !oakEl.classList.contains('meditating')) {
    oakEl.classList.add('meditating');
  } else if (!oak.isMeditating && oakEl.classList.contains('meditating')) {
    oakEl.classList.remove('meditating');
  }

  // Day/Night on container
  const container = document.getElementById('oak-visual-container');
  if (dayPhase === 'night') {
    container.classList.add('night-time');
  } else {
    container.classList.remove('night-time');
  }

  // Stats
  document.getElementById('stat-ht').textContent = oak.height.toFixed(1) + 'm';
  document.getElementById('stat-gi').textContent = oak.trunkGirth.toFixed(1);
  document.getElementById('stat-lv').textContent = formatNumber(oak.leaves);
  document.getElementById('stat-fe').textContent = Math.floor(oak.fertility);
  document.getElementById('stat-ch').textContent = Math.floor(oak.charisma);
  document.getElementById('stat-ac').textContent = oak.acorns;

  // Action buttons disable state
  document.getElementById('btn-grow').disabled = oak.energy < 50 || oak.height >= 100;
  document.getElementById('btn-acorn').disabled = oak.energy < 30;
}

/* ══════════════════════════════════════════════════════════
   renderHUD — top bar stats
   ══════════════════════════════════════════════════════════ */
export function renderHUD(oak, casino, population, gameDay, dayPhase) {
  document.getElementById('oak-name-display').textContent =
    `${oak.name} — ${oak.getStageName()}`;
  document.getElementById('hud-age').textContent = Math.floor(oak.age);
  document.getElementById('hud-height').textContent = oak.height.toFixed(1);

  const ePct = (oak.energy / oak.maxEnergy) * 100;
  document.getElementById('energy-fill').style.width = ePct + '%';
  document.getElementById('hud-energy').textContent = Math.floor(oak.energy);
  document.getElementById('hud-max-energy').textContent = Math.floor(oak.maxEnergy);

  document.getElementById('hud-coins').textContent = formatNumber(casino.totalCoins);
  document.getElementById('hud-dna').textContent = formatNumber(oak.dnaPoints);
  document.getElementById('hud-day').textContent = gameDay;
  document.getElementById('hud-day-icon').textContent = dayPhase === 'day' ? '☀️' : '🌙';
}

/* ══════════════════════════════════════════════════════════
   renderMachines — slot machine cards + NPC mood
   ══════════════════════════════════════════════════════════ */
export function renderMachines(casino) {
  const grid = document.getElementById('machines-grid');
  // Only rebuild if machine count changes; otherwise update in-place
  if (grid.children.length !== casino.machines.length) {
    grid.innerHTML = '';
    for (const m of casino.machines) {
      grid.appendChild(createMachineCard(m, casino));
    }
  } else {
    for (const m of casino.machines) {
      updateMachineCard(m, casino);
    }
  }
}

function createMachineCard(machine, casino) {
  const npc = casino.npcs.find(n => n.machineId === machine.id);
  const card = document.createElement('div');
  card.className = 'slot-card' + (machine.isLocked ? ' locked' : '') + (machine.isBroken ? ' broken' : '');
  card.id = `machine-${machine.id}`;

  card.innerHTML = `
    <div class="slot-header">${machine.name}</div>
    <div class="slot-screen" id="screen-${machine.id}">
      <div class="reel-column"><div class="reel-strip" id="reel-${machine.id}-0"><div class="reel-symbol">${machine.currentReels[0]}</div></div></div>
      <div class="reel-column"><div class="reel-strip" id="reel-${machine.id}-1"><div class="reel-symbol">${machine.currentReels[1]}</div></div></div>
      <div class="reel-column"><div class="reel-strip" id="reel-${machine.id}-2"><div class="reel-symbol">${machine.currentReels[2]}</div></div></div>
    </div>
    <div class="slot-info">
      <span class="slot-rev" id="rev-${machine.id}">💰 ${formatNumber(machine.revenuePerMinute)}/min</span>
      <span class="slot-status ${machine.isBroken ? 'status-broken' : machine.isLocked ? 'status-locked' : 'status-ok'}" id="status-${machine.id}">
        ${machine.isBroken ? '🔧 BROKEN' : machine.isLocked ? '🔒 LOCKED' : '✅ ACTIVE'}
      </span>
    </div>
    ${machine.isBroken ? `<button class="slot-repair-btn" data-machine="${machine.id}">🔧 Repair (150🪙)</button>` : ''}
    ${npc ? `
    <div class="npc-avatar">
      <div class="npc-icon" style="background: ${getNpcBg(npc)}">🧔</div>
      <div class="npc-details">
        <div class="npc-name">${npc.name}</div>
        <div class="npc-mood-bar"><div class="npc-mood-fill" id="mood-${npc.id}" style="width:${npc.mood}%;background:${getMoodColor(npc.mood)}"></div></div>
        <div class="npc-quote" id="quote-${npc.id}">"${npc.currentQuote}"</div>
      </div>
    </div>` : ''}
  `;
  return card;
}

function updateMachineCard(machine, casino) {
  const card = document.getElementById(`machine-${machine.id}`);
  if (!card) return;

  // Update class
  card.className = 'slot-card' + (machine.isLocked ? ' locked' : '') + (machine.isBroken ? ' broken' : '');

  // Reels
  for (let i = 0; i < 3; i++) {
    const strip = document.getElementById(`reel-${machine.id}-${i}`);
    if (strip) {
      const sym = strip.querySelector('.reel-symbol');
      if (sym) sym.textContent = machine.currentReels[i];
      if (machine.spinning) {
        strip.classList.add('spinning');
      } else {
        strip.classList.remove('spinning');
      }
    }
  }

  // Revenue
  const revEl = document.getElementById(`rev-${machine.id}`);
  if (revEl) revEl.textContent = `💰 ${formatNumber(machine.revenuePerMinute)}/min`;

  // Status
  const statusEl = document.getElementById(`status-${machine.id}`);
  if (statusEl) {
    statusEl.className = 'slot-status ' + (machine.isBroken ? 'status-broken' : machine.isLocked ? 'status-locked' : 'status-ok');
    statusEl.textContent = machine.isBroken ? '🔧 BROKEN' : machine.isLocked ? '🔒 LOCKED' : '✅ ACTIVE';
  }

  // NPC mood & quote
  const npc = casino.npcs.find(n => n.machineId === machine.id);
  if (npc) {
    const moodFill = document.getElementById(`mood-${npc.id}`);
    if (moodFill) {
      moodFill.style.width = npc.mood + '%';
      moodFill.style.background = getMoodColor(npc.mood);
    }
    const quoteEl = document.getElementById(`quote-${npc.id}`);
    if (quoteEl) quoteEl.textContent = `"${npc.currentQuote}"`;
  }

  // Win/Jackpot flash
  if (machine.lastResult) {
    if (machine.lastResult.jackpot) {
      card.classList.add('active-jackpot');
      setTimeout(() => card.classList.remove('active-jackpot'), 3000);
    } else if (machine.lastResult.win) {
      card.classList.add('active-win');
      setTimeout(() => card.classList.remove('active-win'), 1500);
    }
  }
}

function getNpcBg(npc) {
  const map = { devout: '#1e3a5f', nervous: '#4a3728', sleepy: '#2a2a3a', greedy: '#3a2a0a', paranoid: '#1a2a1a', authoritative: '#3a1a2a' };
  return map[npc.personality] || '#2a2a2a';
}

function getMoodColor(mood) {
  if (mood > 70) return 'var(--success-green)';
  if (mood > 40) return 'var(--warning-orange)';
  return 'var(--danger-red)';
}

/* ══════════════════════════════════════════════════════════
   renderPartners — breeding tab partner grid
   ══════════════════════════════════════════════════════════ */
export function renderPartners(partners, oak) {
  const grid = document.getElementById('partners-grid');
  grid.innerHTML = '';

  for (const p of partners) {
    const locked = (p.category === 'animal' && !oak.canBreedAnimals()) ||
                   (p.category === 'taliban' && !oak.canBreedTaliban());
    const compatClass = p.compatibility >= 70 ? 'compat-high' : p.compatibility >= 40 ? 'compat-medium' : 'compat-low';

    const card = document.createElement('div');
    card.className = `partner-card type-${p.category}${locked ? ' locked' : ''}`;
    card.dataset.partnerId = p.id;

    card.innerHTML = `
      <div class="partner-emoji">${p.emoji}</div>
      <div class="partner-name">${p.name}</div>
      <div class="partner-type">${p.category}</div>
      <div class="partner-compat ${compatClass}">${p.compatibility}% compat</div>
      <button class="breed-btn" ${locked || oak.energy < 40 || oak.acorns < 1 ? 'disabled' : ''} data-partner="${p.id}">
        🌱 Breed (40⚡ + 1🌰)
      </button>
    `;
    grid.appendChild(card);
  }

  // Breed status counters
  document.getElementById('breed-acorns').textContent = oak.acorns;
  document.getElementById('breed-children').textContent = '—';
}

/* ══════════════════════════════════════════════════════════
   renderFamily — family tab offspring list
   ══════════════════════════════════════════════════════════ */
export function renderFamily(offspring) {
  const list = document.getElementById('family-list');
  const countEl = document.getElementById('family-count');
  countEl.textContent = `${offspring.length} offspring`;

  list.innerHTML = '';
  for (const o of offspring) {
    const roleDef = ROLES.find(r => r.id === o.role);
    const card = document.createElement('div');
    card.className = 'offspring-card';
    card.innerHTML = `
      <div class="offspring-avatar">${o.emoji}</div>
      <div class="offspring-info">
        <div class="offspring-name">${o.name}</div>
        <div class="offspring-type">${o.type} (Gen ${o.generation})</div>
        <div class="offspring-stats">
          <span class="offspring-stat">❤️ ${o.stats.health}</span>
          <span class="offspring-stat">⚡ ${o.stats.energy}</span>
          <span class="offspring-stat">💪 ${o.stats.strength}</span>
          <span class="offspring-stat">💎 ${o.stats.charisma}</span>
          <span class="offspring-stat">⚡ ${o.stats.speed}</span>
          <span class="offspring-stat">🍀 ${o.stats.luck}</span>
        </div>
        <div class="offspring-traits">🧬 ${o.traits.join(', ')}</div>
        ${roleDef ? `<div class="offspring-role">${roleDef.icon} ${roleDef.name}</div>` : '<div class="offspring-role" style="color:var(--text-muted)">No role assigned</div>'}
        <div class="offspring-desc">${o.description}</div>
        <div class="offspring-actions">
          <button class="assign-role-btn" data-offspring="${o.id}">📋 Assign Role</button>
          ${o.role ? `<button class="fire-role-btn" data-offspring="${o.id}">❌ Remove Role</button>` : ''}
        </div>
      </div>
    `;
    list.appendChild(card);
  }
}

/* ══════════════════════════════════════════════════════════
   renderUpgrades — DNA upgrade list
   ══════════════════════════════════════════════════════════ */
export function renderUpgrades(upgrades, dnaPoints) {
  const list = document.getElementById('upgrades-list');
  list.innerHTML = '';

  for (const u of upgrades) {
    const item = document.createElement('div');
    item.className = 'upgrade-item' + (u.purchased ? ' purchased' : '');
    item.innerHTML = `
      <div class="upgrade-info">
        <div class="upgrade-name">${u.name}</div>
        <div class="upgrade-desc">${u.desc}</div>
      </div>
      <button class="upgrade-buy" ${u.purchased || dnaPoints < u.cost ? 'disabled' : ''} data-upgrade="${u.id}">
        ${u.purchased ? '✅' : `${u.cost} 🧬`}
      </button>
    `;
    list.appendChild(item);
  }
}

/* ══════════════════════════════════════════════════════════
   renderRevenue — revenue tab charts
   ══════════════════════════════════════════════════════════ */
export function renderRevenue(casino) {
  // Per-machine bars
  const revBars = document.getElementById('revenue-bars');
  revBars.innerHTML = '';
  const maxRev = Math.max(1, ...casino.machines.map(m => m.revenuePerMinute));
  for (const m of casino.machines) {
    const pct = (m.revenuePerMinute / maxRev) * 100;
    const wrapper = document.createElement('div');
    wrapper.className = 'rev-bar-wrapper';
    wrapper.innerHTML = `
      <div class="rev-bar-value">${formatNumber(m.revenuePerMinute)}</div>
      <div class="rev-bar" style="height: ${Math.max(4, pct)}%"></div>
      <div class="rev-bar-label">M${m.id}</div>
    `;
    revBars.appendChild(wrapper);
  }

  // Daily history
  const dailyBars = document.getElementById('daily-bars');
  dailyBars.innerHTML = '';
  const maxDaily = Math.max(1, ...casino.dailyHistory);
  casino.dailyHistory.forEach((val, i) => {
    const pct = (val / maxDaily) * 100;
    const wrapper = document.createElement('div');
    wrapper.className = 'rev-bar-wrapper';
    wrapper.innerHTML = `
      <div class="rev-bar-value">${formatNumber(val)}</div>
      <div class="rev-bar" style="height: ${Math.max(4, pct)}%"></div>
      <div class="rev-bar-label">D${i + 1}</div>
    `;
    dailyBars.appendChild(wrapper);
  });

  // Casino totals
  document.getElementById('casino-rpm').textContent = formatNumber(casino.getTotalRPM());
  document.getElementById('casino-today').textContent = formatNumber(casino.todayRevenue);
}

/* ══════════════════════════════════════════════════════════
   renderStaff — staff tab
   ══════════════════════════════════════════════════════════ */
export function renderStaff(casino, offspring) {
  // Operators
  const opList = document.getElementById('staff-operators');
  opList.innerHTML = '';
  for (const npc of casino.npcs) {
    const card = document.createElement('div');
    card.className = 'staff-card';
    card.innerHTML = `
      <div class="npc-icon" style="background:${getNpcBg(npc)}">🧔</div>
      <div>
        <div class="staff-name">${npc.name}</div>
        <div class="staff-role">${npc.personality} — Machine ${npc.machineId}</div>
      </div>
      <div class="staff-mood"><div class="staff-mood-fill" style="width:${npc.mood}%;background:${getMoodColor(npc.mood)}"></div></div>
    `;
    opList.appendChild(card);
  }

  // Dealers (offspring with roles)
  const dealerList = document.getElementById('staff-dealers');
  dealerList.innerHTML = '';
  const assigned = offspring.filter(o => o.role);
  if (assigned.length === 0) {
    dealerList.innerHTML = '<div style="color:var(--text-muted);font-size:.8rem;padding:8px">No offspring assigned yet.</div>';
    return;
  }
  for (const o of assigned) {
    const roleDef = ROLES.find(r => r.id === o.role);
    const card = document.createElement('div');
    card.className = 'staff-card';
    card.innerHTML = `
      <div class="offspring-avatar" style="width:32px;height:32px;font-size:1.2rem">${o.emoji}</div>
      <div>
        <div class="staff-name">${o.name}</div>
        <div class="staff-role">${roleDef ? roleDef.name : o.role}</div>
      </div>
    `;
    dealerList.appendChild(card);
  }
}

/* ══════════════════════════════════════════════════════════
   renderEventToast — create + animated toast
   ══════════════════════════════════════════════════════════ */
export function renderEventToast(evt) {
  const area = document.getElementById('toast-area');
  const toast = document.createElement('div');
  const cls = evt.type === 'positive' ? 'toast-positive' : evt.type === 'negative' ? 'toast-negative' : 'toast-weird';
  toast.className = `toast ${cls}`;
  toast.innerHTML = `<span class="toast-icon">${evt.name.split(' ')[0]}</span><span class="toast-text"><strong>${evt.name}</strong><br>${evt.desc}</span>`;
  area.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

/* ══════════════════════════════════════════════════════════
   addEventLogEntry — event log footer
   ══════════════════════════════════════════════════════════ */
export function addEventLogEntry(evt) {
  const list = document.getElementById('event-list');
  const item = document.createElement('div');
  const cls = evt.type === 'positive' ? 'event-positive' : evt.type === 'negative' ? 'event-negative' : 'event-weird';
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  item.className = `event-item ${cls}`;
  item.innerHTML = `<span class="event-time">${time}</span>${evt.name} — ${evt.desc}`;
  list.prepend(item);
  // Keep max 30
  while (list.children.length > 30) list.lastChild.remove();
}

/* ══════════════════════════════════════════════════════════
   spawnCoinBurst — particle effect on slot win
   ══════════════════════════════════════════════════════════ */
export function spawnCoinBurst(machineId) {
  const card = document.getElementById(`machine-${machineId}`);
  if (!card) return;
  for (let i = 0; i < 8; i++) {
    const coin = document.createElement('div');
    coin.className = 'coin-particle';
    coin.textContent = '🪙';
    coin.style.setProperty('--coin-dx', `${(Math.random() - 0.5) * 80}px`);
    coin.style.setProperty('--coin-dy', `${-30 - Math.random() * 60}px`);
    coin.style.left = '50%';
    coin.style.top = '40%';
    card.appendChild(coin);
    setTimeout(() => coin.remove(), 1600);
  }
}
