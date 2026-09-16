import { formatNumber } from './utils';
import { ROLES } from './breeding';
import type { OakTree } from './oak';
import type { Casino } from './casino';
import type { PopulationManager } from './population';
import type { Offspring, MachineState, Partner, DNAUpgrade } from '../types/game';

export function renderOak(oak: OakTree, dayPhase: string): void {
  const oakEl = document.getElementById('oak-visual');
  if (!oakEl) return;
  const stageClass = oak.getStageClass();
  const currentClass = oakEl.getAttribute('data-stage') || '';
  if (currentClass !== stageClass) {
    oakEl.setAttribute('data-stage', stageClass);
    oakEl.className = stageClass;
    oakEl.style.animation = 'none';
    void oakEl.offsetHeight;
    oakEl.style.animation = 'oakGrow 1s ease';
  }
  if (oak.isMeditating) {
    oakEl.classList.add('meditating');
  } else {
    oakEl.classList.remove('meditating');
  }
  const container = document.getElementById('oak-visual-container');
  if (dayPhase === 'night') {
    container?.classList.add('night-time');
  } else {
    container?.classList.remove('night-time');
  }
  const sht = document.getElementById('stat-ht'); if (sht) sht.textContent = oak.height.toFixed(1) + 'm';
  const sgi = document.getElementById('stat-gi'); if (sgi) sgi.textContent = oak.trunkGirth.toFixed(1);
  const slv = document.getElementById('stat-lv'); if (slv) slv.textContent = formatNumber(oak.leaves);
  const sfe = document.getElementById('stat-fe'); if (sfe) sfe.textContent = Math.floor(oak.fertility).toString();
  const sch = document.getElementById('stat-ch'); if (sch) sch.textContent = Math.floor(oak.charisma).toString();
  const sac = document.getElementById('stat-ac'); if (sac) sac.textContent = oak.acorns.toString();
  const bg = document.getElementById('btn-grow') as HTMLButtonElement; if (bg) bg.disabled = oak.energy < 50 || oak.height >= 100;
  const ba = document.getElementById('btn-acorn') as HTMLButtonElement; if (ba) ba.disabled = oak.energy < 30;
}

export function renderHUD(oak: OakTree, casino: Casino, population: PopulationManager, gameDay: number, dayPhase: string): void {
  const one = document.getElementById('oak-name-display'); if (one) one.textContent = `${oak.name} — ${oak.getStageName()}`;
  const ha = document.getElementById('hud-age'); if (ha) ha.textContent = Math.floor(oak.age).toString();
  const hh = document.getElementById('hud-height'); if (hh) hh.textContent = oak.height.toFixed(1);
  const ePct = (oak.energy / oak.maxEnergy) * 100;
  const ef = document.getElementById('energy-fill'); if (ef) ef.style.width = ePct + '%';
  const he = document.getElementById('hud-energy'); if (he) he.textContent = Math.floor(oak.energy).toString();
  const hme = document.getElementById('hud-max-energy'); if (hme) hme.textContent = Math.floor(oak.maxEnergy).toString();
  const hc = document.getElementById('hud-coins'); if (hc) hc.textContent = formatNumber(casino.totalCoins);
  const hd = document.getElementById('hud-dna'); if (hd) hd.textContent = formatNumber(oak.dnaPoints);
  const hdy = document.getElementById('hud-day'); if (hdy) hdy.textContent = gameDay.toString();
  const hdi = document.getElementById('hud-day-icon'); if (hdi) hdi.textContent = dayPhase === 'day' ? '☀️' : '🌙';
}

export function renderMachines(casino: Casino): void {
  const grid = document.getElementById('machines-grid');
  if (!grid) return;
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

function createMachineCard(machine: MachineState, casino: Casino): HTMLElement {
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
    ${npc ? `<div class="npc-avatar"><div class="npc-icon" style="background: ${getNpcBg(npc)}">🧔</div><div class="npc-details"><div class="npc-name">${npc.name}</div><div class="npc-mood-bar"><div class="npc-mood-fill" id="mood-${npc.id}" style="width:${npc.mood}%;background:${getMoodColor(npc.mood)}"></div></div><div class="npc-quote" id="quote-${npc.id}">"${npc.currentQuote}"</div></div></div>` : ''}
  `;
  return card;
}

function updateMachineCard(machine: MachineState, casino: Casino): void {
  const card = document.getElementById(`machine-${machine.id}`);
  if (!card) return;
  card.className = 'slot-card' + (machine.isLocked ? ' locked' : '') + (machine.isBroken ? ' broken' : '');
  for (let i = 0; i < 3; i++) {
    const strip = document.getElementById(`reel-${machine.id}-${i}`);
    if (strip) {
      const sym = strip.querySelector('.reel-symbol');
      if (sym) sym.textContent = machine.currentReels[i];
      if (machine.spinning) { strip.classList.add('spinning'); } else { strip.classList.remove('spinning'); }
    }
  }
  const revEl = document.getElementById(`rev-${machine.id}`);
  if (revEl) revEl.textContent = `💰 ${formatNumber(machine.revenuePerMinute)}/min`;
  const statusEl = document.getElementById(`status-${machine.id}`);
  if (statusEl) {
    statusEl.className = 'slot-status ' + (machine.isBroken ? 'status-broken' : machine.isLocked ? 'status-locked' : 'status-ok');
    statusEl.textContent = machine.isBroken ? '🔧 BROKEN' : machine.isLocked ? '🔒 LOCKED' : '✅ ACTIVE';
  }
  const npc = casino.npcs.find(n => n.machineId === machine.id);
  if (npc) {
    const moodFill = document.getElementById(`mood-${npc.id}`);
    if (moodFill) { moodFill.style.width = npc.mood + '%'; moodFill.style.background = getMoodColor(npc.mood); }
    const quoteEl = document.getElementById(`quote-${npc.id}`);
    if (quoteEl) quoteEl.textContent = `"${npc.currentQuote}"`;
  }
  if (machine.lastResult) {
    if (machine.lastResult.jackpot) { card.classList.add('active-jackpot'); setTimeout(() => card.classList.remove('active-jackpot'), 3000); }
    else if (machine.lastResult.win) { card.classList.add('active-win'); setTimeout(() => card.classList.remove('active-win'), 1500); }
  }
}

function getNpcBg(npc: { personality: string }): string {
  const map: Record<string, string> = { devout: '#1e3a5f', nervous: '#4a3728', sleepy: '#2a2a3a', greedy: '#3a2a0a', paranoid: '#1a2a1a', authoritative: '#3a1a2a' };
  return map[npc.personality] || '#2a2a2a';
}

function getMoodColor(mood: number): string {
  if (mood > 70) return 'var(--success-green)';
  if (mood > 40) return 'var(--warning-orange)';
  return 'var(--danger-red)';
}

export function renderPartners(partners: Partner[], oak: OakTree): void {
  const grid = document.getElementById('partners-grid');
  if (!grid) return;
  grid.innerHTML = '';
  for (const p of partners) {
    const locked = (p.category === 'animal' && !oak.canBreedAnimals()) || (p.category === 'taliban' && !oak.canBreedTaliban());
    const compatClass = p.compatibility >= 70 ? 'compat-high' : p.compatibility >= 40 ? 'compat-medium' : 'compat-low';
    const card = document.createElement('div');
    card.className = `partner-card type-${p.category}${locked ? ' locked' : ''}`;
    card.dataset.partnerId = p.id;
    card.innerHTML = `
      <div class="partner-emoji">${p.emoji}</div>
      <div class="partner-name">${p.name}</div>
      <div class="partner-type">${p.category}</div>
      <div class="partner-compat ${compatClass}">${p.compatibility}% compat</div>
      <button class="breed-btn" ${locked || oak.energy < 40 || oak.acorns < 1 ? 'disabled' : ''} data-partner="${p.id}">🌱 Breed (40⚡ + 1🌰)</button>
    `;
    grid.appendChild(card);
  }
  const ba = document.getElementById('breed-acorns'); if (ba) ba.textContent = oak.acorns.toString();
  const bc = document.getElementById('breed-children'); if (bc) bc.textContent = '—';
}

export function renderFamily(offspring: Offspring[]): void {
  const list = document.getElementById('family-list');
  const countEl = document.getElementById('family-count');
  if (countEl) countEl.textContent = `${offspring.length} offspring`;
  if (!list) return;
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

export function renderUpgrades(upgrades: DNAUpgrade[], dnaPoints: number): void {
  const list = document.getElementById('upgrades-list');
  if (!list) return;
  list.innerHTML = '';
  for (const u of upgrades) {
    const item = document.createElement('div');
    item.className = 'upgrade-item' + (u.purchased ? ' purchased' : '');
    item.innerHTML = `
      <div class="upgrade-info"><div class="upgrade-name">${u.name}</div><div class="upgrade-desc">${u.desc}</div></div>
      <button class="upgrade-buy" ${u.purchased || dnaPoints < u.cost ? 'disabled' : ''} data-upgrade="${u.id}">${u.purchased ? '✅' : `${u.cost} 🧬`}</button>
    `;
    list.appendChild(item);
  }
}

export function renderRevenue(casino: Casino): void {
  const revBars = document.getElementById('revenue-bars');
  if (revBars) {
    revBars.innerHTML = '';
    const maxRev = Math.max(1, ...casino.machines.map(m => m.revenuePerMinute));
    for (const m of casino.machines) {
      const pct = (m.revenuePerMinute / maxRev) * 100;
      const wrapper = document.createElement('div');
      wrapper.className = 'rev-bar-wrapper';
      wrapper.innerHTML = `<div class="rev-bar-value">${formatNumber(m.revenuePerMinute)}</div><div class="rev-bar" style="height: ${Math.max(4, pct)}%"></div><div class="rev-bar-label">M${m.id}</div>`;
      revBars.appendChild(wrapper);
    }
  }
  const dailyBars = document.getElementById('daily-bars');
  if (dailyBars) {
    dailyBars.innerHTML = '';
    const maxDaily = Math.max(1, ...casino.dailyHistory);
    casino.dailyHistory.forEach((val, i) => {
      const pct = (val / maxDaily) * 100;
      const wrapper = document.createElement('div');
      wrapper.className = 'rev-bar-wrapper';
      wrapper.innerHTML = `<div class="rev-bar-value">${formatNumber(val)}</div><div class="rev-bar" style="height: ${Math.max(4, pct)}%"></div><div class="rev-bar-label">D${i + 1}</div>`;
      dailyBars.appendChild(wrapper);
    });
  }
  const cr = document.getElementById('casino-rpm'); if (cr) cr.textContent = formatNumber(casino.getTotalRPM());
  const ct = document.getElementById('casino-today'); if (ct) ct.textContent = formatNumber(casino.todayRevenue);
}

export function renderStaff(casino: Casino, offspring: Offspring[]): void {
  const opList = document.getElementById('staff-operators');
  if (opList) {
    opList.innerHTML = '';
    for (const npc of casino.npcs) {
      const card = document.createElement('div');
      card.className = 'staff-card';
      card.innerHTML = `<div class="npc-icon" style="background:${getNpcBg(npc)}">🧔</div><div><div class="staff-name">${npc.name}</div><div class="staff-role">${npc.personality} — Machine ${npc.machineId}</div></div><div class="staff-mood"><div class="staff-mood-fill" style="width:${npc.mood}%;background:${getMoodColor(npc.mood)}"></div></div>`;
      opList.appendChild(card);
    }
  }
  const dealerList = document.getElementById('staff-dealers');
  if (dealerList) {
    dealerList.innerHTML = '';
    const assigned = offspring.filter(o => o.role);
    if (assigned.length === 0) { dealerList.innerHTML = '<div style="color:var(--text-muted);font-size:.8rem;padding:8px">No offspring assigned yet.</div>'; return; }
    for (const o of assigned) {
      const roleDef = ROLES.find(r => r.id === o.role);
      const card = document.createElement('div');
      card.className = 'staff-card';
      card.innerHTML = `<div class="offspring-avatar" style="width:32px;height:32px;font-size:1.2rem">${o.emoji}</div><div><div class="staff-name">${o.name}</div><div class="staff-role">${roleDef ? roleDef.name : o.role}</div></div>`;
      dealerList.appendChild(card);
    }
  }
}

export function renderEventToast(evt: { type: string; name: string; desc: string }): void {
  const area = document.getElementById('toast-area');
  if (!area) return;
  const toast = document.createElement('div');
  const cls = evt.type === 'positive' ? 'toast-positive' : evt.type === 'negative' ? 'toast-negative' : 'toast-weird';
  toast.className = `toast ${cls}`;
  toast.innerHTML = `<span class="toast-icon">${evt.name.split(' ')[0]}</span><span class="toast-text"><strong>${evt.name}</strong><br>${evt.desc}</span>`;
  area.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

export function addEventLogEntry(evt: { type: string; name: string; desc: string }): void {
  const list = document.getElementById('event-list');
  if (!list) return;
  const item = document.createElement('div');
  const cls = evt.type === 'positive' ? 'event-positive' : evt.type === 'negative' ? 'event-negative' : 'event-weird';
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  item.className = `event-item ${cls}`;
  item.innerHTML = `<span class="event-time">${time}</span>${evt.name} — ${evt.desc}`;
  list.prepend(item);
  while (list.children.length > 30) list.lastChild?.remove();
}

export function spawnCoinBurst(machineId: number): void {
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
