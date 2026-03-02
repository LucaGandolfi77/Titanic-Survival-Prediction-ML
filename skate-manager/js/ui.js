/* ===== UI rendering, modals, toasts, panel updates ===== */
import { GameState } from './state.js';
import { formatMoney, formatMoneyFull, overallColor, formDotClass, clamp } from './utils.js';
import { getSquadAvgOverall, getSquadAvgMorale, getTeamCohesion, getTotalWages } from './skaters.js';
import { getAvailableFormations, getLockedFormations } from './formations.js';
import { getThisWeekCompetition, canEnterCompetition } from './competitions.js';
import { getAvailableSponsors, canNegotiate, getTotalSponsorIncome, SPONSORS } from './sponsors.js';
import { getCohesion, getWeeklyWages } from './squad.js';

// ===== Toast system =====
let toastCounter = 0;
export function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.id = `toast-${++toastCounter}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('out');
    setTimeout(() => toast.remove(), 400);
  }, duration);
}

// ===== Modal system =====
export function showModal(html, options = {}) {
  const overlay = document.getElementById('modal-overlay');
  const content = document.getElementById('modal-content');
  content.innerHTML = html;
  overlay.classList.remove('hidden');
  overlay.classList.add('visible');

  // Close on overlay click (unless persistent)
  if (!options.persistent) {
    overlay.onclick = (e) => {
      if (e.target === overlay) hideModal();
    };
  }
}

export function hideModal() {
  const overlay = document.getElementById('modal-overlay');
  overlay.classList.remove('visible');
  overlay.classList.add('hidden');
  overlay.onclick = null;
}

export function confirmModal(title, message, onConfirm, onCancel) {
  const html = `
    <h3 class="modal-title">${title}</h3>
    <p class="modal-message">${message}</p>
    <div class="modal-buttons">
      <button class="modal-btn confirm" id="modal-confirm">✔ Confirm</button>
      <button class="modal-btn cancel" id="modal-cancel">✖ Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  document.getElementById('modal-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm();
  });
  document.getElementById('modal-cancel').addEventListener('click', () => {
    hideModal();
    if (onCancel) onCancel();
  });
}

// ===== Skater detail modal =====
export function showSkaterDetail(skater, actions = []) {
  const stats = skater.stats;
  const html = `
    <div class="skater-detail">
      <div class="skater-detail-header">
        <span class="skater-avatar-lg">${skater.avatar}</span>
        <div>
          <h3>${skater.name}</h3>
          <span>${skater.nationality.flag} ${skater.nationality.country} &middot; Age ${skater.age}</span>
        </div>
        <span class="overall-badge ${overallColor(skater.overall)}">${skater.overall}</span>
      </div>
      <div class="skater-detail-stats">
        ${statBar('Technique', stats.technique, '#7dd3fc')}
        ${statBar('Stamina', stats.stamina, '#34d399')}
        ${statBar('Rhythm', stats.rhythm, '#f472b6')}
        ${statBar('Sync', stats.sync, '#a78bfa')}
        ${statBar('Charisma', stats.charisma, '#fbbf24')}
      </div>
      <div class="skater-detail-info">
        <span>💰 Value: ${formatMoneyFull(skater.value)}</span>
        <span>💵 Wage: ${formatMoneyFull(skater.wage)}/wk</span>
        <span>📊 Form: ${skater.form}%</span>
        <span>😊 Morale: ${skater.morale}%</span>
        <span>📝 Contract: ${skater.contract.weeksRemaining} wks</span>
        ${skater.injuryWeeks > 0 ? `<span class="injury-text">🏥 Injured: ${skater.injuryWeeks} wks</span>` : ''}
      </div>
      ${actions.length > 0 ? `<div class="modal-buttons">${actions.map(a =>
        `<button class="modal-btn ${a.class || 'confirm'}" data-action="${a.id}">${a.label}</button>`
      ).join('')}<button class="modal-btn cancel" data-action="close">Close</button></div>` : '<div class="modal-buttons"><button class="modal-btn cancel" data-action="close">Close</button></div>'}
    </div>
  `;
  showModal(html);
  // Wire up buttons
  document.querySelectorAll('#modal-content .modal-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'close') { hideModal(); return; }
      const handler = actions.find(a => a.id === action);
      if (handler && handler.fn) {
        hideModal();
        handler.fn(skater);
      }
    });
  });
}

function statBar(label, value, color) {
  return `
    <div class="stat-row-detail">
      <span class="stat-label">${label}</span>
      <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${value}%;background:${color}"></div></div>
      <span class="stat-value">${value}</span>
    </div>
  `;
}

// ===== Sell skater modal =====
export function showSellModal(skater, onConfirm) {
  const minPrice = Math.round(skater.value * 0.5);
  const maxPrice = Math.round(skater.value * 2);
  const defaultPrice = skater.value;
  const html = `
    <h3 class="modal-title">List ${skater.name} for Sale</h3>
    <p class="modal-message">Set your asking price:</p>
    <div class="sell-price-control">
      <input type="range" id="sell-price-slider" min="${minPrice}" max="${maxPrice}" value="${defaultPrice}" step="500" />
      <span id="sell-price-display" class="sell-price">${formatMoneyFull(defaultPrice)}</span>
    </div>
    <p class="modal-message" style="font-size:0.8rem;color:var(--text-dim)">Market value: ${formatMoneyFull(skater.value)}</p>
    <div class="modal-buttons">
      <button class="modal-btn danger" id="modal-sell-confirm">🏷️ List for Sale</button>
      <button class="modal-btn cancel" id="modal-sell-cancel">Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  const slider = document.getElementById('sell-price-slider');
  const display = document.getElementById('sell-price-display');
  slider.addEventListener('input', () => {
    display.textContent = formatMoneyFull(parseInt(slider.value));
  });
  document.getElementById('modal-sell-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm(parseInt(slider.value));
  });
  document.getElementById('modal-sell-cancel').addEventListener('click', () => hideModal());
}

// ===== Buy confirmation modal =====
export function showBuyModal(skater, onConfirm) {
  const price = skater.askingPrice || skater.value;
  const html = `
    <h3 class="modal-title">Sign ${skater.name}?</h3>
    <div class="skater-detail-header" style="margin-bottom:12px;">
      <span class="skater-avatar-lg">${skater.avatar}</span>
      <div>
        <span>${skater.nationality.flag} ${skater.nationality.country} &middot; Age ${skater.age}</span>
        <span class="overall-badge ${overallColor(skater.overall)}">${skater.overall}</span>
      </div>
    </div>
    <p class="modal-message">Transfer fee: <strong>${formatMoneyFull(price)}</strong></p>
    <p class="modal-message">Weekly wage: <strong>${formatMoneyFull(skater.wage)}/wk</strong></p>
    <p class="modal-message" style="font-size:0.8rem;color:var(--text-dim)">Your balance: ${formatMoneyFull(GameState.money)}</p>
    <div class="modal-buttons">
      <button class="modal-btn confirm" id="modal-buy-confirm">💰 Sign Player</button>
      <button class="modal-btn cancel" id="modal-buy-cancel">Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  document.getElementById('modal-buy-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm();
  });
  document.getElementById('modal-buy-cancel').addEventListener('click', () => hideModal());
}

// ===== Competition entry confirmation =====
export function showCompEntryModal(comp, onConfirm) {
  const html = `
    <h3 class="modal-title">Enter ${comp.name}?</h3>
    <div class="modal-message">
      <p>Tier ${comp.tier} competition</p>
      <p>Entry Fee: <strong>${formatMoneyFull(comp.entryFee)}</strong></p>
      <p>1st Prize: <strong>${formatMoneyFull(comp.prizes[1])}</strong></p>
      <p>Min Overall: <strong>${comp.minOverall}</strong></p>
    </div>
    <div class="modal-buttons">
      <button class="modal-btn gold" id="modal-enter-confirm">🏆 Enter Competition</button>
      <button class="modal-btn cancel" id="modal-enter-cancel">Cancel</button>
    </div>
  `;
  showModal(html, { persistent: true });
  document.getElementById('modal-enter-confirm').addEventListener('click', () => {
    hideModal();
    if (onConfirm) onConfirm();
  });
  document.getElementById('modal-enter-cancel').addEventListener('click', () => hideModal());
}

// ===== Render: Skater card =====
export function skaterCardHTML(skater, mode = 'squad') {
  const s = skater.stats;
  const injured = skater.injuryWeeks > 0;
  const formClass = formDotClass(skater.form);
  const ovClass = overallColor(skater.overall);

  let actionsHTML = '';
  if (mode === 'active') {
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn demote" data-id="${skater.id}" title="Demote to Reserve">⬇</button>
        <button class="card-btn sell" data-id="${skater.id}" title="List for Sale">🏷️</button>
        <button class="card-btn release" data-id="${skater.id}" title="Release">✖</button>
      </div>`;
  } else if (mode === 'reserve') {
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn promote" data-id="${skater.id}" title="Promote to Active">⬆</button>
        <button class="card-btn sell" data-id="${skater.id}" title="List for Sale">🏷️</button>
        <button class="card-btn release" data-id="${skater.id}" title="Release">✖</button>
      </div>`;
  } else if (mode === 'market') {
    const price = skater.askingPrice || skater.value;
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn buy" data-id="${skater.id}" title="Buy">💰 ${formatMoney(price)}</button>
      </div>`;
  } else if (mode === 'listed') {
    actionsHTML = `
      <div class="card-actions">
        <button class="card-btn cancel-listing" data-id="${skater.id}" title="Cancel Listing">↩ Cancel</button>
        <span class="listed-price">${formatMoney(skater.askingPrice)}</span>
      </div>`;
  }

  return `
    <div class="skater-card ${injured ? 'injured' : ''} ${skater.scouted ? 'scouted' : ''}" data-id="${skater.id}">
      <div class="card-top">
        <span class="card-avatar">${skater.avatar}</span>
        <div class="card-info">
          <span class="card-name">${skater.name}</span>
          <span class="card-meta">${skater.nationality.flag} ${skater.age}y</span>
        </div>
        <span class="card-overall ${ovClass}">${skater.overall}</span>
      </div>
      <div class="card-stats">
        <div class="mini-bar" title="Technique ${s.technique}"><div class="mini-fill tech" style="width:${s.technique}%"></div></div>
        <div class="mini-bar" title="Stamina ${s.stamina}"><div class="mini-fill stam" style="width:${s.stamina}%"></div></div>
        <div class="mini-bar" title="Rhythm ${s.rhythm}"><div class="mini-fill rhy" style="width:${s.rhythm}%"></div></div>
        <div class="mini-bar" title="Sync ${s.sync}"><div class="mini-fill syn" style="width:${s.sync}%"></div></div>
        <div class="mini-bar" title="Charisma ${s.charisma}"><div class="mini-fill cha" style="width:${s.charisma}%"></div></div>
      </div>
      <div class="card-bottom">
        <span class="card-wage">${formatMoney(skater.wage)}/wk</span>
        <span class="card-form">
          <span class="form-dot ${formClass}"></span>
          ${skater.form}%
        </span>
        ${injured ? '<span class="injury-badge">🏥</span>' : ''}
      </div>
      ${actionsHTML}
    </div>
  `;
}

// ===== Panel: Header resources =====
export function updateHeader() {
  document.getElementById('header-team-name').textContent = GameState.teamName;
  document.getElementById('header-week').textContent = `Week ${GameState.week} / ${GameState.maxWeeks} — S${GameState.season}`;
  document.getElementById('res-money').textContent = `💰 ${formatMoneyFull(GameState.money)}`;
  document.getElementById('res-fame').textContent = `⭐ ${GameState.fame}`;
  document.getElementById('res-points').textContent = `🏆 ${GameState.points}`;
}

// ===== Panel: Overview =====
export function renderOverview() {
  const avgOv = getSquadAvgOverall(GameState.activeSquad);
  const avgMorale = getSquadAvgMorale(GameState.activeSquad);
  const cohesion = getCohesion();
  const wages = getWeeklyWages();
  const sponsorIncome = getTotalSponsorIncome();
  const netIncome = sponsorIncome - wages;

  document.getElementById('overview-stats-grid').innerHTML = `
    <div class="stat-card">
      <div class="stat-value-big ${overallColor(avgOv)}">${avgOv}</div>
      <div class="stat-label-sm">Avg Overall</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${avgMorale}%</div>
      <div class="stat-label-sm">Avg Morale</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${cohesion}%</div>
      <div class="stat-label-sm">Cohesion</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big ${netIncome >= 0 ? 'text-green' : 'text-red'}">${formatMoney(netIncome)}</div>
      <div class="stat-label-sm">Net Weekly</div>
    </div>
  `;

  // Next competition
  const nextComp = getThisWeekCompetition();
  if (nextComp && nextComp.comp.name) {
    const comp = nextComp.comp;
    const entered = nextComp.entered;
    document.getElementById('overview-next-comp').innerHTML = `
      <div class="next-comp-card">
        <h4>📅 This Week: ${comp.name}</h4>
        <span>Tier ${comp.tier} &middot; Entry: ${formatMoneyFull(comp.entryFee)} &middot; 1st: ${formatMoneyFull(comp.prizes[1])}</span>
        ${entered ? '<span class="badge-entered">✔ ENTERED</span>' : `<span class="badge-not-entered">⬜ NOT ENTERED</span>`}
      </div>
    `;
    // Show compete button
    const btnCompete = document.getElementById('btn-compete');
    btnCompete.style.display = entered ? 'inline-block' : 'none';
  } else {
    document.getElementById('overview-next-comp').innerHTML = `
      <div class="next-comp-card">
        <h4>📅 This Week: Training Week</h4>
        <span>No competition scheduled</span>
      </div>
    `;
    document.getElementById('btn-compete').style.display = 'none';
  }

  // Sponsors summary
  const activeSponsors = GameState.activeSponsors;
  document.getElementById('overview-sponsors-summary').innerHTML = `
    <div class="sponsors-summary-card">
      <h4>💼 Sponsors (${activeSponsors.length}/${GameState.maxSimultaneous})</h4>
      ${activeSponsors.length > 0
        ? activeSponsors.map(d =>
          `<span>${d.sponsor.icon} ${d.sponsor.name}: +${formatMoney(d.sponsor.weeklyIncome)}/wk (${d.weeksRemaining}wk left)</span>`
        ).join('')
        : '<span>No active sponsors</span>'
      }
      <span>Total: <strong>+${formatMoney(sponsorIncome)}/wk</strong></span>
    </div>
  `;

  // Event log
  renderEventLog();
}

// ===== Panel: Event log =====
export function renderEventLog() {
  const log = GameState.eventLog.slice(-5).reverse();
  document.getElementById('overview-event-log').innerHTML = `
    <div class="event-log">
      <h4>📋 Recent Events</h4>
      ${log.length > 0
        ? log.map(e => `<div class="event-entry">${e}</div>`).join('')
        : '<div class="event-entry dim">No events yet</div>'
      }
    </div>
  `;
}

// ===== Panel: Squad =====
export function renderSquad(onCardAction) {
  // Cohesion bar
  const cohesion = getCohesion();
  document.getElementById('cohesion-fill').style.width = cohesion + '%';
  document.getElementById('cohesion-value').textContent = cohesion + '%';
  document.getElementById('weekly-wages').textContent = `Wages: ${formatMoneyFull(getWeeklyWages())}/week`;

  // Active squad
  const activeGrid = document.getElementById('active-squad-grid');
  if (GameState.activeSquad.length === 0) {
    activeGrid.innerHTML = '<div class="empty-slot">No active skaters</div>';
  } else {
    activeGrid.innerHTML = GameState.activeSquad.map(sk => skaterCardHTML(sk, 'active')).join('');
    // Empty slots
    for (let i = GameState.activeSquad.length; i < 16; i++) {
      activeGrid.innerHTML += '<div class="empty-slot">Empty Slot</div>';
    }
  }

  // Reserve
  const reserveGrid = document.getElementById('reserve-grid');
  if (GameState.reserveBench.length === 0) {
    reserveGrid.innerHTML = '<div class="empty-slot">No reserves</div>';
  } else {
    reserveGrid.innerHTML = GameState.reserveBench.map(sk => skaterCardHTML(sk, 'reserve')).join('');
  }

  // Wire up card actions
  wireCardActions(onCardAction);
}

function wireCardActions(handler) {
  document.querySelectorAll('.card-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const action = btn.classList.contains('promote') ? 'promote' :
                     btn.classList.contains('demote') ? 'demote' :
                     btn.classList.contains('sell') ? 'sell' :
                     btn.classList.contains('release') ? 'release' :
                     btn.classList.contains('buy') ? 'buy' :
                     btn.classList.contains('cancel-listing') ? 'cancel-listing' : null;
      if (handler && action) handler(action, id);
    });
  });

  // Click on card (not buttons) = show detail
  document.querySelectorAll('.skater-card').forEach(card => {
    card.addEventListener('click', (e) => {
      if (e.target.closest('.card-btn')) return;
      const id = card.dataset.id;
      const skater = findSkaterById(id);
      if (skater) showSkaterDetail(skater);
    });
  });
}

function findSkaterById(id) {
  return GameState.activeSquad.find(s => s.id === id) ||
         GameState.reserveBench.find(s => s.id === id) ||
         GameState.marketSkaters.find(s => s.id === id) ||
         GameState.listedSkaters.find(s => s.id === id);
}

// ===== Panel: Market =====
export function renderMarket(onCardAction) {
  const refreshIn = GameState.marketRefreshWeek > 0 ? GameState.marketRefreshWeek : 'now';
  document.getElementById('market-refresh-info').textContent = `Refreshes in: ${refreshIn} week(s)`;

  // Available skaters
  const grid = document.getElementById('market-available-grid');
  if (GameState.marketSkaters.length === 0) {
    grid.innerHTML = '<div class="empty-slot">No skaters on the market</div>';
  } else {
    grid.innerHTML = GameState.marketSkaters.map(sk => skaterCardHTML(sk, 'market')).join('');
  }

  // Listed skaters
  const listedGrid = document.getElementById('market-listed-grid');
  if (GameState.listedSkaters.length === 0) {
    listedGrid.innerHTML = '<div class="empty-slot">No skaters listed for sale</div>';
  } else {
    listedGrid.innerHTML = GameState.listedSkaters.map(sk => skaterCardHTML(sk, 'listed')).join('');
  }

  wireCardActions(onCardAction);
}

// ===== Panel: Calendar =====
export function renderCalendar() {
  const grid = document.getElementById('competition-calendar-grid');
  grid.innerHTML = '';
  GameState.calendar.forEach((comp, idx) => {
    const isCurrent = idx === GameState.week - 1;
    const isPast = idx < GameState.week - 1;
    const entered = !!GameState.enteredCompetitions[idx];
    const hasResult = comp.competition;

    let statusBadge = '';
    if (hasResult) {
      const place = hasResult.placement;
      statusBadge = `<span class="badge-result place-${place <= 3 ? place : 'other'}">#${place}</span>`;
    } else if (entered) {
      statusBadge = '<span class="badge-entered">✔ ENTERED</span>';
    }

    const div = document.createElement('div');
    div.className = `calendar-card ${isCurrent ? 'current' : ''} ${isPast ? 'past' : ''}`;
    div.innerHTML = `
      <div class="cal-week">Week ${comp.week}</div>
      ${comp.name
        ? `<div class="cal-name">${comp.name}</div>
           <div class="cal-tier">Tier ${comp.tier}</div>
           <div class="cal-fee">Entry: ${formatMoney(comp.entryFee)}</div>
           <div class="cal-prize">1st: ${formatMoney(comp.prizes[1])}</div>
           ${statusBadge}`
        : `<div class="cal-name training">Training Week</div>`
      }
    `;
    grid.appendChild(div);
  });
}

// ===== Panel: Sponsors =====
export function renderSponsors(onNegotiate) {
  // Active deals
  const activeCont = document.getElementById('sponsors-active');
  if (GameState.activeSponsors.length === 0) {
    activeCont.innerHTML = '<div class="empty-slot">No active sponsor deals</div>';
  } else {
    activeCont.innerHTML = GameState.activeSponsors.map(deal => {
      const sp = deal.sponsor;
      return `
        <div class="sponsor-card active-deal">
          <span class="sponsor-icon">${sp.icon}</span>
          <div class="sponsor-info">
            <span class="sponsor-name">${sp.name}</span>
            <span class="sponsor-income">+${formatMoney(sp.weeklyIncome)}/wk</span>
            <span class="sponsor-remaining">${deal.weeksRemaining} weeks left</span>
            ${deal.breachCount > 0 ? '<span class="sponsor-warning">⚠ Warning!</span>' : ''}
          </div>
        </div>
      `;
    }).join('');
  }

  // Available sponsors
  const availCont = document.getElementById('sponsors-available');
  const available = getAvailableSponsors();
  availCont.innerHTML = available.map(sp => {
    const check = canNegotiate(sp.id);
    const locked = GameState.fame < sp.requiredFame;
    const alreadyActive = GameState.activeSponsors.find(d => d.sponsor.id === sp.id);
    return `
      <div class="sponsor-card ${locked ? 'locked' : ''} ${alreadyActive ? 'active-deal' : ''}">
        <span class="sponsor-icon">${sp.icon}</span>
        <div class="sponsor-info">
          <span class="sponsor-name">${sp.name}</span>
          <span class="sponsor-desc">${sp.description}</span>
          <span class="sponsor-income">${formatMoney(sp.weeklyIncome)}/wk &middot; ${sp.duration} weeks</span>
          ${locked
            ? `<span class="sponsor-lock">🔒 Requires Fame ${sp.requiredFame}</span>`
            : alreadyActive
              ? '<span class="sponsor-active-label">Active</span>'
              : `<button class="card-btn negotiate" data-sponsor="${sp.id}">🤝 Negotiate</button>`
          }
        </div>
      </div>
    `;
  }).join('');

  // Fame indicator
  document.getElementById('fame-indicator').innerHTML = `
    <div class="fame-bar-wrapper">
      <span>⭐ Fame: ${GameState.fame}</span>
      <div class="fame-progress"><div class="fame-fill" style="width:${clamp(GameState.fame, 0, 120)}%"></div></div>
    </div>
  `;

  // Wire negotiate buttons
  document.querySelectorAll('.card-btn.negotiate').forEach(btn => {
    btn.addEventListener('click', () => {
      if (onNegotiate) onNegotiate(btn.dataset.sponsor);
    });
  });
}

// ===== Panel: Standings =====
export function renderStandings() {
  const tbody = document.getElementById('standings-body');
  const entries = [
    { name: GameState.teamName, points: GameState.points, fame: GameState.fame, wins: GameState.competitionResults.filter(r => r.placement === 1).length, isPlayer: true },
    ...GameState.rivals.map(r => ({ name: r.name, points: r.points, fame: r.fame, wins: r.wins, isPlayer: false }))
  ];
  entries.sort((a, b) => b.points - a.points);

  tbody.innerHTML = entries.map((e, i) => `
    <tr class="${e.isPlayer ? 'player-row' : ''}">
      <td>${i + 1}</td>
      <td>${e.isPlayer ? '👤 ' : ''}${e.name}</td>
      <td>${e.points}</td>
      <td>⭐ ${e.fame}</td>
      <td>${e.wins}</td>
    </tr>
  `).join('');

  // Season history
  const historyDiv = document.getElementById('season-history');
  if (GameState.seasonHistory.length === 0) {
    historyDiv.innerHTML = '<div class="empty-slot">No previous seasons</div>';
  } else {
    historyDiv.innerHTML = GameState.seasonHistory.map(sh => `
      <div class="history-card">
        <span>Season ${sh.season}: Rank #${sh.rank} — ${sh.points} pts, ⭐ ${sh.fame}</span>
      </div>
    `).join('');
  }
}

// ===== Results screen =====
export function renderResults(result) {
  document.getElementById('results-title').textContent = `${result.competition} — Results`;

  const leaderboard = document.getElementById('results-leaderboard');
  leaderboard.innerHTML = result.leaderboard.map(e => {
    const medals = { 1: '🥇', 2: '🥈', 3: '🥉' };
    const medal = medals[e.placement] || `#${e.placement}`;
    return `
      <div class="results-row ${e.isPlayer ? 'player-result' : ''}">
        <span class="result-place">${medal}</span>
        <span class="result-team">${e.team}</span>
        <span class="result-score">${Math.round(e.score).toLocaleString()}</span>
      </div>
    `;
  }).join('');

  // Score breakdown
  document.getElementById('results-score-breakdown').innerHTML = `
    <div class="score-breakdown">
      <h4>Score Breakdown</h4>
      <div class="breakdown-row"><span>Base Score</span><span>${result.baseScore.toLocaleString()}</span></div>
      <div class="breakdown-row"><span>Music Bonus</span><span>+${result.musicBonus.toLocaleString()}</span></div>
      <div class="breakdown-row"><span>Sync Bonus</span><span>+${result.syncBonus.toLocaleString()}</span></div>
      <div class="breakdown-row negative"><span>Wobble Penalty</span><span>-${result.wobblePenalty.toLocaleString()}</span></div>
      ${result.perfectBonus > 0 ? `<div class="breakdown-row perfect"><span>Perfect Bonus ✨</span><span>+${result.perfectBonus}</span></div>` : ''}
      <div class="breakdown-row total"><span>TOTAL</span><span>${result.score.toLocaleString()}</span></div>
      <div class="breakdown-stats">
        <span>Formations: ${result.formationsCompleted}</span>
        <span>Saves: ${result.wobblesSaved}</span>
        <span>Falls: ${result.wobblesFailed}</span>
      </div>
    </div>
  `;

  // Prize
  const placement = result.placement;
  const prizeMoney = result.prizeMoney;
  const pointsAwarded = result.pointsAwarded;
  const fameAwarded = result.fameAwarded;
  document.getElementById('results-prize').innerHTML = `
    <div class="prize-card ${placement <= 3 ? 'podium' : ''}">
      <h3>Your Placement: ${placement <= 3 ? ['','🥇 1st','🥈 2nd','🥉 3rd'][placement] : `#${placement}`}</h3>
      ${prizeMoney > 0 ? `<span>Prize: +${formatMoneyFull(prizeMoney)}</span>` : ''}
      <span>Points: +${pointsAwarded}</span>
      ${fameAwarded > 0 ? `<span>Fame: +${fameAwarded}</span>` : ''}
    </div>
  `;
}

// ===== Season end screen =====
export function renderSeasonEnd() {
  const results = GameState.competitionResults.filter(r => r.season === GameState.season);
  const wins = results.filter(r => r.placement === 1).length;
  const podiums = results.filter(r => r.placement <= 3).length;
  const totalPrize = results.reduce((s, r) => s + (r.prizeMoney || 0), 0);

  // Overall standings
  const entries = [
    { name: GameState.teamName, points: GameState.points, isPlayer: true },
    ...GameState.rivals.map(r => ({ name: r.name, points: r.points, isPlayer: false }))
  ];
  entries.sort((a, b) => b.points - a.points);
  const rank = entries.findIndex(e => e.isPlayer) + 1;

  document.getElementById('season-summary').innerHTML = `
    <div class="season-summary-grid">
      <div class="summary-stat"><span class="big">${results.length}</span><span>Competitions</span></div>
      <div class="summary-stat"><span class="big">${wins}</span><span>Wins</span></div>
      <div class="summary-stat"><span class="big">${podiums}</span><span>Podiums</span></div>
      <div class="summary-stat"><span class="big">${formatMoney(totalPrize)}</span><span>Prize Money</span></div>
      <div class="summary-stat"><span class="big">⭐ ${GameState.fame}</span><span>Fame</span></div>
      <div class="summary-stat"><span class="big">🏆 ${GameState.points}</span><span>Points</span></div>
    </div>
  `;

  document.getElementById('championship-result').innerHTML = `
    <div class="championship-card ${rank === 1 ? 'champion' : ''}">
      <h3>${rank === 1 ? '🏆 CHAMPION! 🏆' : `Season Rank: #${rank}`}</h3>
      ${entries.map((e, i) => `
        <div class="champ-row ${e.isPlayer ? 'player-row' : ''}">
          <span>#${i + 1}</span>
          <span>${e.name}</span>
          <span>${e.points} pts</span>
        </div>
      `).join('')}
    </div>
  `;

  return { rank, wins, podiums, totalPrize };
}

// ===== Screen management =====
export function showScreen(screenId) {
  document.querySelectorAll('.screen').forEach(s => {
    s.classList.remove('active');
    s.classList.add('hidden');
  });
  const screen = document.getElementById(screenId);
  if (screen) {
    screen.classList.remove('hidden');
    screen.classList.add('active');
  }
}

export function switchTab(tabId) {
  // Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  // Tab panels
  document.querySelectorAll('.tab-panel').forEach(panel => {
    const id = panel.id.replace('tab-', '');
    panel.classList.toggle('active', id === tabId);
    panel.classList.toggle('hidden', id !== tabId);
  });
}

// ===== Full UI refresh =====
export function refreshAllPanels(cardActionHandler, negotiateHandler) {
  updateHeader();
  renderOverview();
  renderSquad(cardActionHandler);
  renderMarket(cardActionHandler);
  renderCalendar();
  renderSponsors(negotiateHandler);
  renderStandings();
}
