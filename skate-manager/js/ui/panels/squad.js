/* ===== Panel: Squad =====
 * Card actions and detail clicks are handled by a single delegated listener
 * in main.js — this module only renders HTML.
 */

import { GameState } from '../../state.js';
import { formatMoneyFull } from '../../utils.js';
import { skaterCardHTML } from '../skater-card.js';
import { getCohesion, getWeeklyWages } from '../../squad.js';
import { ACTIVE_SQUAD_SIZE, RESERVE_SIZE, STAFF } from '../../config.js';
import { t } from '../../i18n.js';

export function renderSquad() {
  // Section titles reflect the live roster size
  document.getElementById('active-squad-title').textContent =
    t('panel.activeSquad', { n: GameState.activeSquad.length, max: ACTIVE_SQUAD_SIZE });
  document.getElementById('reserve-title').textContent =
    t('panel.reserveBench', { n: GameState.reserveBench.length, max: RESERVE_SIZE });

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
    for (let i = GameState.activeSquad.length; i < ACTIVE_SQUAD_SIZE; i++) {
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

  // Coaching staff
  const staffGrid = document.getElementById('staff-grid');
  staffGrid.innerHTML = STAFF.map(member => {
    const hired = GameState.staff.includes(member.id);
    return `
      <div class="sponsor-card ${hired ? 'active-deal' : ''}">
        <span class="sponsor-icon">${member.icon}</span>
        <div class="sponsor-info">
          <span class="sponsor-name">${member.name}</span>
          <span class="sponsor-desc">${member.description}</span>
          <span class="sponsor-income">${formatMoneyFull(member.cost)}</span>
          ${hired
            ? '<span class="sponsor-active-label">Hired ✓</span>'
            : `<button class="card-btn hire-staff" data-staff="${member.id}">👔 Hire</button>`
          }
        </div>
      </div>
    `;
  }).join('');
}
