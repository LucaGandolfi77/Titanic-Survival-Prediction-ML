/* ===== Panel: Sponsors =====
 * Negotiate buttons are handled by a single delegated listener in main.js.
 */

import { GameState } from '../../state.js';
import { formatMoney, clamp } from '../../utils.js';
import { getAvailableSponsors } from '../../sponsors.js';

export function renderSponsors() {
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
}
