/* ===== Panel: Season calendar ===== */

import { GameState } from '../../state.js';
import { formatMoney } from '../../utils.js';

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
