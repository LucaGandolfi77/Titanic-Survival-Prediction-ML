/* ===== Panel: Overview (header, stats, next competition, sponsors, log) ===== */

import { GameState } from '../../state.js';
import { formatMoney, formatMoneyFull, overallColor } from '../../utils.js';
import { getSquadAvgOverall, getSquadAvgMorale } from '../../skaters.js';
import { getCohesion, getWeeklyWages } from '../../squad.js';
import { getTotalSponsorIncome } from '../../sponsors.js';
import { getThisWeekCompetition } from '../../competitions.js';
import { MAX_ACTIVE_SPONSORS } from '../../config.js';
import { t } from '../../i18n.js';

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
      <div class="stat-label-sm">${t('ov.avgOverall')}</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${avgMorale}%</div>
      <div class="stat-label-sm">${t('ov.avgMorale')}</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${cohesion}%</div>
      <div class="stat-label-sm">${t('ov.cohesion')}</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big ${netIncome >= 0 ? 'text-green' : 'text-red'}">${formatMoney(netIncome)}</div>
      <div class="stat-label-sm">${t('ov.netWeekly')}</div>
    </div>
  `;

  // Next competition
  const nextComp = getThisWeekCompetition();
  if (nextComp && nextComp.comp.name) {
    const comp = nextComp.comp;
    const entered = nextComp.entered;
    const competed = !!comp.competition;
    document.getElementById('overview-next-comp').innerHTML = `
      <div class="next-comp-card">
        <h4>📅 ${t('ov.thisWeek')} ${comp.name}</h4>
        <span>Tier ${comp.tier} &middot; Entry: ${formatMoneyFull(comp.entryFee)} &middot; 1st: ${formatMoneyFull(comp.prizes[1])}</span>
        ${competed
          ? '<span class="badge-entered">✔ COMPETED</span>'
          : entered
            ? '<span class="badge-entered">✔ ENTERED</span><span class="badge-withdraw">↩ WITHDRAW</span>'
            : '<span class="badge-not-entered">⬜ NOT ENTERED</span>'
        }
      </div>
    `;
    // Show compete / quick-sim buttons only while the competition can still be played
    const canPlay = entered && !competed;
    document.getElementById('btn-compete').style.display = canPlay ? 'inline-block' : 'none';
    document.getElementById('btn-quick-sim').style.display = canPlay ? 'inline-block' : 'none';
  } else {
    document.getElementById('overview-next-comp').innerHTML = `
      <div class="next-comp-card">
        <h4>📅 ${t('ov.thisWeek')} ${t('ov.trainingWeek')}</h4>
        <span>${t('ov.noCompScheduled')}</span>
      </div>
    `;
    document.getElementById('btn-compete').style.display = 'none';
    document.getElementById('btn-quick-sim').style.display = 'none';
  }

  // Sponsors summary
  const activeSponsors = GameState.activeSponsors;
  document.getElementById('overview-sponsors-summary').innerHTML = `
    <div class="sponsors-summary-card">
      <h4>💼 Sponsors (${activeSponsors.length}/${MAX_ACTIVE_SPONSORS})</h4>
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
