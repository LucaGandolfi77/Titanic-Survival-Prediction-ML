/* ===== Panel: Stats — club records, score progression, hall of fame ===== */

import { GameState } from '../../state.js';
import { formatMoneyFull } from '../../utils.js';

export function renderStats() {
  // Club records
  const rec = GameState.records;
  document.getElementById('stats-records-grid').innerHTML = `
    <div class="stat-card">
      <div class="stat-value-big text-green">${rec.bestScore.toLocaleString()}</div>
      <div class="stat-label-sm">Best Routine Score</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${rec.bestPlacement === 99 ? '—' : '#' + rec.bestPlacement}</div>
      <div class="stat-label-sm">Best Placement</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${rec.totalWins}</div>
      <div class="stat-label-sm">Total Wins</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${rec.totalPodiums}</div>
      <div class="stat-label-sm">Total Podiums</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${formatMoneyFull(rec.totalPrize)}</div>
      <div class="stat-label-sm">Total Prize Money</div>
    </div>
    <div class="stat-card">
      <div class="stat-value-big">${GameState.seasonHistory.length}</div>
      <div class="stat-label-sm">Seasons Played</div>
    </div>
  `;

  // Score progression (current season, simple bar chart)
  const seasonResults = GameState.competitionResults
    .filter(r => r.season === GameState.season)
    .sort((a, b) => a.week - b.week);
  const chart = document.getElementById('stats-score-chart');
  if (seasonResults.length === 0) {
    chart.innerHTML = '<div class="empty-slot">No competitions skated this season yet</div>';
  } else {
    const maxScore = Math.max(...seasonResults.map(r => r.playerScore), 1);
    chart.innerHTML = seasonResults.map(r => {
      const pct = Math.max(4, Math.round((r.playerScore / maxScore) * 100));
      const medal = { 1: '🥇', 2: '🥈', 3: '🥉' }[r.placement] || `#${r.placement}`;
      return `
        <div class="score-chart-row">
          <span class="score-chart-label">W${r.week} ${r.competition}</span>
          <div class="score-chart-bar"><div class="score-chart-fill" style="width:${pct}%"></div></div>
          <span class="score-chart-value">${medal} ${r.playerScore.toLocaleString()}</span>
        </div>
      `;
    }).join('');
  }

  // Most-fielded skaters
  const skaters = [...GameState.activeSquad, ...GameState.reserveBench]
    .filter(sk => (sk.appearances || 0) > 0)
    .sort((a, b) => (b.appearances || 0) - (a.appearances || 0))
    .slice(0, 8);
  const appDiv = document.getElementById('stats-appearances');
  if (skaters.length === 0) {
    appDiv.innerHTML = '<div class="empty-slot">No competition appearances yet</div>';
  } else {
    appDiv.innerHTML = skaters.map(sk => `
      <div class="history-card">
        <span>${sk.avatar} ${sk.name} — ${sk.appearances} routine${sk.appearances === 1 ? '' : 's'} (${sk.overall} OVR)</span>
      </div>
    `).join('');
  }

  // Hall of fame
  const hofDiv = document.getElementById('stats-hall-of-fame');
  if (GameState.hallOfFame.length === 0) {
    hofDiv.innerHTML = '<div class="empty-slot">No legends yet — retire a skater with 75+ OVR</div>';
  } else {
    hofDiv.innerHTML = GameState.hallOfFame.map(l => `
      <div class="history-card">
        <span>🏅 ${l.name} — ${l.overall} OVR (retired at ${l.age})</span>
      </div>
    `).join('');
  }
}
