/* ===== Results & season-end screens ===== */

import { GameState } from '../../state.js';
import { formatMoneyFull } from '../../utils.js';

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
      ${result.tempoBoost > 0 ? `<div class="breakdown-row"><span>🥤 CoolBreeze Boost</span><span>+${result.tempoBoost.toLocaleString()}</span></div>` : ''}
      <div class="breakdown-row"><span>Sync Bonus</span><span>+${result.syncBonus.toLocaleString()}</span></div>
      <div class="breakdown-row negative"><span>Wobble Penalty (applied during routine)</span><span>-${result.wobblePenalty.toLocaleString()}</span></div>
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
  document.getElementById('results-prize').innerHTML = `
    <div class="prize-card ${placement <= 3 ? 'podium' : ''}">
      <h3>Your Placement: ${placement <= 3 ? ['','🥇 1st','🥈 2nd','🥉 3rd'][placement] : `#${placement}`}</h3>
      ${result.prizeMoney > 0 ? `<span>Prize: +${formatMoneyFull(result.prizeMoney)}</span>` : ''}
      <span>Points: +${result.pointsAwarded}</span>
      ${result.fameAwarded > 0 ? `<span>Fame: +${result.fameAwarded}</span>` : ''}
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
      <div class="summary-stat"><span class="big">${formatMoneyFull(totalPrize)}</span><span>Prize Money</span></div>
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
