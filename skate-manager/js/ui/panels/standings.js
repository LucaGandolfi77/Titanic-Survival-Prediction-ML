/* ===== Panel: Standings ===== */

import { GameState } from '../../state.js';

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
