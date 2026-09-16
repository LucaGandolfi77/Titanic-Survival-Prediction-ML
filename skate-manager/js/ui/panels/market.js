/* ===== Panel: Market =====
 * Card actions are handled by a single delegated listener in main.js.
 */

import { GameState } from '../../state.js';
import { skaterCardHTML } from '../skater-card.js';

export function renderMarket() {
  const weeksToRefresh = Math.max(0, GameState.marketRefreshWeek - GameState.week);
  document.getElementById('market-refresh-info').textContent =
    weeksToRefresh === 0 ? 'Refreshes when you advance' : `Refreshes in: ${weeksToRefresh} week${weeksToRefresh > 1 ? 's' : ''}`;

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
}
