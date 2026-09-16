/* ===== Full UI refresh ===== */

import { updateHeader, renderOverview } from './panels/overview.js';
import { renderSquad } from './panels/squad.js';
import { renderMarket } from './panels/market.js';
import { renderCalendar } from './panels/calendar.js';
import { renderSponsors } from './panels/sponsors.js';
import { renderStandings } from './panels/standings.js';
import { renderStats } from './panels/stats.js';

export function refreshAllPanels() {
  updateHeader();
  renderOverview();
  renderSquad();
  renderMarket();
  renderCalendar();
  renderSponsors();
  renderStandings();
  renderStats();
}
