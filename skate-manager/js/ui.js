/* ===== UI facade — re-exports the split UI modules =====
 *
 * Consumers (main.js) import from here; implementations live in ui/*.
 * Card actions & skater detail clicks are handled by a single delegated
 * listener in main.js, so render functions no longer take handler params.
 */

export { showScreen, switchTab } from './ui/screens.js';
export { showToast, showModal, hideModal, confirmModal } from './ui/feedback.js';
export { showSkaterDetail, showSellModal, showBuyModal, showCompEntryModal } from './ui/modals.js';
export { skaterCardHTML } from './ui/skater-card.js';
export { updateHeader, renderOverview } from './ui/panels/overview.js';
export { renderSquad } from './ui/panels/squad.js';
export { renderMarket } from './ui/panels/market.js';
export { renderCalendar } from './ui/panels/calendar.js';
export { renderSponsors } from './ui/panels/sponsors.js';
export { renderStandings } from './ui/panels/standings.js';
export { renderResults, renderSeasonEnd } from './ui/panels/results.js';
export { renderStats } from './ui/panels/stats.js';
export { refreshAllPanels } from './ui/refresh.js';
