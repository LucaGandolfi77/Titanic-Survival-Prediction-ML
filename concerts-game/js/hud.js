/* hud.js — HUD display updates */
window.G = window.G || {};

G.updateHUD = function() {
  if (!G.state) return;
  document.getElementById('hud-points').textContent = G.state.points.toLocaleString();
  document.getElementById('hud-budget').textContent = '€' + G.state.budget.toLocaleString();
  document.getElementById('hud-attended').textContent = G.state.concertsAttended + '/' + G.state.totalConcerts;
  document.getElementById('hud-date').textContent = G.formatDate(G.state.currentDate);

  var next = G.getNextConcert();
  var el = document.getElementById('hud-next');
  if (next) {
    el.textContent = next.concert.city + ' in ' + next.daysUntil + 'd';
  } else {
    el.textContent = '—';
  }
};
