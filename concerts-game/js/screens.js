/* screens.js — Screen navigation, modal helpers */
window.G = window.G || {};

G.currentScreen = null;

G.SCREEN_IDS = [
  'screen-menu','screen-map','screen-calendar','screen-discovery',
  'screen-booking','screen-concert','screen-result','screen-organise','screen-gameover'
];

/** Switch visible screen */
G.showScreen = function(id) {
  G.SCREEN_IDS.forEach(function(sid) {
    document.getElementById(sid).classList.remove('active');
  });
  document.getElementById(id).classList.add('active');
  G.currentScreen = id;
  // Show/hide HUD (hidden on menu and gameover)
  var hud = document.getElementById('hud');
  hud.style.display = (id === 'screen-menu' || id === 'screen-gameover') ? 'none' : 'flex';
  // If navigating to map, re-render dots
  if (id === 'screen-map') G.renderMap();
  // If navigating to calendar, render it
  if (id === 'screen-calendar') G.renderCalendar();
};

/** Show a modal overlay */
G.showModal = function(id) {
  document.getElementById(id).classList.add('active');
};

/** Close a modal overlay */
G.closeModal = function(id) {
  document.getElementById(id).classList.remove('active');
};

/**
 * Show a non-blocking toast notification.
 * @param {string} msg   - Text to display
 * @param {number} [dur] - Duration in ms (default 2500)
 */
G.toast = function(msg, dur) {
  var container = document.getElementById('toast-container');
  if (!container) return;
  var t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  container.appendChild(t);
  // Trigger entrance animation on next frame
  requestAnimationFrame(function() {
    requestAnimationFrame(function() { t.classList.add('toast-show'); });
  });
  setTimeout(function() {
    t.classList.remove('toast-show');
    t.classList.add('toast-hide');
    setTimeout(function() {
      if (t.parentNode) t.parentNode.removeChild(t);
    }, 350);
  }, dur || 2500);
};
