/* ===== Screen & tab management ===== */

export function showScreen(screenId) {
  document.querySelectorAll('.screen').forEach(s => {
    s.classList.remove('active');
    s.classList.add('hidden');
  });
  const screen = document.getElementById(screenId);
  if (screen) {
    screen.classList.remove('hidden');
    screen.classList.add('active');
  }
}

export function switchTab(tabId) {
  // Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  // Tab panels
  document.querySelectorAll('.tab-panel').forEach(panel => {
    const id = panel.id.replace('tab-', '');
    panel.classList.toggle('active', id === tabId);
    panel.classList.toggle('hidden', id !== tabId);
  });
}
