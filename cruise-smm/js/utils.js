/* ===== UTILITY HELPERS ===== */

export function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function randomFloat(min, max) {
  return Math.random() * (max - min) + min;
}

export function randomPick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function formatMoney(n) {
  return `€${n.toLocaleString('en-US')}`;
}

export function formatFame(n) {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return n.toString();
}

export function formatFameExact(n) {
  return n.toLocaleString('en-US');
}

export function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function $(sel) {
  return document.querySelector(sel);
}

export function $$(sel) {
  return document.querySelectorAll(sel);
}

export function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (typeof c === 'string') e.appendChild(document.createTextNode(c));
    else if (c) e.appendChild(c);
  }
  return e;
}

export function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => {
    s.classList.remove('active');
    s.classList.add('hidden');
  });
  const screen = document.getElementById(id);
  if (screen) {
    screen.classList.remove('hidden');
    screen.classList.add('active');
  }
}

export function showPanel(id) {
  const panel = document.getElementById(id);
  if (panel) panel.classList.remove('hidden');
}

export function hidePanel(id) {
  const panel = document.getElementById(id);
  if (panel) panel.classList.add('hidden');
}

export function showModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.classList.remove('hidden');
}

export function hideModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.classList.add('hidden');
}

export function saveGame(state) {
  try {
    localStorage.setItem('cruise-smm-save', JSON.stringify(state));
  } catch (e) {
    console.warn('Save failed:', e);
  }
}

export function loadGame() {
  try {
    const data = localStorage.getItem('cruise-smm-save');
    return data ? JSON.parse(data) : null;
  } catch (e) {
    console.warn('Load failed:', e);
    return null;
  }
}

export function saveRecords(records) {
  try {
    localStorage.setItem('cruise-smm-records', JSON.stringify(records));
  } catch (e) {
    console.warn('Records save failed:', e);
  }
}

export function loadRecords() {
  try {
    const data = localStorage.getItem('cruise-smm-records');
    return data ? JSON.parse(data) : [];
  } catch (e) {
    return [];
  }
}

export function getDayName(day) {
  const names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  return names[(day - 1) % 7];
}

export function getTimeLabel(slotIndex) {
  const times = ['08:00', '09:00', '10:30', '12:00', '13:30', '15:00', '16:30', '18:00', '22:00'];
  return times[slotIndex] || '??:??';
}

export function getSlotName(slotIndex) {
  const names = [
    'Morning Briefing', 'Slot 1', 'Slot 2', 'Lunch Break',
    'Slot 3', 'Slot 4', 'Slot 5', 'Evening Free Time', 'Night'
  ];
  return names[slotIndex] || 'Unknown';
}

export function getSlotIcon(slotIndex) {
  const icons = ['☀️', '📋', '📋', '🍽️', '📋', '📋', '📋', '🌅', '🌙'];
  return icons[slotIndex] || '📋';
}
