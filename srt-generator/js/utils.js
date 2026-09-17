/* Shared JavaScript Utilities for SRT Generator Suite */

export function fmtTime(s) {
  const h = Math.floor(s / 3600);
  s %= 3600;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.round((s - Math.floor(s)) * 1000);
  return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0') + ',' + String(ms).padStart(3, '0');
}

export function parseTime(t) {
  const m = (t || '').trim().match(/(\d+):(\d+):(\d+),(\d+)/);
  if (!m) return 0;
  return parseInt(m[1]) * 3600 + parseInt(m[2]) * 60 + parseInt(m[3]) + parseInt(m[4]) / 1000;
}

export function parseSRT(text) {
  if (!text) return [];
  const parts = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  const items = [];
  for (const p of parts) {
    const lines = p.split(/\n/);
    let idx = 0;
    if (/^\d+$/.test(lines[0].trim())) idx = 1;
    if (idx >= lines.length) continue;
    const times = lines[idx].split('-->');
    if (times.length < 2) continue;
    const start = parseTime(times[0]);
    const end = parseTime(times[1]);
    const textLines = lines.slice(idx + 1).join('\n');
    items.push({ start, end, text: textLines });
  }
  return items;
}

export function buildSRT(items) {
  return items.map((it, i) => `${i + 1}\n${fmtTime(it.start)} --> ${fmtTime(it.end)}\n${it.text}`).join('\n\n');
}

export function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(2) + ' MB';
}

export function showToast(msg, err) {
  var t = document.createElement('div');
  t.className = 'toast' + (err ? ' error' : ' success');
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 300); }, 2500);
}

export function addHaptic(duration) {
  if (navigator.vibrate) {
    try { navigator.vibrate(duration || 10); } catch (e) { /* not supported */ }
  }
}

/* ── Keyboard Shortcuts System ───────────────────────── */
export function initKeyboardShortcuts(shortcuts, context) {
  var handler = function(e) {
    var tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      if (e.key !== 'Escape') return;
    }
    var key = e.key.toLowerCase();
    var mod = e.ctrlKey || e.metaKey;
    var combo = (mod ? 'ctrl+' : '') + key;
    var comboShift = (mod ? 'ctrl+shift+' : 'shift+') + key;
    var map = {
      'escape': 'esc',
      ' ': 'space',
      'arrowup': 'up',
      'arrowdown': 'down',
      'arrowleft': 'left',
      'arrowright': 'right',
      'delete': 'del',
      'backspace': 'backspace'
    };
    var keyName = map[key] || key;
    var id = combo;
    if (e.shiftKey && !mod) id = comboShift;
    if (shortcuts[id]) {
      e.preventDefault();
      shortcuts[id](e, context);
      addHaptic(5);
    }
  };
  document.addEventListener('keydown', handler);
  return function() { document.removeEventListener('keydown', handler); };
}

export function showShortcutsModal(shortcuts, toolName) {
  var existing = document.getElementById('shortcuts-modal');
  if (existing) { existing.remove(); return; }
  var modal = document.createElement('div');
  modal.id = 'shortcuts-modal';
  modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:10000;display:flex;align-items:center;justify-content:center';
  var content = document.createElement('div');
  content.style.cssText = 'background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;max-width:480px;width:90%;max-height:80vh;overflow-y:auto';
  var html = '<h3 style="margin-bottom:16px;color:var(--accent);font-size:16px">⌨️ Scorciatoie da tastiera — ' + (toolName || 'Generico') + '</h3>';
  html += '<div style="display:flex;flex-direction:column;gap:6px">';
  for (var key in shortcuts) {
    if (!shortcuts.hasOwnProperty(key)) continue;
    var parts = key.split('+');
    var keyDisplay = parts[parts.length - 1].toUpperCase();
    var modDisplay = parts.length > 1 ? parts.slice(0, -1).join(' + ') + ' + ' : '';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border)">';
    html += '<span style="font-size:13px;color:var(--text)">' + shortcuts[key].label + '</span>';
    html += '<code style="background:var(--surface2);padding:2px 8px;border-radius:4px;font-size:12px;color:var(--accent)">' + modDisplay + keyDisplay + '</code>';
    html += '</div>';
  }
  html += '</div>';
  html += '<p style="margin-top:14px;font-size:11px;color:var(--muted);text-align:center">Premi ? o ESC per chiudere</p>';
  content.innerHTML = html;
  modal.appendChild(content);
  modal.addEventListener('click', function(e) { if (e.target === modal) modal.remove(); });
  document.body.appendChild(modal);
}

export function isInputFocused() {
  var tag = (document.activeElement && document.activeElement.tagName || '').toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select';
}

/* ── Auto-Save ───────────────────────────────────── */
export function autoSave(key, value, delay) {
  delay = delay || 1000;
  var timer = null;
  return function() {
    var v = value();
    clearTimeout(timer);
    timer = setTimeout(function() {
      try { localStorage.setItem(key, v); } catch (e) { /* quota */ }
    }, delay);
  };
}

export function autoLoad(key, fallback) {
  try {
    var v = localStorage.getItem(key);
    return v !== null ? v : fallback;
  } catch (e) { return fallback; }
}

export function autoSaveClear(key) {
  try { localStorage.removeItem(key); } catch (e) { /* */ }
}

/* ── Theme Toggle ────────────────────────────────── */
export var THEME_DARK = 'dark';
export var THEME_LIGHT = 'light';

export function getTheme() {
  return localStorage.getItem('srt_theme') || THEME_DARK;
}

export function setTheme(theme) {
  localStorage.setItem('srt_theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
  if (theme === THEME_LIGHT) {
    document.documentElement.style.setProperty('--bg', '#f5f5f7');
    document.documentElement.style.setProperty('--surface', '#ffffff');
    document.documentElement.style.setProperty('--surface2', '#f0f0f3');
    document.documentElement.style.setProperty('--border', '#d2d2d7');
    document.documentElement.style.setProperty('--text', '#1d1d1f');
    document.documentElement.style.setProperty('--muted', '#6e6e73');
  } else {
    document.documentElement.style.setProperty('--bg', '#0f1117');
    document.documentElement.style.setProperty('--surface', '#1a1d27');
    document.documentElement.style.setProperty('--surface2', '#21253a');
    document.documentElement.style.setProperty('--border', '#2e3348');
    document.documentElement.style.setProperty('--text', '#e2e6f0');
    document.documentElement.style.setProperty('--muted', '#7880a0');
  }
  addHaptic(5);
}

export function toggleTheme() {
  setTheme(getTheme() === THEME_DARK ? THEME_LIGHT : THEME_DARK);
}

export function initTheme() {
  var t = getTheme();
  document.documentElement.setAttribute('data-theme', t);
  if (t === THEME_LIGHT) setTheme(THEME_LIGHT);
}

/* ── SRT Validation ──────────────────────────────── */
export function validateSRT(text) {
  var errors = [];
  var warnings = [];
  if (!text || !text.trim()) {
    errors.push('Il file SRT è vuoto');
    return { valid: false, errors: errors, warnings: warnings, blocks: 0 };
  }

  var lines = text.split(/\r?\n/);
  var blockStart = -1;
  var blockCount = 0;
  var lastEnd = 0;
  var indices = [];

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i].trim();

    // Index line
    if (/^\d+$/.test(line)) {
      var num = parseInt(line);
      indices.push(num);
      if (blockStart === -1) blockStart = i;
      // Check sequential indices
      if (blockCount > 0 && num !== indices[0] + blockCount) {
        warnings.push('Riga ' + (i + 1) + ': indice non sequenziale (' + num + ')');
      }
      continue;
    }

    // Time line
    if (line.includes('-->')) {
      var times = line.split('-->');
      if (times.length !== 2) {
        errors.push('Riga ' + (i + 1) + ': formato tempo non valido');
        continue;
      }
      var start = parseTime(times[0]);
      var end = parseTime(times[1]);
      if (isNaN(start) || isNaN(end)) {
        errors.push('Riga ' + (i + 1) + ': tempi non validi');
        continue;
      }
      if (end <= start) {
        errors.push('Riga ' + (i + 1) + ': fine deve essere dopo inizio');
      }
      if (start < lastEnd - 0.01) {
        warnings.push('Riga ' + (i + 1) + ': sovrapposizione temporale');
      }
      lastEnd = end;
      blockCount++;
      continue;
    }

    // Empty line = block separator
    if (line === '' && blockStart !== -1) {
      blockStart = -1;
    }
  }

  if (indices.length === 0 && blockCount === 0) {
    errors.push('Nessun blocco SRT valido trovato');
  }

  return {
    valid: errors.length === 0,
    errors: errors,
    warnings: warnings,
    blocks: blockCount,
    indices: indices
  };
}

/* ── Export Formats ──────────────────────────────── */
export function srtToVTT(srtText) {
  var items = parseSRT(srtText);
  return items.map(function(it, i) {
    return (i + 1) + '\n' + fmtTime(it.start) + ' --> ' + fmtTime(it.end) + '\n' + it.text;
  }).join('\n\n');
}

export function srtToTXT(srtText) {
  var items = parseSRT(srtText);
  return items.map(function(it) {
    return it.text;
  }).join('\n');
}

export function srtToWEBVTT(srtText) {
  var items = parseSRT(srtText);
  var vtt = 'WEBVTT\n\n';
  vtt += items.map(function(it, i) {
    return i + 1 + '\n' + fmtTime(it.start).replace(',', '.') + ' --> ' + fmtTime(it.end).replace(',', '.') + '\n' + it.text;
  }).join('\n\n');
  return vtt;
}

/* ── Undo/Redo ───────────────────────────────────── */
export function createUndoManager(maxSteps) {
  maxSteps = maxSteps || 50;
  var history = [];
  var index = -1;
  return {
    push: function(state) {
      history.splice(index + 1);
      history.push(JSON.stringify(state));
      if (history.length > maxSteps) history.shift();
      index = history.length - 1;
    },
    undo: function() {
      if (index > 0) { index--; return JSON.parse(history[index]); }
      return null;
    },
    redo: function() {
      if (index < history.length - 1) { index++; return JSON.parse(history[index]); }
      return null;
    },
    canUndo: function() { return index > 0; },
    canRedo: function() { return index < history.length - 1; },
    clear: function() { history = []; index = -1; }
  };
}
