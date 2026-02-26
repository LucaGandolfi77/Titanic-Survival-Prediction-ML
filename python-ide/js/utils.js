/* =====================================================
   UTILS — Shared utility functions
   ===================================================== */
window.IDE = window.IDE || {};

IDE.utils = (() => {
  'use strict';

  function debounce(fn, ms) {
    let timer = null;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  function throttle(fn, ms) {
    let last = 0;
    return function (...args) {
      const now = Date.now();
      if (now - last >= ms) {
        last = now;
        fn.apply(this, args);
      }
    };
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function formatTime(ms) {
    if (ms < 1) return '<1ms';
    if (ms < 1000) return Math.round(ms) + 'ms';
    return (ms / 1000).toFixed(2) + 's';
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function generateId() {
    return 'id_' + Math.random().toString(36).substr(2, 9);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  /* --- Event Bus --- */
  const _listeners = {};

  function on(event, callback) {
    if (!_listeners[event]) _listeners[event] = [];
    _listeners[event].push(callback);
    return () => off(event, callback);
  }

  function off(event, callback) {
    if (!_listeners[event]) return;
    _listeners[event] = _listeners[event].filter(cb => cb !== callback);
  }

  function emit(event, data) {
    if (_listeners[event]) {
      _listeners[event].forEach(cb => {
        try { cb(data); } catch (e) { console.error(`Event handler error [${event}]:`, e); }
      });
    }
  }

  /* --- Toast notifications --- */
  function toast(message, type = 'info', durationMs = 3500) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    const icons = {
      info: 'fa-circle-info',
      success: 'fa-circle-check',
      error: 'fa-circle-xmark',
      warning: 'fa-triangle-exclamation'
    };
    el.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i><span>${escapeHtml(message)}</span>`;
    container.appendChild(el);
    requestAnimationFrame(() => el.classList.add('toast-visible'));
    setTimeout(() => {
      el.classList.remove('toast-visible');
      el.classList.add('toast-exit');
      setTimeout(() => el.remove(), 400);
    }, durationMs);
  }

  /* --- Fuzzy match for command palette --- */
  function fuzzyMatch(query, text) {
    const q = query.toLowerCase();
    const t = text.toLowerCase();
    let qi = 0;
    let score = 0;
    let prevMatch = -1;
    for (let ti = 0; ti < t.length && qi < q.length; ti++) {
      if (t[ti] === q[qi]) {
        score += (prevMatch === ti - 1) ? 10 : 1;
        if (ti === 0 || t[ti - 1] === ' ' || t[ti - 1] === '.') score += 5;
        prevMatch = ti;
        qi++;
      }
    }
    return qi === q.length ? score : 0;
  }

  /* --- Resizable splitter --- */
  function initSplitter(splitterEl, beforeEl, afterEl, direction, opts = {}) {
    const isHorizontal = direction === 'horizontal';
    const minBefore = opts.minBefore || 100;
    const minAfter = opts.minAfter || 100;
    const storageKey = opts.storageKey || null;

    if (storageKey) {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        if (isHorizontal) {
          beforeEl.style.width = saved + 'px';
        } else {
          beforeEl.style.height = saved + 'px';
        }
      }
    }

    let startPos = 0;
    let startSize = 0;

    function onMouseDown(e) {
      e.preventDefault();
      startPos = isHorizontal ? e.clientX : e.clientY;
      startSize = isHorizontal ? beforeEl.getBoundingClientRect().width : beforeEl.getBoundingClientRect().height;
      document.body.style.cursor = isHorizontal ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    }

    function onMouseMove(e) {
      const delta = (isHorizontal ? e.clientX : e.clientY) - startPos;
      let newSize = startSize + delta;
      const parentSize = isHorizontal
        ? splitterEl.parentElement.getBoundingClientRect().width
        : splitterEl.parentElement.getBoundingClientRect().height;
      const splitterSize = isHorizontal ? splitterEl.offsetWidth : splitterEl.offsetHeight;
      const maxBefore = parentSize - splitterSize - minAfter;
      newSize = clamp(newSize, minBefore, maxBefore);
      if (isHorizontal) {
        beforeEl.style.width = newSize + 'px';
      } else {
        beforeEl.style.height = newSize + 'px';
      }
      emit('splitter-resize', { direction, size: newSize });
    }

    function onMouseUp() {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      if (storageKey) {
        const size = isHorizontal ? beforeEl.getBoundingClientRect().width : beforeEl.getBoundingClientRect().height;
        localStorage.setItem(storageKey, Math.round(size));
      }
    }

    splitterEl.addEventListener('mousedown', onMouseDown);
  }

  /* --- Download blob helper --- */
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
  }

  return {
    debounce, throttle, formatBytes, formatTime, escapeHtml,
    generateId, clamp, on, off, emit, toast, fuzzyMatch,
    initSplitter, downloadBlob
  };
})();
