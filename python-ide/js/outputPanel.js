/* =====================================================
   OUTPUT PANEL — Console, Plots, Variables, Problems,
   History — multi-tab output area
   ===================================================== */
window.IDE = window.IDE || {};

IDE.outputPanel = (() => {
  'use strict';

  let _activeOutputTab = 'console';
  let _consoleLines = [];
  let _plots = [];
  let _problems = [];
  let _history = [];
  let _maxLines = 1000;

  /* ---- Tab switching ---- */
  function _switchTab(tabId) {
    _activeOutputTab = tabId;
    document.querySelectorAll('.output-tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    document.querySelectorAll('.output-tab-content').forEach(el => {
      el.classList.toggle('active', el.id === 'output-' + tabId);
    });
  }

  /* ================= CONSOLE ================= */
  function appendConsole(text, type) {
    type = type || 'stdout';
    const entry = { text, type, timestamp: Date.now() };
    _consoleLines.push(entry);
    if (_consoleLines.length > _maxLines) {
      _consoleLines = _consoleLines.slice(-_maxLines);
    }
    _renderConsoleLine(entry);
    _scrollConsoleToBottom();
  }

  function _renderConsoleLine(entry) {
    const consoleEl = document.getElementById('console-output');
    if (!consoleEl) return;

    const div = document.createElement('div');
    div.className = 'console-line console-' + entry.type;

    if (entry.type === 'image') {
      const img = document.createElement('img');
      img.src = entry.text;
      img.className = 'console-plot-inline';
      img.addEventListener('click', () => _showPlotZoom(entry.text));
      div.appendChild(img);
    } else if (entry.type === 'dataframe') {
      div.innerHTML = entry.text;
      div.className = 'console-line console-dataframe';
    } else if (entry.type === 'html') {
      div.innerHTML = entry.text;
    } else {
      const pre = document.createElement('pre');
      pre.textContent = entry.text;
      div.appendChild(pre);
    }
    consoleEl.appendChild(div);
  }

  function _scrollConsoleToBottom() {
    const consoleEl = document.getElementById('console-output');
    if (consoleEl) {
      requestAnimationFrame(() => {
        consoleEl.scrollTop = consoleEl.scrollHeight;
      });
    }
  }

  function clearConsole() {
    _consoleLines = [];
    const consoleEl = document.getElementById('console-output');
    if (consoleEl) consoleEl.innerHTML = '';
  }

  function _copyConsole() {
    const text = _consoleLines
      .filter(l => l.type !== 'image' && l.type !== 'dataframe')
      .map(l => l.text)
      .join('\n');
    navigator.clipboard.writeText(text).then(() => {
      IDE.utils.toast('Console output copied', 'success');
    });
  }

  /* ---- REPL input ---- */
  function _handleReplInput(e) {
    if (e.key !== 'Enter') return;
    const input = e.target;
    const code = input.value.trim();
    if (!code) return;

    appendConsole('>>> ' + code, 'repl-input');
    input.value = '';

    /* Execute in Python */
    IDE.pythonEngine.execute(code, 'repl').then(result => {
      if (result && result.result !== undefined && result.result !== null && result.result !== 'None') {
        appendConsole(String(result.result), 'result');
      }
    }).catch(err => {
      appendConsole(err.message || String(err), 'stderr');
    });

    /* Save to history */
    _history.push({
      code,
      timestamp: Date.now(),
      type: 'repl'
    });
    _renderHistory();
  }

  /* ================= PLOTS ================= */
  function addPlot(base64Src) {
    _plots.push({
      src: base64Src,
      timestamp: Date.now()
    });
    _renderPlots();
    /* Also show inline in console */
    appendConsole(base64Src, 'image');
    /* Switch to plots tab */
    _updateBadge('plots', _plots.length);
  }

  function _renderPlots() {
    const plotsEl = document.getElementById('output-plots');
    if (!plotsEl) return;
    if (_plots.length === 0) {
      plotsEl.innerHTML = '<div class="output-empty"><i class="fa-solid fa-chart-line"></i><p>No plots yet. Use matplotlib to create visualizations.</p></div>';
      return;
    }
    plotsEl.innerHTML = '<div class="plots-gallery">' +
      _plots.map((p, i) => `<div class="plot-item" data-idx="${i}">
        <img src="${p.src}" alt="Plot ${i + 1}">
        <div class="plot-actions">
          <button class="plot-download" data-idx="${i}" title="Download"><i class="fa-solid fa-download"></i></button>
          <button class="plot-zoom" data-idx="${i}" title="Zoom"><i class="fa-solid fa-expand"></i></button>
        </div>
      </div>`).join('') +
      '</div>';

    plotsEl.querySelectorAll('.plot-zoom').forEach(btn => {
      btn.addEventListener('click', () => {
        _showPlotZoom(_plots[parseInt(btn.dataset.idx, 10)].src);
      });
    });
    plotsEl.querySelectorAll('.plot-download').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx, 10);
        _downloadPlot(_plots[idx].src, `plot_${idx + 1}.png`);
      });
    });
  }

  function _showPlotZoom(src) {
    let modal = document.getElementById('plot-zoom-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'plot-zoom-modal';
      modal.className = 'modal-overlay';
      modal.innerHTML = `<div class="modal plot-zoom-dialog">
        <div class="modal-header"><h3>Plot Viewer</h3><button class="modal-close"><i class="fa-solid fa-xmark"></i></button></div>
        <div class="plot-zoom-body"><img id="plot-zoom-img" src="" alt="Plot"></div>
      </div>`;
      document.body.appendChild(modal);
      modal.querySelector('.modal-close').addEventListener('click', () => modal.classList.remove('visible'));
      modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('visible'); });
    }
    modal.querySelector('#plot-zoom-img').src = src;
    modal.classList.add('visible');
  }

  function _downloadPlot(src, filename) {
    const a = document.createElement('a');
    a.href = src;
    a.download = filename;
    a.click();
  }

  function clearPlots() {
    _plots = [];
    _renderPlots();
    _updateBadge('plots', 0);
  }

  /* ================= PROBLEMS ================= */
  function addProblem(error) {
    _problems.push({
      type: error.type || 'Error',
      message: error.message || String(error),
      line: error.line || null,
      filename: error.filename || null,
      timestamp: Date.now()
    });
    _renderProblems();
    _updateBadge('problems', _problems.length);
  }

  function clearProblems() {
    _problems = [];
    _renderProblems();
    _updateBadge('problems', 0);
  }

  function _renderProblems() {
    const el = document.getElementById('output-problems');
    if (!el) return;
    if (_problems.length === 0) {
      el.innerHTML = '<div class="output-empty"><i class="fa-solid fa-circle-check"></i><p>No problems detected.</p></div>';
      return;
    }
    el.innerHTML = _problems.map(p => {
      const lineInfo = p.line ? ` <span class="problem-line" data-line="${p.line}">line ${p.line}</span>` : '';
      return `<div class="problem-item problem-${p.type.toLowerCase().includes('warning') ? 'warning' : 'error'}">
        <i class="fa-solid ${p.type.toLowerCase().includes('warning') ? 'fa-triangle-exclamation' : 'fa-circle-xmark'}"></i>
        <span class="problem-type">${IDE.utils.escapeHtml(p.type)}</span>
        <span class="problem-msg">${IDE.utils.escapeHtml(p.message)}</span>
        ${lineInfo}
      </div>`;
    }).join('');

    el.querySelectorAll('.problem-line').forEach(span => {
      span.style.cursor = 'pointer';
      span.addEventListener('click', () => {
        const ln = parseInt(span.dataset.line, 10);
        if (ln) IDE.editor.goToLine(ln);
      });
    });
  }

  /* ================= HISTORY ================= */
  function addHistory(code, type) {
    _history.push({
      code,
      timestamp: Date.now(),
      type: type || 'run'
    });
    _renderHistory();
  }

  function _renderHistory() {
    const el = document.getElementById('output-history');
    if (!el) return;
    if (_history.length === 0) {
      el.innerHTML = '<div class="output-empty"><i class="fa-solid fa-clock-rotate-left"></i><p>No execution history yet.</p></div>';
      return;
    }
    el.innerHTML = _history.slice().reverse().map((h, i) => {
      const time = new Date(h.timestamp).toLocaleTimeString();
      const preview = h.code.split('\n')[0].substring(0, 80);
      return `<div class="history-item" data-idx="${_history.length - 1 - i}">
        <div class="history-meta">
          <span class="history-type"><i class="fa-solid ${h.type === 'repl' ? 'fa-terminal' : 'fa-play'}"></i> ${h.type}</span>
          <span class="history-time">${time}</span>
        </div>
        <pre class="history-code">${IDE.utils.escapeHtml(preview)}${h.code.length > 80 ? '…' : ''}</pre>
        <div class="history-actions">
          <button class="history-rerun" data-idx="${_history.length - 1 - i}" title="Run again"><i class="fa-solid fa-arrow-rotate-right"></i></button>
          <button class="history-copy" data-idx="${_history.length - 1 - i}" title="Copy"><i class="fa-solid fa-copy"></i></button>
        </div>
      </div>`;
    }).join('');

    el.querySelectorAll('.history-rerun').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx, 10);
        const entry = _history[idx];
        if (entry) IDE.utils.emit('run-code', { code: entry.code });
      });
    });
    el.querySelectorAll('.history-copy').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx, 10);
        const entry = _history[idx];
        if (entry) {
          navigator.clipboard.writeText(entry.code);
          IDE.utils.toast('Code copied', 'success');
        }
      });
    });
  }

  /* ---- Badge ---- */
  function _updateBadge(tabName, count) {
    const btn = document.querySelector(`.output-tab-btn[data-tab="${tabName}"]`);
    if (!btn) return;
    let badge = btn.querySelector('.tab-badge');
    if (count > 0) {
      if (!badge) {
        badge = document.createElement('span');
        badge.className = 'tab-badge';
        btn.appendChild(badge);
      }
      badge.textContent = count;
    } else if (badge) {
      badge.remove();
    }
  }

  /* ---- DataFrame display ---- */
  function renderDataFrame(html) {
    appendConsole(html, 'dataframe');
  }

  /* ---- Execution result display ---- */
  function showExecResult(result, elapsed) {
    if (IDE.settings.get('showExecTime') && elapsed !== undefined) {
      appendConsole(`\n✓ Executed in ${IDE.utils.formatTime(elapsed)}`, 'info');
    }
  }

  /* ================= INIT ================= */
  function init() {
    _maxLines = IDE.settings.get('maxOutputLines');

    /* Tab buttons */
    document.querySelectorAll('.output-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => _switchTab(btn.dataset.tab));
    });

    /* Console action buttons */
    const clearBtn = document.getElementById('btn-clear-console');
    if (clearBtn) clearBtn.addEventListener('click', clearConsole);

    const copyBtn = document.getElementById('btn-copy-console');
    if (copyBtn) copyBtn.addEventListener('click', _copyConsole);

    /* REPL input */
    const replInput = document.getElementById('repl-input');
    if (replInput) replInput.addEventListener('keydown', _handleReplInput);

    /* Listen for Python events */
    IDE.utils.on('python-stdout', (data) => appendConsole(data.text, 'stdout'));
    IDE.utils.on('python-stderr', (data) => appendConsole(data.text, 'stderr'));
    IDE.utils.on('python-plot', (data) => addPlot(data.src));
    IDE.utils.on('python-dataframe', (data) => renderDataFrame(data.html));
    IDE.utils.on('python-error', (data) => {
      appendConsole(data.traceback || data.message, 'stderr');
      addProblem(data);
    });
    IDE.utils.on('python-result', (data) => {
      if (data.result !== null && data.result !== undefined && data.result !== 'None') {
        appendConsole(String(data.result), 'result');
      }
      showExecResult(data.result, data.elapsed);
      addHistory(data.code || '', 'run');
    });

    /* Settings change */
    IDE.utils.on('setting-changed', ({ key, value }) => {
      if (key === 'maxOutputLines') _maxLines = value;
    });

    /* Initial render */
    _renderPlots();
    _renderProblems();
    _renderHistory();
  }

  return {
    init, appendConsole, clearConsole,
    addPlot, clearPlots,
    addProblem, clearProblems,
    addHistory, renderDataFrame,
    showExecResult,
    switchTab: _switchTab
  };
})();
