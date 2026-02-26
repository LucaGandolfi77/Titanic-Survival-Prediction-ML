/* =====================================================
   CELL MODE — Jupyter-style # %% delimiters,
   per-cell run buttons, inline output
   ===================================================== */
window.IDE = window.IDE || {};

IDE.cellMode = (() => {
  'use strict';

  let _enabled = false;
  let _cells = [];

  /* ---- Parse cells from code ---- */
  function parseCells(code) {
    const lines = code.split('\n');
    const cells = [];
    let currentCell = { startLine: 1, endLine: 0, code: [], title: 'Cell 1' };
    let cellCount = 1;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const isDelimiter = /^# %%|^# ---|^# In\[/.test(line);

      if (isDelimiter && i > 0) {
        /* Close current cell */
        currentCell.endLine = i;
        currentCell.code = currentCell.code.join('\n');
        cells.push(currentCell);

        /* Start new cell */
        cellCount++;
        const titleMatch = line.match(/^# %%\s*(.*)/);
        const title = titleMatch && titleMatch[1] ? titleMatch[1].trim() : `Cell ${cellCount}`;
        currentCell = {
          startLine: i + 1,
          endLine: 0,
          code: [],
          title
        };
      } else {
        currentCell.code.push(line);
      }
    }

    /* Close last cell */
    currentCell.endLine = lines.length;
    currentCell.code = currentCell.code.join('\n');
    cells.push(currentCell);

    _cells = cells;
    return cells;
  }

  /* ---- Run a single cell ---- */
  async function runCell(cellIndex) {
    if (cellIndex < 0 || cellIndex >= _cells.length) return;
    const cell = _cells[cellIndex];
    const code = cell.code.trim();
    if (!code) return;

    IDE.outputPanel.appendConsole(`\n▶ Running ${cell.title} (lines ${cell.startLine}-${cell.endLine})`, 'info');

    try {
      const result = await IDE.pythonEngine.execute(code, 'cell');
      if (result && result.result !== null && result.result !== undefined && result.result !== 'None') {
        IDE.outputPanel.appendConsole(String(result.result), 'result');
      }
      IDE.outputPanel.showExecResult(result?.result, result?.elapsed);

      _updateCellStatus(cellIndex, 'success');
    } catch (err) {
      IDE.outputPanel.appendConsole(err.message || String(err), 'stderr');
      _updateCellStatus(cellIndex, 'error');
    }
  }

  /* ---- Run all cells ---- */
  async function runAllCells() {
    for (let i = 0; i < _cells.length; i++) {
      await runCell(i);
    }
  }

  /* ---- Run cells above ---- */
  async function runCellsAbove(cellIndex) {
    for (let i = 0; i < cellIndex; i++) {
      await runCell(i);
    }
  }

  /* ---- Update cell visual status ---- */
  function _updateCellStatus(cellIndex, status) {
    const cellElements = document.querySelectorAll('.cell-marker');
    if (cellElements[cellIndex]) {
      cellElements[cellIndex].className = 'cell-marker cell-' + status;
    }
  }

  /* ---- Render cell mode overlay ---- */
  function renderOverlay() {
    if (!_enabled) {
      _removeOverlay();
      return;
    }

    const tab = IDE.editor.getActiveTab();
    if (!tab) return;

    const cells = parseCells(tab.content);
    const container = document.getElementById('cell-overlay');
    if (!container) return;

    container.innerHTML = '';
    container.style.display = 'block';

    cells.forEach((cell, i) => {
      const marker = document.createElement('div');
      marker.className = 'cell-marker';
      marker.style.top = ((cell.startLine - 1) * 22) + 'px';
      marker.innerHTML = `
        <div class="cell-header">
          <span class="cell-title">${IDE.utils.escapeHtml(cell.title)}</span>
          <div class="cell-actions">
            <button class="cell-run" data-cell="${i}" title="Run Cell (Shift+Enter)">
              <i class="fa-solid fa-play"></i>
            </button>
            <button class="cell-run-above" data-cell="${i}" title="Run Cells Above">
              <i class="fa-solid fa-angles-up"></i>
            </button>
          </div>
        </div>`;
      container.appendChild(marker);
    });

    /* Bind run buttons */
    container.querySelectorAll('.cell-run').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        runCell(parseInt(btn.dataset.cell, 10));
      });
    });
    container.querySelectorAll('.cell-run-above').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        runCellsAbove(parseInt(btn.dataset.cell, 10));
      });
    });
  }

  function _removeOverlay() {
    const container = document.getElementById('cell-overlay');
    if (container) {
      container.innerHTML = '';
      container.style.display = 'none';
    }
  }

  /* ---- Insert cell delimiter ---- */
  function insertCellDelimiter(title) {
    const textarea = document.querySelector('.editor-textarea');
    if (!textarea) return;
    const pos = textarea.selectionStart;
    const before = textarea.value.substring(0, pos);
    const after = textarea.value.substring(pos);
    const prefix = before.endsWith('\n') || before === '' ? '' : '\n';
    const delimiter = `${prefix}# %% ${title || ''}\n`;
    textarea.value = before + delimiter + after;
    textarea.selectionStart = textarea.selectionEnd = pos + delimiter.length;
    textarea.dispatchEvent(new Event('input'));
  }

  /* ---- Get cell at cursor ---- */
  function getCellAtLine(lineNum) {
    for (let i = 0; i < _cells.length; i++) {
      if (lineNum >= _cells[i].startLine && lineNum <= _cells[i].endLine) {
        return i;
      }
    }
    return -1;
  }

  /* ---- Toggle cell mode ---- */
  function toggle() {
    _enabled = !_enabled;
    IDE.settings.set('cellMode', _enabled);
    renderOverlay();
    _updateToggleBtn();
    IDE.utils.emit('cell-mode-changed', { enabled: _enabled });
    IDE.utils.toast(_enabled ? 'Cell mode enabled' : 'Cell mode disabled', 'info');
  }

  function _updateToggleBtn() {
    const btn = document.getElementById('btn-cell-mode');
    if (btn) {
      btn.classList.toggle('active', _enabled);
      btn.title = _enabled ? 'Disable Cell Mode' : 'Enable Cell Mode';
    }
  }

  /* ---- Init ---- */
  function init() {
    _enabled = IDE.settings.get('cellMode') || false;
    _updateToggleBtn();

    const toggleBtn = document.getElementById('btn-cell-mode');
    if (toggleBtn) toggleBtn.addEventListener('click', toggle);

    /* Re-render overlay when content changes */
    IDE.utils.on('editor-content-changed', () => {
      if (_enabled) renderOverlay();
    });
    IDE.utils.on('editor-tab-changed', () => {
      if (_enabled) renderOverlay();
    });
  }

  return {
    init, toggle, parseCells,
    runCell, runAllCells, runCellsAbove,
    insertCellDelimiter, getCellAtLine,
    renderOverlay,
    isEnabled: () => _enabled,
    getCells: () => [..._cells]
  };
})();
