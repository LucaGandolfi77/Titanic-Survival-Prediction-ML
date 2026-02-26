/* =====================================================
   COMMAND PALETTE — Ctrl+Shift+P fuzzy search
   for all IDE actions
   ===================================================== */
window.IDE = window.IDE || {};

IDE.commandPalette = (() => {
  'use strict';

  let _commands = [];
  let _selectedIndex = 0;
  let _visible = false;

  /* ---- Register all commands ---- */
  function _registerCommands() {
    _commands = [
      /* File operations */
      { id: 'file.new', label: 'New File', icon: 'fa-file-circle-plus', category: 'File', shortcut: '', action: () => IDE.editor.newFile() },
      { id: 'file.save', label: 'Save File', icon: 'fa-floppy-disk', category: 'File', shortcut: 'Ctrl+S', action: () => IDE.utils.emit('save-file') },
      { id: 'file.upload', label: 'Upload Files', icon: 'fa-upload', category: 'File', shortcut: '', action: () => IDE.fileExplorer.uploadFiles() },

      /* Run operations */
      { id: 'run.file', label: 'Run File', icon: 'fa-play', category: 'Run', shortcut: 'Ctrl+Enter / F5', action: () => IDE.utils.emit('run-file') },
      { id: 'run.selection', label: 'Run Selection', icon: 'fa-play', category: 'Run', shortcut: 'Shift+Enter', action: () => IDE.utils.emit('run-selection') },
      { id: 'run.cell', label: 'Run Current Cell', icon: 'fa-play', category: 'Run', shortcut: '', action: () => {
        const tab = IDE.editor.getActiveTab();
        if (tab) {
          const ci = IDE.cellMode.getCellAtLine(tab.cursorLine);
          if (ci >= 0) IDE.cellMode.runCell(ci);
        }
      }},
      { id: 'run.allCells', label: 'Run All Cells', icon: 'fa-forward', category: 'Run', shortcut: '', action: () => IDE.cellMode.runAllCells() },

      /* Debug operations */
      { id: 'debug.start', label: 'Start Debugging', icon: 'fa-bug', category: 'Debug', shortcut: 'F5', action: () => IDE.debugger.start() },
      { id: 'debug.stop', label: 'Stop Debugging', icon: 'fa-stop', category: 'Debug', shortcut: '', action: () => IDE.debugger.stop() },
      { id: 'debug.breakpoint', label: 'Toggle Breakpoint', icon: 'fa-circle-dot', category: 'Debug', shortcut: 'F9', action: () => {
        const tab = IDE.editor.getActiveTab();
        if (tab) IDE.editor.toggleBreakpoint(tab.id, tab.cursorLine);
      }},

      /* Editor operations */
      { id: 'editor.goToLine', label: 'Go to Line', icon: 'fa-arrow-down-1-9', category: 'Editor', shortcut: 'Ctrl+G', action: () => {
        const input = prompt('Go to line:');
        if (input) IDE.editor.goToLine(parseInt(input, 10));
      }},
      { id: 'editor.toggleComment', label: 'Toggle Comment', icon: 'fa-comment-slash', category: 'Editor', shortcut: 'Ctrl+/', action: () => {
        /* Trigger via keyboard simulation */
        document.querySelector('.editor-textarea')?.dispatchEvent(new KeyboardEvent('keydown', { key: '/', ctrlKey: true }));
      }},
      { id: 'editor.duplicateLine', label: 'Duplicate Line', icon: 'fa-clone', category: 'Editor', shortcut: 'Ctrl+D', action: () => {} },

      /* View operations */
      { id: 'view.toggleSidebar', label: 'Toggle Sidebar', icon: 'fa-sidebar', category: 'View', shortcut: 'Ctrl+B', action: () => {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.toggle('collapsed');
      }},
      { id: 'view.toggleInspector', label: 'Toggle Variable Inspector', icon: 'fa-flask', category: 'View', shortcut: '', action: () => {
        const panel = document.getElementById('inspector-panel');
        if (panel) panel.classList.toggle('collapsed');
      }},
      { id: 'view.toggleOutput', label: 'Toggle Output Panel', icon: 'fa-terminal', category: 'View', shortcut: 'Ctrl+`', action: () => {
        const panel = document.getElementById('output-panel');
        if (panel) panel.classList.toggle('collapsed');
      }},
      { id: 'view.cellMode', label: 'Toggle Cell Mode', icon: 'fa-table-cells', category: 'View', shortcut: '', action: () => IDE.cellMode.toggle() },

      /* Tools */
      { id: 'tools.packages', label: 'Package Manager', icon: 'fa-cube', category: 'Tools', shortcut: '', action: () => IDE.packageManager.showModal() },
      { id: 'tools.templates', label: 'Code Templates', icon: 'fa-wand-magic-sparkles', category: 'Tools', shortcut: '', action: () => IDE.templates.showPicker((t) => IDE.editor.newFile(t.name.replace(/\s+/g, '_').toLowerCase() + '.py', t.code)) },
      { id: 'tools.settings', label: 'Settings', icon: 'fa-gear', category: 'Tools', shortcut: 'Ctrl+,', action: () => IDE.settings.showModal() },

      /* Console */
      { id: 'console.clear', label: 'Clear Console', icon: 'fa-eraser', category: 'Console', shortcut: '', action: () => IDE.outputPanel.clearConsole() },
      { id: 'console.clearPlots', label: 'Clear Plots', icon: 'fa-broom', category: 'Console', shortcut: '', action: () => IDE.outputPanel.clearPlots() },

      /* Theme */
      ...IDE.themes.list().map(t => ({
        id: 'theme.' + t.id,
        label: 'Theme: ' + t.name,
        icon: 'fa-palette',
        category: 'Theme',
        shortcut: '',
        action: () => IDE.themes.apply(t.id)
      }))
    ];
  }

  /* ---- Show/Hide ---- */
  function show() {
    _registerCommands();
    _visible = true;
    _selectedIndex = 0;

    let modal = document.getElementById('cmd-palette');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'cmd-palette';
      modal.className = 'cmd-palette-overlay';
      modal.innerHTML = `<div class="cmd-palette-dialog">
        <div class="cmd-palette-input-wrap">
          <i class="fa-solid fa-angle-right"></i>
          <input type="text" id="cmd-palette-input" placeholder="Type a command..." autocomplete="off" spellcheck="false">
        </div>
        <div class="cmd-palette-list" id="cmd-palette-list"></div>
      </div>`;
      document.body.appendChild(modal);

      modal.addEventListener('click', (e) => {
        if (e.target === modal) hide();
      });
    }

    const input = modal.querySelector('#cmd-palette-input');
    input.value = '';
    _renderList('');

    modal.classList.add('visible');
    input.focus();

    /* Bind input events */
    input.oninput = () => {
      _selectedIndex = 0;
      _renderList(input.value);
    };
    input.onkeydown = (e) => {
      if (e.key === 'Escape') { hide(); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); _selectNext(); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); _selectPrev(); return; }
      if (e.key === 'Enter') { e.preventDefault(); _executeSelected(); return; }
    };
  }

  function hide() {
    _visible = false;
    const modal = document.getElementById('cmd-palette');
    if (modal) modal.classList.remove('visible');
  }

  function toggle() {
    if (_visible) hide(); else show();
  }

  /* ---- Render command list ---- */
  function _renderList(query) {
    const listEl = document.getElementById('cmd-palette-list');
    if (!listEl) return;

    let filtered;
    if (!query.trim()) {
      filtered = _commands;
    } else {
      filtered = _commands
        .map(cmd => ({
          ...cmd,
          score: IDE.utils.fuzzyMatch(query, cmd.label + ' ' + cmd.category)
        }))
        .filter(cmd => cmd.score > 0)
        .sort((a, b) => b.score - a.score);
    }

    if (filtered.length === 0) {
      listEl.innerHTML = '<div class="cmd-palette-empty">No matching commands</div>';
      return;
    }

    listEl.innerHTML = filtered.map((cmd, i) => `<div class="cmd-palette-item${i === _selectedIndex ? ' selected' : ''}" data-idx="${i}">
      <i class="fa-solid ${cmd.icon}"></i>
      <div class="cmd-label">
        <span class="cmd-name">${IDE.utils.escapeHtml(cmd.label)}</span>
        <span class="cmd-category">${IDE.utils.escapeHtml(cmd.category)}</span>
      </div>
      ${cmd.shortcut ? `<span class="cmd-shortcut">${cmd.shortcut}</span>` : ''}
    </div>`).join('');

    /* Click handlers */
    listEl.querySelectorAll('.cmd-palette-item').forEach(el => {
      el.addEventListener('click', () => {
        _selectedIndex = parseInt(el.dataset.idx, 10);
        _executeSelected();
      });
      el.addEventListener('mouseenter', () => {
        _selectedIndex = parseInt(el.dataset.idx, 10);
        _renderList(document.getElementById('cmd-palette-input')?.value || '');
      });
    });

    /* Store filtered for execution */
    listEl._filteredCommands = filtered;
  }

  function _selectNext() {
    const listEl = document.getElementById('cmd-palette-list');
    const items = listEl?.querySelectorAll('.cmd-palette-item');
    if (!items) return;
    _selectedIndex = Math.min(_selectedIndex + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle('selected', i === _selectedIndex));
    items[_selectedIndex]?.scrollIntoView({ block: 'nearest' });
  }

  function _selectPrev() {
    const listEl = document.getElementById('cmd-palette-list');
    const items = listEl?.querySelectorAll('.cmd-palette-item');
    if (!items) return;
    _selectedIndex = Math.max(_selectedIndex - 1, 0);
    items.forEach((el, i) => el.classList.toggle('selected', i === _selectedIndex));
    items[_selectedIndex]?.scrollIntoView({ block: 'nearest' });
  }

  function _executeSelected() {
    const listEl = document.getElementById('cmd-palette-list');
    const filtered = listEl?._filteredCommands;
    if (!filtered || !filtered[_selectedIndex]) return;
    hide();
    filtered[_selectedIndex].action();
  }

  /* ---- Init ---- */
  function init() {
    IDE.utils.on('open-command-palette', show);
  }

  return { init, show, hide, toggle };
})();
