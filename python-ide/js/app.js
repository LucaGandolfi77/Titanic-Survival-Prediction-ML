/* =====================================================
   APP — Main application init, global state, menu bar
   wiring, DOMContentLoaded bootstrap
   ===================================================== */
window.IDE = window.IDE || {};

IDE.app = (() => {
  'use strict';

  let _pyReady = false;
  let _running = false;

  /* ---- Splash screen ---- */
  function _showSplash() {
    const splash = document.getElementById('splash-screen');
    if (splash) splash.classList.add('visible');
  }

  function _hideSplash() {
    const splash = document.getElementById('splash-screen');
    if (splash) {
      splash.classList.add('fade-out');
      setTimeout(() => {
        splash.classList.remove('visible', 'fade-out');
        splash.style.display = 'none';
      }, 500);
    }
  }

  function _updateSplash(text, progress) {
    const msgEl = document.getElementById('splash-message');
    const barEl = document.getElementById('splash-progress-bar');
    if (msgEl) msgEl.textContent = text;
    if (barEl) barEl.style.width = progress + '%';
  }

  /* ---- Initialize all modules ---- */
  async function init() {
    _showSplash();
    _updateSplash('Loading IDE...', 5);

    try {
      /* 1. Apply saved theme */
      _updateSplash('Applying theme...', 10);
      const savedTheme = localStorage.getItem('ide-theme') || 'dark';
      IDE.themes.apply(savedTheme);

      /* 2. Init settings */
      _updateSplash('Loading settings...', 15);

      /* 3. Init editor */
      _updateSplash('Initializing editor...', 20);
      IDE.editor.init();

      /* 4. Init output panel */
      _updateSplash('Setting up output panel...', 25);
      IDE.outputPanel.init();

      /* 5. Init file system (IndexedDB) */
      _updateSplash('Loading file system...', 30);
      await IDE.fileSystem.init();

      /* 6. Init file explorer */
      _updateSplash('Building file explorer...', 35);
      IDE.fileExplorer.init();

      /* 7. Init variable inspector */
      _updateSplash('Setting up inspector...', 40);
      IDE.variableInspector.init();

      /* 8. Init debugger */
      _updateSplash('Initializing debugger...', 45);
      IDE.debugger.init();

      /* 9. Init package manager */
      _updateSplash('Loading package manager...', 50);
      IDE.packageManager.init();

      /* 10. Init cell mode */
      _updateSplash('Setting up cell mode...', 55);
      IDE.cellMode.init();

      /* 11. Init command palette */
      _updateSplash('Loading command palette...', 60);
      IDE.commandPalette.init();

      /* 12. Init outline */
      _updateSplash('Building outline...', 65);
      IDE.outline.init();

      /* 13. Setup toolbar & menu bindings */
      _updateSplash('Wiring up toolbar...', 70);
      _setupToolbar();
      _setupSplitters();

      /* 14. Init Python engine (Pyodide download) */
      _updateSplash('Loading Python (Pyodide WebAssembly)...', 75);

      IDE.utils.on('python-loading', (data) => {
        _updateSplash(data.message || 'Loading Python...', 75 + Math.round((data.progress || 0) * 20));
      });

      await IDE.pythonEngine.init();
      _pyReady = true;

      _updateSplash('Python ready!', 100);

      /* Update status bar */
      const statusPy = document.getElementById('status-python');
      if (statusPy) statusPy.innerHTML = '<i class="fab fa-python"></i> Python 3.12 (Pyodide)';

      IDE.outputPanel.appendConsole('Python 3.12 (Pyodide) ready ✓', 'info');
      IDE.outputPanel.appendConsole('Type code below or open a file and press Ctrl+Enter to run.\n', 'info');

    } catch (err) {
      console.error('IDE init error:', err);
      _updateSplash('Error loading Python: ' + err.message, 100);
      IDE.outputPanel.appendConsole('⚠ Python engine failed to load: ' + err.message, 'stderr');
      IDE.outputPanel.appendConsole('Some features may still work without Python execution.', 'warning');
    }

    _hideSplash();
    _updateStatusBar();
  }

  /* ---- Toolbar setup ---- */
  function _setupToolbar() {
    /* Run button */
    _bindBtn('btn-run', () => _runFile());
    _bindBtn('btn-stop', () => _stopExecution());

    /* File operations */
    _bindBtn('btn-new', () => IDE.editor.newFile());
    _bindBtn('btn-templates', () => {
      IDE.templates.showPicker((t) => {
        IDE.editor.newFile(t.name.replace(/\s+/g, '_').toLowerCase() + '.py', t.code);
      });
    });

    /* View toggles */
    _bindBtn('btn-toggle-sidebar', () => {
      const sidebar = document.getElementById('sidebar');
      if (sidebar) {
        sidebar.classList.toggle('collapsed');
        IDE.settings.set('showSidebar', !sidebar.classList.contains('collapsed'));
      }
    });

    /* Sidebar sub-tabs */
    document.querySelectorAll('.sidebar-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sidebar-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.sidebar-tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const tabId = btn.dataset.tab;
        const content = document.getElementById('sidebar-' + tabId);
        if (content) content.classList.add('active');
      });
    });

    /* Listen for run events */
    IDE.utils.on('run-file', () => _runFile());
    IDE.utils.on('run-selection', () => _runSelection());
    IDE.utils.on('run-code', (data) => _runCode(data.code));
    IDE.utils.on('save-file', (data) => _saveFile(data));
  }

  function _bindBtn(id, fn) {
    const btn = document.getElementById(id);
    if (btn) btn.addEventListener('click', fn);
  }

  /* ---- Run file ---- */
  async function _runFile() {
    if (_running) {
      IDE.utils.toast('Code is already running', 'warning');
      return;
    }
    if (!_pyReady) {
      IDE.utils.toast('Python is still loading...', 'warning');
      return;
    }

    const tab = IDE.editor.getActiveTab();
    if (!tab) {
      IDE.utils.toast('No file open', 'warning');
      return;
    }

    const code = tab.content;
    if (!code.trim()) return;

    _running = true;
    _updateRunButton(true);
    IDE.outputPanel.clearProblems();
    IDE.outputPanel.appendConsole(`\n▶ Running ${tab.filename}...`, 'info');

    /* Auto-detect and install missing packages */
    if (IDE.settings.get('autoInstallPackages')) {
      const imports = IDE.packageManager.detectImports(code);
      const toInstall = imports.filter(pkg =>
        !['os', 'sys', 'math', 'json', 'time', 'datetime', 'collections',
          'itertools', 'functools', 'random', 'string', 'io', 're',
          'pathlib', 'typing', 'abc', 'copy', 'operator', 'enum',
          'dataclasses', 'decimal', 'fractions', 'statistics',
          'csv', 'hashlib', 'hmac', 'secrets', 'struct', 'codecs',
          'unicodedata', 'textwrap', 'pprint', 'difflib',
          'logging', 'warnings', 'traceback', 'inspect',
          'unittest', 'doctest', 'contextlib', 'concurrent',
          'asyncio', 'socket', 'http', 'urllib', 'email',
          'html', 'xml', 'base64', 'binascii', 'pickle',
          'shelve', 'sqlite3', 'zlib', 'gzip', 'tarfile', 'zipfile',
          'tempfile', 'shutil', 'glob', 'fnmatch', 'platform',
          'subprocess', 'signal', 'threading', 'multiprocessing',
          'queue', 'heapq', 'bisect', 'array', 'weakref',
          'types', 'numbers', 'cmath', 'builtins',
          'numpy', 'pandas', 'matplotlib', 'micropip'
        ].includes(pkg)
      );
      for (const pkg of toInstall) {
        try {
          IDE.outputPanel.appendConsole(`📦 Auto-installing ${pkg}...`, 'info');
          await IDE.pythonEngine.installPackages([pkg]);
        } catch {
          /* Package might not be available */
        }
      }
    }

    try {
      const result = await IDE.pythonEngine.execute(code, 'run');
      if (result && result.result !== null && result.result !== undefined && result.result !== 'None') {
        IDE.outputPanel.appendConsole(String(result.result), 'result');
      }
      IDE.outputPanel.showExecResult(result?.result, result?.elapsed);
      IDE.outputPanel.addHistory(code, 'run');
    } catch (err) {
      IDE.outputPanel.appendConsole(err.traceback || err.message || String(err), 'stderr');
      IDE.outputPanel.addProblem(err);
    }

    _running = false;
    _updateRunButton(false);
  }

  /* ---- Run selection ---- */
  async function _runSelection() {
    if (!_pyReady || _running) return;
    const selected = IDE.editor.getSelectedText();
    if (!selected.trim()) {
      IDE.utils.toast('No text selected', 'info');
      return;
    }
    await _runCode(selected);
  }

  /* ---- Run arbitrary code ---- */
  async function _runCode(code) {
    if (!_pyReady || _running) return;
    _running = true;
    _updateRunButton(true);

    try {
      const result = await IDE.pythonEngine.execute(code, 'run');
      if (result && result.result !== null && result.result !== undefined && result.result !== 'None') {
        IDE.outputPanel.appendConsole(String(result.result), 'result');
      }
      IDE.outputPanel.showExecResult(result?.result, result?.elapsed);
    } catch (err) {
      IDE.outputPanel.appendConsole(err.traceback || err.message || String(err), 'stderr');
    }

    _running = false;
    _updateRunButton(false);
  }

  /* ---- Stop execution ---- */
  function _stopExecution() {
    if (_running) {
      IDE.pythonEngine.interrupt();
      _running = false;
      _updateRunButton(false);
      IDE.outputPanel.appendConsole('⏹ Execution interrupted', 'warning');
    }
  }

  /* ---- Save file ---- */
  async function _saveFile() {
    const tab = IDE.editor.getActiveTab();
    if (!tab) return;

    try {
      await IDE.fileSystem.writeFile('/' + tab.filename, tab.content);
      tab.dirty = false;
      IDE.utils.toast(`Saved ${tab.filename}`, 'success');
    } catch (err) {
      IDE.utils.toast('Save failed: ' + err.message, 'error');
    }
  }

  /* ---- UI helpers ---- */
  function _updateRunButton(running) {
    const runBtn = document.getElementById('btn-run');
    const stopBtn = document.getElementById('btn-stop');
    if (runBtn) runBtn.disabled = running;
    if (stopBtn) stopBtn.disabled = !running;

    const statusRun = document.getElementById('status-running');
    if (statusRun) {
      statusRun.style.display = running ? 'inline-flex' : 'none';
      statusRun.innerHTML = running ? '<i class="fa-solid fa-spinner fa-spin"></i> Running' : '';
    }
  }

  function _updateStatusBar() {
    const statusReady = document.getElementById('status-ready');
    if (statusReady) statusReady.innerHTML = '<i class="fa-solid fa-circle-check"></i> Ready';
  }

  /* ---- Splitters ---- */
  function _setupSplitters() {
    /* Sidebar splitter */
    const sidebarSplitter = document.getElementById('sidebar-splitter');
    if (sidebarSplitter) {
      IDE.utils.initSplitter(sidebarSplitter, 'horizontal', {
        panel: document.getElementById('sidebar'),
        min: 150,
        max: 400,
        storageKey: 'sidebar-width'
      });
    }

    /* Inspector splitter */
    const inspectorSplitter = document.getElementById('inspector-splitter');
    if (inspectorSplitter) {
      IDE.utils.initSplitter(inspectorSplitter, 'horizontal', {
        panel: document.getElementById('inspector-panel'),
        min: 180,
        max: 450,
        storageKey: 'inspector-width',
        reverse: true
      });
    }

    /* Output splitter */
    const outputSplitter = document.getElementById('output-splitter');
    if (outputSplitter) {
      IDE.utils.initSplitter(outputSplitter, 'vertical', {
        panel: document.getElementById('output-panel'),
        min: 100,
        max: 600,
        storageKey: 'output-height',
        reverse: true
      });
    }
  }

  /* ---- Boot ---- */
  document.addEventListener('DOMContentLoaded', () => {
    init().catch(err => {
      console.error('Fatal IDE init error:', err);
    });
  });

  return {
    init,
    isReady: () => _pyReady,
    isRunning: () => _running
  };
})();
