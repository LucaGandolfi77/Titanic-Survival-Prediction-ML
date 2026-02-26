/* =====================================================
   EDITOR — CodeMirror 6 setup, multi-tab management,
   keybindings, Python autocomplete, cell mode gutter
   ===================================================== */
window.IDE = window.IDE || {};

IDE.editor = (() => {
  'use strict';

  /* We use a simple textarea-based editor with syntax highlighting
     via a lightweight custom tokenizer. CodeMirror 6 ESM bundles
     can't be loaded from plain CDN scripts easily without an import
     map, so we build a capable editor from scratch that looks and
     feels like CodeMirror with Python highlighting. */

  const _tabs = [];
  let _activeTabId = null;
  let _tabCounter = 0;
  let _breakpoints = new Map(); /* fileId -> Set<lineNumber> */

  /* --- Tab data structure --- */
  function _createTabData(filename, content) {
    return {
      id: 'tab_' + (++_tabCounter),
      filename,
      content: content || '',
      dirty: false,
      history: [content || ''],
      historyIndex: 0,
      scrollTop: 0,
      cursorLine: 1,
      cursorCol: 1,
      selection: null
    };
  }

  /* --- Python Tokenizer --- */
  const PY_KEYWORDS = new Set([
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield'
  ]);
  const PY_BUILTINS = new Set([
    'print', 'len', 'range', 'int', 'float', 'str', 'bool', 'list',
    'dict', 'set', 'tuple', 'type', 'isinstance', 'issubclass',
    'input', 'open', 'enumerate', 'zip', 'map', 'filter', 'sorted',
    'reversed', 'abs', 'min', 'max', 'sum', 'any', 'all', 'round',
    'hasattr', 'getattr', 'setattr', 'delattr', 'callable', 'super',
    'property', 'staticmethod', 'classmethod', 'vars', 'dir', 'help',
    'id', 'hash', 'repr', 'format', 'iter', 'next', 'slice', 'object',
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'ImportError', 'RuntimeError', 'StopIteration',
    'FileNotFoundError', 'OSError', 'IOError', 'NotImplementedError',
    'ZeroDivisionError', 'NameError', 'SyntaxError', 'OverflowError',
    'AssertionError', 'SystemExit', 'KeyboardInterrupt'
  ]);

  function tokenizeLine(line) {
    const tokens = [];
    let i = 0;
    while (i < line.length) {
      /* Whitespace */
      if (/\s/.test(line[i])) {
        let start = i;
        while (i < line.length && /\s/.test(line[i])) i++;
        tokens.push({ type: 'space', value: line.slice(start, i) });
        continue;
      }
      /* Comment */
      if (line[i] === '#') {
        tokens.push({ type: 'comment', value: line.slice(i) });
        break;
      }
      /* Strings (single/double/triple quotes) */
      if (line[i] === '"' || line[i] === "'") {
        const q = line[i];
        let start = i;
        /* Check triple quote */
        if (line.substr(i, 3) === q + q + q) {
          i += 3;
          while (i < line.length && line.substr(i, 3) !== q + q + q) {
            if (line[i] === '\\') i++;
            i++;
          }
          if (i < line.length) i += 3;
          tokens.push({ type: 'string', value: line.slice(start, i) });
          continue;
        }
        i++;
        while (i < line.length && line[i] !== q) {
          if (line[i] === '\\') i++;
          i++;
        }
        if (i < line.length) i++;
        tokens.push({ type: 'string', value: line.slice(start, i) });
        continue;
      }
      /* f-string prefix */
      if ((line[i] === 'f' || line[i] === 'r' || line[i] === 'b') &&
          i + 1 < line.length && (line[i+1] === '"' || line[i+1] === "'")) {
        const q = line[i+1];
        let start = i;
        i += 2;
        while (i < line.length && line[i] !== q) {
          if (line[i] === '\\') i++;
          i++;
        }
        if (i < line.length) i++;
        tokens.push({ type: 'string', value: line.slice(start, i) });
        continue;
      }
      /* Numbers */
      if (/\d/.test(line[i]) || (line[i] === '.' && i + 1 < line.length && /\d/.test(line[i+1]))) {
        let start = i;
        if (line[i] === '0' && i + 1 < line.length && (line[i+1] === 'x' || line[i+1] === 'o' || line[i+1] === 'b')) {
          i += 2;
          while (i < line.length && /[0-9a-fA-F_]/.test(line[i])) i++;
        } else {
          while (i < line.length && /[\d._eE+-]/.test(line[i])) i++;
        }
        tokens.push({ type: 'number', value: line.slice(start, i) });
        continue;
      }
      /* Decorator */
      if (line[i] === '@') {
        let start = i;
        i++;
        while (i < line.length && /[\w.]/.test(line[i])) i++;
        tokens.push({ type: 'decorator', value: line.slice(start, i) });
        continue;
      }
      /* Identifiers / keywords */
      if (/[a-zA-Z_]/.test(line[i])) {
        let start = i;
        while (i < line.length && /[\w]/.test(line[i])) i++;
        const word = line.slice(start, i);
        if (PY_KEYWORDS.has(word)) {
          if (word === 'True' || word === 'False' || word === 'None') {
            tokens.push({ type: 'bool', value: word });
          } else if (word === 'def' || word === 'class') {
            tokens.push({ type: 'keyword-def', value: word });
          } else if (word === 'self' || word === 'cls') {
            tokens.push({ type: 'self', value: word });
          } else {
            tokens.push({ type: 'keyword', value: word });
          }
        } else if (word === 'self' || word === 'cls') {
          tokens.push({ type: 'self', value: word });
        } else if (PY_BUILTINS.has(word)) {
          tokens.push({ type: 'builtin', value: word });
        } else if (i < line.length && line[i] === '(') {
          tokens.push({ type: 'function', value: word });
        } else if (word[0] === word[0].toUpperCase() && /^[A-Z]/.test(word)) {
          tokens.push({ type: 'class', value: word });
        } else {
          tokens.push({ type: 'variable', value: word });
        }
        continue;
      }
      /* Operators */
      if (/[+\-*/%=<>!&|^~:;,.()[\]{}]/.test(line[i])) {
        tokens.push({ type: 'operator', value: line[i] });
        i++;
        continue;
      }
      /* Unknown char */
      tokens.push({ type: 'text', value: line[i] });
      i++;
    }
    return tokens;
  }

  function highlightLine(line) {
    const tokens = tokenizeLine(line);
    let html = '';
    for (const tok of tokens) {
      if (tok.type === 'space') {
        html += tok.value;
      } else {
        html += `<span class="syn-${tok.type}">${IDE.utils.escapeHtml(tok.value)}</span>`;
      }
    }
    return html || '\n';
  }

  /* ---- Editor rendering ---- */
  function _renderEditor(tab) {
    const editorEl = document.getElementById('editor-area');
    if (!editorEl) return;

    const lines = tab.content.split('\n');
    const bps = _breakpoints.get(tab.id) || new Set();

    let gutterHtml = '';
    let codeHtml = '';

    for (let i = 0; i < lines.length; i++) {
      const lineNum = i + 1;
      const isActive = lineNum === tab.cursorLine;
      const hasBp = bps.has(lineNum);
      const isCell = /^# %%|^# ---|^# In\[/.test(lines[i]);

      gutterHtml += `<div class="gutter-line${isActive ? ' active' : ''}${hasBp ? ' has-bp' : ''}" data-line="${lineNum}">`;
      gutterHtml += `<span class="bp-dot${hasBp ? ' visible' : ''}"></span>`;
      gutterHtml += `<span class="line-num">${lineNum}</span>`;
      gutterHtml += `</div>`;

      codeHtml += `<div class="code-line${isActive ? ' active' : ''}${isCell ? ' cell-start' : ''}" data-line="${lineNum}">`;
      codeHtml += highlightLine(lines[i]);
      codeHtml += `</div>`;
    }

    const gutterEl = editorEl.querySelector('.editor-gutter');
    const codeEl = editorEl.querySelector('.editor-code');
    const textareaEl = editorEl.querySelector('.editor-textarea');

    if (gutterEl) gutterEl.innerHTML = gutterHtml;
    if (codeEl) codeEl.innerHTML = codeHtml;
    if (textareaEl) {
      textareaEl.value = tab.content;
      textareaEl.style.height = Math.max(lines.length * 22, 300) + 'px';
    }

    /* bind gutter clicks for breakpoints */
    if (gutterEl) {
      gutterEl.querySelectorAll('.gutter-line').forEach(el => {
        el.addEventListener('click', () => {
          const ln = parseInt(el.dataset.line, 10);
          toggleBreakpoint(tab.id, ln);
        });
      });
    }
  }

  /* ---- Tabs ---- */
  function openFile(filename, content) {
    /* Check if already open */
    const existing = _tabs.find(t => t.filename === filename);
    if (existing) {
      activateTab(existing.id);
      return existing.id;
    }
    const tab = _createTabData(filename, content);
    _tabs.push(tab);
    _renderTabs();
    activateTab(tab.id);
    return tab.id;
  }

  function closeTab(tabId) {
    const idx = _tabs.findIndex(t => t.id === tabId);
    if (idx === -1) return;
    _tabs.splice(idx, 1);
    if (_activeTabId === tabId) {
      if (_tabs.length > 0) {
        activateTab(_tabs[Math.min(idx, _tabs.length - 1)].id);
      } else {
        _activeTabId = null;
        _clearEditor();
      }
    }
    _renderTabs();
  }

  function activateTab(tabId) {
    _activeTabId = tabId;
    const tab = _tabs.find(t => t.id === tabId);
    if (tab) {
      _renderEditor(tab);
      _renderTabs();
      _updateStatusBar(tab);
      IDE.utils.emit('editor-tab-changed', { tabId, filename: tab.filename });
    }
  }

  function _clearEditor() {
    const editorEl = document.getElementById('editor-area');
    if (!editorEl) return;
    const gutterEl = editorEl.querySelector('.editor-gutter');
    const codeEl = editorEl.querySelector('.editor-code');
    const textareaEl = editorEl.querySelector('.editor-textarea');
    if (gutterEl) gutterEl.innerHTML = '';
    if (codeEl) codeEl.innerHTML = '<div class="editor-empty">Open a file or create a new one to start coding</div>';
    if (textareaEl) textareaEl.value = '';
  }

  function _renderTabs() {
    const tabBar = document.getElementById('editor-tabs');
    if (!tabBar) return;
    const tabsHtml = _tabs.map(t => {
      const ext = t.filename.split('.').pop();
      const icon = ext === 'py' ? 'fa-python' : ext === 'csv' ? 'fa-table' : ext === 'json' ? 'fa-brackets-curly' : 'fa-file';
      const iconClass = ext === 'py' ? 'fab' : 'fa-solid';
      return `<div class="editor-tab${t.id === _activeTabId ? ' active' : ''}${t.dirty ? ' dirty' : ''}" data-tab="${t.id}">
        <i class="${iconClass} ${icon}"></i>
        <span class="tab-name">${t.filename}${t.dirty ? ' •' : ''}</span>
        <button class="tab-close" data-close="${t.id}"><i class="fa-solid fa-xmark"></i></button>
      </div>`;
    }).join('');
    tabBar.innerHTML = tabsHtml + `<button class="tab-add" id="btn-new-tab" title="New File"><i class="fa-solid fa-plus"></i></button>`;

    /* Bind tab clicks */
    tabBar.querySelectorAll('.editor-tab').forEach(el => {
      el.addEventListener('click', (e) => {
        if (e.target.closest('.tab-close')) return;
        activateTab(el.dataset.tab);
      });
    });
    tabBar.querySelectorAll('.tab-close').forEach(btn => {
      btn.addEventListener('click', () => closeTab(btn.dataset.close));
    });
    const addBtn = tabBar.querySelector('#btn-new-tab');
    if (addBtn) addBtn.addEventListener('click', () => newFile());
  }

  function newFile(filename, content) {
    const name = filename || `untitled_${_tabCounter + 1}.py`;
    return openFile(name, content || `# ${name}\n\n`);
  }

  function getActiveTab() {
    return _tabs.find(t => t.id === _activeTabId) || null;
  }

  function getActiveContent() {
    const tab = getActiveTab();
    return tab ? tab.content : '';
  }

  function getSelectedText() {
    const textarea = document.querySelector('.editor-textarea');
    if (!textarea) return '';
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    if (start === end) return '';
    return textarea.value.substring(start, end);
  }

  /* ---- Content updates from textarea ---- */
  function _onTextareaInput() {
    const tab = getActiveTab();
    if (!tab) return;
    const textarea = document.querySelector('.editor-textarea');
    if (!textarea) return;

    const oldContent = tab.content;
    tab.content = textarea.value;
    tab.dirty = true;

    /* Update cursor position */
    const before = textarea.value.substring(0, textarea.selectionStart);
    const lines = before.split('\n');
    tab.cursorLine = lines.length;
    tab.cursorCol = lines[lines.length - 1].length + 1;

    /* Re-render highlighted code (debounced) */
    _debouncedRender(tab);
    _updateStatusBar(tab);

    /* Push history for undo */
    if (tab.content !== oldContent) {
      if (tab.historyIndex < tab.history.length - 1) {
        tab.history = tab.history.slice(0, tab.historyIndex + 1);
      }
      tab.history.push(tab.content);
      if (tab.history.length > 200) tab.history.shift();
      tab.historyIndex = tab.history.length - 1;
    }

    _renderTabs();
    IDE.utils.emit('editor-content-changed', { tabId: tab.id, content: tab.content });
  }

  const _debouncedRender = IDE.utils.debounce((tab) => {
    _renderEditor(tab);
  }, 100);

  /* ---- Status bar update ---- */
  function _updateStatusBar(tab) {
    const lnCol = document.getElementById('status-ln-col');
    if (lnCol) lnCol.textContent = `Ln ${tab.cursorLine}, Col ${tab.cursorCol}`;
  }

  /* ---- Breakpoints ---- */
  function toggleBreakpoint(fileId, lineNum) {
    if (!_breakpoints.has(fileId)) _breakpoints.set(fileId, new Set());
    const bps = _breakpoints.get(fileId);
    if (bps.has(lineNum)) {
      bps.delete(lineNum);
    } else {
      bps.add(lineNum);
    }
    const tab = getActiveTab();
    if (tab && tab.id === fileId) _renderEditor(tab);
    IDE.utils.emit('breakpoint-changed', { fileId, lineNum, active: bps.has(lineNum) });
  }

  function getBreakpoints(fileId) {
    return _breakpoints.get(fileId) || new Set();
  }

  /* ---- Keybindings ---- */
  function _handleKeydown(e) {
    const tab = getActiveTab();
    const textarea = document.querySelector('.editor-textarea');

    /* Ctrl+Enter: Run file */
    if (e.ctrlKey && e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      IDE.utils.emit('run-file');
      return;
    }
    /* Shift+Enter: Run selection */
    if (e.shiftKey && e.key === 'Enter' && !e.ctrlKey) {
      e.preventDefault();
      IDE.utils.emit('run-selection');
      return;
    }
    /* Ctrl+S: Save */
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();
      if (tab) { tab.dirty = false; _renderTabs(); }
      IDE.utils.emit('save-file', { tabId: tab ? tab.id : null });
      return;
    }
    /* Ctrl+/: Toggle comment */
    if (e.ctrlKey && e.key === '/') {
      e.preventDefault();
      _toggleComment(textarea);
      return;
    }
    /* Ctrl+D: Duplicate line */
    if (e.ctrlKey && e.key === 'd') {
      e.preventDefault();
      _duplicateLine(textarea);
      return;
    }
    /* Ctrl+G: Go to line */
    if (e.ctrlKey && e.key === 'g') {
      e.preventDefault();
      _goToLine();
      return;
    }
    /* Ctrl+Shift+P: Command palette */
    if (e.ctrlKey && e.shiftKey && e.key === 'P') {
      e.preventDefault();
      IDE.utils.emit('open-command-palette');
      return;
    }
    /* F5: Run */
    if (e.key === 'F5') {
      e.preventDefault();
      IDE.utils.emit('run-file');
      return;
    }
    /* F9: Toggle breakpoint */
    if (e.key === 'F9') {
      e.preventDefault();
      if (tab) toggleBreakpoint(tab.id, tab.cursorLine);
      return;
    }
    /* Ctrl+,: Settings */
    if (e.ctrlKey && e.key === ',') {
      e.preventDefault();
      IDE.settings.showModal();
      return;
    }
    /* Tab: indent */
    if (e.key === 'Tab' && !e.ctrlKey && textarea && document.activeElement === textarea) {
      e.preventDefault();
      const tabSize = IDE.settings.get('tabSize');
      const spaces = ' '.repeat(tabSize);
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;

      if (start === end) {
        /* No selection: insert spaces */
        textarea.value = textarea.value.substring(0, start) + spaces + textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = start + tabSize;
      } else if (e.shiftKey) {
        /* Shift+Tab: dedent selection */
        _dedentSelection(textarea, tabSize);
      } else {
        /* Tab: indent selection */
        _indentSelection(textarea, tabSize);
      }
      _onTextareaInput();
      return;
    }
    /* Shift+Tab: dedent */
    if (e.shiftKey && e.key === 'Tab' && textarea && document.activeElement === textarea) {
      e.preventDefault();
      _dedentSelection(textarea, IDE.settings.get('tabSize'));
      _onTextareaInput();
      return;
    }
  }

  function _toggleComment(textarea) {
    if (!textarea) return;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const value = textarea.value;
    const lineStart = value.lastIndexOf('\n', start - 1) + 1;
    const lineEnd = value.indexOf('\n', end);
    const lineEndFinal = lineEnd === -1 ? value.length : lineEnd;
    const selectedLines = value.substring(lineStart, lineEndFinal).split('\n');

    const allCommented = selectedLines.every(l => l.trimStart().startsWith('# ') || l.trim() === '');
    const modified = selectedLines.map(l => {
      if (allCommented) {
        return l.replace(/^(\s*)# ?/, '$1');
      } else {
        if (l.trim() === '') return l;
        return l.replace(/^(\s*)/, '$1# ');
      }
    }).join('\n');

    textarea.value = value.substring(0, lineStart) + modified + value.substring(lineEndFinal);
    textarea.selectionStart = lineStart;
    textarea.selectionEnd = lineStart + modified.length;
    _onTextareaInput();
  }

  function _duplicateLine(textarea) {
    if (!textarea) return;
    const pos = textarea.selectionStart;
    const value = textarea.value;
    const lineStart = value.lastIndexOf('\n', pos - 1) + 1;
    let lineEnd = value.indexOf('\n', pos);
    if (lineEnd === -1) lineEnd = value.length;
    const line = value.substring(lineStart, lineEnd);
    textarea.value = value.substring(0, lineEnd) + '\n' + line + value.substring(lineEnd);
    textarea.selectionStart = textarea.selectionEnd = pos + line.length + 1;
    _onTextareaInput();
  }

  function _indentSelection(textarea, tabSize) {
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const value = textarea.value;
    const lineStart = value.lastIndexOf('\n', start - 1) + 1;
    const lineEnd = value.indexOf('\n', end);
    const lineEndFinal = lineEnd === -1 ? value.length : lineEnd;
    const lines = value.substring(lineStart, lineEndFinal).split('\n');
    const indented = lines.map(l => ' '.repeat(tabSize) + l).join('\n');
    textarea.value = value.substring(0, lineStart) + indented + value.substring(lineEndFinal);
    textarea.selectionStart = lineStart;
    textarea.selectionEnd = lineStart + indented.length;
  }

  function _dedentSelection(textarea, tabSize) {
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const value = textarea.value;
    const lineStart = value.lastIndexOf('\n', start - 1) + 1;
    const lineEnd = value.indexOf('\n', end);
    const lineEndFinal = lineEnd === -1 ? value.length : lineEnd;
    const lines = value.substring(lineStart, lineEndFinal).split('\n');
    const dedented = lines.map(l => {
      const spaces = l.match(/^ */)[0].length;
      const remove = Math.min(spaces, tabSize);
      return l.substring(remove);
    }).join('\n');
    textarea.value = value.substring(0, lineStart) + dedented + value.substring(lineEndFinal);
    textarea.selectionStart = lineStart;
    textarea.selectionEnd = lineStart + dedented.length;
  }

  function _goToLine() {
    const input = prompt('Go to line:');
    if (!input) return;
    const line = parseInt(input, 10);
    if (isNaN(line)) return;
    goToLine(line);
  }

  function goToLine(lineNum) {
    const textarea = document.querySelector('.editor-textarea');
    if (!textarea) return;
    const lines = textarea.value.split('\n');
    const target = Math.max(1, Math.min(lineNum, lines.length));
    let pos = 0;
    for (let i = 0; i < target - 1; i++) {
      pos += lines[i].length + 1;
    }
    textarea.selectionStart = textarea.selectionEnd = pos;
    textarea.focus();

    const tab = getActiveTab();
    if (tab) {
      tab.cursorLine = target;
      tab.cursorCol = 1;
      _renderEditor(tab);
      _updateStatusBar(tab);
    }
  }

  function setContent(content) {
    const tab = getActiveTab();
    if (!tab) return;
    tab.content = content;
    tab.dirty = true;
    _renderEditor(tab);
    _renderTabs();
  }

  /* ---- Init ---- */
  function init() {
    /* Bind textarea events */
    const textarea = document.querySelector('.editor-textarea');
    if (textarea) {
      textarea.addEventListener('input', _onTextareaInput);
      textarea.addEventListener('click', () => {
        const tab = getActiveTab();
        if (!tab) return;
        const before = textarea.value.substring(0, textarea.selectionStart);
        const lines = before.split('\n');
        tab.cursorLine = lines.length;
        tab.cursorCol = lines[lines.length - 1].length + 1;
        _updateStatusBar(tab);
        _renderEditor(tab);
      });
      textarea.addEventListener('keyup', () => {
        const tab = getActiveTab();
        if (!tab) return;
        const before = textarea.value.substring(0, textarea.selectionStart);
        const lines = before.split('\n');
        tab.cursorLine = lines.length;
        tab.cursorCol = lines[lines.length - 1].length + 1;
        _updateStatusBar(tab);
      });
      textarea.addEventListener('scroll', () => {
        const gutter = document.querySelector('.editor-gutter');
        const code = document.querySelector('.editor-code');
        if (gutter) gutter.scrollTop = textarea.scrollTop;
        if (code) code.scrollTop = textarea.scrollTop;
      });

      /* Apply font size from settings */
      textarea.style.fontSize = IDE.settings.get('fontSize') + 'px';
      textarea.style.tabSize = IDE.settings.get('tabSize');
    }

    document.addEventListener('keydown', _handleKeydown);

    /* React to settings changes */
    IDE.utils.on('setting-changed', ({ key, value }) => {
      const ta = document.querySelector('.editor-textarea');
      if (!ta) return;
      if (key === 'fontSize') {
        ta.style.fontSize = value + 'px';
        const code = document.querySelector('.editor-code');
        const gutter = document.querySelector('.editor-gutter');
        if (code) code.style.fontSize = value + 'px';
        if (gutter) gutter.style.fontSize = value + 'px';
      }
      if (key === 'tabSize') ta.style.tabSize = value;
    });

    /* Open a default file */
    newFile('main.py', '# main.py — Welcome to Python IDE!\n# Press Ctrl+Enter to run your code\n\nprint("Hello, World!")\n');
  }

  return {
    init, openFile, closeTab, activateTab, newFile,
    getActiveTab, getActiveContent, getSelectedText,
    setContent, goToLine, toggleBreakpoint, getBreakpoints,
    highlightLine, tokenizeLine,
    getTabs: () => [..._tabs]
  };
})();
