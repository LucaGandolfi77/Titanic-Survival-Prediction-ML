/* =====================================================
   OUTLINE — Parse Python functions, classes, variables,
   TODOs from current file
   ===================================================== */
window.IDE = window.IDE || {};

IDE.outline = (() => {
  'use strict';

  let _items = [];

  /* ---- Parse outline from Python code ---- */
  function parse(code) {
    _items = [];
    if (!code) return _items;
    const lines = code.split('\n');

    let currentClass = null;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineNum = i + 1;

      /* Class definition */
      const classMatch = line.match(/^class\s+(\w+)\s*[:(]/);
      if (classMatch) {
        currentClass = classMatch[1];
        _items.push({
          type: 'class',
          name: classMatch[1],
          line: lineNum,
          icon: 'fa-diamond',
          color: 'var(--syn-class)',
          indent: 0,
          parent: null
        });
        continue;
      }

      /* Function/method definition */
      const funcMatch = line.match(/^(\s*)def\s+(\w+)\s*\(/);
      if (funcMatch) {
        const indent = funcMatch[1].length;
        const isMethod = indent > 0 && currentClass;
        _items.push({
          type: isMethod ? 'method' : 'function',
          name: funcMatch[2],
          line: lineNum,
          icon: isMethod ? 'fa-cube' : 'fa-code',
          color: isMethod ? 'var(--syn-function)' : 'var(--syn-def)',
          indent,
          parent: isMethod ? currentClass : null
        });
        continue;
      }

      /* Async function */
      const asyncMatch = line.match(/^(\s*)async\s+def\s+(\w+)\s*\(/);
      if (asyncMatch) {
        const indent = asyncMatch[1].length;
        const isMethod = indent > 0 && currentClass;
        _items.push({
          type: isMethod ? 'async-method' : 'async-function',
          name: asyncMatch[2],
          line: lineNum,
          icon: 'fa-bolt',
          color: 'var(--syn-keyword)',
          indent,
          parent: isMethod ? currentClass : null
        });
        continue;
      }

      /* If line has no indentation and no class context, reset class */
      if (line.length > 0 && !line.startsWith(' ') && !line.startsWith('\t') && !line.startsWith('#')) {
        currentClass = null;
      }

      /* Top-level variable/constant */
      const varMatch = line.match(/^([A-Z_][A-Z_0-9]*)\s*=/);
      if (varMatch) {
        _items.push({
          type: 'constant',
          name: varMatch[1],
          line: lineNum,
          icon: 'fa-lock',
          color: 'var(--syn-number)',
          indent: 0,
          parent: null
        });
        continue;
      }

      /* Import statements */
      const importMatch = line.match(/^(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)/);
      if (importMatch) {
        const moduleName = importMatch[1] || importMatch[2].split(',')[0].trim();
        _items.push({
          type: 'import',
          name: moduleName,
          line: lineNum,
          icon: 'fa-arrow-right-to-bracket',
          color: 'var(--syn-keyword)',
          indent: 0,
          parent: null
        });
        continue;
      }

      /* TODO/FIXME/HACK/NOTE comments */
      const todoMatch = line.match(/#\s*(TODO|FIXME|HACK|NOTE|XXX|BUG)\s*[:\x2D]?\s*(.*)/i);
      if (todoMatch) {
        _items.push({
          type: 'todo',
          name: todoMatch[1].toUpperCase() + ': ' + todoMatch[2].trim(),
          line: lineNum,
          icon: todoMatch[1].toUpperCase() === 'TODO' ? 'fa-square-check' :
                todoMatch[1].toUpperCase() === 'FIXME' ? 'fa-wrench' :
                todoMatch[1].toUpperCase() === 'BUG' ? 'fa-bug' : 'fa-note-sticky',
          color: todoMatch[1].toUpperCase() === 'TODO' ? '#a6e3a1' :
                 todoMatch[1].toUpperCase() === 'FIXME' ? '#f38ba8' :
                 todoMatch[1].toUpperCase() === 'BUG' ? '#f38ba8' : '#f9e2af',
          indent: 0,
          parent: null
        });
        continue;
      }

      /* Decorator */
      const decoMatch = line.match(/^(\s*)@(\w+)/);
      if (decoMatch) {
        _items.push({
          type: 'decorator',
          name: '@' + decoMatch[2],
          line: lineNum,
          icon: 'fa-at',
          color: 'var(--syn-decorator)',
          indent: decoMatch[1].length,
          parent: currentClass
        });
      }
    }

    return _items;
  }

  /* ---- Render outline ---- */
  function render() {
    const container = document.getElementById('outline-list');
    if (!container) return;

    const tab = IDE.editor.getActiveTab();
    if (!tab) {
      container.innerHTML = '<div class="outline-empty"><i class="fa-solid fa-list"></i><p>Open a file to see its outline.</p></div>';
      return;
    }

    const items = parse(tab.content);

    if (items.length === 0) {
      container.innerHTML = '<div class="outline-empty"><p>No symbols found.</p></div>';
      return;
    }

    /* Group by type */
    const groups = {};
    for (const item of items) {
      const groupName = _getGroupName(item.type);
      if (!groups[groupName]) groups[groupName] = [];
      groups[groupName].push(item);
    }

    let html = '';
    const groupOrder = ['Imports', 'Classes', 'Functions', 'Constants', 'Decorators', 'TODOs'];
    for (const groupName of groupOrder) {
      if (!groups[groupName]) continue;
      html += `<div class="outline-group">
        <div class="outline-group-header">${groupName} (${groups[groupName].length})</div>`;
      for (const item of groups[groupName]) {
        const indent = item.parent ? 16 : 0;
        html += `<div class="outline-item" data-line="${item.line}" style="padding-left: ${indent + 8}px">
          <i class="fa-solid ${item.icon}" style="color: ${item.color}"></i>
          <span class="outline-name">${IDE.utils.escapeHtml(item.name)}</span>
          <span class="outline-line">:${item.line}</span>
        </div>`;
      }
      html += `</div>`;
    }

    container.innerHTML = html;

    /* Click to navigate */
    container.querySelectorAll('.outline-item').forEach(el => {
      el.addEventListener('click', () => {
        const ln = parseInt(el.dataset.line, 10);
        IDE.editor.goToLine(ln);
      });
    });
  }

  function _getGroupName(type) {
    switch (type) {
      case 'class': return 'Classes';
      case 'function': case 'method': case 'async-function': case 'async-method': return 'Functions';
      case 'constant': return 'Constants';
      case 'import': return 'Imports';
      case 'decorator': return 'Decorators';
      case 'todo': return 'TODOs';
      default: return 'Other';
    }
  }

  /* ---- Init ---- */
  function init() {
    IDE.utils.on('editor-content-changed', IDE.utils.debounce(render, 300));
    IDE.utils.on('editor-tab-changed', render);
  }

  return { init, parse, render };
})();
