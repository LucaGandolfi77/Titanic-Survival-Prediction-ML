/* =====================================================
   VARIABLE INSPECTOR — Right panel showing variables
   after each execution, DataFrame viewer, tree explorer
   ===================================================== */
window.IDE = window.IDE || {};

IDE.variableInspector = (() => {
  'use strict';

  let _variables = {};

  /* ---- Render variables table ---- */
  function render(vars) {
    _variables = vars || _variables;
    const container = document.getElementById('variable-list');
    if (!container) return;

    const entries = Object.entries(_variables);
    if (entries.length === 0) {
      container.innerHTML = `<div class="inspector-empty">
        <i class="fa-solid fa-flask"></i>
        <p>No variables yet.<br>Run some code to inspect variables.</p>
      </div>`;
      return;
    }

    let html = `<table class="var-table">
      <thead><tr><th>Name</th><th>Type</th><th>Value</th></tr></thead>
      <tbody>`;

    for (const [name, info] of entries) {
      /* Skip private/magic variables */
      if (name.startsWith('__') && name.endsWith('__')) continue;
      if (name.startsWith('_')) continue;

      const typeClass = _getTypeClass(info.type);
      const displayValue = _formatValue(info);
      const shape = info.shape ? `<span class="var-shape">${info.shape}</span>` : '';
      const clickable = _isClickable(info.type);

      html += `<tr class="var-row ${typeClass}${clickable ? ' clickable' : ''}" data-name="${IDE.utils.escapeHtml(name)}">
        <td class="var-name">${IDE.utils.escapeHtml(name)}</td>
        <td class="var-type"><span class="type-badge type-${typeClass}">${IDE.utils.escapeHtml(info.type)}</span>${shape}</td>
        <td class="var-value" title="${IDE.utils.escapeHtml(String(info.value))}">${displayValue}</td>
      </tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;

    /* Bind click handlers for expandable types */
    container.querySelectorAll('.var-row.clickable').forEach(row => {
      row.addEventListener('click', () => {
        const name = row.dataset.name;
        const info = _variables[name];
        if (info) _showDetailModal(name, info);
      });
    });
  }

  function _getTypeClass(type) {
    const t = type.toLowerCase();
    if (t.includes('int') || t.includes('float') || t.includes('complex')) return 'number';
    if (t === 'str') return 'string';
    if (t === 'bool') return 'bool';
    if (t === 'list' || t === 'tuple') return 'list';
    if (t === 'dict') return 'dict';
    if (t === 'set' || t === 'frozenset') return 'set';
    if (t === 'nonetype') return 'none';
    if (t.includes('ndarray') || t.includes('array')) return 'array';
    if (t.includes('dataframe') || t.includes('series')) return 'dataframe';
    if (t === 'function' || t === 'method') return 'function';
    if (t === 'module') return 'module';
    return 'object';
  }

  function _isClickable(type) {
    const t = type.toLowerCase();
    return t.includes('dataframe') || t.includes('series') ||
           t.includes('ndarray') || t.includes('array') ||
           t === 'dict' || t === 'list' || t === 'tuple' || t === 'set';
  }

  function _formatValue(info) {
    const val = String(info.value || '');
    if (val.length > 60) return IDE.utils.escapeHtml(val.substring(0, 57)) + '…';
    return IDE.utils.escapeHtml(val);
  }

  /* ---- Detail modal for complex types ---- */
  function _showDetailModal(name, info) {
    let modal = document.getElementById('var-detail-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'var-detail-modal';
      modal.className = 'modal-overlay';
      modal.innerHTML = `<div class="modal var-detail-dialog">
        <div class="modal-header">
          <h3 id="var-detail-title">Variable</h3>
          <button class="modal-close"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="modal-body" id="var-detail-body"></div>
      </div>`;
      document.body.appendChild(modal);
      modal.querySelector('.modal-close').addEventListener('click', () => modal.classList.remove('visible'));
      modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('visible'); });
    }

    modal.querySelector('#var-detail-title').textContent = `${name} (${info.type})`;
    const body = modal.querySelector('#var-detail-body');

    const t = info.type.toLowerCase();

    if (t.includes('dataframe')) {
      /* Request full DataFrame HTML from Python */
      _renderDataFrameDetail(body, name);
    } else if (t.includes('ndarray') || t.includes('array')) {
      _renderArrayDetail(body, name, info);
    } else if (t === 'dict' || t === 'list' || t === 'tuple' || t === 'set') {
      _renderCollectionDetail(body, name, info);
    } else {
      body.innerHTML = `<pre class="var-detail-pre">${IDE.utils.escapeHtml(String(info.value))}</pre>`;
    }

    modal.classList.add('visible');
  }

  function _renderDataFrameDetail(body, name) {
    body.innerHTML = '<div class="loading-inline"><i class="fa-solid fa-spinner fa-spin"></i> Loading DataFrame...</div>';
    IDE.pythonEngine.execute(`${name}.to_html(max_rows=100, max_cols=50)`, 'eval').then(result => {
      if (result && result.result) {
        body.innerHTML = `<div class="df-viewer">${result.result}</div>`;
      } else {
        body.innerHTML = '<p>Could not render DataFrame</p>';
      }
    }).catch(() => {
      body.innerHTML = '<p>Error loading DataFrame</p>';
    });
  }

  function _renderArrayDetail(body, name, info) {
    body.innerHTML = `<div class="array-viewer">
      <div class="array-meta">
        <span><strong>Shape:</strong> ${info.shape || 'unknown'}</span>
        <span><strong>Dtype:</strong> ${info.dtype || 'unknown'}</span>
      </div>
      <pre class="var-detail-pre">${IDE.utils.escapeHtml(String(info.value))}</pre>
    </div>`;
  }

  function _renderCollectionDetail(body, name, info) {
    body.innerHTML = '<div class="loading-inline"><i class="fa-solid fa-spinner fa-spin"></i> Loading...</div>';
    IDE.pythonEngine.execute(`import json; json.dumps(${name}, default=str, indent=2)`, 'eval').then(result => {
      if (result && result.result) {
        try {
          const parsed = JSON.parse(result.result);
          body.innerHTML = `<pre class="var-detail-pre">${IDE.utils.escapeHtml(JSON.stringify(parsed, null, 2))}</pre>`;
        } catch {
          body.innerHTML = `<pre class="var-detail-pre">${IDE.utils.escapeHtml(result.result)}</pre>`;
        }
      }
    }).catch(() => {
      body.innerHTML = `<pre class="var-detail-pre">${IDE.utils.escapeHtml(String(info.value))}</pre>`;
    });
  }

  /* ---- Init ---- */
  function init() {
    /* Listen for namespace updates */
    IDE.utils.on('python-namespace', (data) => {
      render(data.variables || data);
    });

    /* Toggle inspector panel */
    const toggleBtn = document.getElementById('btn-toggle-inspector');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        const panel = document.getElementById('inspector-panel');
        if (panel) {
          const visible = !panel.classList.contains('collapsed');
          panel.classList.toggle('collapsed', visible);
          IDE.settings.set('showInspector', !visible);
        }
      });
    }

    /* Initial empty state */
    render({});
  }

  return { init, render };
})();
