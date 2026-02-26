/* =====================================================
   SETTINGS — Settings modal, user preferences,
              localStorage persistence
   ===================================================== */
window.IDE = window.IDE || {};

IDE.settings = (() => {
  'use strict';

  const DEFAULTS = {
    fontSize: 14,
    tabSize: 4,
    wordWrap: false,
    lineNumbers: true,
    minimap: false,
    autoComplete: true,
    autoBrackets: true,
    formatOnSave: false,
    autoInstallPackages: true,
    maxOutputLines: 1000,
    executionTimeout: 30,
    showExecTime: true,
    theme: 'dark',
    sidebarWidth: 220,
    outputHeight: 250,
    inspectorWidth: 260,
    showInspector: true,
    showSidebar: true,
    cellMode: false,
    recentFiles: [],
    installedPackages: []
  };

  let _settings = {};

  function _load() {
    try {
      const saved = localStorage.getItem('ide-settings');
      _settings = saved ? { ...DEFAULTS, ...JSON.parse(saved) } : { ...DEFAULTS };
    } catch {
      _settings = { ...DEFAULTS };
    }
  }

  function _save() {
    localStorage.setItem('ide-settings', JSON.stringify(_settings));
  }

  function get(key) {
    return key in _settings ? _settings[key] : DEFAULTS[key];
  }

  function set(key, value) {
    _settings[key] = value;
    _save();
    IDE.utils.emit('setting-changed', { key, value });
  }

  function getAll() { return { ..._settings }; }

  function reset() {
    _settings = { ...DEFAULTS };
    _save();
    IDE.utils.emit('settings-reset');
  }

  /* --- Settings Modal UI --- */
  function showModal() {
    let modal = document.getElementById('settings-modal');
    if (modal) { modal.classList.add('visible'); return; }

    modal = document.createElement('div');
    modal.id = 'settings-modal';
    modal.className = 'modal-overlay';
    modal.innerHTML = `
      <div class="modal-container modal-lg">
        <div class="modal-header">
          <h2><i class="fa-solid fa-gear"></i> Settings</h2>
          <button class="modal-close" id="settings-close"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="modal-body settings-body">
          <div class="settings-tabs">
            <button class="settings-tab active" data-tab="editor">Editor</button>
            <button class="settings-tab" data-tab="python">Python</button>
            <button class="settings-tab" data-tab="theme">Theme</button>
          </div>

          <div class="settings-panel active" data-panel="editor">
            <div class="setting-row">
              <label>Font Size</label>
              <div class="setting-control">
                <input type="range" min="10" max="24" value="${get('fontSize')}" id="set-fontsize">
                <span class="setting-value" id="set-fontsize-val">${get('fontSize')}px</span>
              </div>
            </div>
            <div class="setting-row">
              <label>Tab Size</label>
              <div class="setting-control">
                <select id="set-tabsize">
                  <option value="2" ${get('tabSize') === 2 ? 'selected' : ''}>2 spaces</option>
                  <option value="4" ${get('tabSize') === 4 ? 'selected' : ''}>4 spaces</option>
                </select>
              </div>
            </div>
            <div class="setting-row">
              <label>Word Wrap</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-wordwrap" ${get('wordWrap') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
            <div class="setting-row">
              <label>Line Numbers</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-linenums" ${get('lineNumbers') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
            <div class="setting-row">
              <label>Minimap</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-minimap" ${get('minimap') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
            <div class="setting-row">
              <label>Auto-Complete</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-autocomplete" ${get('autoComplete') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
            <div class="setting-row">
              <label>Auto-Close Brackets</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-autobrackets" ${get('autoBrackets') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
            <div class="setting-row">
              <label>Format on Save</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-formatonsave" ${get('formatOnSave') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
          </div>

          <div class="settings-panel" data-panel="python">
            <div class="setting-row">
              <label>Auto-Install Packages on Startup</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-autoinstall" ${get('autoInstallPackages') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
            <div class="setting-row">
              <label>Max Output Lines</label>
              <div class="setting-control">
                <input type="number" min="100" max="10000" value="${get('maxOutputLines')}" id="set-maxoutput" class="input-sm">
              </div>
            </div>
            <div class="setting-row">
              <label>Execution Timeout (seconds)</label>
              <div class="setting-control">
                <input type="number" min="5" max="300" value="${get('executionTimeout')}" id="set-timeout" class="input-sm">
              </div>
            </div>
            <div class="setting-row">
              <label>Show Execution Time</label>
              <div class="setting-control">
                <label class="toggle"><input type="checkbox" id="set-showtime" ${get('showExecTime') ? 'checked' : ''}><span class="toggle-slider"></span></label>
              </div>
            </div>
          </div>

          <div class="settings-panel" data-panel="theme">
            <div class="theme-grid" id="theme-grid"></div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    requestAnimationFrame(() => modal.classList.add('visible'));

    /* tab switching */
    modal.querySelectorAll('.settings-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        modal.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
        modal.querySelectorAll('.settings-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        modal.querySelector(`[data-panel="${tab.dataset.tab}"]`).classList.add('active');
      });
    });

    /* close */
    modal.querySelector('#settings-close').addEventListener('click', () => modal.classList.remove('visible'));
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('visible'); });

    /* font size slider */
    const fsSlider = modal.querySelector('#set-fontsize');
    fsSlider.addEventListener('input', () => {
      const v = parseInt(fsSlider.value, 10);
      modal.querySelector('#set-fontsize-val').textContent = v + 'px';
      set('fontSize', v);
    });

    /* tab size */
    modal.querySelector('#set-tabsize').addEventListener('change', function () {
      set('tabSize', parseInt(this.value, 10));
    });

    /* toggles */
    const toggles = [
      ['set-wordwrap', 'wordWrap'],
      ['set-linenums', 'lineNumbers'],
      ['set-minimap', 'minimap'],
      ['set-autocomplete', 'autoComplete'],
      ['set-autobrackets', 'autoBrackets'],
      ['set-formatonsave', 'formatOnSave'],
      ['set-autoinstall', 'autoInstallPackages'],
      ['set-showtime', 'showExecTime']
    ];
    toggles.forEach(([elId, key]) => {
      const el = modal.querySelector('#' + elId);
      if (el) el.addEventListener('change', () => set(key, el.checked));
    });

    /* number inputs */
    modal.querySelector('#set-maxoutput').addEventListener('change', function () {
      set('maxOutputLines', parseInt(this.value, 10));
    });
    modal.querySelector('#set-timeout').addEventListener('change', function () {
      set('executionTimeout', parseInt(this.value, 10));
    });

    /* theme grid */
    _renderThemeGrid(modal.querySelector('#theme-grid'));
  }

  function _renderThemeGrid(container) {
    const themes = IDE.themes.list();
    container.innerHTML = '';
    themes.forEach(t => {
      const card = document.createElement('div');
      card.className = 'theme-card' + (t.id === IDE.themes.current() ? ' active' : '');
      const vars = IDE.themes.THEMES[t.id].vars;
      card.innerHTML = `
        <div class="theme-preview" style="background:${vars['--bg-primary']};border-color:${vars['--border']}">
          <div class="tp-sidebar" style="background:${vars['--bg-secondary']}"></div>
          <div class="tp-editor" style="background:${vars['--editor-bg']}">
            <div class="tp-line" style="background:${vars['--syn-keyword']}"></div>
            <div class="tp-line" style="background:${vars['--syn-string']}"></div>
            <div class="tp-line" style="background:${vars['--syn-function']}"></div>
          </div>
        </div>
        <span class="theme-name">${t.name}</span>
      `;
      card.addEventListener('click', () => {
        IDE.themes.apply(t.id);
        set('theme', t.id);
        container.querySelectorAll('.theme-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
      });
      container.appendChild(card);
    });
  }

  function init() {
    _load();
  }

  return { get, set, getAll, reset, showModal, init };
})();
