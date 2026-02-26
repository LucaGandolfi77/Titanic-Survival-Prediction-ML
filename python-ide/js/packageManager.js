/* =====================================================
   PACKAGE MANAGER — Search, install, list packages,
   requirements.txt support
   ===================================================== */
window.IDE = window.IDE || {};

IDE.packageManager = (() => {
  'use strict';

  let _installedPackages = [];
  let _installing = false;

  /* ---- Built-in Pyodide packages (used by isBuiltinPackage) ---- */
  const BUILTIN_PACKAGES = [
    'numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'sympy',
    'networkx', 'pillow', 'statsmodels', 'seaborn', 'sqlalchemy',
    'beautifulsoup4', 'html5lib', 'lxml', 'pyyaml', 'jsonschema',
    'packaging', 'pyparsing', 'six', 'certifi', 'charset-normalizer',
    'idna', 'urllib3', 'requests', 'regex', 'pytz', 'dateutil',
    'jinja2', 'markupsafe'
  ];

  /* ---- Popular packages for suggestions ---- */
  const POPULAR_PACKAGES = [
    { name: 'numpy', desc: 'Fundamental package for scientific computing' },
    { name: 'pandas', desc: 'Data analysis and manipulation library' },
    { name: 'matplotlib', desc: 'Comprehensive plotting library' },
    { name: 'scipy', desc: 'Scientific and technical computing' },
    { name: 'scikit-learn', desc: 'Machine learning algorithms' },
    { name: 'sympy', desc: 'Symbolic mathematics' },
    { name: 'seaborn', desc: 'Statistical data visualization' },
    { name: 'networkx', desc: 'Network/graph analysis' },
    { name: 'pillow', desc: 'Image processing library' },
    { name: 'beautifulsoup4', desc: 'HTML/XML parser' },
    { name: 'statsmodels', desc: 'Statistical models and tests' },
    { name: 'pyyaml', desc: 'YAML parser and emitter' },
    { name: 'regex', desc: 'Alternative regular expression module' },
    { name: 'jsonschema', desc: 'JSON Schema validation' }
  ];

  /* ---- Show package manager modal ---- */
  function showModal() {
    let modal = document.getElementById('package-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'package-modal';
      modal.className = 'modal-overlay';
      modal.innerHTML = `<div class="modal package-dialog">
        <div class="modal-header">
          <h3><i class="fa-solid fa-cube"></i> Package Manager</h3>
          <button class="modal-close"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="modal-body">
          <div class="pkg-search-bar">
            <i class="fa-solid fa-magnifying-glass"></i>
            <input type="text" id="pkg-search-input" placeholder="Search packages (e.g., numpy, pandas)...">
            <button id="btn-pkg-install" class="btn btn-primary" disabled>
              <i class="fa-solid fa-download"></i> Install
            </button>
          </div>
          <div class="pkg-tabs">
            <button class="pkg-tab-btn active" data-tab="installed">Installed</button>
            <button class="pkg-tab-btn" data-tab="available">Available</button>
            <button class="pkg-tab-btn" data-tab="requirements">requirements.txt</button>
          </div>
          <div id="pkg-tab-installed" class="pkg-tab-content active"></div>
          <div id="pkg-tab-available" class="pkg-tab-content"></div>
          <div id="pkg-tab-requirements" class="pkg-tab-content">
            <textarea id="pkg-requirements-text" placeholder="numpy&#10;pandas>=1.5&#10;matplotlib&#10;scikit-learn"></textarea>
            <button id="btn-install-requirements" class="btn btn-primary">
              <i class="fa-solid fa-download"></i> Install All
            </button>
          </div>
          <div id="pkg-install-log" class="pkg-log"></div>
        </div>
      </div>`;
      document.body.appendChild(modal);

      /* Bind events */
      modal.querySelector('.modal-close').addEventListener('click', () => modal.classList.remove('visible'));
      modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('visible'); });

      /* Tab switching */
      modal.querySelectorAll('.pkg-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          modal.querySelectorAll('.pkg-tab-btn').forEach(b => b.classList.remove('active'));
          modal.querySelectorAll('.pkg-tab-content').forEach(c => c.classList.remove('active'));
          btn.classList.add('active');
          document.getElementById('pkg-tab-' + btn.dataset.tab).classList.add('active');
        });
      });

      /* Search */
      const searchInput = modal.querySelector('#pkg-search-input');
      searchInput.addEventListener('input', () => {
        _renderAvailable(searchInput.value);
        const installBtn = modal.querySelector('#btn-pkg-install');
        installBtn.disabled = !searchInput.value.trim();
      });

      /* Install button */
      modal.querySelector('#btn-pkg-install').addEventListener('click', () => {
        const name = searchInput.value.trim();
        if (name) installPackage(name);
      });

      /* Install requirements */
      modal.querySelector('#btn-install-requirements').addEventListener('click', () => {
        const text = modal.querySelector('#pkg-requirements-text').value;
        _installFromRequirements(text);
      });

      /* Enter key in search */
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const name = searchInput.value.trim();
          if (name) installPackage(name);
        }
      });
    }

    _renderInstalled();
    _renderAvailable('');
    modal.classList.add('visible');
    modal.querySelector('#pkg-search-input').focus();
  }

  /* ---- Render installed packages ---- */
  function _renderInstalled() {
    const el = document.getElementById('pkg-tab-installed');
    if (!el) return;

    if (_installedPackages.length === 0) {
      el.innerHTML = `<div class="pkg-empty">
        <i class="fa-solid fa-box-open"></i>
        <p>No additional packages installed.<br>Default: numpy, pandas, matplotlib, micropip</p>
      </div>`;
      return;
    }

    el.innerHTML = `<div class="pkg-list">` +
      _installedPackages.map(pkg => `<div class="pkg-item installed">
        <div class="pkg-info">
          <span class="pkg-name">${IDE.utils.escapeHtml(pkg.name)}</span>
          <span class="pkg-version">${pkg.version || ''}</span>
        </div>
        <span class="pkg-status"><i class="fa-solid fa-circle-check"></i> Installed</span>
      </div>`).join('') +
      `</div>`;
  }

  /* ---- Render available packages ---- */
  function _renderAvailable(query) {
    const el = document.getElementById('pkg-tab-available');
    if (!el) return;

    const q = query.toLowerCase().trim();
    const filtered = q ?
      POPULAR_PACKAGES.filter(p => p.name.includes(q) || p.desc.toLowerCase().includes(q)) :
      POPULAR_PACKAGES;

    if (filtered.length === 0) {
      el.innerHTML = `<div class="pkg-empty">
        <p>No matching packages found.<br>You can still try installing "${IDE.utils.escapeHtml(query)}" via micropip.</p>
      </div>`;
      return;
    }

    el.innerHTML = `<div class="pkg-list">` +
      filtered.map(pkg => {
        const isInstalled = _installedPackages.some(p => p.name === pkg.name);
        return `<div class="pkg-item${isInstalled ? ' installed' : ''}">
          <div class="pkg-info">
            <span class="pkg-name">${IDE.utils.escapeHtml(pkg.name)}</span>
            <span class="pkg-desc">${IDE.utils.escapeHtml(pkg.desc)}</span>
          </div>
          ${isInstalled ?
            '<span class="pkg-status"><i class="fa-solid fa-circle-check"></i></span>' :
            `<button class="btn btn-sm pkg-install-btn" data-pkg="${pkg.name}"><i class="fa-solid fa-download"></i></button>`
          }
        </div>`;
      }).join('') +
      `</div>`;

    el.querySelectorAll('.pkg-install-btn').forEach(btn => {
      btn.addEventListener('click', () => installPackage(btn.dataset.pkg));
    });
  }

  /* ---- Install package ---- */
  async function installPackage(name) {
    if (_installing) {
      IDE.utils.toast('Installation already in progress', 'warning');
      return;
    }

    _installing = true;
    const logEl = document.getElementById('pkg-install-log');
    if (logEl) {
      logEl.style.display = 'block';
      logEl.textContent = `Installing ${name}...\n`;
    }

    try {
      await IDE.pythonEngine.installPackages([name]);
      _installedPackages.push({ name, version: 'latest' });

      /* Save to settings */
      const current = IDE.settings.get('installedPackages') || [];
      if (!current.includes(name)) {
        current.push(name);
        IDE.settings.set('installedPackages', current);
      }

      if (logEl) logEl.textContent += `✓ ${name} installed successfully\n`;
      IDE.utils.toast(`${name} installed`, 'success');
      _renderInstalled();
      _renderAvailable(document.getElementById('pkg-search-input')?.value || '');
    } catch (err) {
      if (logEl) logEl.textContent += `✗ Failed to install ${name}: ${err.message}\n`;
      IDE.utils.toast(`Failed to install ${name}`, 'error');
    }

    _installing = false;
  }

  /* ---- Install from requirements.txt ---- */
  async function _installFromRequirements(text) {
    const packages = text.split('\n')
      .map(l => l.trim())
      .filter(l => l && !l.startsWith('#'))
      .map(l => l.split(/[>=<!\s]/)[0]); /* strip version specifiers */

    if (packages.length === 0) {
      IDE.utils.toast('No packages to install', 'warning');
      return;
    }

    const logEl = document.getElementById('pkg-install-log');
    if (logEl) {
      logEl.style.display = 'block';
      logEl.textContent = `Installing ${packages.length} packages...\n`;
    }

    let success = 0;
    let failed = 0;
    for (const pkg of packages) {
      try {
        await IDE.pythonEngine.installPackages([pkg]);
        _installedPackages.push({ name: pkg, version: 'latest' });
        if (logEl) logEl.textContent += `  ✓ ${pkg}\n`;
        success++;
      } catch (err) {
        if (logEl) logEl.textContent += `  ✗ ${pkg}: ${err.message}\n`;
        failed++;
      }
    }

    if (logEl) logEl.textContent += `\nDone: ${success} installed, ${failed} failed\n`;
    _renderInstalled();
    IDE.utils.toast(`Installed ${success}/${packages.length} packages`, success === packages.length ? 'success' : 'warning');
  }

  /* ---- Auto-detect imports ---- */
  function detectImports(code) {
    const imports = new Set();
    const lines = code.split('\n');
    for (const line of lines) {
      const m1 = line.match(/^import\s+([\w.]+)/);
      if (m1) imports.add(m1[1].split('.')[0]);
      const m2 = line.match(/^from\s+([\w.]+)\s+import/);
      if (m2) imports.add(m2[1].split('.')[0]);
    }
    return Array.from(imports);
  }

  /* ---- Check if a package is built-in to Pyodide ---- */
  function isBuiltin(name) {
    return BUILTIN_PACKAGES.includes(name.toLowerCase());
  }

  /* ---- Init ---- */
  function init() {
    /* Load saved installed packages */
    _installedPackages = (IDE.settings.get('installedPackages') || []).map(name => ({
      name, version: 'latest'
    }));

    /* Listen for install progress */
    IDE.utils.on('python-install-progress', (data) => {
      const logEl = document.getElementById('pkg-install-log');
      if (logEl) logEl.textContent += data.message + '\n';
    });

    /* Open package manager button */
    const openBtn = document.getElementById('btn-packages');
    if (openBtn) openBtn.addEventListener('click', showModal);
  }

  return { init, showModal, installPackage, detectImports, isBuiltin };
})();
