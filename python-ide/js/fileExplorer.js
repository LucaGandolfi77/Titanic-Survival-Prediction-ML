/* =====================================================
   FILE EXPLORER — Tree view, context menu, drag-drop,
   upload, icons, click-to-open
   ===================================================== */
window.IDE = window.IDE || {};

IDE.fileExplorer = (() => {
  'use strict';

  let _expandedDirs = new Set(['/projects', '/projects/demo']);
  let _selectedPath = null;

  /* ---- File icon mapping ---- */
  function _getIcon(name, type) {
    if (type === 'directory') return '<i class="fa-solid fa-folder icon-folder"></i>';
    const ext = name.split('.').pop().toLowerCase();
    const map = {
      'py':   '<i class="fab fa-python icon-python"></i>',
      'js':   '<i class="fab fa-js icon-js"></i>',
      'json': '<i class="fa-solid fa-brackets-curly icon-json"></i>',
      'csv':  '<i class="fa-solid fa-table icon-csv"></i>',
      'txt':  '<i class="fa-solid fa-file-lines icon-txt"></i>',
      'md':   '<i class="fa-solid fa-file-lines icon-md"></i>',
      'html': '<i class="fa-solid fa-code icon-html"></i>',
      'css':  '<i class="fa-solid fa-paintbrush icon-css"></i>',
      'yaml': '<i class="fa-solid fa-file-code icon-yaml"></i>',
      'yml':  '<i class="fa-solid fa-file-code icon-yaml"></i>',
      'toml': '<i class="fa-solid fa-file-code icon-toml"></i>',
      'ini':  '<i class="fa-solid fa-gear icon-ini"></i>',
      'png':  '<i class="fa-solid fa-image icon-img"></i>',
      'jpg':  '<i class="fa-solid fa-image icon-img"></i>',
      'svg':  '<i class="fa-solid fa-image icon-img"></i>',
      'zip':  '<i class="fa-solid fa-file-zipper icon-zip"></i>'
    };
    return map[ext] || '<i class="fa-solid fa-file icon-default"></i>';
  }

  /* ---- Render tree ---- */
  function render() {
    const container = document.getElementById('file-tree');
    if (!container) return;

    const tree = IDE.fileSystem.getTree();
    container.innerHTML = _renderNode(tree, 0);
    _bindTreeEvents(container);
  }

  function _renderNode(node, depth) {
    if (depth === 0) {
      /* Root node — render children only */
      return (node.children || []).map(c => _renderNode(c, depth)).join('');
    }

    const isDir = node.type === 'directory';
    const isExpanded = _expandedDirs.has(node.path);
    const isSelected = _selectedPath === node.path;
    const pad = (depth - 1) * 16;

    let html = `<div class="tree-item${isSelected ? ' selected' : ''}${isDir ? ' is-dir' : ''}" 
      data-path="${node.path}" data-type="${node.type}" 
      style="padding-left: ${pad + 8}px" draggable="true">`;

    if (isDir) {
      html += `<span class="tree-arrow ${isExpanded ? 'expanded' : ''}"><i class="fa-solid fa-chevron-right"></i></span>`;
      html += `<span class="tree-icon">${_getIcon(node.name, 'directory')}</span>`;
    } else {
      html += `<span class="tree-arrow-spacer"></span>`;
      html += `<span class="tree-icon">${_getIcon(node.name, 'file')}</span>`;
    }

    html += `<span class="tree-name">${IDE.utils.escapeHtml(node.name)}</span>`;
    html += `</div>`;

    if (isDir && isExpanded && node.children) {
      html += node.children.map(c => _renderNode(c, depth + 1)).join('');
    }

    return html;
  }

  function _bindTreeEvents(container) {
    /* Click to select / open */
    container.querySelectorAll('.tree-item').forEach(el => {
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const path = el.dataset.path;
        const type = el.dataset.type;
        _selectedPath = path;

        if (type === 'directory') {
          if (_expandedDirs.has(path)) {
            _expandedDirs.delete(path);
          } else {
            _expandedDirs.add(path);
          }
          render();
        } else {
          _openFileInEditor(path);
          render(); /* re-render for selection highlight */
        }
      });

      /* Right-click context menu */
      el.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        _selectedPath = el.dataset.path;
        render();
        _showContextMenu(e.clientX, e.clientY, el.dataset.path, el.dataset.type);
      });

      /* Drag start */
      el.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/plain', el.dataset.path);
        e.dataTransfer.effectAllowed = 'move';
      });

      /* Drop on directories */
      if (el.dataset.type === 'directory') {
        el.addEventListener('dragover', (e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'move';
          el.classList.add('drag-over');
        });
        el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
        el.addEventListener('drop', async (e) => {
          e.preventDefault();
          el.classList.remove('drag-over');
          const srcPath = e.dataTransfer.getData('text/plain');
          const targetDir = el.dataset.path;
          const fileName = srcPath.split('/').pop();
          const newPath = targetDir + '/' + fileName;
          if (srcPath !== newPath) {
            try {
              await IDE.fileSystem.rename(srcPath, newPath);
              render();
              IDE.utils.toast(`Moved to ${targetDir}`, 'success');
            } catch (err) {
              IDE.utils.toast('Move failed: ' + err.message, 'error');
            }
          }
        });
      }
    });
  }

  /* ---- Open file in editor ---- */
  async function _openFileInEditor(path) {
    try {
      const content = await IDE.fileSystem.readFile(path);
      const name = path.split('/').pop();
      IDE.editor.openFile(name, content);
    } catch (err) {
      IDE.utils.toast('Could not open file: ' + err.message, 'error');
    }
  }

  /* ---- Context menu ---- */
  function _showContextMenu(x, y, path, type) {
    _removeContextMenu();

    const menu = document.createElement('div');
    menu.id = 'file-context-menu';
    menu.className = 'context-menu';

    const items = [];
    if (type === 'directory') {
      items.push({ label: 'New File', icon: 'fa-file-circle-plus', action: () => _promptNewFile(path) });
      items.push({ label: 'New Folder', icon: 'fa-folder-plus', action: () => _promptNewFolder(path) });
      items.push({ divider: true });
    }
    items.push({ label: 'Rename', icon: 'fa-pen', action: () => _promptRename(path) });
    items.push({ label: 'Delete', icon: 'fa-trash', action: () => _promptDelete(path, type) });
    items.push({ divider: true });
    items.push({ label: 'Copy Path', icon: 'fa-copy', action: () => { navigator.clipboard.writeText(path); IDE.utils.toast('Path copied', 'success'); } });
    if (type === 'file') {
      items.push({ label: 'Download', icon: 'fa-download', action: () => _downloadFile(path) });
    }

    for (const item of items) {
      if (item.divider) {
        const hr = document.createElement('div');
        hr.className = 'context-menu-divider';
        menu.appendChild(hr);
        continue;
      }
      const btn = document.createElement('button');
      btn.className = 'context-menu-item';
      btn.innerHTML = `<i class="fa-solid ${item.icon}"></i> ${item.label}`;
      btn.addEventListener('click', () => {
        _removeContextMenu();
        item.action();
      });
      menu.appendChild(btn);
    }

    /* Position */
    menu.style.left = x + 'px';
    menu.style.top = y + 'px';
    document.body.appendChild(menu);

    /* Adjust if offscreen */
    requestAnimationFrame(() => {
      const rect = menu.getBoundingClientRect();
      if (rect.right > window.innerWidth) menu.style.left = (x - rect.width) + 'px';
      if (rect.bottom > window.innerHeight) menu.style.top = (y - rect.height) + 'px';
    });

    /* Close on click outside */
    setTimeout(() => {
      document.addEventListener('click', _removeContextMenu, { once: true });
    }, 10);
  }

  function _removeContextMenu() {
    const menu = document.getElementById('file-context-menu');
    if (menu) menu.remove();
  }

  /* ---- Prompt dialogs ---- */
  function _promptNewFile(dirPath) {
    const name = prompt('New file name:', 'new_file.py');
    if (!name) return;
    const fullPath = dirPath + '/' + name;
    IDE.fileSystem.writeFile(fullPath, '').then(() => {
      _expandedDirs.add(dirPath);
      render();
      _openFileInEditor(fullPath);
      IDE.utils.toast(`Created ${name}`, 'success');
    });
  }

  function _promptNewFolder(dirPath) {
    const name = prompt('New folder name:', 'new_folder');
    if (!name) return;
    IDE.fileSystem.createDirectory(dirPath + '/' + name).then(() => {
      _expandedDirs.add(dirPath);
      render();
      IDE.utils.toast(`Created folder ${name}`, 'success');
    });
  }

  function _promptRename(path) {
    const oldName = path.split('/').pop();
    const newName = prompt('Rename to:', oldName);
    if (!newName || newName === oldName) return;
    const parent = path.split('/').slice(0, -1).join('/');
    IDE.fileSystem.rename(path, parent + '/' + newName).then(() => {
      render();
      IDE.utils.toast(`Renamed to ${newName}`, 'success');
    });
  }

  function _promptDelete(path, type) {
    const name = path.split('/').pop();
    if (!confirm(`Delete ${type} "${name}"?`)) return;
    const fn = type === 'directory' ? IDE.fileSystem.deleteDirectory : IDE.fileSystem.deleteFile;
    fn(path).then(() => {
      render();
      IDE.utils.toast(`Deleted ${name}`, 'success');
    });
  }

  async function _downloadFile(path) {
    try {
      const content = await IDE.fileSystem.readFile(path);
      const name = path.split('/').pop();
      IDE.utils.downloadBlob(new Blob([content], { type: 'text/plain' }), name);
    } catch (err) {
      IDE.utils.toast('Download failed: ' + err.message, 'error');
    }
  }

  /* ---- Upload ---- */
  function uploadFiles() {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.py,.txt,.csv,.json,.md,.yaml,.yml,.toml,.html,.css,.js';
    input.addEventListener('change', async () => {
      const files = Array.from(input.files);
      const targetDir = _selectedPath && _selectedPath !== '/' ?
        (_selectedPath.endsWith('/') ? _selectedPath : _selectedPath.substring(0, _selectedPath.lastIndexOf('/')) || '/projects') :
        '/projects';

      for (const file of files) {
        const content = await file.text();
        await IDE.fileSystem.writeFile(targetDir + '/' + file.name, content);
      }
      render();
      IDE.utils.toast(`Uploaded ${files.length} file(s)`, 'success');
    });
    input.click();
  }

  /* ---- Init ---- */
  function init() {
    IDE.utils.on('fs-ready', () => render());
    IDE.utils.on('fs-changed', () => render());

    /* Toolbar buttons */
    const newFileBtn = document.getElementById('btn-new-file');
    if (newFileBtn) newFileBtn.addEventListener('click', () => _promptNewFile('/projects'));

    const newFolderBtn = document.getElementById('btn-new-folder');
    if (newFolderBtn) newFolderBtn.addEventListener('click', () => _promptNewFolder('/projects'));

    const uploadBtn = document.getElementById('btn-upload');
    if (uploadBtn) uploadBtn.addEventListener('click', uploadFiles);

    const refreshBtn = document.getElementById('btn-refresh-tree');
    if (refreshBtn) refreshBtn.addEventListener('click', render);
  }

  return { init, render, uploadFiles };
})();
