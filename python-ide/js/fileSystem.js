/* =====================================================
   FILE SYSTEM — IndexedDB-backed virtual filesystem
   CRUD ops, syncs with Pyodide Emscripten FS
   ===================================================== */
window.IDE = window.IDE || {};

IDE.fileSystem = (() => {
  'use strict';

  const DB_NAME = 'python-ide-fs';
  const DB_VERSION = 1;
  const STORE_NAME = 'files';
  let _db = null;
  let _fileTree = {}; /* path -> { name, content, type, size, modified } */

  /* ---- Open/Init IndexedDB ---- */
  function _openDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'path' });
          store.createIndex('type', 'type', { unique: false });
          store.createIndex('parent', 'parent', { unique: false });
        }
      };
      req.onsuccess = (e) => {
        _db = e.target.result;
        resolve(_db);
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  /* ---- CRUD operations ---- */
  function _tx(mode) {
    const tx = _db.transaction(STORE_NAME, mode);
    return tx.objectStore(STORE_NAME);
  }

  function _parentPath(path) {
    const parts = path.split('/').filter(Boolean);
    parts.pop();
    return '/' + parts.join('/');
  }

  function writeFile(path, content) {
    return new Promise((resolve, reject) => {
      const normalPath = _normalizePath(path);
      const name = normalPath.split('/').pop();
      const parent = _parentPath(normalPath);
      const entry = {
        path: normalPath,
        name,
        parent,
        content: content || '',
        type: 'file',
        size: (content || '').length,
        modified: Date.now(),
        created: Date.now()
      };
      /* Check if exists to preserve created time */
      const getReq = _tx('readonly').get(normalPath);
      getReq.onsuccess = () => {
        if (getReq.result) {
          entry.created = getReq.result.created;
        }
        const putReq = _tx('readwrite').put(entry);
        putReq.onsuccess = () => {
          _fileTree[normalPath] = entry;
          _syncToPyodide(normalPath, content);
          IDE.utils.emit('fs-changed', { action: 'write', path: normalPath });
          resolve(entry);
        };
        putReq.onerror = (e) => reject(e.target.error);
      };
    });
  }

  function readFile(path) {
    return new Promise((resolve, reject) => {
      const normalPath = _normalizePath(path);
      const req = _tx('readonly').get(normalPath);
      req.onsuccess = () => {
        if (req.result) {
          resolve(req.result.content);
        } else {
          reject(new Error('File not found: ' + normalPath));
        }
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  function deleteFile(path) {
    return new Promise((resolve, reject) => {
      const normalPath = _normalizePath(path);
      const req = _tx('readwrite').delete(normalPath);
      req.onsuccess = () => {
        delete _fileTree[normalPath];
        IDE.utils.emit('fs-changed', { action: 'delete', path: normalPath });
        resolve();
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  function createDirectory(path) {
    return new Promise((resolve, reject) => {
      const normalPath = _normalizePath(path);
      const name = normalPath.split('/').filter(Boolean).pop();
      const parent = _parentPath(normalPath);
      const entry = {
        path: normalPath,
        name,
        parent,
        content: null,
        type: 'directory',
        size: 0,
        modified: Date.now(),
        created: Date.now()
      };
      const req = _tx('readwrite').put(entry);
      req.onsuccess = () => {
        _fileTree[normalPath] = entry;
        IDE.utils.emit('fs-changed', { action: 'mkdir', path: normalPath });
        resolve(entry);
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  function deleteDirectory(path) {
    return new Promise(async (resolve, reject) => {
      const normalPath = _normalizePath(path);
      try {
        /* Delete all children first */
        const children = await listDir(normalPath, true);
        const store = _tx('readwrite');
        for (const child of children) {
          store.delete(child.path);
          delete _fileTree[child.path];
        }
        /* Delete the directory itself */
        const store2 = _tx('readwrite');
        store2.delete(normalPath);
        delete _fileTree[normalPath];
        IDE.utils.emit('fs-changed', { action: 'rmdir', path: normalPath });
        resolve();
      } catch (e) {
        reject(e);
      }
    });
  }

  function rename(oldPath, newPath) {
    return new Promise(async (resolve, reject) => {
      const normalOld = _normalizePath(oldPath);
      const normalNew = _normalizePath(newPath);
      try {
        const content = await readFile(normalOld);
        await writeFile(normalNew, content);
        await deleteFile(normalOld);
        IDE.utils.emit('fs-changed', { action: 'rename', oldPath: normalOld, newPath: normalNew });
        resolve();
      } catch (e) {
        reject(e);
      }
    });
  }

  function exists(path) {
    return new Promise((resolve) => {
      const normalPath = _normalizePath(path);
      const req = _tx('readonly').get(normalPath);
      req.onsuccess = () => resolve(!!req.result);
      req.onerror = () => resolve(false);
    });
  }

  function listDir(path, recursive) {
    return new Promise((resolve, reject) => {
      const normalPath = _normalizePath(path);
      const store = _tx('readonly');
      const index = store.index('parent');
      const req = index.getAll(normalPath);
      req.onsuccess = async () => {
        let results = req.result || [];
        if (recursive) {
          const dirs = results.filter(r => r.type === 'directory');
          for (const dir of dirs) {
            const children = await listDir(dir.path, true);
            results = results.concat(children);
          }
        }
        resolve(results);
      };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  function getAllFiles() {
    return new Promise((resolve, reject) => {
      const req = _tx('readonly').getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = (e) => reject(e.target.error);
    });
  }

  /* ---- Path normalization ---- */
  function _normalizePath(path) {
    if (!path.startsWith('/')) path = '/' + path;
    /* Remove trailing slash unless root */
    if (path.length > 1 && path.endsWith('/')) path = path.slice(0, -1);
    /* Collapse double slashes */
    path = path.replace(/\/+/g, '/');
    return path;
  }

  /* ---- Sync to Pyodide FS ---- */
  function _syncToPyodide(path, content) {
    if (IDE.pythonEngine && IDE.pythonEngine.writeFile) {
      IDE.pythonEngine.writeFile('/files' + path, content).catch(() => {
        /* Pyodide may not be ready yet, silently ignore */
      });
    }
  }

  /* ---- Load all files into cache ---- */
  async function _loadTree() {
    const allFiles = await getAllFiles();
    _fileTree = {};
    for (const f of allFiles) {
      _fileTree[f.path] = f;
    }
  }

  /* ---- Default files ---- */
  async function _createDefaults() {
    const count = await new Promise((resolve) => {
      const req = _tx('readonly').count();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(0);
    });

    if (count === 0) {
      await createDirectory('/projects');
      await createDirectory('/projects/demo');
      await writeFile('/projects/demo/hello.py', `# Hello World Demo
print("Hello from Python IDE!")
print("Running Python via Pyodide (WebAssembly)")

for i in range(5):
    print(f"  Count: {i + 1}")

print("\\nDone!")
`);
      await writeFile('/projects/demo/data_analysis.py', `# Data Analysis Demo
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': np.random.randint(20, 60, 5),
    'Score': np.random.uniform(50, 100, 5).round(1),
    'Grade': ['A', 'B', 'A', 'C', 'B']
}

df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)
print("\\nStatistics:")
print(df.describe())
print("\\nMean Score:", df['Score'].mean().round(2))
`);
      await writeFile('/projects/demo/plotting.py', `# Matplotlib Demo
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(x, y1, 'b-', label='sin(x)')
ax1.plot(x, y2, 'r--', label='cos(x)')
ax1.set_title('Trigonometric Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(['A', 'B', 'C', 'D'], [23, 45, 56, 78], color=['#cba6f7', '#f38ba8', '#a6e3a1', '#89b4fa'])
ax2.set_title('Bar Chart')

plt.tight_layout()
plt.show()
`);
    }
  }

  /* ---- Build tree structure for file explorer ---- */
  function getTree() {
    const tree = { name: '/', type: 'directory', children: [], path: '/' };
    const entries = Object.values(_fileTree).sort((a, b) => a.path.localeCompare(b.path));

    for (const entry of entries) {
      const parts = entry.path.split('/').filter(Boolean);
      let current = tree;
      let currentPath = '';
      for (let i = 0; i < parts.length; i++) {
        currentPath += '/' + parts[i];
        if (i === parts.length - 1) {
          current.children.push({
            name: parts[i],
            type: entry.type,
            path: entry.path,
            size: entry.size,
            modified: entry.modified
          });
        } else {
          let child = current.children.find(c => c.name === parts[i] && c.type === 'directory');
          if (!child) {
            child = { name: parts[i], type: 'directory', children: [], path: currentPath };
            current.children.push(child);
          }
          current = child;
        }
      }
    }

    /* Sort: directories first, then alphabetically */
    function sortTree(node) {
      if (node.children) {
        node.children.sort((a, b) => {
          if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
          return a.name.localeCompare(b.name);
        });
        node.children.forEach(sortTree);
      }
    }
    sortTree(tree);
    return tree;
  }

  /* ---- Init ---- */
  async function init() {
    await _openDB();
    await _loadTree();
    await _createDefaults();
    IDE.utils.emit('fs-ready');
  }

  return {
    init, writeFile, readFile, deleteFile,
    createDirectory, deleteDirectory,
    rename, exists, listDir, getAllFiles,
    getTree
  };
})();
