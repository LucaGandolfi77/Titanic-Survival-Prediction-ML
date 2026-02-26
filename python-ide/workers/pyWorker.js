/* =====================================================
   PYWORKER — Pyodide Web Worker
   Runs Python code off the main thread
   ===================================================== */
/* eslint-disable no-redeclare */

let pyodide = null;
let ready = false;
let executing = false;

self.onmessage = async function (e) {
  const msg = e.data;
  switch (msg.type) {
    case 'INIT':
      await initPyodide(msg.packages || []);
      break;
    case 'EXEC':
      await executePython(msg.code, msg.id, msg.mode || 'exec');
      break;
    case 'INSTALL':
      await installPackage(msg.packages, msg.id);
      break;
    case 'INSPECT':
      inspectNamespace(msg.id);
      break;
    case 'STDIN':
      handleStdinResponse(msg.value);
      break;
    case 'INTERRUPT':
      interruptExecution();
      break;
    case 'WRITE_FILE':
      writeFile(msg.path, msg.content, msg.id);
      break;
    case 'READ_FILE':
      readFile(msg.path, msg.id);
      break;
    case 'LIST_DIR':
      listDir(msg.path, msg.id);
      break;
    default:
      break;
  }
};

async function initPyodide(packages) {
  try {
    self.postMessage({ type: 'INIT_PROGRESS', step: 'Loading Pyodide runtime...', progress: 0.1 });
    importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.0/full/pyodide.js');

    self.postMessage({ type: 'INIT_PROGRESS', step: 'Initializing Python 3.12...', progress: 0.3 });
    pyodide = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.0/full/',
      stdout: (text) => self.postMessage({ type: 'STDOUT', text }),
      stderr: (text) => self.postMessage({ type: 'STDERR', text })
    });

    /* Set up /files directory for user files */
    pyodide.FS.mkdirTree('/files');

    /* Set up matplotlib Agg backend and patch plt.show */
    self.postMessage({ type: 'INIT_PROGRESS', step: 'Configuring Python environment...', progress: 0.5 });
    await pyodide.runPythonAsync(`
import sys
import io

# Patch input() to communicate with JS
import builtins
_original_input = builtins.input

def _patched_input(prompt=''):
    import js
    js.postMessage({"type": "INPUT_REQUEST", "prompt": str(prompt)})
    # Block: we rely on Atomics for SharedArrayBuffer or a sync approach
    # For now, we use a simple approach — input is handled via pre-set values
    raise EOFError("input() requires the synchronous input bridge. Use the console REPL instead.")

builtins.input = _patched_input
`);

    /* Install default packages */
    if (packages.length > 0) {
      self.postMessage({ type: 'INIT_PROGRESS', step: 'Installing packages (numpy, pandas, matplotlib)...', progress: 0.6 });
      try {
        await pyodide.loadPackage(packages, {
          messageCallback: (msg) => {
            self.postMessage({ type: 'INIT_PROGRESS', step: msg, progress: 0.7 });
          }
        });
      } catch (err) {
        self.postMessage({ type: 'STDERR', text: 'Warning: Some packages failed to load: ' + err.message });
      }
    }

    /* Configure matplotlib after loading */
    self.postMessage({ type: 'INIT_PROGRESS', step: 'Configuring matplotlib...', progress: 0.85 });
    await pyodide.runPythonAsync(`
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _orig_show = plt.show

    def _patched_show(*args, **kwargs):
        import io, base64
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor() if fig.get_facecolor() != (1.0, 1.0, 1.0, 0.0) else '#1e1e2e',
                        edgecolor='none')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            import js
            js.postMessage({"type": "PLOT", "data": img_b64})
            buf.close()
        plt.close('all')

    plt.show = _patched_show
except ImportError:
    pass
`);

    /* Install micropip for PyPI packages */
    try {
      await pyodide.loadPackage(['micropip']);
    } catch {
      /* micropip may fail, that's ok */
    }

    ready = true;
    self.postMessage({
      type: 'READY',
      version: pyodide.version,
      pythonVersion: pyodide.runPython('import sys; sys.version')
    });
  } catch (err) {
    self.postMessage({ type: 'ERROR', error: 'Failed to initialize Pyodide: ' + err.message });
  }
}

async function executePython(code, execId, mode) {
  if (!ready) {
    self.postMessage({ type: 'EXEC_ERROR', id: execId, error: 'Pyodide not ready' });
    return;
  }
  if (executing) {
    self.postMessage({ type: 'EXEC_ERROR', id: execId, error: 'Already executing' });
    return;
  }

  executing = true;
  const startTime = performance.now();

  try {
    let result;
    if (mode === 'eval') {
      /* Try eval first (expression), fall back to exec (statements) */
      try {
        result = await pyodide.runPythonAsync(code);
      } catch {
        result = await pyodide.runPythonAsync(code);
      }
    } else {
      result = await pyodide.runPythonAsync(code);
    }

    const elapsed = performance.now() - startTime;

    /* Check if result is a DataFrame or has special rendering */
    let resultRepr = null;

    if (result !== undefined && result !== null) {
      try {
        const pyResult = result;
        if (pyResult && typeof pyResult.toJs === 'function') {
          resultRepr = pyResult.toString();
        } else if (result !== undefined) {
          resultRepr = String(result);
        }
      } catch {
        if (result !== undefined) resultRepr = String(result);
      }

      /* Check for DataFrame */
      try {
        pyodide.runPython(`
import sys
_last = sys.modules.get('__main__').__dict__.get('_')
_last_result = None
try:
    import pandas as pd
    _exec_result = ${JSON.stringify(resultRepr || '')}
except:
    pass
False
`);
      } catch { /* ignore */ }
    }

    self.postMessage({
      type: 'EXEC_RESULT',
      id: execId,
      result: resultRepr,
      elapsed: elapsed
    });
  } catch (err) {
    const elapsed = performance.now() - startTime;
    let errorInfo = {
      type: 'PythonError',
      message: err.message || String(err),
      traceback: ''
    };

    /* Parse Python traceback */
    const msg = err.message || '';
    const tbMatch = msg.match(/Traceback \(most recent call last\):\n([\s\S]*?)(\w+Error:.*)/);
    if (tbMatch) {
      errorInfo.traceback = tbMatch[1].trim();
      errorInfo.message = tbMatch[2].trim();
      errorInfo.type = tbMatch[2].split(':')[0];
    }

    /* Extract line number from traceback */
    const lineMatch = msg.match(/line (\d+)/);
    if (lineMatch) errorInfo.line = parseInt(lineMatch[1], 10);

    self.postMessage({
      type: 'EXEC_ERROR',
      id: execId,
      error: errorInfo,
      elapsed: elapsed
    });
  } finally {
    executing = false;
  }
}

async function installPackage(packages, installId) {
  if (!ready) {
    self.postMessage({ type: 'INSTALL_ERROR', id: installId, error: 'Pyodide not ready' });
    return;
  }

  for (const pkg of packages) {
    try {
      self.postMessage({ type: 'INSTALL_PROGRESS', id: installId, package: pkg, status: 'installing' });

      /* Try loadPackage first (built-in), then micropip */
      try {
        await pyodide.loadPackage([pkg]);
      } catch {
        await pyodide.runPythonAsync(`
import micropip
await micropip.install('${pkg}')
`);
      }

      self.postMessage({ type: 'INSTALL_PROGRESS', id: installId, package: pkg, status: 'installed' });
    } catch (err) {
      self.postMessage({ type: 'INSTALL_ERROR', id: installId, package: pkg, error: err.message });
    }
  }
  self.postMessage({ type: 'INSTALL_DONE', id: installId });
}

function inspectNamespace(inspectId) {
  if (!ready) return;
  try {
    const snapshotJson = pyodide.runPython(`
import json, types, sys

_snapshot = {}
_ns = sys.modules.get('__main__').__dict__

for _name, _val in _ns.items():
    if _name.startswith('_') or isinstance(_val, types.ModuleType):
        continue
    if callable(_val) and isinstance(_val, type):
        _snapshot[_name] = {'type': 'class', 'value': _name, 'members': str(len(dir(_val)))}
    elif callable(_val):
        _snapshot[_name] = {'type': 'function', 'value': _name + '()'}
    elif isinstance(_val, (int, float, str, bool)):
        _snapshot[_name] = {'type': type(_val).__name__, 'value': repr(_val)[:200]}
    elif hasattr(_val, 'shape'):
        try:
            _snapshot[_name] = {'type': type(_val).__name__, 'value': str(_val.shape), 'dtype': str(getattr(_val, 'dtype', ''))}
        except:
            _snapshot[_name] = {'type': type(_val).__name__, 'value': '...'}
    elif isinstance(_val, (list, tuple, set, frozenset)):
        _snapshot[_name] = {'type': type(_val).__name__, 'value': f'len={len(_val)}'}
    elif isinstance(_val, dict):
        _snapshot[_name] = {'type': 'dict', 'value': f'len={len(_val)}'}
    else:
        _snapshot[_name] = {'type': type(_val).__name__, 'value': repr(_val)[:100]}

json.dumps(_snapshot)
`);
    self.postMessage({ type: 'NAMESPACE', id: inspectId, data: JSON.parse(snapshotJson) });
  } catch (err) {
    self.postMessage({ type: 'NAMESPACE', id: inspectId, data: {}, error: err.message });
  }
}

let _stdinResolve = null;

function handleStdinResponse(value) {
  if (_stdinResolve) {
    _stdinResolve(value);
    _stdinResolve = null;
  }
}

function interruptExecution() {
  if (pyodide && executing) {
    try {
      pyodide.runPython('raise KeyboardInterrupt');
    } catch { /* expected */ }
  }
}

function writeFile(path, content, id) {
  if (!ready) return;
  try {
    if (typeof content === 'string') {
      pyodide.FS.writeFile(path, content);
    } else {
      pyodide.FS.writeFile(path, new Uint8Array(content));
    }
    self.postMessage({ type: 'FILE_WRITTEN', id, path });
  } catch (err) {
    self.postMessage({ type: 'FILE_ERROR', id, error: err.message });
  }
}

function readFile(path, id) {
  if (!ready) return;
  try {
    const content = pyodide.FS.readFile(path, { encoding: 'utf8' });
    self.postMessage({ type: 'FILE_READ', id, path, content });
  } catch (err) {
    self.postMessage({ type: 'FILE_ERROR', id, error: err.message });
  }
}

function listDir(path, id) {
  if (!ready) return;
  try {
    const entries = pyodide.FS.readdir(path).filter(e => e !== '.' && e !== '..');
    const result = entries.map(name => {
      const fullPath = path.endsWith('/') ? path + name : path + '/' + name;
      try {
        const stat = pyodide.FS.stat(fullPath);
        return {
          name,
          isDir: pyodide.FS.isDir(stat.mode),
          size: stat.size
        };
      } catch {
        return { name, isDir: false, size: 0 };
      }
    });
    self.postMessage({ type: 'DIR_LIST', id, path, entries: result });
  } catch (err) {
    self.postMessage({ type: 'FILE_ERROR', id, error: err.message });
  }
}
