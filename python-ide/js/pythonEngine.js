/* =====================================================
   PYTHON ENGINE — Worker manager, execution queue,
   communication bridge between UI and pyWorker
   ===================================================== */
window.IDE = window.IDE || {};

IDE.pythonEngine = (() => {
  'use strict';

  let worker = null;
  let _ready = false;
  let _execCounter = 0;
  let _pendingExec = {};
  let _pendingInstall = {};
  let _pendingFile = {};
  let _onReady = null;

  const DEFAULT_PACKAGES = ['numpy', 'pandas', 'matplotlib', 'micropip'];

  function init() {
    return new Promise((resolve) => {
      _onReady = resolve;
      worker = new Worker('workers/pyWorker.js');
      worker.onmessage = _handleMessage;
      worker.onerror = (err) => {
        console.error('Worker error:', err);
        IDE.utils.emit('python-error', { message: 'Worker crashed: ' + err.message });
      };
      worker.postMessage({ type: 'INIT', packages: DEFAULT_PACKAGES });
    });
  }

  function _handleMessage(e) {
    const msg = e.data;
    switch (msg.type) {
      case 'INIT_PROGRESS':
        IDE.utils.emit('python-init-progress', msg);
        break;

      case 'READY':
        _ready = true;
        IDE.utils.emit('python-ready', { version: msg.version, pythonVersion: msg.pythonVersion });
        if (_onReady) { _onReady(); _onReady = null; }
        break;

      case 'STDOUT':
        IDE.utils.emit('python-stdout', { text: msg.text });
        break;

      case 'STDERR':
        IDE.utils.emit('python-stderr', { text: msg.text });
        break;

      case 'PLOT':
        IDE.utils.emit('python-plot', { data: msg.data });
        break;

      case 'INPUT_REQUEST':
        IDE.utils.emit('python-input-request', { prompt: msg.prompt });
        break;

      case 'EXEC_RESULT': {
        const cb = _pendingExec[msg.id];
        if (cb) {
          cb.resolve({ result: msg.result, elapsed: msg.elapsed });
          delete _pendingExec[msg.id];
        }
        IDE.utils.emit('python-exec-done', { id: msg.id, result: msg.result, elapsed: msg.elapsed });
        /* Automatically inspect namespace after execution */
        inspectNamespace();
        break;
      }

      case 'EXEC_ERROR': {
        const cb2 = _pendingExec[msg.id];
        if (cb2) {
          cb2.reject(msg.error);
          delete _pendingExec[msg.id];
        }
        IDE.utils.emit('python-exec-error', { id: msg.id, error: msg.error, elapsed: msg.elapsed });
        inspectNamespace();
        break;
      }

      case 'INSTALL_PROGRESS':
        IDE.utils.emit('python-install-progress', msg);
        break;

      case 'INSTALL_ERROR':
        IDE.utils.emit('python-install-error', msg);
        break;

      case 'INSTALL_DONE': {
        const icb = _pendingInstall[msg.id];
        if (icb) { icb(); delete _pendingInstall[msg.id]; }
        IDE.utils.emit('python-install-done', msg);
        break;
      }

      case 'NAMESPACE':
        IDE.utils.emit('python-namespace', { data: msg.data });
        break;

      case 'FILE_WRITTEN':
        if (_pendingFile[msg.id]) { _pendingFile[msg.id].resolve(msg.path); delete _pendingFile[msg.id]; }
        break;

      case 'FILE_READ':
        if (_pendingFile[msg.id]) { _pendingFile[msg.id].resolve(msg.content); delete _pendingFile[msg.id]; }
        break;

      case 'DIR_LIST':
        if (_pendingFile[msg.id]) { _pendingFile[msg.id].resolve(msg.entries); delete _pendingFile[msg.id]; }
        break;

      case 'FILE_ERROR':
        if (_pendingFile[msg.id]) { _pendingFile[msg.id].reject(new Error(msg.error)); delete _pendingFile[msg.id]; }
        break;

      case 'ERROR':
        IDE.utils.emit('python-error', { message: msg.error });
        break;

      default:
        break;
    }
  }

  function execute(code, mode = 'exec') {
    return new Promise((resolve, reject) => {
      if (!_ready) { reject(new Error('Python not ready')); return; }
      const id = 'exec_' + (++_execCounter);
      _pendingExec[id] = { resolve, reject };
      worker.postMessage({ type: 'EXEC', code, id, mode });
      IDE.utils.emit('python-exec-start', { id, code });
    });
  }

  function installPackages(packages) {
    return new Promise((resolve) => {
      if (!_ready) return;
      const id = 'inst_' + (++_execCounter);
      _pendingInstall[id] = resolve;
      worker.postMessage({ type: 'INSTALL', packages, id });
    });
  }

  function inspectNamespace() {
    if (!_ready || !worker) return;
    worker.postMessage({ type: 'INSPECT', id: 'ns_' + Date.now() });
  }

  function sendStdin(value) {
    if (worker) worker.postMessage({ type: 'STDIN', value });
  }

  function interrupt() {
    if (worker) worker.postMessage({ type: 'INTERRUPT' });
  }

  function writeFile(path, content) {
    return new Promise((resolve, reject) => {
      if (!_ready) { reject(new Error('Python not ready')); return; }
      const id = 'wf_' + (++_execCounter);
      _pendingFile[id] = { resolve, reject };
      worker.postMessage({ type: 'WRITE_FILE', path, content, id });
    });
  }

  function readFile(path) {
    return new Promise((resolve, reject) => {
      if (!_ready) { reject(new Error('Python not ready')); return; }
      const id = 'rf_' + (++_execCounter);
      _pendingFile[id] = { resolve, reject };
      worker.postMessage({ type: 'READ_FILE', path, id });
    });
  }

  function listDir(path) {
    return new Promise((resolve, reject) => {
      if (!_ready) { reject(new Error('Python not ready')); return; }
      const id = 'ld_' + (++_execCounter);
      _pendingFile[id] = { resolve, reject };
      worker.postMessage({ type: 'LIST_DIR', path, id });
    });
  }

  function isReady() { return _ready; }

  return {
    init, execute, installPackages, inspectNamespace,
    sendStdin, interrupt, writeFile, readFile, listDir, isReady
  };
})();
