/* =====================================================
   DEBUGGER — Simple Python debugger using sys.settrace,
   breakpoints, step controls, variable state per frame
   ===================================================== */
window.IDE = window.IDE || {};

IDE.debugger = (() => {
  'use strict';

  let _isDebugging = false;
  let _isPaused = false;
  let _currentLine = null;
  let _callStack = [];
  let _localVars = {};

  /* ---- Debug wrapper code ---- */
  function _buildDebugCode(code, breakpoints) {
    const bpList = Array.from(breakpoints).join(', ');
    return `
import sys
import json

_ide_breakpoints = {${bpList}}
_ide_debug_state = {"paused": False, "step_mode": None, "current_line": 0}
_ide_call_stack = []
_ide_should_stop = False

def _ide_trace(frame, event, arg):
    global _ide_should_stop
    if _ide_should_stop:
        return None
    
    if event == 'line':
        lineno = frame.f_lineno
        _ide_debug_state["current_line"] = lineno
        
        # Build call stack
        _ide_call_stack.clear()
        f = frame
        while f is not None:
            _ide_call_stack.append({
                "function": f.f_code.co_name,
                "line": f.f_lineno,
                "filename": f.f_code.co_filename
            })
            f = f.f_back
        
        if lineno in _ide_breakpoints or _ide_debug_state["step_mode"]:
            # Capture local variables
            local_vars = {}
            for k, v in frame.f_locals.items():
                if not k.startswith('_ide_'):
                    try:
                        local_vars[k] = {"type": type(v).__name__, "value": repr(v)[:200]}
                    except:
                        local_vars[k] = {"type": type(v).__name__, "value": "..."}
            
            # Send debug info to main thread
            import js
            js.postMessage(type="DEBUG_BREAK", data=json.dumps({
                "line": lineno,
                "callStack": list(reversed(_ide_call_stack)),
                "locals": local_vars
            }))
    
    elif event == 'call':
        pass
    elif event == 'return':
        pass
    elif event == 'exception':
        pass
    
    return _ide_trace

sys.settrace(_ide_trace)
try:
${code.split('\n').map(l => '    ' + l).join('\n')}
finally:
    sys.settrace(None)
`;
  }

  /* ---- Start debugging ---- */
  function start() {
    const tab = IDE.editor.getActiveTab();
    if (!tab) {
      IDE.utils.toast('No file open to debug', 'warning');
      return;
    }

    const breakpoints = IDE.editor.getBreakpoints(tab.id);
    if (breakpoints.size === 0) {
      IDE.utils.toast('Set at least one breakpoint (F9 or click gutter)', 'info');
      /* Still run in debug mode — will just run through */
    }

    _isDebugging = true;
    _isPaused = false;
    _updateUI();

    IDE.outputPanel.clearConsole();
    IDE.outputPanel.clearProblems();
    IDE.outputPanel.appendConsole('🐛 Debug session started', 'info');

    /* For now, run normally with tracing info */
    const code = tab.content;
    IDE.pythonEngine.execute(code, 'debug').then(result => {
      _isDebugging = false;
      _isPaused = false;
      _currentLine = null;
      _updateUI();
      IDE.outputPanel.appendConsole('🐛 Debug session ended', 'info');
      if (result) IDE.outputPanel.showExecResult(result.result, result.elapsed);
    }).catch(err => {
      _isDebugging = false;
      _isPaused = false;
      _currentLine = null;
      _updateUI();
      IDE.outputPanel.appendConsole('🐛 Debug error: ' + (err.message || err), 'stderr');
    });
  }

  function stop() {
    if (!_isDebugging) return;
    IDE.pythonEngine.interrupt();
    _isDebugging = false;
    _isPaused = false;
    _currentLine = null;
    _callStack = [];
    _localVars = {};
    _updateUI();
    IDE.outputPanel.appendConsole('🐛 Debug session stopped', 'info');
  }

  function stepOver() {
    if (!_isDebugging) return;
    IDE.utils.emit('debug-step', { type: 'over' });
  }

  function stepInto() {
    if (!_isDebugging) return;
    IDE.utils.emit('debug-step', { type: 'into' });
  }

  function stepOut() {
    if (!_isDebugging) return;
    IDE.utils.emit('debug-step', { type: 'out' });
  }

  function continueExec() {
    if (!_isDebugging) return;
    _isPaused = false;
    IDE.utils.emit('debug-continue');
    _updateUI();
  }

  /* ---- Handle debug break from worker ---- */
  function _onDebugBreak(data) {
    _isPaused = true;
    _currentLine = data.line;
    _callStack = data.callStack || [];
    _localVars = data.locals || {};
    _updateUI();
    _renderDebugPanel();

    /* Highlight current line in editor */
    IDE.editor.goToLine(data.line);
  }

  /* ---- UI ---- */
  function _updateUI() {
    const toolbar = document.getElementById('debug-toolbar');
    if (!toolbar) return;

    toolbar.classList.toggle('active', _isDebugging);

    const btns = {
      'btn-debug-start': !_isDebugging,
      'btn-debug-stop': _isDebugging,
      'btn-debug-continue': _isPaused,
      'btn-debug-step-over': _isPaused,
      'btn-debug-step-into': _isPaused,
      'btn-debug-step-out': _isPaused
    };

    for (const [id, enabled] of Object.entries(btns)) {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = !enabled;
    }

    /* Status bar indicator */
    const statusDebug = document.getElementById('status-debug');
    if (statusDebug) {
      if (_isDebugging) {
        statusDebug.innerHTML = `<i class="fa-solid fa-bug"></i> ${_isPaused ? 'Paused L' + _currentLine : 'Running'}`;
        statusDebug.classList.add('active');
      } else {
        statusDebug.textContent = '';
        statusDebug.classList.remove('active');
      }
    }
  }

  function _renderDebugPanel() {
    const panel = document.getElementById('debug-panel');
    if (!panel) return;

    let html = '';

    /* Call Stack */
    html += `<div class="debug-section">
      <h4><i class="fa-solid fa-layer-group"></i> Call Stack</h4>
      <div class="debug-callstack">`;
    for (const frame of _callStack) {
      const isCurrent = frame.line === _currentLine;
      html += `<div class="callstack-frame${isCurrent ? ' current' : ''}">
        <span class="frame-func">${IDE.utils.escapeHtml(frame.function)}</span>
        <span class="frame-line">:${frame.line}</span>
      </div>`;
    }
    html += `</div></div>`;

    /* Local Variables */
    html += `<div class="debug-section">
      <h4><i class="fa-solid fa-cube"></i> Local Variables</h4>
      <div class="debug-locals">`;
    for (const [name, info] of Object.entries(_localVars)) {
      html += `<div class="debug-var">
        <span class="debug-var-name">${IDE.utils.escapeHtml(name)}</span>
        <span class="debug-var-type">${IDE.utils.escapeHtml(info.type)}</span>
        <span class="debug-var-value">${IDE.utils.escapeHtml(String(info.value).substring(0, 100))}</span>
      </div>`;
    }
    if (Object.keys(_localVars).length === 0) {
      html += '<p class="debug-empty">No local variables</p>';
    }
    html += `</div></div>`;

    /* Breakpoints */
    const tab = IDE.editor.getActiveTab();
    if (tab) {
      const bps = IDE.editor.getBreakpoints(tab.id);
      html += `<div class="debug-section">
        <h4><i class="fa-solid fa-circle-dot"></i> Breakpoints</h4>
        <div class="debug-breakpoints">`;
      for (const ln of bps) {
        html += `<div class="debug-bp">
          <i class="fa-solid fa-circle bp-icon"></i>
          <span>${tab.filename}:${ln}</span>
          <button class="debug-bp-remove" data-line="${ln}"><i class="fa-solid fa-xmark"></i></button>
        </div>`;
      }
      if (bps.size === 0) {
        html += '<p class="debug-empty">No breakpoints set</p>';
      }
      html += `</div></div>`;
    }

    panel.innerHTML = html;

    /* Bind breakpoint remove */
    panel.querySelectorAll('.debug-bp-remove').forEach(btn => {
      btn.addEventListener('click', () => {
        const ln = parseInt(btn.dataset.line, 10);
        if (tab) IDE.editor.toggleBreakpoint(tab.id, ln);
        _renderDebugPanel();
      });
    });
  }

  /* ---- Init ---- */
  function init() {
    /* Debug toolbar buttons */
    const startBtn = document.getElementById('btn-debug-start');
    if (startBtn) startBtn.addEventListener('click', start);

    const stopBtn = document.getElementById('btn-debug-stop');
    if (stopBtn) stopBtn.addEventListener('click', stop);

    const continueBtn = document.getElementById('btn-debug-continue');
    if (continueBtn) continueBtn.addEventListener('click', continueExec);

    const stepOverBtn = document.getElementById('btn-debug-step-over');
    if (stepOverBtn) stepOverBtn.addEventListener('click', stepOver);

    const stepIntoBtn = document.getElementById('btn-debug-step-into');
    if (stepIntoBtn) stepIntoBtn.addEventListener('click', stepInto);

    const stepOutBtn = document.getElementById('btn-debug-step-out');
    if (stepOutBtn) stepOutBtn.addEventListener('click', stepOut);

    /* Listen for debug events from worker */
    IDE.utils.on('debug-break', _onDebugBreak);

    _updateUI();
  }

  return {
    init, start, stop,
    stepOver, stepInto, stepOut, continueExec,
    isDebugging: () => _isDebugging,
    isPaused: () => _isPaused
  };
})();
