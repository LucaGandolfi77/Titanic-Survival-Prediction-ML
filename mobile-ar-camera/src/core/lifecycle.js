export function createLifecycleManager() {
  const hooks = {
    beforeUnload: [],
    shutdown: [],
    error: [],
  };

  function on(event, handler) {
    if (hooks[event]) hooks[event].push(handler);
    return () => {
      hooks[event] = hooks[event].filter((h) => h !== handler);
    };
  }

  function emit(event, ...args) {
    (hooks[event] || []).forEach((handler) => {
      try {
        handler(...args);
      } catch (err) {
        console.error(`Lifecycle hook error [${event}]:`, err);
      }
    });
  }

  function registerDefaults(window, document) {
    window.addEventListener('beforeunload', () => {
      emit('beforeUnload');
      runCleanup();
    });
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        emit('shutdown');
      }
    });
    window.addEventListener('error', (e) => {
      emit('error', e.error);
    });
    window.addEventListener('unhandledrejection', (e) => {
      emit('error', e.reason);
    });
  }

  function runCleanup() {
    emit('shutdown');
  }

  return { on, emit, registerDefaults, runCleanup };
}
