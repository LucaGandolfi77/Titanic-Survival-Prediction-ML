async function registerSW() {
  if (!('serviceWorker' in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.register('/static/sw.js');
    console.log('SW registered:', reg.scope);

    if ('SyncManager' in window) {
      reg.addEventListener('updatefound', () => {
        const newWorker = reg.installing;
        newWorker?.addEventListener('statechange', () => {
          if (newWorker.state === 'activated') {
            console.log('SW activated');
          }
        });
      });
    }
  } catch (err) {
    console.warn('SW registration failed:', err);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', registerSW);
} else {
  registerSW();
}
