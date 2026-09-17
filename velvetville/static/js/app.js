window.addEventListener('online', () => {
  if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
    navigator.serviceWorker.ready.then((reg) => {
      if ('sync' in reg) {
        reg.sync.register('sync-votes').catch(console.warn);
      }
    });
  }
});

async function queueVote(target, score) {
  if (!('indexedDB' in window)) return false;
  try {
    const data = await idbGet('pending-votes');
    const vote = { target, score };
    await idbAdd('pending-votes', vote);

    if (navigator.onLine && 'serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.ready.then((reg) => {
        if ('sync' in reg) reg.sync.register('sync-votes').catch(() => {});
      });
    }
    return true;
  } catch (e) {
    console.warn('Queue vote failed:', e);
    return false;
  }
}

function idbGet(store) {
  return new Promise((resolve) => {
    const req = indexedDB.open('velvetville', 1);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(store)) db.createObjectStore(store, { keyPath: 'id', autoIncrement: true });
    };
    req.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction(store, 'readonly');
      const st = tx.objectStore(store);
      const g = st.getAll();
      g.onsuccess = () => resolve(g.result);
      g.onerror = () => resolve([]);
    };
    req.onerror = () => resolve([]);
  });
}

function idbAdd(store, value) {
  return new Promise((resolve) => {
    const req = indexedDB.open('velvetville', 1);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(store)) db.createObjectStore(store, { keyPath: 'id', autoIncrement: true });
    };
    req.onsuccess = (e) => {
      const db = e.target.result;
      const tx = db.transaction(store, 'readwrite');
      tx.objectStore(store).add(value);
      tx.oncomplete = () => resolve();
    };
  });
}

window.queueVote = queueVote;
window.registerSW = registerSW;
