// Offline milestone queue: IndexedDB + Background Sync API.
// Game milestones are queued locally and synced when connectivity returns.
// The Service Worker drains the queue on the 'sync' event; when the API is
// missing but the page is online, the queue drains immediately as a fallback.
// A future cloud backend endpoint plugs into `drainMilestones()`.
// This module is SW-safe: no window references outside feature guards.

const DB_NAME = 'speed-crush';
const STORE = 'milestones';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) {
        req.result.createObjectStore(STORE, { autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function addToStore(milestone) {
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).add(milestone);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

/** Ask the Service Worker to sync when connectivity allows. */
export async function requestSync() {
  try {
    if (typeof window === 'undefined' || !('SyncManager' in window)) return false;
    if (!('serviceWorker' in navigator)) return false;
    const reg = await navigator.serviceWorker.ready;
    await reg.sync.register('sync-milestones');
    return true;
  } catch {
    return false;
  }
}

/**
 * Read + clear the queue. The future cloud backend call goes here.
 * Called from the SW sync handler and from the immediate fallback.
 */
export async function drainMilestones() {
  const db = await openDb();
  const items = await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).clear();
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
  return items;
}

/** Queue a milestone; returns how it will be delivered. */
export async function queueMilestone(milestone) {
  await addToStore({ ...milestone, queuedAt: Date.now() });
  const scheduled = await requestSync();
  if (!scheduled && typeof navigator !== 'undefined' && navigator.onLine) {
    const items = await drainMilestones();
    return { scheduled: false, immediate: true, count: items.length };
  }
  return { scheduled, immediate: false, count: 0 };
}
