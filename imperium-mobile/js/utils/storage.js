export const Storage = {
  _db: null,

  async init() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('imperium-db', 1);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('saves')) {
          db.createObjectStore('saves', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('citizens')) {
          db.createObjectStore('citizens', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('chronicle')) {
          db.createObjectStore('chronicle', { keyPath: 'id', autoIncrement: true });
        }
      };
      req.onsuccess = (e) => { this._db = e.target.result; resolve(); };
      req.onerror = () => reject(req.error);
    });
  },

  async save(key, data) {
    localStorage.setItem(`imperium:${key}`, JSON.stringify(data));
  },

  async load(key) {
    try {
      const raw = localStorage.getItem(`imperium:${key}`);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  },

  async saveIndexed(store, data) {
    if (!this._db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(store, 'readwrite');
      tx.objectStore(store).put(data);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  },

  async loadIndexed(store, key) {
    if (!this._db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(store, 'readonly');
      const req = tx.objectStore(store).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  },

  async saveChronicle(entry) {
    await this.saveIndexed('chronicle', entry);
  },

  async getChronicle() {
    if (!this._db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction('chronicle', 'readonly');
      const req = tx.objectStore('chronicle').getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  },

  async removeSave(key) {
    localStorage.removeItem(`imperium:${key}`);
  },
};
