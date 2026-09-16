// IndexedDB wrapper for persistent storage
const DB_NAME = 'hds_db';
const DB_VERSION = 1;
const STORE_SCORES = 'scores';
const STORE_SETTINGS = 'settings';

export class IDBStorage {
    constructor() {
        this.db = null;
        this._ready = this._init();
    }

    async _init() {
        return new Promise((resolve, reject) => {
            if (!window.indexedDB) {
                console.warn('IndexedDB not available');
                resolve(false);
                return;
            }
            const request = indexedDB.open(DB_NAME, DB_VERSION);
            request.onerror = () => {
                console.error('IndexedDB open error');
                resolve(false);
            };
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_SCORES)) {
                    db.createObjectStore(STORE_SCORES, { keyPath: 'id', autoIncrement: true });
                }
                if (!db.objectStoreNames.contains(STORE_SETTINGS)) {
                    db.createObjectStore(STORE_SETTINGS, { keyPath: 'key' });
                }
            };
            request.onsuccess = (e) => {
                this.db = e.target.result;
                resolve(true);
            };
        });
    }

    async ready() {
        return this._ready;
    }

    async saveScores(scores) {
        await this._ready;
        if (!this.db) return;
        return new Promise((resolve, reject) => {
            try {
                const tx = this.db.transaction(STORE_SCORES, 'readwrite');
                const store = tx.objectStore(STORE_SCORES);
                store.clear();
                scores.forEach((s, i) => {
                    store.put({ ...s, order: i });
                });
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            } catch (e) {
                reject(e);
            }
        });
    }

    async loadScores() {
        await this._ready;
        if (!this.db) return [];
        return new Promise((resolve, reject) => {
            try {
                const tx = this.db.transaction(STORE_SCORES, 'readonly');
                const store = tx.objectStore(STORE_SCORES);
                const request = store.getAll();
                request.onsuccess = () => {
                    const results = request.result.sort((a, b) => (a.order || 0) - (b.order || 0));
                    resolve(results);
                };
                request.onerror = () => reject(request.error);
            } catch (e) {
                reject(e);
            }
        });
    }

    async saveSetting(key, value) {
        await this._ready;
        if (!this.db) return;
        return new Promise((resolve, reject) => {
            try {
                const tx = this.db.transaction(STORE_SETTINGS, 'readwrite');
                const store = tx.objectStore(STORE_SETTINGS);
                store.put({ key, value });
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            } catch (e) {
                reject(e);
            }
        });
    }

    async loadSetting(key) {
        await this._ready;
        if (!this.db) return null;
        return new Promise((resolve, reject) => {
            try {
                const tx = this.db.transaction(STORE_SETTINGS, 'readonly');
                const store = tx.objectStore(STORE_SETTINGS);
                const request = store.get(key);
                request.onsuccess = () => resolve(request.result?.value ?? null);
                request.onerror = () => reject(request.error);
            } catch (e) {
                reject(e);
            }
        });
    }
}
