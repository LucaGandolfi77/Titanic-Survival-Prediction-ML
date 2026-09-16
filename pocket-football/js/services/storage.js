const DB_NAME = 'PocketFootballDB';
const DB_VERSION = 1;

const STORES = {
  PROGRESSION: 'progression',
  RECORDS: 'records',
  TOURNAMENTS: 'tournaments',
  SETTINGS: 'settings',
  ARENAS: 'arenas'
};

class StorageService {
  constructor() {
    this.db = null;
    this._ready = this._init();
  }

  async _init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.warn('IndexedDB failed, falling back to localStorage', request.error);
        this.db = null;
        resolve();
      };

      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORES.PROGRESSION)) {
          db.createObjectStore(STORES.PROGRESSION, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(STORES.RECORDS)) {
          db.createObjectStore(STORES.RECORDS, { keyPath: 'id', autoIncrement: true });
        }
        if (!db.objectStoreNames.contains(STORES.TOURNAMENTS)) {
          db.createObjectStore(STORES.TOURNAMENTS, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(STORES.SETTINGS)) {
          db.createObjectStore(STORES.SETTINGS, { keyPath: 'key' });
        }
        if (!db.objectStoreNames.contains(STORES.ARENAS)) {
          db.createObjectStore(STORES.ARENAS, { keyPath: 'id' });
        }
      };

      request.onsuccess = (e) => {
        this.db = e.target.result;
        resolve();
      };
    });
  }

  async ready() {
    await this._ready;
  }

  async _getStore(storeName, mode) {
    await this._ready;
    const tx = this.db.transaction(storeName, mode);
    return tx.objectStore(storeName);
  }

  async _promisify(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async set(storeName, data) {
    const store = await this._getStore(storeName, 'readwrite');
    return this._promisify(store.put(data));
  }

  async get(storeName, key) {
    const store = await this._getStore(storeName, 'readonly');
    return this._promisify(store.get(key));
  }

  async getAll(storeName) {
    const store = await this._getStore(storeName, 'readonly');
    return this._promisify(store.getAll());
  }

  async delete(storeName, key) {
    const store = await this._getStore(storeName, 'readwrite');
    return this._promisify(store.delete(key));
  }

  async clear(storeName) {
    const store = await this._getStore(storeName, 'readwrite');
    return this._promisify(store.clear());
  }

  async saveProgression(progression) {
    return this.set(STORES.PROGRESSION, progression);
  }

  async getProgression() {
    const data = await this.get(STORES.PROGRESSION, 'player');
    return data || this._defaultProgression();
  }

  async saveRecord(record) {
    return this.set(STORES.RECORDS, record);
  }

  async getRecords(limit = 20) {
    const all = await this.getAll(STORES.RECORDS);
    return all.sort((a, b) => b.id - a.id).slice(0, limit);
  }

  async saveTournament(tournament) {
    return this.set(STORES.TOURNAMENTS, tournament);
  }

  async getTournament(id) {
    return this.get(STORES.TOURNAMENTS, id);
  }

  async getTournaments() {
    return this.getAll(STORES.TOURNAMENTS);
  }

  async setSetting(key, value) {
    return this.set(STORES.SETTINGS, { key, value });
  }

  async getSetting(key) {
    const data = await this.get(STORES.SETTINGS, key);
    return data ? data.value : null;
  }

  async saveArena(arena) {
    return this.set(STORES.ARENAS, arena);
  }

  async getArenas() {
    return this.getAll(STORES.ARENAS);
  }

  _defaultProgression() {
    return {
      id: 'player',
      level: 1,
      xp: 0,
      xpToNext: 100,
      totalMatches: 0,
      totalWins: 0,
      totalGoals: 0,
      totalPasses: 0,
      totalTackles: 0,
      currentStreak: 0,
      bestStreak: 0,
      badges: [],
      matchesPlayed: [],
      levelRewardsClaimed: []
    };
  }
}

export const storage = new StorageService();
