const STORAGE_KEY = 'chaos-knots-save';

const DEFAULT_STATE = {
  threads: 150,
  roster: ['ember', 'ripple', 'flamy'],
  unravelerLevels: { ember: 1, ripple: 1, flamy: 1 },
  threadShards: {},
  currentLevel: 1,
  levelStars: {},
  highScore: 0,
  totalMatches: 0,
  pullCount: 0,
  pityRare: 0,
  pityEpic: 0,
  pityLegendary: 0,
  hasFreePull: true,
  firstInstall: false,
  lastPlayDate: null,
};

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_STATE };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_STATE, ...parsed };
  } catch (e) {
    return { ...DEFAULT_STATE };
  }
}

function saveState(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    console.warn('Failed to save state', e);
  }
}

function getState() {
  return loadState();
}

function updateState(updates) {
  const state = loadState();
  Object.assign(state, updates);
  saveState(state);
  return state;
}

function addThread(amount) {
  const state = loadState();
  state.threads += amount;
  saveState(state);
  return state.threads;
}

function spendThreads(amount) {
  const state = loadState();
  if (state.threads < amount) return false;
  state.threads -= amount;
  saveState(state);
  return true;
}
