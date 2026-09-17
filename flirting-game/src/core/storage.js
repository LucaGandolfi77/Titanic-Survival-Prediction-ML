// Safe localStorage wrapper: personal records + preferences survive reloads,
// and never crash in private mode or when storage is disabled.

const KEY = 'speed-crush:v1';

const DEFAULTS = {
  bestScore: 0,
  bestStreak: 0,
  gamesPlayed: 0,
  endings: [],
  achievements: [],
  dailyStreak: null,
  reminders: false,
  aiModelDownloaded: false,
  difficulty: { avgReaction: 3.2, samples: 0, timeLimit: 8, threshold: 0.65 },
  prefs: { playerGender: 'male', interestGender: 'female', aiMode: false, language: null }
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return clone(DEFAULTS);
    const parsed = JSON.parse(raw);
    return { ...clone(DEFAULTS), ...parsed };
  } catch {
    return clone(DEFAULTS);
  }
}

function write(data) {
  try {
    localStorage.setItem(KEY, JSON.stringify(data));
  } catch {
    /* storage unavailable (private mode) — silently ignore */
  }
}

export const storage = {
  load: read,
  save(patch) {
    write({ ...read(), ...patch });
  },
  update(fn) {
    const next = fn(read());
    if (next) write(next);
  },
  savePrefs(playerGender, interestGender, aiMode = false, language = null) {
    this.save({ prefs: { playerGender, interestGender, aiMode, language } });
  }
};
