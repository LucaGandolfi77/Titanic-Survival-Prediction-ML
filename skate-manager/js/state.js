/* ===== Global game state ===== */

import {
  SAVE_KEY,
  SAVE_VERSION,
  STARTING_MONEY,
  SEASON_WEEKS,
  DEFAULT_VOLUME
} from './config.js';

// Fields that are never persisted (transient session state)
const TRANSIENT_KEYS = ['minigameActive', 'currentCompetition'];

export const GameState = {
  // Meta
  teamName: 'Ice Stars',
  teamColor: '#7dd3fc',
  difficulty: 'semi-pro',
  season: 1,
  week: 1,
  maxWeeks: SEASON_WEEKS,
  saveVersion: SAVE_VERSION,

  // Resources
  money: STARTING_MONEY['semi-pro'],
  fame: 0,
  points: 0,

  // Roster
  activeSquad: [],   // max 16
  reserveBench: [],  // max 8

  // Market
  marketSkaters: [],
  listedSkaters: [],       // your skaters listed for sale
  marketRefreshWeek: 2,    // next week at which the market refreshes
  scoutedThisWeek: false,

  // Competitions
  calendar: [],             // 12 week schedule
  enteredCompetitions: {},  // { weekIndex: true }
  competitionResults: [],   // history

  // Sponsors
  activeSponsors: [],       // { sponsor, weeksRemaining, breachCount }

  // AI rivals
  rivals: [],

  // Coaching staff (ids of hired staff, see config.js STAFF)
  staff: [],

  // Team records
  records: {
    bestScore: 0,
    bestPlacement: 99,
    totalWins: 0,
    totalPodiums: 0,
    totalPrize: 0
  },

  // Hall of fame — retired legends (retired with overall >= 75)
  hallOfFame: [],

  // Event log
  eventLog: [],

  // Season history
  seasonHistory: [],

  // Settings
  sfxEnabled: true,
  volume: DEFAULT_VOLUME,
  autosave: true,
  language: 'en',

  // Mini-game state (transient, never saved)
  minigameActive: false,
  currentCompetition: null
};

// Save/Load
// Slots: slot 1 = legacy key (backward compatible), slots 2–3 = extra slots
function slotKey(slot) {
  return slot === 1 ? SAVE_KEY : `${SAVE_KEY}-${slot}`;
}

export function serializeState() {
  const data = {};
  for (const [key, value] of Object.entries(GameState)) {
    if (!TRANSIENT_KEYS.includes(key)) data[key] = value;
  }
  return data;
}

export function saveGame(slot = 1) {
  try {
    localStorage.setItem(slotKey(slot), JSON.stringify(serializeState()));
    return true;
  } catch {
    return false;
  }
}

export function isValidSave(parsed) {
  return !!parsed && typeof parsed === 'object' &&
    typeof parsed.week === 'number' &&
    typeof parsed.season === 'number' &&
    Array.isArray(parsed.activeSquad) &&
    Array.isArray(parsed.reserveBench) &&
    Array.isArray(parsed.calendar);
}

// Upgrade older saves to the current schema (mutates and returns `parsed`)
export function migrateState(parsed) {
  const version = parsed.saveVersion || 1;

  if (version < 2) {
    // v2: market refresh cadence is driven by marketRefreshWeek (next refresh week)
    const week = typeof parsed.week === 'number' ? parsed.week : 1;
    if (typeof parsed.marketRefreshWeek !== 'number' || parsed.marketRefreshWeek < week + 1) {
      parsed.marketRefreshWeek = week + 1;
    }
  }

  // Fill in any keys missing from older saves with current defaults
  for (const [key, value] of Object.entries(GameState)) {
    if (!(key in parsed) && !TRANSIENT_KEYS.includes(key)) parsed[key] = value;
  }

  parsed.saveVersion = SAVE_VERSION;
  return parsed;
}

export function loadGame(slot = 1) {
  try {
    const data = localStorage.getItem(slotKey(slot));
    if (!data) return false;
    const parsed = JSON.parse(data);
    if (!isValidSave(parsed)) return false;
    Object.assign(GameState, migrateState(parsed));
    // Never restore transient session state
    GameState.minigameActive = false;
    GameState.currentCompetition = null;
    return true;
  } catch {
    return false;
  }
}

export function hasSave(slot = 1) {
  return !!localStorage.getItem(slotKey(slot));
}

export function hasAnySave() {
  return hasSave(1) || hasSave(2) || hasSave(3);
}

/** Light metadata for the save-slot picker (does not touch GameState). */
export function getSaveInfo(slot = 1) {
  try {
    const data = localStorage.getItem(slotKey(slot));
    if (!data) return null;
    const parsed = JSON.parse(data);
    return {
      teamName: parsed.teamName || '—',
      season: parsed.season || 1,
      week: parsed.week || 1,
      difficulty: parsed.difficulty || 'semi-pro'
    };
  } catch {
    return null;
  }
}

export function resetState() {
  GameState.season = 1;
  GameState.week = 1;
  GameState.money = STARTING_MONEY['semi-pro'];
  GameState.fame = 0;
  GameState.points = 0;
  GameState.activeSquad = [];
  GameState.reserveBench = [];
  GameState.marketSkaters = [];
  GameState.listedSkaters = [];
  GameState.marketRefreshWeek = 2;
  GameState.scoutedThisWeek = false;
  GameState.calendar = [];
  GameState.enteredCompetitions = {};
  GameState.competitionResults = [];
  GameState.activeSponsors = [];
  GameState.rivals = [];
  GameState.staff = [];
  GameState.records = {
    bestScore: 0,
    bestPlacement: 99,
    totalWins: 0,
    totalPodiums: 0,
    totalPrize: 0
  };
  GameState.hallOfFame = [];
  GameState.eventLog = [];
  GameState.seasonHistory = [];
  GameState.minigameActive = false;
  GameState.currentCompetition = null;
}
