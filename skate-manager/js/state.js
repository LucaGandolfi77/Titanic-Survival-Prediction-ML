/* ===== Global game state ===== */

export const GameState = {
  // Meta
  teamName: 'Ice Stars',
  teamColor: '#7dd3fc',
  difficulty: 'semi-pro',
  season: 1,
  week: 1,
  maxWeeks: 12,

  // Resources
  money: 50000,
  fame: 0,
  points: 0,

  // Roster
  activeSquad: [],   // max 16
  reserveBench: [],  // max 8

  // Market
  marketSkaters: [],
  listedSkaters: [],       // your skaters listed for sale
  marketRefreshWeek: 1,    // market refreshes every 2 weeks
  scoutedThisWeek: false,

  // Competitions
  calendar: [],             // 12 week schedule
  enteredCompetitions: {},  // { weekIndex: true }
  competitionResults: [],   // history

  // Sponsors
  activeSponsors: [],       // { sponsor, weeksRemaining, breachCount }
  maxSimultaneous: 3,

  // AI rivals
  rivals: [],

  // Event log
  eventLog: [],

  // Season history
  seasonHistory: [],

  // Settings
  soundEnabled: true,
  sfxEnabled: true,
  volume: 0.3,
  autosave: true,

  // Mini-game state (transient)
  minigameActive: false,
  currentCompetition: null
};

// Save/Load
export function saveGame() {
  try {
    const data = JSON.stringify(GameState);
    localStorage.setItem('skate-manager-save', data);
    return true;
  } catch(e) {
    return false;
  }
}

export function loadGame() {
  try {
    const data = localStorage.getItem('skate-manager-save');
    if (!data) return false;
    const parsed = JSON.parse(data);
    Object.assign(GameState, parsed);
    return true;
  } catch(e) {
    return false;
  }
}

export function hasSave() {
  return !!localStorage.getItem('skate-manager-save');
}

export function resetState() {
  GameState.season = 1;
  GameState.week = 1;
  GameState.money = 50000;
  GameState.fame = 0;
  GameState.points = 0;
  GameState.activeSquad = [];
  GameState.reserveBench = [];
  GameState.marketSkaters = [];
  GameState.listedSkaters = [];
  GameState.marketRefreshWeek = 1;
  GameState.scoutedThisWeek = false;
  GameState.calendar = [];
  GameState.enteredCompetitions = {};
  GameState.competitionResults = [];
  GameState.activeSponsors = [];
  GameState.rivals = [];
  GameState.eventLog = [];
  GameState.seasonHistory = [];
  GameState.minigameActive = false;
  GameState.currentCompetition = null;
}
