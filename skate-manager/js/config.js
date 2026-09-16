/* ===== Central configuration & balance values =====
 *
 * Single source of truth for tunable game constants. Game-design changes
 * should happen here, not as magic numbers scattered across modules.
 */

// ---- Save ----
export const SAVE_KEY = 'skate-manager-save';
export const SAVE_VERSION = 2;

// ---- Squad & roster ----
export const ACTIVE_SQUAD_SIZE = 16;
export const RESERVE_SIZE = 8;
export const MAX_ROSTER = 24;           // active + reserve combined
export const TRAIN_TEAM_COST = 5000;
export const SCOUT_COST = 2000;

// ---- Economy ----
export const STARTING_MONEY = {
  amateur: 150000,
  'semi-pro': 100000,
  elite: 80000
};

// ---- Weekly tick ----
export const SEASON_WEEKS = 12;
export const FORM_FLUCTUATION = 10;      // ± form per week
export const MORALE_FLUCTUATION = 5;     // ± morale per week
export const CONTRACT_EXPIRY_WARN_WEEKS = 2;
export const RETIREMENT_AGE = 35;
export const RETIREMENT_CHANCE = 0.3;

// ---- Sponsors ----
export const MAX_ACTIVE_SPONSORS = 3;
export const SPONSOR_BREACH_LIMIT = 2;   // warnings before a deal is cancelled
export const TEMPO_BONUS_PERK = 0.25;    // CoolBreeze: +25% on high-tempo bonus

// ---- Skater generation ----
export const STAT_RANGES = {
  1: [30, 50],
  2: [50, 70],
  3: [70, 90],
  4: [85, 99]
};
export const AGE_RANGES = {
  1: [16, 20],
  2: [19, 28],
  3: [22, 30],
  4: [24, 32]
};

// ---- Mini-game ----
export const ROUTINE_DURATION = 60;      // seconds
export const FORMATION_DURATION = 8;     // seconds per formation
export const FORMATION_COOLDOWN = 5;     // seconds between formations
export const PERFECT_BONUS = 300;
export const MUSIC_BONUS_SCALE = 1.3;
export const FALL_PENALTY = 50;
export const SAVE_BONUS = 10;
export const SCORE_MILESTONE = 5000;     // play a jingle every N points
export const WOBBLE_MAX_TIME = 2.0;      // seconds to tap a wobbling skater
export const FALLEN_MAX_TIME = 5.0;      // seconds a skater stays down
export const WOBBLE_BASE_CHANCE = 0.002; // per-second base chance before tempo/stamina modifiers
export const ENTRANCE_PHASE = 3;         // seconds of entrance animation

export const TEMPO_MULTIPLIERS = { slow: 1.0, medium: 1.5, fast: 2.0, max: 2.5 };
export const TEMPO_RISK = { slow: 0, medium: 0.3, fast: 0.6, max: 1.0 };
export const TEMPO_LABELS = {
  slow: '🎵 SLOW ×1.0',
  medium: '🎶 MED ×1.5',
  fast: '🎸 FAST ×2.0',
  max: '🔥 MAX ×2.5'
};

// ---- Audio ----
export const DEFAULT_VOLUME = 0.3;
export const MUSIC_TEMPO_CONFIG = {
  slow:   { bpm: 120, freq: 220, detune: 0 },
  medium: { bpm: 150, freq: 330, detune: 200 },
  fast:   { bpm: 180, freq: 440, detune: 400 },
  max:    { bpm: 210, freq: 523, detune: 600 }
};
