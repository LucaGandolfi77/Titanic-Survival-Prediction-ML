// Adaptive difficulty: a local heuristic that tracks reaction times and
// adjusts the timer window per chapter, keeping the game in the "flow zone".
// Bounded so it never becomes unfair: 4–12s window, 50–80% fast threshold.

const LIMITS = { min: 4, max: 12 };
const THRESHOLDS = { min: 0.5, max: 0.8 };
const EMA_ALPHA = 0.35;

export function createProfile(meta = {}) {
  return {
    avgReaction: (meta.timerSeconds || 8) * 0.4,
    samples: 0,
    timeLimit: meta.timerSeconds || 8,
    threshold: meta.fastThreshold || 0.65
  };
}

/** Fold one reaction (seconds) into the moving average. */
export function observe(profile, reactionSeconds) {
  const r = Math.max(0, reactionSeconds);
  profile.avgReaction =
    profile.samples === 0 ? r : profile.avgReaction + EMA_ALPHA * (r - profile.avgReaction);
  profile.samples += 1;
  return profile;
}

/**
 * Recalibrate for the next chapter: if the player answers consistently fast,
 * tighten the window; if they struggle, widen it.
 */
export function calibrateForChapter(profile) {
  if (profile.samples < 2) return profile; // not enough evidence yet
  const ratio = profile.avgReaction / profile.timeLimit;
  let delta = 0;
  if (ratio < 0.28) delta = -1.0; // too easy → tighten
  else if (ratio > 0.55) delta = +1.0; // too hard → widen
  profile.timeLimit = round1(profile.timeLimit + delta, LIMITS.min, LIMITS.max);
  profile.threshold = round2(profile.threshold - delta * 0.02, THRESHOLDS.min, THRESHOLDS.max);
  return profile;
}

function round1(value, min, max) {
  return Math.round(Math.min(Math.max(value, min), max) * 10) / 10;
}

function round2(value, min, max) {
  return Math.round(Math.min(Math.max(value, min), max) * 100) / 100;
}
