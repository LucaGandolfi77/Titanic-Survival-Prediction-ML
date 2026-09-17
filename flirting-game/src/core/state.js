// Central game state with a tiny pub/sub emitter.
// Mutate numeric fields freely; use setState() for structural patches
// (scene changes, phase switches) that subscribers may react to.

const listeners = new Set();

export const state = {
  playerGender: 'male',
  interestGender: 'female',
  aiMode: false,
  sharedPrompt: null,
  duetActive: false,
  voiceUsed: false,
  character: null,
  scene: null,
  score: 0,
  streak: 0,
  bestStreak: 0,
  secrets: 0,
  // Game config, loaded from dialogues.json meta block
  timeLimit: 8,
  fastThreshold: 0.65,
  // Phase flags
  awaitingChoice: false,
  responsePhase: false,
  lastEndingTier: null,
  // Choice log for the swipe-to-review history
  history: []
};

export function setState(patch) {
  Object.assign(state, patch);
  listeners.forEach((fn) => fn(state));
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function resetRound() {
  Object.assign(state, {
    score: 0,
    streak: 0,
    bestStreak: 0,
    secrets: 0,
    awaitingChoice: false,
    responsePhase: false,
    lastEndingTier: null,
    history: []
  });
}
