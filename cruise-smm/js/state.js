/* ===== GLOBAL GAME STATE ===== */

const defaultState = {
  playerName: 'Alex Morgan',
  teamName: 'Aurora Creative',
  day: 1,
  currentSlot: 0,
  money: 3000,
  fame: 0,
  energy: 100,
  focus: 0,

  difficulty: 'Normal', // Easy, Normal, Hard
  soundOn: true,
  musicVolume: 50,
  autoDialogue: false,

  teamHappiness: 60,
  teamMembers: {
    jessica:  { happiness: 80, status: 'available', assignedTo: null },
    marta:    { happiness: 75, status: 'available', assignedTo: null },
    valentina:{ happiness: 78, status: 'available', assignedTo: null },
    chiara:   { happiness: 72, status: 'available', assignedTo: null },
  },

  characters: {
    carmen:   { met: false, love: 10, unlockDay: 1, interviewed: false, giftGiven: false, mood: 'neutral' },
    baptiste: { met: false, love: 5,  unlockDay: 2, interviewed: false, giftGiven: false, mood: 'neutral' },
    marina:   { met: false, love: 0,  unlockDay: 2, interviewed: false, giftGiven: false, mood: 'neutral' },
    theo:     { met: false, love: 20, unlockDay: 1, interviewed: false, giftGiven: false, mood: 'happy'   },
    isabel:   { met: false, love: 0,  unlockDay: 3, interviewed: false, giftGiven: false, mood: 'neutral' },
    james:    { met: false, love: 0,  unlockDay: 3, interviewed: false, giftGiven: false, mood: 'neutral' },
    luna:     { met: false, love: 15, unlockDay: 1, interviewed: false, giftGiven: false, mood: 'happy'   },
    captain:  { met: false, love: 0,  unlockDay: 1, interviewed: false, giftGiven: false, mood: 'neutral' },
  },

  equipment: {
    proCamera: false,
    cinemaLens: false,
    ledPanel: false,
    audioInterface: false,
    socialSuite: false,
    drone: false,
  },

  tasksCompleted: 0,
  tasksCompletedToday: 0,
  dailyFame: 0,
  dailyMoneyEarned: 0,
  dailyMoneySpent: 0,
  dailyEvents: [],

  daySlots: [],   // filled per day
  lunchDone: false,
  eveningDone: false,

  captainTrust: 0,
  viralCampaignReady: false,

  // Milestones
  milestones: {
    '10000': false,
    '50000': false,
    '100000': false,
    '500000': false,
    '1000000': false,
  },

  gamePhase: 'menu', // menu, setup, dayStart, playing, lunch, date, minigame, evening, ending
};

let gameState = null;

export function getState() {
  return gameState;
}

export function setState(newState) {
  gameState = newState;
}

export function createNewState() {
  gameState = JSON.parse(JSON.stringify(defaultState));
  return gameState;
}

export function resetDayStats() {
  if (!gameState) return;
  gameState.tasksCompletedToday = 0;
  gameState.dailyFame = 0;
  gameState.dailyMoneyEarned = 0;
  gameState.dailyMoneySpent = 0;
  gameState.dailyEvents = [];
  gameState.lunchDone = false;
  gameState.eveningDone = false;
  gameState.currentSlot = 0;
  gameState.energy = 100;
  gameState.focus = 0;

  // Reset team statuses
  for (const m of Object.values(gameState.teamMembers)) {
    m.status = 'available';
    m.assignedTo = null;
  }
}

export function getTeamHappiness() {
  if (!gameState) return 60;
  const members = Object.values(gameState.teamMembers);
  const avg = members.reduce((s, m) => s + m.happiness, 0) / members.length;
  gameState.teamHappiness = Math.round(avg);
  return gameState.teamHappiness;
}

export function getCharLoveLevel(charId) {
  if (!gameState || !gameState.characters[charId]) return 'stranger';
  const love = gameState.characters[charId].love;
  if (love >= 81) return 'romance';
  if (love >= 61) return 'close';
  if (love >= 41) return 'friend';
  if (love >= 21) return 'acquaintance';
  return 'stranger';
}

export function getFameMilestone(fame) {
  if (fame >= 1000000) return '👑';
  if (fame >= 500000) return '🔥';
  if (fame >= 100000) return '💫';
  if (fame >= 50000) return '🌟';
  if (fame >= 10000) return '⭐';
  return '';
}

export function getDifficultyMultiplier() {
  if (!gameState) return 1;
  switch (gameState.difficulty) {
    case 'Easy': return 1.3;
    case 'Hard': return 0.7;
    default: return 1;
  }
}
