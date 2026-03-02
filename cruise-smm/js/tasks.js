/* ===== TASK SYSTEM ===== */
import { getState, getDifficultyMultiplier, getTeamHappiness } from './state.js';
import { CHARACTERS } from './characters.js';

export const TASK_CATEGORIES = {
  photo: {
    id: 'photo',
    name: 'Photo Shootings',
    icon: '📸',
    minigame: 'framePerfect',
    requires: ['photographer', 'assistant'],
  },
  safety: {
    id: 'safety',
    name: 'Safety Courses',
    icon: '🛟',
    minigame: 'quickEdit',
    requires: ['coordinator'],
  },
  interview: {
    id: 'interview',
    name: 'Crew Interviews',
    icon: '🎤',
    minigame: 'perfectQuestion',
    requires: ['camera'],
  },
  social: {
    id: 'social',
    name: 'Social Media',
    icon: '📱',
    minigame: 'hashtagRush',
    requires: [],
  },
};

export const TASKS = {
  // PHOTO SHOOTINGS
  photo_pool: {
    id: 'photo_pool', category: 'photo', name: 'Pool Deck Shoot',
    icon: '🏊', location: 'Pool Deck', baseFame: 200, money: 150,
    bg: 'linear-gradient(135deg, #0ea5e9, #06b6d4)',
    unlockDay: 1, unlockCondition: null,
    description: 'Capture stunning poolside content with gorgeous Mediterranean backdrop.',
  },
  photo_restaurant: {
    id: 'photo_restaurant', category: 'photo', name: 'Restaurant Shoot',
    icon: '🍽️', location: 'Restaurant', baseFame: 180, money: 130,
    bg: 'linear-gradient(135deg, #92400e, #b45309)',
    unlockDay: 1, unlockCondition: null,
    description: 'Food photography session in the ship\'s luxury restaurant.',
  },
  photo_bridge: {
    id: 'photo_bridge', category: 'photo', name: 'Bridge Shoot',
    icon: '🌉', location: 'Bridge', baseFame: 400, money: 300,
    bg: 'linear-gradient(135deg, #1e3a5f, #0369a1)',
    unlockDay: 3, unlockCondition: null,
    description: 'Exclusive bridge photography session with the navigation team.',
  },
  photo_captain: {
    id: 'photo_captain', category: 'photo', name: "Captain's Deck Shoot",
    icon: '⚓', location: "Captain's Deck", baseFame: 800, money: 500,
    bg: 'linear-gradient(135deg, #92400e, #fbbf24)',
    unlockDay: 4, unlockCondition: (s) => s.captainTrust >= 60,
    unlockText: 'Requires Captain trust ≥ 60',
    description: 'The ultimate shoot at the Captain\'s personal deck. Incredibly prestigious.',
  },
  photo_sunset: {
    id: 'photo_sunset', category: 'photo', name: 'Sunset Balcony Shoot',
    icon: '🌅', location: 'Sunset Balcony', baseFame: 350, money: 250,
    bg: 'linear-gradient(135deg, #f97316, #ec4899, #8b5cf6)',
    unlockDay: 1, unlockCondition: null,
    description: 'Golden hour photography from the sunset balcony. Pure magic.',
  },

  // SAFETY COURSES
  safety_lifejacket: {
    id: 'safety_lifejacket', category: 'safety', name: 'Lifejacket Demo',
    icon: '🦺', location: 'Main Deck', baseFame: 150, money: 100,
    bg: 'linear-gradient(135deg, #f97316, #fbbf24)',
    unlockDay: 1, unlockCondition: null,
    description: 'Film a basic lifejacket demonstration for safety content.',
  },
  safety_overboard: {
    id: 'safety_overboard', category: 'safety', name: 'Man Overboard Drill',
    icon: '🌊', location: 'Main Deck', baseFame: 300, money: 220,
    bg: 'linear-gradient(135deg, #0ea5e9, #1e3a5f)',
    unlockDay: 2, unlockCondition: null,
    description: 'Dramatic man overboard rescue drill — great for engagement.',
  },
  safety_fire: {
    id: 'safety_fire', category: 'safety', name: 'Fire Evacuation',
    icon: '🔥', location: 'Lower Deck', baseFame: 280, money: 200,
    bg: 'linear-gradient(135deg, #ef4444, #f97316)',
    unlockDay: 2, unlockCondition: null,
    description: 'Fire evacuation procedure filming — informative and intense.',
  },
  safety_full: {
    id: 'safety_full', category: 'safety', name: 'Full Emergency Drill',
    icon: '🚨', location: 'All Decks', baseFame: 600, money: 450,
    bg: 'linear-gradient(135deg, #ef4444, #dc2626)',
    unlockDay: 4, unlockCondition: null,
    description: 'Complete emergency drill coverage — maximum content.',
  },

  // SOCIAL MEDIA TASKS
  social_post: {
    id: 'social_post', category: 'social', name: 'Post Design Session',
    icon: '🎨', location: 'Office', baseFame: 120, money: 80,
    bg: 'linear-gradient(135deg, #a78bfa, #ec4899)',
    unlockDay: 1, unlockCondition: null,
    happinessBoost: 3,
    description: 'Design and publish a stunning social media post.',
  },
  social_story: {
    id: 'social_story', category: 'social', name: 'Story Writing',
    icon: '📖', location: 'Office', baseFame: 100, money: 60,
    bg: 'linear-gradient(135deg, #0ea5e9, #a78bfa)',
    unlockDay: 1, unlockCondition: null,
    description: 'Write compelling stories for all social platforms.',
  },
  social_viral: {
    id: 'social_viral', category: 'social', name: 'Viral Campaign',
    icon: '🔥', location: 'Office', baseFame: 500, money: 400,
    bg: 'linear-gradient(135deg, #f97316, #ef4444)',
    unlockDay: 3, unlockCondition: (s) => s.tasksCompleted >= 3,
    unlockText: 'Requires 3 tasks completed',
    description: 'Launch a massive viral campaign across all platforms.',
  },
  social_collab: {
    id: 'social_collab', category: 'social', name: 'Collaboration Post',
    icon: '🤝', location: 'Various', baseFame: 250, money: 180,
    bg: 'linear-gradient(135deg, #22c55e, #0ea5e9)',
    unlockDay: 2, unlockCondition: null,
    fameMultiplier: 1.8,
    description: 'Collaborate with crew members for a cross-promotion post.',
  },
};

export function getAvailableTasks(state) {
  const available = [];
  for (const task of Object.values(TASKS)) {
    if (task.unlockDay > state.day) continue;
    if (task.unlockCondition && !task.unlockCondition(state)) continue;
    available.push(task);
  }
  return available;
}

export function getLockedTasks(state) {
  const locked = [];
  for (const task of Object.values(TASKS)) {
    if (task.unlockDay > state.day) { locked.push(task); continue; }
    if (task.unlockCondition && !task.unlockCondition(state)) { locked.push(task); continue; }
  }
  return locked;
}

export function getAllTasksForSlot(state) {
  const available = getAvailableTasks(state);
  const locked = getLockedTasks(state);
  return { available, locked };
}

export function calculateTaskFame(task, state, minigameMultiplier = 1) {
  let fame = task.baseFame;
  const diff = getDifficultyMultiplier();
  fame *= diff;

  // Equipment bonuses
  if (task.category === 'photo' && state.equipment.proCamera) fame *= 1.3;
  if (task.category === 'photo' && state.equipment.cinemaLens) fame *= 1.4;
  if (task.category === 'photo' && state.equipment.ledPanel && task.location !== 'Pool Deck' && task.location !== 'Sunset Balcony') fame *= 1.25;
  if (task.category === 'interview' && state.equipment.audioInterface) fame *= 1.2;
  if (task.category === 'social' && state.equipment.socialSuite) fame *= 1.35;
  if (state.equipment.drone && (task.id === 'photo_pool' || task.id === 'photo_sunset')) fame *= 1.8;

  // Team happiness effect
  const happiness = getTeamHappiness();
  if (happiness > 70) fame *= 1.2;
  else if (happiness < 40) fame *= 0.7;

  // Minigame multiplier
  fame *= minigameMultiplier;

  // Focus bonus from coffee
  if (state.focus > 0) {
    fame *= (1 + state.focus / 100);
  }

  // Task-specific multipliers
  if (task.fameMultiplier) fame *= task.fameMultiplier;

  // Energy penalty
  if (state.energy < 30) fame *= 0.75;

  // Team member special skills
  const members = state.teamMembers;
  if (task.category === 'photo' && task.id === 'photo_sunset' && members.marco.assignedTo === task.id) {
    fame *= 1.5; // Marco golden hour
  }
  if (task.category === 'photo' && members.yuki.assignedTo === task.id) {
    fame *= 1.4; // Yuki macro vision
  }
  if (task.category === 'social' && members.sofia.assignedTo === task.id) {
    fame *= 1.6; // Sofia viral brain
  }

  return Math.round(fame);
}

export function calculateTaskMoney(task, state) {
  let money = task.money;
  const diff = getDifficultyMultiplier();
  money *= diff;
  return Math.round(money);
}

export function getTaskMinigame(task) {
  const cat = TASK_CATEGORIES[task.category];
  return cat ? cat.minigame : null;
}

export function interviewCharacter(charId, state) {
  const c = CHARACTERS[charId];
  if (!c) return 0;
  const cs = state.characters[charId];
  if (!cs || cs.interviewed) return 0;

  cs.interviewed = true;
  cs.love += 8;
  if (cs.love > 100) cs.love = 100;

  let fameGain = c.interviewFame;
  if (state.equipment.audioInterface) fameGain *= 1.2;
  return Math.round(fameGain * getDifficultyMultiplier());
}
