// Gamification: achievements + daily streaks, evaluated over the game
// records and the playthrough history. Unlocks fire staggered toasts.

import { storage } from '../../core/storage.js';
import { showToast } from '../../ui/toast.js';

export const ACHIEVEMENTS = [
  { id: 'first-move', icon: '💫', name: 'First Move', desc: 'Complete your first story', check: (ctx) => ctx.gamesPlayed >= 1 },
  { id: 'speed-demon', icon: '⚡', name: 'Speed Demon', desc: '5 fast choices in one story', check: (ctx) => ctx.fastCount >= 5 },
  { id: 'secret-keeper', icon: '🔐', name: 'Secret Keeper', desc: 'Unlock a secret scene', check: (ctx) => ctx.secrets >= 1 },
  { id: 'electric', icon: '💘', name: 'Electric', desc: 'Reach the Electric ending', check: (ctx) => ctx.tier === 'great' },
  { id: 'streak-master', icon: '🔥', name: 'Streak Master', desc: 'Fast streak of 4', check: (ctx) => ctx.bestStreak >= 4 },
  { id: 'night-owl', icon: '🌙', name: 'Night Owl', desc: 'Play after 11pm', check: (ctx) => ctx.hour >= 23 || ctx.hour < 5 },
  { id: 'charmer', icon: '✍️', name: 'Custom Charmer', desc: 'Write your own line', check: (ctx) => ctx.customCount >= 1 },
  { id: 'voice-flirt', icon: '🎙️', name: 'Voice Flirt', desc: 'Answer by voice', check: (ctx) => ctx.voiceUsed },
  { id: 'duet', icon: '👯', name: 'Duet Partner', desc: 'Play a co-op duet', check: (ctx) => ctx.duetPlayed },
  { id: 'collector', icon: '🏆', name: 'Collector', desc: 'Unlock all 3 endings', check: (ctx) => ctx.endings.length >= 3 }
];

/**
 * Evaluate achievements against the context; persist + toast the unlocks.
 * Returns the newly unlocked list.
 */
export function evaluateAchievements(context) {
  const data = storage.load();
  const unlocked = data.achievements || [];
  const newly = [];
  for (const achievement of ACHIEVEMENTS) {
    if (!unlocked.includes(achievement.id) && achievement.check(context)) {
      newly.push(achievement);
      unlocked.push(achievement.id);
    }
  }
  if (newly.length) {
    storage.save({ achievements: unlocked });
    newly.forEach((achievement, index) => {
      setTimeout(() => {
        showToast(`${achievement.icon} Achievement: ${achievement.name}`, { duration: 3500 });
      }, index * 1200);
    });
  }
  return newly;
}

function localDateKey(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(
    date.getDate()
  ).padStart(2, '0')}`;
}

/** Daily streak: consecutive days with at least one completed story. */
export function touchDailyStreak() {
  const data = storage.load();
  const streak = data.dailyStreak || { count: 0, lastDay: null };
  const today = localDateKey(new Date());
  if (streak.lastDay === today) return streak.count;
  const yesterday = localDateKey(new Date(Date.now() - 86400000));
  streak.count = streak.lastDay === yesterday ? streak.count + 1 : 1;
  streak.lastDay = today;
  storage.save({ dailyStreak: streak });
  return streak.count;
}

export function dailyStreakBadge() {
  const { dailyStreak } = storage.load();
  const count = dailyStreak?.count || 0;
  return count > 0 ? `🔥 ${count}-day streak` : null;
}

/** All achievements with their unlocked state, for the stats vault. */
export function achievementStates() {
  const unlocked = storage.load().achievements || [];
  return ACHIEVEMENTS.map((a) => ({ ...a, unlocked: unlocked.includes(a.id) }));
}
