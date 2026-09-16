// Achievement / Gamification system
import { IDBStorage } from './storage.js';

export const ACHIEVEMENTS = [
  { id: 'first_delivery', name: 'First Delivery', desc: 'Deliver your first package', icon: '📦', condition: (s) => s.totalDeliveries >= 1 },
  { id: 'speed_demon', name: 'Speed Demon', desc: 'Complete a level in under 45s', icon: '⚡', condition: (s) => s.bestLevelTime > 0 && s.bestLevelTime < 45 },
  { id: 'cell_explorer', name: 'Cell Explorer', desc: 'Visit all 8 cells', icon: '🌐', condition: (s) => s.uniqueCells >= 8 },
  { id: 'porter', name: 'Master Porter', desc: 'Deliver 50 packages total', icon: '🚚', condition: (s) => s.totalDeliveries >= 50 },
  { id: 'level_5', name: 'Dimensional Master', desc: 'Reach Level 5', icon: '🔮', condition: (s) => s.maxLevel >= 5 },
  { id: 'level_8', name: 'Hypercube God', desc: 'Reach Level 8', icon: '👑', condition: (s) => s.maxLevel >= 8 },
  { id: 'perfectionist', name: 'Perfectionist', desc: 'Complete a level with 0 failed deliveries', icon: '✅', condition: (s) => s.perfectRuns >= 1 },
  { id: 'time_warrior', name: 'Time Warrior', desc: 'Deliver with <5s remaining', icon: '⏰', condition: (s) => s.bestTimeRemaining > 0 && s.bestTimeRemaining < 5 },
  { id: 'portal_hopper', name: 'Portal Hopper', desc: 'Use 20 portals', icon: '🚪', condition: (s) => s.portalsUsed >= 20 },
  { id: 'collector', name: 'Score Collector', desc: 'Accumulate 10,000 points', icon: '💎', condition: (s) => s.totalScore >= 10000 },
];

export class AchievementSystem {
  constructor() {
    this.storage = new IDBStorage();
    this.unlocked = [];
    this.stats = this.defaultStats();
  }

  defaultStats() {
    return {
      totalDeliveries: 0,
      totalScore: 0,
      maxLevel: 1,
      uniqueCells: 0,
      bestLevelTime: 0,
      perfectRuns: 0,
      bestTimeRemaining: 999,
      portalsUsed: 0,
      sessionsPlayed: 0,
      cellsVisited: new Set(),
    };
  }

  async init() {
    try {
      const saved = await this.storage.loadSetting('hds_stats');
      if (saved) {
        this.stats = { ...this.defaultStats(), ...saved, cellsVisited: new Set(saved.cellsVisited || []) };
      }
      const unlocked = await this.storage.loadSetting('hds_achievements');
      this.unlocked = unlocked || [];
    } catch {
      this.stats = this.defaultStats();
      this.unlocked = [];
    }
  }

  async save() {
    await this.storage.saveSetting('hds_stats', {
      ...this.stats,
      cellsVisited: [...this.stats.cellsVisited],
    });
    await this.storage.saveSetting('hds_achievements', this.unlocked);
  }

  recordDelivery() {
    this.stats.totalDeliveries++;
    this.checkAll();
    this.save();
  }

  recordScore(points) {
    this.stats.totalScore += points;
    this.checkAll();
    this.save();
  }

  recordLevelComplete(level, timeSpent, deliveriesFailed, cellsVisited) {
    if (!this.stats.bestLevelTime || timeSpent < this.stats.bestLevelTime) {
      this.stats.bestLevelTime = timeSpent;
    }
    if (deliveriesFailed === 0) this.stats.perfectRuns++;
    this.stats.maxLevel = Math.max(this.stats.maxLevel, level);
    this.stats.portalsUsed += 0; // updated elsewhere
    cellsVisited.forEach(c => this.stats.cellsVisited.add(c));
    this.stats.uniqueCells = this.stats.cellsVisited.size;
    this.checkAll();
    this.save();
  }

  recordPortalUse() {
    this.stats.portalsUsed++;
    this.checkAll();
    this.save();
  }

  recordTimeRemaining(time) {
    if (time < this.stats.bestTimeRemaining) {
      this.stats.bestTimeRemaining = time;
    }
    this.checkAll();
    this.save();
  }

  async recordSessionStart() {
    this.stats.sessionsPlayed++;
    await this.save();
  }

  checkAll() {
    for (const achievement of ACHIEVEMENTS) {
      if (!this.unlocked.includes(achievement.id) && achievement.condition(this.stats)) {
        this.unlocked.push(achievement.id);
        this.onAchievementUnlock(achievement);
      }
    }
  }

  onAchievementUnlock(achievement) {
    console.log(`🏆 Achievement unlocked: ${achievement.name}`);
    try {
      if (navigator.vibrate) navigator.vibrate([50, 100, 50, 100, 150]);
    } catch { /* noop */ }
  }

  isUnlocked(id) {
    return this.unlocked.includes(id);
  }

  getProgress(achievementId) {
    const a = ACHIEVEMENTS.find(ach => ach.id === achievementId);
    if (!a) return 0;
    // Simple check: 0 or 100
    return a.condition(this.stats) ? 100 : 0;
  }
}
