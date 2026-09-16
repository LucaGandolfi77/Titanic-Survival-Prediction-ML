const XP_PER_GOAL = 50;
const XP_PER_PASS = 10;
const XP_PER_TACKLE_WIN = 20;
const XP_PER_MATCH = 25;
const XP_PER_WIN = 75;
const XP_PER_CLEAN_SHEET = 60;
const XP_FOR_LEVEL = (level) => 100 * level;

const BADGES = [
  { id: 'first_goal', name: '⚽ Primo Gol', desc: 'Segna il tuo primo gol', icon: '⚽', condition: (p) => p.totalGoals >= 1 },
  { id: 'hat_trick', name: '🍣 Hat Trick', desc: '3 gol in una partita', icon: '🍣', condition: (p, ctx) => ctx.lastMatchGoals >= 3 },
  { id: 'first_win', name: '🏆 Prima Vittoria', desc: 'Vinci la tua prima partita', icon: '🏆', condition: (p) => p.totalWins >= 1 },
  { id: 'win_streak_3', name: '🔥 Serie Vincente', desc: '3 vittorie consecutive', icon: '🔥', condition: (p) => p.currentStreak >= 3 },
  { id: 'win_streak_5', name: '💎 Serie Impeccabile', desc: '5 vittorie consecutive', icon: '💎', condition: (p) => p.currentStreak >= 5 },
  { id: 'defender', name: '🛡️ Muro', desc: 'Clean sheet (0 gol subiti)', icon: '🛡️', condition: (p, ctx) => ctx.lastMatchOpponentGoals === 0 },
  { id: 'top_scorer', name: '🎯 Capocannoniere', desc: '10 gol totali', icon: '🎯', condition: (p) => p.totalGoals >= 10 },
  { id: 'veteran', name: '⭐ Veterano', desc: '50 partite giocate', icon: '⭐', condition: (p) => p.totalMatches >= 50 },
  { id: 'legend', name: '👑 Leggenda', desc: '100 partite e 50 vittorie', icon: '👑', condition: (p) => p.totalMatches >= 100 && p.totalWins >= 50 },
  { id: 'pass_master', name: '🧤 Maestro dei Passaggi', desc: '100 passaggi riusciti', icon: '🧤', condition: (p) => p.totalPasses >= 100 },
  { id: 'tackle_master', name: '⚔️ Maestro delle Rinviate', desc: '100 tackle completati', icon: '⚔️', condition: (p) => p.totalTackles >= 100 },
  { id: 'level_5', name: '🥇 Livello 5', desc: 'Raggiungi il livello 5', icon: '🥇', condition: (p) => p.level >= 5 },
  { id: 'level_10', name: '🥈 Livello 10', desc: 'Raggiungi il livello 10', icon: '🥈', condition: (p) => p.level >= 10 },
];

class GamificationEngine {
  constructor(storageService) {
    this.storage = storageService;
    this.progression = null;
  }

  async init() {
    await this.storage.ready();
    this.progression = await this.storage.getProgression();
    if (!this.progression || this.progression.id !== 'player') {
      this.progression = this.storage._defaultProgression();
      await this.storage.saveProgression(this.progression);
    }
    return this.progression;
  }

  getLevelInfo() {
    return {
      level: this.progression.level,
      xp: this.progression.xp,
      xpToNext: this.progression.xpToNext,
      percent: Math.min(100, Math.round((this.progression.xp / this.progression.xpToNext) * 100))
    };
  }

  addXP(amount) {
    this.progression.xp += amount;
    let leveledUp = false;
    while (this.progression.xp >= this.progression.xpToNext) {
      this.progression.xp -= this.progression.xpToNext;
      this.progression.level++;
      this.progression.xpToNext = XP_FOR_LEVEL(this.progression.level);
      leveledUp = true;
    }
    if (leveledUp) {
      this.checkBadges({ levelUp: true });
      this.storage.saveProgression(this.progression);
    }
    return { leveledUp, newLevel: this.progression.level };
  }

  recordMatch(result) {
    const { goalsScored = 0, goalsConceded = 0, passes = 0, tackles = 0, won = false } = result;

    this.progression.totalMatches++;
    this.progression.totalGoals += goalsScored;
    this.progression.totalPasses += passes;
    this.progression.totalTackles += tackles;

    if (won) {
      this.progression.totalWins++;
      this.progression.currentStreak++;
      this.progression.bestStreak = Math.max(this.progression.bestStreak, this.progression.currentStreak);
    } else {
      this.progression.currentStreak = 0;
    }

    let xpGained = XP_PER_MATCH;
    xpGained += goalsScored * XP_PER_GOAL;
    xpGained += passes * XP_PER_PASS;
    xpGained += tackles * XP_PER_TACKLE_WIN;
    if (won) xpGained += XP_PER_WIN;
    if (goalsConceded === 0 && goalsScored > 0) xpGained += XP_PER_CLEAN_SHEET;

    this.progression.matchesPlayed.push({
      date: new Date().toISOString(),
      goalsScored,
      goalsConceded,
      won,
      xpGained
    });
    if (this.progression.matchesPlayed.length > 100) {
      this.progression.matchesPlayed = this.progression.matchesPlayed.slice(-100);
    }

    const lastMatchGoals = goalsScored;
    const lastMatchOpponentGoals = goalsConceded;

    this.addXP(xpGained);
    this.checkBadges({
      lastMatchGoals,
      lastMatchOpponentGoals
    });

    this.storage.saveProgression(this.progression);
    return { xpGained, newLevel: this.progression.level };
  }

  checkBadges(context = {}) {
    const newBadges = [];
    for (const badge of BADGES) {
      if (!this.progression.badges.includes(badge.id)) {
        try {
          if (badge.condition(this.progression, context)) {
            this.progression.badges.push(badge.id);
            newBadges.push(badge);
          }
        } catch (e) {
          console.warn('Badge check failed for', badge.id, e);
        }
      }
    }
    if (newBadges.length > 0) {
      this.storage.saveProgression(this.progression);
    }
    return newBadges;
  }

  getActiveBadges() {
    return BADGES.filter(b => this.progression.badges.includes(b.id));
  }

  getStats() {
    return {
      totalMatches: this.progression.totalMatches,
      totalWins: this.progression.totalWins,
      totalGoals: this.progression.totalGoals,
      totalPasses: this.progression.totalPasses,
      totalTackles: this.progression.totalTackles,
      currentStreak: this.progression.currentStreak,
      bestStreak: this.progression.bestStreak,
      winRate: this.progression.totalMatches > 0
        ? Math.round((this.progression.totalWins / this.progression.totalMatches) * 100)
        : 0
    };
  }
}

export { GamificationEngine, XP_PER_GOAL, XP_PER_PASS, XP_PER_TACKLE_WIN, XP_PER_MATCH, XP_PER_WIN, XP_PER_CLEAN_SHEET, BADGES };
