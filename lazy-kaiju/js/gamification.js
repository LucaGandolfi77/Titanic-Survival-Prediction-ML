export class AchievementSystem {
    constructor() {
        this.unlocked = this._load();
        this.xp = parseInt(localStorage.getItem('lazykaiju_xp') || '0');
        this.level = parseInt(localStorage.getItem('lazykaiju_xplevel') || '1');
        this.totalTrash = parseInt(localStorage.getItem('lazykaiju_totaltrash') || '0');
        this.totalPlayTime = parseInt(localStorage.getItem('lazykaiju_playtime') || '0');
        this.badgePopup = null;
    }

    _load() {
        try {
            const data = localStorage.getItem('lazykaiju_badges');
            return data ? JSON.parse(data) : [];
        } catch {
            return [];
        }
    }

    _save() {
        try {
            localStorage.setItem('lazykaiju_badges', JSON.stringify(this.unlocked));
            localStorage.setItem('lazykaiju_xp', String(this.xp));
            localStorage.setItem('lazykaiju_xplevel', String(this.level));
            localStorage.setItem('lazykaiju_totaltrash', String(this.totalTrash));
            localStorage.setItem('lazykaiju_playtime', String(this.totalPlayTime));
        } catch {}
    }

    static ACHIEVEMENTS = [
        { id: 'first_play', name: '🎮 Prima Partita', desc: 'Completa la tua prima partita', icon: '🏁', condition: (s) => s.gamesPlayed >= 1 },
        { id: 'veteran', name: '🦎 Veterano', desc: 'Gioca 50 partite', icon: '⭐', condition: (s) => s.gamesPlayed >= 50 },
        { id: 'trash_100', name: '🗑️ Collezionista', desc: 'Raccogli 100 rifiuti totali', icon: '📦', condition: (s) => s.totalTrash >= 100 },
        { id: 'trash_1000', name: '💪 Pulizia Massima', desc: 'Raccogli 1000 rifiuti totali', icon: '🏆', condition: (s) => s.totalTrash >= 1000 },
        { id: 'score_500', name: '🌟 Talento', desc: 'Metti 500 punti in una partita', icon: '✨', condition: (s) => s.bestScore >= 500 },
        { id: 'score_2000', name: '💎 Leggenda', desc: 'Metti 2000 punti in una partita', icon: '💎', condition: (s) => s.bestScore >= 2000 },
        { id: 'karma_max', name: '🌿 Karma Puro', desc: 'Completa una partita a 100 karma', icon: '🌱', condition: (s) => s.finishedKarmaMax },
        { id: 'level_5', name: '🏙️ Cittadino', desc: 'Raggiungi il livello 5', icon: '🏙️', condition: (s) => s.level >= 5 },
        { id: 'level_10', name: '🌆 Metropoli', desc: 'Raggiungi il livello 10', icon: '🌆', condition: (s) => s.level >= 10 },
        { id: 'multi_win', name: '👥 Campione', desc: 'Vinci una partita multiplayer', icon: '👑', condition: (s) => s.multiWins >= 1 },
        { id: 'editor_use', name: '🎨 Architetto', desc: 'Crea un livello nell\'editor', icon: '🏗️', condition: (s) => s.usedEditor },
        { id: 'night_play', name: '🌙 Notte', desc: 'Gioca tra mezzanotte e le 5', icon: '🌙', condition: (s) => s.playedAtNight },
        { id: 'sweep_5', name: '💨 Cinque!', desc: '5 sweep consecutivi senza miss', icon: '💨', condition: (s) => s.maxSweeps >= 5 },
        { id: 'swipe_mobile', name: '📱 Mobile Master', desc: 'Gioca 10 partite su mobile', icon: '📱', condition: (s) => s.mobileGames >= 10 },
        { id: 'speed_run', name: '⚡ Veloce', desc: 'Completa un livello in <60s', icon: '⚡', condition: (s) => s.speedRun },
    ];

    check(stats = {}) {
        const newAchievements = [];
        for (const ach of AchievementSystem.ACHIEVEMENTS) {
            if (this.unlocked.includes(ach.id)) continue;
            if (ach.condition(stats)) {
                this.unlocked.push(ach.id);
                newAchievements.push(ach);
                this._grantXP(50);
            }
        }
        this._save();
        return newAchievements;
    }

    _grantXP(amount) {
        this.xp += amount;
        const nextLevel = this.level * 100;
        if (this.xp >= nextLevel && nextLevel > 0) {
            this.xp -= nextLevel;
            this.level++;
        }
    }

    getProgress() {
        const nextLevelXP = this.level * 100;
        return {
            xp: this.xp,
            nextLevelXP,
            level: this.level,
            percentage: Math.min(100, Math.round((this.xp / (nextLevelXP || 1)) * 100)),
            totalBadges: this.unlocked.length,
            totalAchievements: AchievementSystem.ACHIEVEMENTS.length
        };
    }

    updateSession(stats) {
        this.totalPlayTime += stats.sessionDuration || 0;
        this.totalTrash += stats.trashCleared || 0;
        return this.check(stats);
    }

    getBadgePopup() {
        return this.badgePopup;
    }
}