export const UI = {
  screens: {
    menu: document.getElementById('screen-menu'),
    difficulty: document.getElementById('screen-difficulty'),
    setup: document.getElementById('screen-team-setup'),
    pause: document.getElementById('screen-pause'),
    halftime: document.getElementById('screen-halftime'),
    fulltime: document.getElementById('screen-fulltime'),
    settings: document.getElementById('screen-settings'),
    records: document.getElementById('screen-records')
  },
  
  overlays: {
    hud: document.getElementById('hud'),
    controls: document.getElementById('controls-overlay'),
    event: document.getElementById('match-event-overlay')
  },

  showScreen(name) {
    Object.values(this.screens).forEach(s => s.classList.add('hidden'));
    Object.values(this.screens).forEach(s => s.classList.remove('active'));
    
    if (this.screens[name]) {
      this.screens[name].classList.remove('hidden');
      this.screens[name].classList.add('active');
    }

    if (name === 'menu' || name === 'setup' || name === 'fulltime') {
      this.hideHUD();
    }
  },

  showHUD() {
    this.overlays.hud.classList.remove('hidden');
    this.overlays.controls.classList.remove('hidden');
  },

  hideHUD() {
    this.overlays.hud.classList.add('hidden');
    this.overlays.controls.classList.add('hidden');
  },

  updateScore(home, away) {
    document.getElementById('home-score').textContent = home;
    document.getElementById('away-score').textContent = away;
    const pauseScoreEl = document.getElementById('pause-score');
    const halftimeScoreEl = document.getElementById('halftime-score');
    const fulltimeScoreEl = document.getElementById('fulltime-score');
    if (pauseScoreEl) pauseScoreEl.textContent = `${home} - ${away}`;
    if (halftimeScoreEl) halftimeScoreEl.textContent = `${home} - ${away}`;
    if (fulltimeScoreEl) fulltimeScoreEl.textContent = `${home} - ${away}`;
  },

  updateTime(seconds, half) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    document.getElementById('match-timer').textContent = 
      `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
    document.getElementById('match-half').textContent = half === 1 ? '1ST' : '2ND';
  },

  showEvent(text, duration = 2000) {
    const el = document.getElementById('event-text');
    el.textContent = text;
    this.overlays.event.classList.remove('hidden');
    clearTimeout(this._eventTimeout);
    this._eventTimeout = setTimeout(() => {
      this.overlays.event.classList.add('hidden');
    }, duration);
  },

  showToast(message, duration = 2500) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('visible'), 10);
    setTimeout(() => {
      toast.classList.remove('visible');
      setTimeout(() => toast.remove(), 300);
    }, duration);
  },

  updateProfile(progression, stats, badges, levelInfo) {
    const levelEl = document.getElementById('profile-level-info');
    if (levelEl) {
      levelEl.innerHTML = `
        <div class="level-display">Level ${levelInfo.level}</div>
        <div class="xp-bar-container">
          <div class="xp-bar-fill" style="width: ${levelInfo.percent}%"></div>
        </div>
        <div class="xp-text">${levelInfo.xp} / ${levelInfo.xpToNext} XP</div>
      `;
    }
    const statsEl = document.getElementById('profile-stats');
    if (statsEl) {
      statsEl.innerHTML = `
        <div class="stat-row"><span>⚽ Goals</span><span>${stats.totalGoals}</span></div>
        <div class="stat-row"><span>🧤 Passes</span><span>${stats.totalPasses}</span></div>
        <div class="stat-row"><span>⚔️ Tackles</span><span>${stats.totalTackles}</span></div>
        <div class="stat-row"><span>🏆 Wins</span><span>${stats.totalWins}/${stats.totalMatches}</span></div>
        <div class="stat-row"><span>📈 Win Rate</span><span>${stats.winRate}%</span></div>
        <div class="stat-row"><span>🔥 Streak</span><span>${stats.currentStreak} (Best: ${stats.bestStreak})</span></div>
      `;
    }
    const badgesEl = document.getElementById('profile-badges');
    if (badgesEl) {
      if (badges.length === 0) {
        badgesEl.innerHTML = '<p style="color:var(--text-muted)">No badges yet. Keep playing!</p>';
      } else {
        badgesEl.innerHTML = '<h3 style="margin-bottom:10px">Achievements</h3>' +
          badges.map(b => `<span class="badge-item" title="${b.desc}">${b.icon}</span>`).join('');
      }
    }
  },
  
  setupListeners(callbacks) {
    // Menu
    document.getElementById('btn-play').onclick = () => this.showScreen('difficulty');
    document.getElementById('btn-settings-menu').onclick = () => this.showScreen('settings');
    document.getElementById('btn-records').onclick = () => {
        this.renderRecords();
        this.showScreen('records');
    };

    // Difficulty
    document.querySelectorAll('.diff-card').forEach(card => {
        card.onclick = () => {
            const diff = card.getAttribute('data-diff');
            callbacks.onSelectDifficulty(diff);
            this.showScreen('setup');
        };
    });
    document.getElementById('btn-diff-back').onclick = () => this.showScreen('menu');

    // Setup
    document.querySelectorAll('.color-swatch').forEach(sw => {
        sw.onclick = () => {
            document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
            sw.classList.add('active');
            callbacks.onSelectColor(sw.getAttribute('data-color'));
        };
    });
    document.getElementById('btn-start-match').onclick = () => {
        const name = document.getElementById('team-name-input').value || 'HOME';
        callbacks.onStartMatch(name);
    };
    document.getElementById('btn-setup-back').onclick = () => this.showScreen('difficulty');

    // Pause
    document.getElementById('btn-resume').onclick = callbacks.onResume;
    document.getElementById('btn-restart').onclick = callbacks.onRestart;
        document.getElementById('btn-pause-settings').onclick = () => { UI.showScreen('settings'); };
    document.getElementById('btn-pause-menu').onclick = callbacks.onQuit;
    
    // Halftime
    document.getElementById('btn-second-half').onclick = callbacks.onNextHalf;
    
    // Fulltime
    document.getElementById('btn-rematch').onclick = callbacks.onRestart;
    document.getElementById('btn-ft-menu').onclick = callbacks.onQuit;
    
    // Settings
    document.querySelectorAll('.duration-group .opt-btn').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.duration-group .opt-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            callbacks.onSetDuration(parseInt(btn.getAttribute('data-min')));
        };
    });
    document.getElementById('btn-set-back').onclick = () => this.showScreen('menu');
  },
  
  renderRecords() {
      // Load from localStorage implementation
      const recs = JSON.parse(localStorage.getItem('pf_records') || '[]');
      const container = document.getElementById('records-content');
      if (recs.length === 0) {
          container.innerHTML = '<p>No matches played yet.</p>';
          return;
      }
      container.innerHTML = recs.map(r => `
        <div class="stat-row">
            <span>${r.date}</span>
            <span>${r.result}</span>
            <span>${r.score}</span>
        </div>
      `).join('');
  }
};