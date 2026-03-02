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
    // Also update pause/half/full screens
    document.getElementById('pause-score').textContent = `${home} - ${away}`;
    document.getElementById('halftime-score').textContent = `${home} - ${away}`;
    document.getElementById('fulltime-score').textContent = `${home} - ${away}`;
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
    setTimeout(() => {
      this.overlays.event.classList.add('hidden');
    }, duration);
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
        document.getElementById('btn-pause-step-settings') // typo in previous logic?
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