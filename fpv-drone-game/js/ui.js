/* ── js/ui.js ── Screen management & high scores ── */

/* ══════════════════════════════════════════════════════════
   UIManager — menu / pause / wave-complete / game-over screens
   ══════════════════════════════════════════════════════════ */
export class UIManager {
  constructor() {
    this.screens = {
      menu:         document.getElementById('menu-screen'),
      controlsInfo: document.getElementById('controls-info'),
      pause:        document.getElementById('pause-screen'),
      waveComplete: document.getElementById('wave-complete-screen'),
      gameOver:     document.getElementById('game-over-screen'),
    };

    this.elements = {
      startBtn:        document.getElementById('btn-start'),
      controlsBtn:     document.getElementById('btn-controls'),
      backBtn:         document.getElementById('btn-back-menu'),
      resumeBtn:       document.getElementById('btn-resume'),
      restartPauseBtn: document.getElementById('btn-restart-pause'),
      nextWaveBtn:     document.getElementById('btn-next-wave'),
      restartOverBtn:  document.getElementById('btn-restart-over'),
      menuOverBtn:     document.getElementById('btn-menu-over'),
      finalScore:      document.getElementById('final-score'),
      finalWave:       document.getElementById('final-wave'),
      waveNumDisplay:  document.getElementById('wave-num-display'),
      waveBonusDisplay:document.getElementById('wave-bonus-display'),
      highScoreInput:  document.getElementById('high-score-input'),
      highScoreForm:   document.getElementById('high-score-form'),
      highScoreList:   document.getElementById('high-score-list'),
    };

    // Callbacks (set by main.js)
    this.onStart     = null;
    this.onResume    = null;
    this.onRestart   = null;
    this.onNextWave  = null;
    this.onMainMenu  = null;

    this._bindButtons();
  }

  _bindButtons() {
    const { elements: el } = this;

    el.startBtn?.addEventListener('click', () => this.onStart?.());
    el.controlsBtn?.addEventListener('click', () => {
      this._hideAll();
      this._show('controlsInfo');
    });
    el.backBtn?.addEventListener('click', () => {
      this._hideAll();
      this._show('menu');
    });
    el.resumeBtn?.addEventListener('click', () => this.onResume?.());
    el.restartPauseBtn?.addEventListener('click', () => this.onRestart?.());
    el.nextWaveBtn?.addEventListener('click', () => this.onNextWave?.());
    el.restartOverBtn?.addEventListener('click', () => this.onRestart?.());
    el.menuOverBtn?.addEventListener('click', () => this.onMainMenu?.());

    // High score form
    el.highScoreForm?.addEventListener('submit', (e) => {
      e.preventDefault();
      const name = el.highScoreInput?.value?.trim() || 'PILOT';
      this._saveHighScore(name);
      el.highScoreForm.classList.add('hidden');
      this._renderHighScores();
    });
  }

  /* ── Screen helpers ── */
  _show(key) {
    const s = this.screens[key];
    if (s) s.classList.remove('hidden');
  }

  _hide(key) {
    const s = this.screens[key];
    if (s) s.classList.add('hidden');
  }

  _hideAll() {
    Object.values(this.screens).forEach(s => s?.classList.add('hidden'));
  }

  /* ── Public API ── */
  showMenu() {
    this._hideAll();
    this._show('menu');
  }

  showGame() {
    this._hideAll();
  }

  showPause() {
    this._show('pause');
  }

  hidePause() {
    this._hide('pause');
  }

  showWaveComplete(waveNum, bonus) {
    this._hideAll();
    if (this.elements.waveNumDisplay)   this.elements.waveNumDisplay.textContent = waveNum;
    if (this.elements.waveBonusDisplay) this.elements.waveBonusDisplay.textContent = `+${bonus}`;
    this._show('waveComplete');
  }

  showGameOver(score, wave) {
    this._hideAll();
    if (this.elements.finalScore) this.elements.finalScore.textContent = score;
    if (this.elements.finalWave)  this.elements.finalWave.textContent  = wave;

    // Check if it's a high score
    const scores = this._getHighScores();
    const isHigh = scores.length < 5 || score > (scores[scores.length - 1]?.score ?? 0);
    this._pendingScore = score;
    this._pendingWave  = wave;

    if (isHigh && this.elements.highScoreForm) {
      this.elements.highScoreForm.classList.remove('hidden');
      this.elements.highScoreInput.value = '';
      this.elements.highScoreInput.focus();
    } else {
      this.elements.highScoreForm?.classList.add('hidden');
    }

    this._renderHighScores();
    this._show('gameOver');
  }

  /* ── High Scores (localStorage) ── */
  _getHighScores() {
    try {
      return JSON.parse(localStorage.getItem('droneStrike_highScores') || '[]');
    } catch { return []; }
  }

  _saveHighScore(name) {
    const scores = this._getHighScores();
    scores.push({ name, score: this._pendingScore, wave: this._pendingWave });
    scores.sort((a, b) => b.score - a.score);
    if (scores.length > 5) scores.length = 5;
    localStorage.setItem('droneStrike_highScores', JSON.stringify(scores));
  }

  _renderHighScores() {
    const list = this.elements.highScoreList;
    if (!list) return;
    const scores = this._getHighScores();
    if (scores.length === 0) {
      list.innerHTML = '<li>No scores yet</li>';
      return;
    }
    list.innerHTML = scores.map((s, i) =>
      `<li>${i + 1}. ${s.name} — ${s.score} (W${s.wave})</li>`
    ).join('');
  }
}
