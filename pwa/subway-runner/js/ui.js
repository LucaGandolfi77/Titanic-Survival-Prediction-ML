const HIGH_SCORE_KEY = 'subwayRunner_highScores';
const MAX_SCORES = 10;

export class UIManager {
  constructor() {
    this.screens = {
      menu: document.getElementById('menu-screen'),
      pause: document.getElementById('pause-screen'),
      gameover: document.getElementById('gameover-screen')
    };
    this.toast = document.getElementById('toast');
    this._toastTimeout = null;
  }

  showScreen(name) {
    Object.values(this.screens).forEach(s => s.classList.add('hidden'));
    if (this.screens[name]) {
      this.screens[name].classList.remove('hidden');
    }
  }

  hideAllScreens() {
    Object.values(this.screens).forEach(s => s.classList.add('hidden'));
  }

  showGameOver(score, coins, distance, isNewHighScore) {
    document.getElementById('go-score').textContent = Math.floor(score).toLocaleString();
    document.getElementById('go-coins').textContent = coins.toLocaleString();
    document.getElementById('go-distance').textContent = Math.floor(distance) + 'm';

    const hsEl = document.getElementById('new-highscore');
    if (isNewHighScore) {
      hsEl.classList.remove('hidden');
    } else {
      hsEl.classList.add('hidden');
    }

    this.renderLeaderboard();
    this.showScreen('gameover');
  }

  saveHighScore(score, coins, distance) {
    const scores = this.getHighScores();
    scores.push({
      score: Math.floor(score),
      coins,
      distance: Math.floor(distance),
      date: Date.now()
    });
    scores.sort((a, b) => b.score - a.score);
    if (scores.length > MAX_SCORES) scores.length = MAX_SCORES;
    try {
      localStorage.setItem(HIGH_SCORE_KEY, JSON.stringify(scores));
    } catch (e) { /* quota exceeded */ }
  }

  getHighScores() {
    try {
      return JSON.parse(localStorage.getItem(HIGH_SCORE_KEY)) || [];
    } catch {
      return [];
    }
  }

  isHighScore(score) {
    const scores = this.getHighScores();
    if (scores.length < MAX_SCORES) return true;
    return score > scores[scores.length - 1].score;
  }

  renderLeaderboard() {
    const el = document.getElementById('leaderboard');
    const scores = this.getHighScores();
    if (scores.length === 0) {
      el.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:20px;font-family:var(--font-hud);font-size:13px;">No scores yet</div>';
      return;
    }
    el.innerHTML = scores.map((s, i) =>
      `<div class="leaderboard-entry">
        <span class="rank">#${i + 1}</span>
        <span class="lb-score">${s.score.toLocaleString()}</span>
      </div>`
    ).join('');
  }

  showToast(message, duration = 2500) {
    clearTimeout(this._toastTimeout);
    this.toast.textContent = message;
    this.toast.classList.add('visible');
    this._toastTimeout = setTimeout(() => {
      this.toast.classList.remove('visible');
    }, duration);
  }

  async shareScore(score, distance) {
    const text = `I scored ${Math.floor(score).toLocaleString()} points and ran ${Math.floor(distance)}m in Subway Runner! Can you beat me?`;
    if (navigator.share) {
      try {
        await navigator.share({ title: 'Subway Runner', text });
      } catch (e) { /* user cancelled */ }
    } else if (navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      this.showToast('Score copied to clipboard!');
    }
  }
}
