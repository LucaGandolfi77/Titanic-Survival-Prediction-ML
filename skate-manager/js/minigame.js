/* ===== Mini-game controller — DOM wiring, HUD, sound =====
 *
 * Thin presentation layer over RoutineEngine (rules/state), RinkRenderer
 * (canvas view) and MusicEngine (audio). All HUD updates are cached and
 * only touch the DOM on change.
 */

import { GameState } from './state.js';
import { RinkRenderer } from './rink-renderer.js';
import { FORMATIONS } from './formations.js';
import { MusicEngine } from './music.js';
import { RoutineEngine } from './minigame/engine.js';
import { computeFinalScore } from './minigame/scoring.js';
import { getTeamCohesion } from './skaters.js';
import { getSyncBonus, getTempoBonus } from './sponsors.js';
import { clamp } from './utils.js';
import { TEMPO_LABELS } from './config.js';

export class MiniGame {
  constructor() {
    this.canvas = null;
    this.renderer = null;
    this.engine = null;
    this.music = new MusicEngine();
    this.animFrame = null;
    this.lastTime = 0;
    this.onFinish = null;

    // DOM references (cached in init)
    this.ui = null;
    this.tempoButtons = [];
    this.judges = [];
    this.formationBtns = [];

    // HUD caches — DOM is updated only when values change
    this._lastTimerText = '';
    this._lastScoreText = '';
    this._lastTempoText = '';
    this._lastCooldownCeil = -2;

    // Bind-once guards for static DOM controls
    this._canvasBound = false;
    this._tempoBound = false;
    this._keyboardBound = false;
    this._pauseBound = false;
    this._resizeBound = false;
    this._judgeTimer = null;
  }

  init(competition) {
    // Cache DOM references
    this.canvas = document.getElementById('rink-canvas');
    this.ui = {
      timer: document.getElementById('mg-timer'),
      score: document.getElementById('mg-score'),
      tempoDisplay: document.getElementById('mg-tempo-display'),
      moraleFill: document.getElementById('morale-fill'),
      wobbleAlerts: document.getElementById('wobble-alert-list'),
      formationButtons: document.getElementById('formation-buttons'),
      compName: document.getElementById('mg-comp-name'),
      pauseBtn: document.getElementById('mg-pause-btn')
    };
    this.tempoButtons = [...document.querySelectorAll('.tempo-btn')];
    this.judges = [...document.querySelectorAll('.judge')];
    this.formationBtns = [];

    this.renderer = new RinkRenderer(this.canvas);
    this.renderer.setTeamColor(GameState.teamColor);

    // Reset HUD caches
    this._lastTimerText = '';
    this._lastScoreText = '';
    this._lastTempoText = '';
    this._lastCooldownCeil = -2;

    // Engine with presentation events — logical dimensions from the renderer
    this.engine = new RoutineEngine({
      width: this.renderer.width,
      height: this.renderer.height,
      events: {
        onWobble: (sk) => {
          this.music.playWobbleStart();
          this.addWobbleAlert(sk);
        },
        onSave: (sk) => {
          this.music.playWobbleSave();
          this.removeWobbleAlert(sk.idx);
        },
        onFall: (sk) => {
          this.music.playWobbleFall();
          this.removeWobbleAlert(sk.idx);
          this.setJudges('😬', 1200);
        },
        onFormationStart: (formation) => {
          this.setJudges('😮', 800);
          for (const btn of this.formationBtns) {
            btn.classList.toggle('active', btn.dataset.formationId === formation.id);
          }
        },
        onFormationComplete: () => {
          this.setJudges('👏', 1500);
          for (const btn of this.formationBtns) btn.classList.remove('active');
        },
        onMilestone: () => this.music.playMilestone()
      }
    });
    this.engine.init();

    // Bind canvas handlers once (stored so stop() can remove them)
    if (!this._canvasBound) {
      this._canvasBound = true;
      this._clickHandler = (e) => this.handleClick(e);
      this._touchHandler = (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = this.canvas.getBoundingClientRect();
        this.handleClickAt(touch.clientX - rect.left, touch.clientY - rect.top);
      };
      this.canvas.addEventListener('click', this._clickHandler);
      this.canvas.addEventListener('touchstart', this._touchHandler, { passive: false });
    }

    this.setupFormationButtons();
    this.setupTempoButtons();
    this.resetTempoButtons();
    this.setupKeyboard();
    this.setupPause();
    this.setupResize();

    this.ui.compName.textContent = competition ? competition.name : 'Exhibition';
    if (this.ui.pauseBtn) this.ui.pauseBtn.textContent = '⏸';
    this.setJudges('😐');
  }

  setupFormationButtons() {
    const container = this.ui.formationButtons;
    container.innerHTML = '';
    this.formationBtns = [];

    for (const f of FORMATIONS) {
      const btn = document.createElement('button');
      btn.className = 'formation-btn';
      const unlocked = GameState.fame >= f.unlockFame;
      if (!unlocked) btn.classList.add('locked');
      btn.disabled = !unlocked;
      btn.innerHTML = `
        <span>${f.emoji} ${f.name}</span>
        ${unlocked
          ? `<span class="diff-badge">×${f.difficulty.toFixed(1)}</span>`
          : `<span class="diff-badge">🔒 Fame ${f.unlockFame}</span>`
        }
      `;
      btn.dataset.formationId = f.id;
      btn.addEventListener('click', () => this.executeFormation(f.id));
      container.appendChild(btn);
      this.formationBtns.push(btn);
    }
  }

  setupTempoButtons() {
    // Tempo buttons are static DOM — bind only once per MiniGame instance
    // to avoid stacking duplicate listeners on every competition.
    if (this._tempoBound) return;
    this._tempoBound = true;
    for (const btn of this.tempoButtons) {
      btn.addEventListener('click', () => this.setTempo(btn.dataset.tempo));
    }
  }

  setTempo(tempo) {
    this.engine.changeTempo(tempo);
    this.tempoButtons.forEach(b => b.classList.toggle('active', b.dataset.tempo === tempo));
  }

  /** Keyboard controls: 1–4 tempo, Q–P formations, ESC pause. */
  setupKeyboard() {
    if (this._keyboardBound) return;
    this._keyboardBound = true;
    this._keyHandler = (e) => this.handleKey(e);
    document.addEventListener('keydown', this._keyHandler);
  }

  handleKey(e) {
    if (!this.engine) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      this.togglePause();
      return;
    }
    if (!this.engine.running) return; // remaining keys only during play

    const tempoKeys = { '1': 'slow', '2': 'medium', '3': 'fast', '4': 'max' };
    if (tempoKeys[e.key]) {
      this.setTempo(tempoKeys[e.key]);
      return;
    }
    const formationIdx = 'qwertyuiop'.indexOf(e.key.toLowerCase());
    if (formationIdx !== -1 && this.formationBtns[formationIdx]) {
      this.executeFormation(this.formationBtns[formationIdx].dataset.formationId);
    }
  }

  /** Pause/resume the routine (Esc or the header button). */
  setupPause() {
    if (!this.ui.pauseBtn || this._pauseBound) return;
    this._pauseBound = true;
    this.ui.pauseBtn.addEventListener('click', () => this.togglePause());
  }

  togglePause() {
    if (!this.engine || this.engine.timer <= 0) return; // routine over
    if (this.engine.running) {
      this.engine.pause();
      this.music.stopMusic();
      if (this.ui.pauseBtn) this.ui.pauseBtn.textContent = '▶';
    } else {
      this.engine.resume();
      this.lastTime = performance.now();
      this.music.startMusic(this.engine.tempo);
      if (this.ui.pauseBtn) this.ui.pauseBtn.textContent = '⏸';
      this.loop();
    }
  }

  /** Re-fit the canvas on window resize (HiDPI + logical size). */
  setupResize() {
    if (this._resizeBound) return;
    this._resizeBound = true;
    this._resizeHandler = () => this.handleResize();
    window.addEventListener('resize', this._resizeHandler);
  }

  handleResize() {
    if (!this.renderer || !this.engine) return;
    this.renderer.resize();
    this.engine.setDimensions(this.renderer.width, this.renderer.height);
    // While paused, re-render once so the frame isn't stale
    if (!this.engine.running) {
      this.renderer.render(this.engine.skaters, this.engine.scorePopups);
    }
  }

  resetTempoButtons() {
    for (const b of this.tempoButtons) {
      b.classList.toggle('active', b.dataset.tempo === 'slow');
    }
  }

  start() {
    this.engine.start();
    this.lastTime = performance.now();
    this.music.startMusic('slow');
    this.loop();
  }

  stop() {
    this.engine?.stop();
    this.music.stopMusic();
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    this.animFrame = null;
    if (this.canvas && this._clickHandler) this.canvas.removeEventListener('click', this._clickHandler);
    if (this.canvas && this._touchHandler) this.canvas.removeEventListener('touchstart', this._touchHandler);
    this._canvasBound = false;
  }

  loop() {
    if (!this.engine.running) return;
    const now = performance.now();
    const dt = Math.min((now - this.lastTime) / 1000, 0.05);
    this.lastTime = now;

    this.engine.update(dt);
    this.updateHUD();

    this.renderer.sparkle = this.engine.timer <= 10;
    this.renderer.render(this.engine.skaters, this.engine.scorePopups);

    // TIME UP
    if (this.engine.timer <= 0) {
      this.stop(); // stops the music and removes canvas handlers
      this.finishRoutine();
      return;
    }

    this.animFrame = requestAnimationFrame(() => this.loop());
  }

  /** Update the HUD from engine state (DOM writes only on change). */
  updateHUD() {
    const engine = this.engine;

    // Timer
    const mins = Math.floor(engine.timer / 60);
    const secs = Math.floor(engine.timer % 60);
    const timerText = `⏱ ${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    if (timerText !== this._lastTimerText) {
      this._lastTimerText = timerText;
      this.ui.timer.textContent = timerText;
    }

    // Score
    const scoreText = `🏆 ${Math.round(engine.score).toLocaleString()}`;
    if (scoreText !== this._lastScoreText) {
      this._lastScoreText = scoreText;
      this.ui.score.textContent = scoreText;
    }

    // Tempo display
    const tempoText = TEMPO_LABELS[engine.tempo] || '';
    if (tempoText !== this._lastTempoText) {
      this._lastTempoText = tempoText;
      this.ui.tempoDisplay.textContent = tempoText;
    }

    // Morale bar
    const avgMorale = engine.skaters.reduce((s, sk) => s + sk.ref.morale, 0) / engine.skaters.length;
    this.ui.moraleFill.style.width = avgMorale + '%';
    this.ui.moraleFill.style.backgroundColor =
      avgMorale > 65 ? '#34d399' :
      avgMorale > 35 ? '#fb923c' : '#f87171';

    // Formation cooldown buttons
    this.updateFormationCooldownUI();
  }

  /** Cooldown button state — DOM churn only when the visible value changes. */
  updateFormationCooldownUI() {
    const engine = this.engine;
    const active = engine.cooldown > 0 && !engine.currentFormation;
    const ceil = active ? Math.ceil(engine.cooldown) : -1;
    if (ceil === this._lastCooldownCeil) return;
    this._lastCooldownCeil = ceil;

    for (const btn of this.formationBtns) {
      let text = btn.querySelector('.cooldown-text');
      if (ceil === -1) {
        if (text) text.remove();
        if (!btn.classList.contains('locked')) btn.disabled = false;
      } else {
        if (!text) {
          text = document.createElement('span');
          text.className = 'cooldown-text';
          btn.appendChild(text);
        }
        text.textContent = `${ceil}s`;
        btn.disabled = true;
      }
    }
  }

  // ===== Actions =====

  executeFormation(id) {
    this.engine.executeFormation(id);
  }

  handleClick(e) {
    // Canvas logical size == CSS size, so click coords need no scaling
    const rect = this.canvas.getBoundingClientRect();
    this.handleClickAt(e.clientX - rect.left, e.clientY - rect.top);
  }

  handleClickAt(x, y) {
    // Larger tap targets on small/mobile screens
    const radius = Math.max(25, Math.round(Math.min(this.canvas.clientWidth, this.canvas.clientHeight) * 0.08));
    this.engine.trySaveAt(x, y, radius);
  }

  addWobbleAlert(sk) {
    const squadSk = GameState.activeSquad[sk.idx];
    const name = squadSk ? squadSk.name.split(' ')[0] : `#${sk.idx + 1}`;
    const div = document.createElement('div');
    div.className = 'wobble-alert';
    div.id = `wobble-${sk.idx}`;
    div.textContent = `⚠ ${name} (${sk.idx + 1})`;
    this.ui.wobbleAlerts.appendChild(div);
  }

  removeWobbleAlert(idx) {
    const el = document.getElementById(`wobble-${idx}`);
    if (el) el.remove();
  }

  /** Set judge emojis; optionally auto-reset to neutral after resetAfterMs. */
  setJudges(emoji, resetAfterMs = 0) {
    for (const j of this.judges) {
      j.textContent = emoji;
      j.classList.toggle('happy', emoji === '👏');
    }
    clearTimeout(this._judgeTimer);
    if (resetAfterMs > 0) {
      this._judgeTimer = setTimeout(() => this.setJudges('😐'), resetAfterMs);
    }
  }

  // ===== Finish =====

  finishRoutine() {
    this.renderer.sparkle = false;
    const engine = this.engine;

    const cohesion = getTeamCohesion(GameState.activeSquad);
    const scoring = computeFinalScore({
      baseScore: engine.score,
      highTempoTime: engine.highTempoTime,
      cohesion,
      syncBonusPercent: getSyncBonus(),
      tempoBonusPct: getTempoBonus(),
      wobblesFailed: engine.wobblesFailed,
      perfectRoutine: engine.perfectRoutine
    });

    const result = {
      ...scoring,
      formationsCompleted: engine.formationsCompleted,
      bestDifficulty: engine.bestDifficulty,
      wobblesSaved: engine.wobblesSaved,
      wobblesFailed: engine.wobblesFailed,
      perfectRoutine: engine.perfectRoutine
    };

    // Post-routine effects on skaters (MAX tempo builds stamina)
    if (engine.tempo === 'max' || engine.highTempoTime > 15) {
      for (const sk of GameState.activeSquad) {
        sk.stats.stamina = clamp(sk.stats.stamina + 1, 1, 100);
      }
    }

    if (this.onFinish) this.onFinish(result);
  }
}
