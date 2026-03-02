/* ===== Real-time rink mini-game engine ===== */
import { GameState } from './state.js';
import { RinkRenderer } from './rink-renderer.js';
import { FORMATIONS, getAvailableFormations } from './formations.js';
import { MusicEngine } from './music.js';
import { lerp, clamp, randInt, randFloat } from './utils.js';
import { getTeamCohesion, getSquadAvgMorale } from './skaters.js';
import { getSyncBonus } from './sponsors.js';

export class MiniGame {
  constructor() {
    this.canvas = null;
    this.renderer = null;
    this.music = new MusicEngine();
    this.running = false;
    this.animFrame = null;
    this.lastTime = 0;

    // Timing
    this.duration = 60; // seconds
    this.elapsed = 0;
    this.timer = 60;

    // Score
    this.score = 0;
    this.formationsCompleted = 0;
    this.bestDifficulty = 0;
    this.wobblesSaved = 0;
    this.wobblesFailed = 0;
    this.perfectRoutine = true;
    this.milestoneHit = 0;
    this.highTempoTime = 0;

    // Tempo
    this.tempo = 'slow';
    this.tempoMultiplier = 1.0;
    this.tempoRiskMap = { slow: 0, medium: 0.3, fast: 0.6, max: 1.0 };

    // Formations
    this.currentFormation = null;
    this.formationTimer = 0;
    this.formationDuration = 8;
    this.formationCooldown = 0;
    this.formationCooldownMax = 5;

    // Skaters (16)
    this.skaters = [];

    // Score popups
    this.scorePopups = [];

    // Callback
    this.onFinish = null;
  }

  init(competition) {
    this.canvas = document.getElementById('rink-canvas');
    this.renderer = new RinkRenderer(this.canvas);
    this.renderer.setTeamColor(GameState.teamColor);

    this.elapsed = 0;
    this.timer = this.duration;
    this.score = 0;
    this.formationsCompleted = 0;
    this.bestDifficulty = 0;
    this.wobblesSaved = 0;
    this.wobblesFailed = 0;
    this.perfectRoutine = true;
    this.milestoneHit = 0;
    this.highTempoTime = 0;

    this.tempo = 'slow';
    this.tempoMultiplier = 1.0;
    this.currentFormation = null;
    this.formationTimer = 0;
    this.formationCooldown = 0;
    this.scorePopups = [];

    // Initialize 16 skater objects for the rink
    this.skaters = [];
    const squad = GameState.activeSquad;
    for (let i = 0; i < 16; i++) {
      const sk = squad[i] || { stats: { technique: 50, stamina: 50, rhythm: 50, sync: 50, charisma: 50 }, morale: 60 };
      const angle = (i / 16) * Math.PI * 2;
      // Lazy oval path params
      const cx = this.canvas.width / 2;
      const cy = this.canvas.height / 2;
      const rx = 180 + randFloat(-20, 20);
      const ry = 120 + randFloat(-15, 15);
      const phase = angle;

      this.skaters.push({
        idx: i,
        stats: sk.stats,
        morale: sk.morale || 60,
        // Position
        renderX: cx + rx * Math.cos(phase),
        renderY: cy + ry * Math.sin(phase),
        // Oval movement
        cx, cy, rx, ry,
        phase,
        phaseSpeed: 0.4 + randFloat(-0.05, 0.05),
        // State
        state: 'skating', // skating | formation | wobbling | fallen
        // Formation targets
        targetX: undefined,
        targetY: undefined,
        // Wobble
        wobbleTimer: 0,
        wobbleMax: 2.0,
        // Fallen
        fallenTimer: 0,
        fallenMax: 5.0
      });
    }

    // Set up canvas click handler
    this._clickHandler = (e) => this.handleClick(e);
    this.canvas.addEventListener('click', this._clickHandler);
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const rect = this.canvas.getBoundingClientRect();
      this.handleClickAt(touch.clientX - rect.left, touch.clientY - rect.top);
    }, { passive: false });

    // Setup formation buttons
    this.setupFormationButtons();
    this.setupTempoButtons();

    // Update comp name
    document.getElementById('mg-comp-name').textContent = competition ? competition.name : 'Exhibition';

    // Init judge reactions
    this.setJudges('😐');
  }

  setupFormationButtons() {
    const container = document.getElementById('formation-buttons');
    container.innerHTML = '';
    const available = getAvailableFormations(GameState.fame);
    const allFormations = FORMATIONS;

    for (const f of allFormations) {
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
    }
  }

  setupTempoButtons() {
    const buttons = document.querySelectorAll('.tempo-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tempo = btn.dataset.tempo;
        this.changeTempo(tempo);
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });
  }

  start() {
    this.running = true;
    this.lastTime = performance.now();
    this.music.startMusic('slow');

    // 5-second entrance animation
    const cx = this.canvas.width / 2;
    for (let i = 0; i < this.skaters.length; i++) {
      this.skaters[i].renderX = -30;
      this.skaters[i].renderY = this.canvas.height / 2 + (i - 8) * 10;
    }

    this.loop();
  }

  stop() {
    this.running = false;
    this.music.stopMusic();
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    this.canvas.removeEventListener('click', this._clickHandler);
  }

  loop() {
    if (!this.running) return;
    const now = performance.now();
    const dt = Math.min((now - this.lastTime) / 1000, 0.05);
    this.lastTime = now;

    this.update(dt);
    this.render();

    this.animFrame = requestAnimationFrame(() => this.loop());
  }

  update(dt) {
    this.elapsed += dt;
    this.timer = Math.max(0, this.duration - this.elapsed);

    // Update timer UI
    const mins = Math.floor(this.timer / 60);
    const secs = Math.floor(this.timer % 60);
    document.getElementById('mg-timer').textContent = `⏱ ${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    document.getElementById('mg-score').textContent = `🏆 ${Math.round(this.score).toLocaleString()}`;

    // Track high tempo time
    if (this.tempoMultiplier > 1.3) {
      this.highTempoTime += dt;
    }

    // Entrance phase (first 3 seconds)
    if (this.elapsed < 3) {
      for (let i = 0; i < this.skaters.length; i++) {
        const sk = this.skaters[i];
        const targetAngle = (i / 16) * Math.PI * 2;
        const tx = sk.cx + sk.rx * Math.cos(targetAngle);
        const ty = sk.cy + sk.ry * Math.sin(targetAngle);
        sk.renderX = lerp(sk.renderX, tx, dt * 2);
        sk.renderY = lerp(sk.renderY, ty, dt * 2);
      }
      return;
    }

    // FINALE: last 10 seconds
    if (this.timer <= 10 && this.timer > 0) {
      this.renderer.sparkle = true;
    }

    // TIME UP
    if (this.timer <= 0) {
      this.running = false;
      this.music.stopMusic();
      this.finishRoutine();
      return;
    }

    // Update formation cooldown
    if (this.formationCooldown > 0) {
      this.formationCooldown -= dt;
      this.updateFormationCooldownUI();
    }

    // Update formation state
    if (this.currentFormation) {
      this.formationTimer -= dt;
      // Accumulate score while in formation
      const syncBonus = getSyncBonus();
      this.score += this.currentFormation.difficulty * 10 * dt * (1 + syncBonus);

      if (this.formationTimer <= 0) {
        // Formation ends
        this.formationsCompleted++;
        if (this.currentFormation.difficulty > this.bestDifficulty) {
          this.bestDifficulty = this.currentFormation.difficulty;
        }
        this.currentFormation = null;
        this.formationCooldown = this.formationCooldownMax;
        // Reset skaters to skating
        for (const sk of this.skaters) {
          if (sk.state === 'formation') sk.state = 'skating';
          sk.targetX = undefined;
          sk.targetY = undefined;
        }
        this.setJudges('👏');
        setTimeout(() => this.setJudges('😐'), 1500);
      }
    }

    // Update skater positions
    for (const sk of this.skaters) {
      if (sk.state === 'skating') {
        // Lazy oval
        sk.phase += sk.phaseSpeed * this.tempoMultiplier * dt;
        const tx = sk.cx + sk.rx * Math.cos(sk.phase);
        const ty = sk.cy + sk.ry * Math.sin(sk.phase);
        sk.renderX = lerp(sk.renderX, tx, dt * 4);
        sk.renderY = lerp(sk.renderY, ty, dt * 4);

        // Wobble chance
        const tempoRisk = this.tempoRiskMap[this.tempo];
        const wobbleChance = tempoRisk * (1 - sk.stats.stamina / 100) * 0.002;
        const moraleMod = sk.morale < 40 ? 1.5 : 1.0;
        if (Math.random() < wobbleChance * moraleMod * 60 * dt) {
          sk.state = 'wobbling';
          sk.wobbleTimer = sk.wobbleMax;
          this.music.playWobbleStart();
          this.addWobbleAlert(sk.idx);
        }
      } else if (sk.state === 'formation') {
        // Lerp to formation target
        if (sk.targetX !== undefined) {
          sk.renderX = lerp(sk.renderX, sk.targetX, dt * 2.5);
          sk.renderY = lerp(sk.renderY, sk.targetY, dt * 2.5);

          // Drift based on sync stat
          if (sk.stats.sync < 50) {
            const drift = (50 - sk.stats.sync) * 0.01;
            sk.renderX += (Math.random() - 0.5) * drift;
            sk.renderY += (Math.random() - 0.5) * drift;
          }
        }

        // Wobble chance during formation (lower)
        const tempoRisk = this.tempoRiskMap[this.tempo];
        const wobbleChance = tempoRisk * (1 - sk.stats.stamina / 100) * 0.001;
        if (Math.random() < wobbleChance * 60 * dt) {
          sk.state = 'wobbling';
          sk.wobbleTimer = sk.wobbleMax;
          this.music.playWobbleStart();
          this.addWobbleAlert(sk.idx);
          // Formation broken
          this.currentFormation = null;
          this.formationCooldown = this.formationCooldownMax;
          for (const s of this.skaters) {
            if (s.state === 'formation') s.state = 'skating';
            s.targetX = undefined;
            s.targetY = undefined;
          }
        }
      } else if (sk.state === 'wobbling') {
        sk.wobbleTimer -= dt;
        if (sk.wobbleTimer <= 0) {
          // Not tapped in time → fall
          sk.state = 'fallen';
          sk.fallenTimer = sk.fallenMax;
          this.wobblesFailed++;
          this.perfectRoutine = false;
          this.score -= 50;
          this.music.playWobbleFall();
          this.setJudges('😬');
          setTimeout(() => this.setJudges('😐'), 1200);
          this.removeWobbleAlert(sk.idx);
          // Morale hit for all
          for (const s of this.skaters) s.morale = clamp(s.morale - 5, 0, 100);
          this.addScorePopup(sk.renderX, sk.renderY, '-50');
        }
      } else if (sk.state === 'fallen') {
        sk.fallenTimer -= dt;
        if (sk.fallenTimer <= 0) {
          sk.state = 'skating';
        }
      }
    }

    // Milestone check
    const newMilestone = Math.floor(this.score / 5000);
    if (newMilestone > this.milestoneHit && this.score > 0) {
      this.milestoneHit = newMilestone;
      this.music.playMilestone();
    }

    // Update morale display
    const avgMorale = this.skaters.reduce((s, sk) => s + sk.morale, 0) / this.skaters.length;
    const moraleFill = document.getElementById('morale-fill');
    moraleFill.style.width = avgMorale + '%';
    moraleFill.style.backgroundColor =
      avgMorale > 65 ? '#34d399' :
      avgMorale > 35 ? '#fb923c' : '#f87171';

    // Update tempo display
    const tempoLabels = { slow: '🎵 SLOW ×1.0', medium: '🎶 MED ×1.5', fast: '🎸 FAST ×2.0', max: '🔥 MAX ×2.5' };
    document.getElementById('mg-tempo-display').textContent = tempoLabels[this.tempo] || '';

    // Update score popups
    for (let i = this.scorePopups.length - 1; i >= 0; i--) {
      this.scorePopups[i].y -= 30 * dt;
      this.scorePopups[i].alpha -= dt * 0.8;
      if (this.scorePopups[i].alpha <= 0) this.scorePopups.splice(i, 1);
    }
  }

  render() {
    this.renderer.render(this.skaters, this.scorePopups);
  }

  // ===== Actions =====

  executeFormation(id) {
    if (this.formationCooldown > 0) return;
    if (this.currentFormation) return;

    const formation = FORMATIONS.find(f => f.id === id);
    if (!formation || GameState.fame < formation.unlockFame) return;

    this.currentFormation = formation;
    this.formationTimer = this.formationDuration;
    this.music.playFormationChime();

    // Set targets for each skater
    for (let i = 0; i < this.skaters.length; i++) {
      const sk = this.skaters[i];
      if (sk.state === 'fallen' || sk.state === 'wobbling') continue;
      sk.state = 'formation';
      const pos = formation.positions[i];
      sk.targetX = pos.x * this.canvas.width;
      sk.targetY = pos.y * this.canvas.height;
    }

    // Highlight active formation button
    document.querySelectorAll('.formation-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.formationId === id);
    });

    this.setJudges('😮');
    setTimeout(() => this.setJudges('😐'), 800);
  }

  changeTempo(tempo) {
    const multipliers = { slow: 1.0, medium: 1.5, fast: 2.0, max: 2.5 };
    this.tempo = tempo;
    this.tempoMultiplier = multipliers[tempo] || 1.0;
    this.music.changeTempo(tempo);
  }

  handleClick(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    this.handleClickAt(x, y);
  }

  handleClickAt(x, y) {
    // Check if clicked on a wobbling skater
    for (const sk of this.skaters) {
      if (sk.state !== 'wobbling') continue;
      const dx = sk.renderX - x;
      const dy = sk.renderY - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 25) {
        // Saved!
        sk.state = 'skating';
        sk.wobbleTimer = 0;
        this.wobblesSaved++;
        this.score += 10;
        this.music.playWobbleSave();
        this.removeWobbleAlert(sk.idx);
        this.addScorePopup(sk.renderX, sk.renderY, '+10');
        sk.morale = clamp(sk.morale - 2, 0, 100);
        return;
      }
    }
  }

  addWobbleAlert(idx) {
    const list = document.getElementById('wobble-alert-list');
    const name = GameState.activeSquad[idx]
      ? GameState.activeSquad[idx].name.split(' ')[0]
      : `#${idx + 1}`;
    const div = document.createElement('div');
    div.className = 'wobble-alert';
    div.id = `wobble-${idx}`;
    div.textContent = `⚠ ${name} (#${idx + 1})`;
    list.appendChild(div);
  }

  removeWobbleAlert(idx) {
    const el = document.getElementById(`wobble-${idx}`);
    if (el) el.remove();
  }

  addScorePopup(x, y, text) {
    this.scorePopups.push({ x, y: y - 10, text, alpha: 1 });
  }

  setJudges(emoji) {
    const judges = document.querySelectorAll('.judge');
    judges.forEach(j => {
      j.textContent = emoji;
      j.classList.remove('happy');
      if (emoji === '👏') {
        j.classList.add('happy');
      }
    });
  }

  updateFormationCooldownUI() {
    document.querySelectorAll('.formation-btn').forEach(btn => {
      const cooldownText = btn.querySelector('.cooldown-text');
      if (this.formationCooldown > 0 && !this.currentFormation) {
        if (!cooldownText) {
          const span = document.createElement('span');
          span.className = 'cooldown-text';
          span.textContent = `${Math.ceil(this.formationCooldown)}s`;
          btn.appendChild(span);
        } else {
          cooldownText.textContent = `${Math.ceil(this.formationCooldown)}s`;
        }
        btn.disabled = true;
      } else {
        if (cooldownText) cooldownText.remove();
        if (!btn.classList.contains('locked')) btn.disabled = false;
      }
    });
  }

  // ===== Finish =====
  finishRoutine() {
    this.renderer.sparkle = false;
    const cohesion = getTeamCohesion(GameState.activeSquad);
    const syncBonusPercent = getSyncBonus();

    // Calculate final score
    const baseScore = this.score;
    const musicBonus = Math.round(this.highTempoTime / this.duration * 100) * 1.3;
    const syncBonus = cohesion * 0.5;
    const wobblePenalty = this.wobblesFailed * 50;
    const perfectBonus = this.perfectRoutine ? 300 : 0;

    const finalScore = Math.round(
      (baseScore + musicBonus + syncBonus) - wobblePenalty + perfectBonus
    );

    const result = {
      score: Math.max(0, finalScore),
      baseScore: Math.round(baseScore),
      musicBonus: Math.round(musicBonus),
      syncBonus: Math.round(syncBonus),
      wobblePenalty,
      perfectBonus,
      formationsCompleted: this.formationsCompleted,
      bestDifficulty: this.bestDifficulty,
      wobblesSaved: this.wobblesSaved,
      wobblesFailed: this.wobblesFailed,
      perfectRoutine: this.perfectRoutine
    };

    // Post-routine effects on skaters
    for (let i = 0; i < GameState.activeSquad.length; i++) {
      const sk = GameState.activeSquad[i];
      // MAX tempo training effect
      if (this.tempo === 'max' || this.highTempoTime > 15) {
        sk.stats.stamina = clamp(sk.stats.stamina + 1, 1, 100);
      }
    }

    if (this.onFinish) this.onFinish(result);
  }
}
