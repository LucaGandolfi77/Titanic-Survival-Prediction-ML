/* ===== Routine engine — mini-game rules & state (no DOM access) =====
 *
 * Owns all mini-game state and rules. Emits events for the controller to
 * present (sound, HUD, alerts). Rendering is delegated to RinkRenderer.
 */

import { GameState } from '../state.js';
import { FORMATIONS } from '../formations.js';
import { lerp, clamp, randFloat } from '../utils.js';
import { getSyncBonus } from '../sponsors.js';
import {
  ACTIVE_SQUAD_SIZE,
  ROUTINE_DURATION,
  FORMATION_DURATION,
  FORMATION_COOLDOWN,
  FALL_PENALTY,
  SAVE_BONUS,
  SCORE_MILESTONE,
  WOBBLE_MAX_TIME,
  FALLEN_MAX_TIME,
  WOBBLE_BASE_CHANCE,
  ENTRANCE_PHASE,
  TEMPO_RISK,
  TEMPO_MULTIPLIERS
} from '../config.js';

export class RoutineEngine {
  /**
   * @param {{width:number, height:number, events?:Object}} opts
   *        events: onWobble(sk), onSave(sk), onFall(sk),
   *                onFormationStart(formation), onFormationComplete(formation),
   *                onMilestone()
   */
  constructor({ width, height, events = {} }) {
    this.width = width;
    this.height = height;
    this.events = events;

    this.running = false;
    this.duration = ROUTINE_DURATION;
    this.elapsed = 0;
    this.timer = ROUTINE_DURATION;

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
    this.cooldown = 0;

    this.skaters = [];
    this.scorePopups = [];
  }

  /** Build the rink skaters from the active squad and reset all counters. */
  init() {
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
    this.cooldown = 0;
    this.scorePopups = [];

    this.skaters = [];
    const squad = GameState.activeSquad;
    for (let i = 0; i < ACTIVE_SQUAD_SIZE; i++) {
      // Injured skaters keep their real ref; dummies fill missing slots
      const sk = squad[i] || { stats: { technique: 50, stamina: 50, rhythm: 50, sync: 50, charisma: 50 }, morale: 60 };
      const angle = (i / ACTIVE_SQUAD_SIZE) * Math.PI * 2;
      const cx = this.width / 2;
      const cy = this.height / 2;
      const rx = 180 + randFloat(-20, 20);
      const ry = 120 + randFloat(-15, 15);
      const phase = angle;

      this.skaters.push({
        idx: i,
        ref: sk, // live reference to the GameState skater — stat AND morale changes persist
        stats: sk.stats,
        renderX: cx + rx * Math.cos(phase),
        renderY: cy + ry * Math.sin(phase),
        cx, cy, rx, ry,
        phase,
        phaseSpeed: 0.4 + randFloat(-0.05, 0.05),
        state: sk.injuryWeeks > 0 ? 'injured' : 'skating', // skating | formation | wobbling | fallen | injured
        targetX: undefined,
        targetY: undefined,
        wobbleTimer: 0,
        wobbleMax: WOBBLE_MAX_TIME,
        fallenTimer: 0,
        fallenMax: FALLEN_MAX_TIME
      });
    }
  }

  /** Begin the routine with a 5-second entrance animation. */
  start() {
    this.running = true;
    for (let i = 0; i < this.skaters.length; i++) {
      this.skaters[i].renderX = -30;
      this.skaters[i].renderY = this.height / 2 + (i - 8) * 10;
    }
  }

  stop() {
    this.running = false;
  }

  /** Pause the simulation (timer stops advancing). */
  pause() {
    this.running = false;
  }

  /** Resume after pause(). */
  resume() {
    this.running = true;
  }

  /**
   * Rescale all positions to a new logical canvas size (e.g. after a window
   * resize). Keeps relative positions, paths and formation targets intact.
   */
  setDimensions(width, height) {
    if (this.skaters.length === 0 || (width === this.width && height === this.height)) return;
    const sx = width / this.width;
    const sy = height / this.height;
    for (const sk of this.skaters) {
      sk.renderX *= sx;
      sk.renderY *= sy;
      sk.cx *= sx;
      sk.cy *= sy;
      sk.rx *= sx;
      sk.ry *= sy;
      if (sk.targetX !== undefined) {
        sk.targetX *= sx;
        sk.targetY *= sy;
      }
    }
    this.width = width;
    this.height = height;
  }

  /** Advance the simulation by dt seconds. */
  update(dt) {
    if (!this.running) return;
    this.elapsed += dt;
    this.timer = Math.max(0, this.duration - this.elapsed);

    if (this.tempoMultiplier > 1.3) this.highTempoTime += dt;

    // Entrance phase
    if (this.elapsed < ENTRANCE_PHASE) {
      for (const sk of this.skaters) {
        const targetAngle = (sk.idx / ACTIVE_SQUAD_SIZE) * Math.PI * 2;
        const tx = sk.cx + sk.rx * Math.cos(targetAngle);
        const ty = sk.cy + sk.ry * Math.sin(targetAngle);
        sk.renderX = lerp(sk.renderX, tx, dt * 2);
        sk.renderY = lerp(sk.renderY, ty, dt * 2);
      }
      return;
    }

    if (this.cooldown > 0) this.cooldown -= dt;

    // Formation scoring (tempo multiplies points; sync sponsor adds a fraction)
    if (this.currentFormation) {
      this.formationTimer -= dt;
      this.score += this.currentFormation.difficulty * 10 * dt * this.tempoMultiplier * (1 + getSyncBonus());
      if (this.formationTimer <= 0) this._completeFormation();
    }

    for (const sk of this.skaters) {
      if (sk.state === 'skating') {
        sk.phase += sk.phaseSpeed * this.tempoMultiplier * dt;
        const tx = sk.cx + sk.rx * Math.cos(sk.phase);
        const ty = sk.cy + sk.ry * Math.sin(sk.phase);
        sk.renderX = lerp(sk.renderX, tx, dt * 4);
        sk.renderY = lerp(sk.renderY, ty, dt * 4);

        const tempoRisk = TEMPO_RISK[this.tempo];
        const wobbleChance = tempoRisk * (1 - sk.stats.stamina / 100) * WOBBLE_BASE_CHANCE;
        const moraleMod = sk.ref.morale < 40 ? 1.5 : 1.0;
        if (Math.random() < wobbleChance * moraleMod * 60 * dt) this._startWobble(sk);
      } else if (sk.state === 'formation') {
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
        const tempoRisk = TEMPO_RISK[this.tempo];
        const wobbleChance = tempoRisk * (1 - sk.stats.stamina / 100) * WOBBLE_BASE_CHANCE;
        if (Math.random() < wobbleChance * 60 * dt) {
          this._startWobble(sk);
          this._breakFormation();
        }
      } else if (sk.state === 'wobbling') {
        sk.wobbleTimer -= dt;
        if (sk.wobbleTimer <= 0) {
          // Not tapped in time → fall
          sk.state = 'fallen';
          sk.fallenTimer = FALLEN_MAX_TIME;
          this.wobblesFailed++;
          this.perfectRoutine = false;
          this.score -= FALL_PENALTY;
          this.addScorePopup(sk.renderX, sk.renderY, `-${FALL_PENALTY}`);
          this.events.onFall?.(sk);
          // Morale hit for all (persists to GameState via ref)
          for (const s of this.skaters) s.ref.morale = clamp(s.ref.morale - 5, 0, 100);
        }
      } else if (sk.state === 'fallen') {
        sk.fallenTimer -= dt;
        if (sk.fallenTimer <= 0) sk.state = 'skating';
      }
      // 'injured' skaters sit out the routine
    }

    // Milestone jingles
    const newMilestone = Math.floor(this.score / SCORE_MILESTONE);
    if (newMilestone > this.milestoneHit && this.score > 0) {
      this.milestoneHit = newMilestone;
      this.events.onMilestone?.();
    }

    // Score popups
    for (let i = this.scorePopups.length - 1; i >= 0; i--) {
      this.scorePopups[i].y -= 30 * dt;
      this.scorePopups[i].alpha -= dt * 0.8;
      if (this.scorePopups[i].alpha <= 0) this.scorePopups.splice(i, 1);
    }
  }

  changeTempo(tempo) {
    this.tempo = tempo;
    this.tempoMultiplier = TEMPO_MULTIPLIERS[tempo] || 1.0;
  }

  /** Command skaters into a formation. Returns false if unavailable. */
  executeFormation(id) {
    if (this.cooldown > 0) return false;
    if (this.currentFormation) return false;

    const formation = FORMATIONS.find(f => f.id === id);
    if (!formation || GameState.fame < formation.unlockFame) return false;

    this.currentFormation = formation;
    this.formationTimer = FORMATION_DURATION;

    for (const sk of this.skaters) {
      if (sk.state === 'fallen' || sk.state === 'wobbling' || sk.state === 'injured') continue;
      sk.state = 'formation';
      const pos = formation.positions[sk.idx];
      sk.targetX = pos.x * this.width;
      sk.targetY = pos.y * this.height;
    }

    this.events.onFormationStart?.(formation);
    return true;
  }

  /** Try to save a wobbling skater by tapping within `radius` px. Returns the skater or null. */
  trySaveAt(x, y, radius = 25) {
    for (const sk of this.skaters) {
      if (sk.state !== 'wobbling') continue;
      const dx = sk.renderX - x;
      const dy = sk.renderY - y;
      if (Math.sqrt(dx * dx + dy * dy) < radius) {
        sk.state = 'skating';
        sk.wobbleTimer = 0;
        this.wobblesSaved++;
        this.score += SAVE_BONUS;
        this.addScorePopup(sk.renderX, sk.renderY, `+${SAVE_BONUS}`);
        sk.ref.morale = clamp(sk.ref.morale - 2, 0, 100);
        this.events.onSave?.(sk);
        return sk;
      }
    }
    return null;
  }

  addScorePopup(x, y, text) {
    this.scorePopups.push({ x, y: y - 10, text, alpha: 1 });
  }

  // ===== internals =====

  _startWobble(sk) {
    sk.state = 'wobbling';
    sk.wobbleTimer = sk.wobbleMax;
    this.events.onWobble?.(sk);
  }

  _completeFormation() {
    this.formationsCompleted++;
    if (this.currentFormation.difficulty > this.bestDifficulty) {
      this.bestDifficulty = this.currentFormation.difficulty;
    }
    const finished = this.currentFormation;
    this.currentFormation = null;
    this.cooldown = FORMATION_COOLDOWN;
    for (const sk of this.skaters) {
      if (sk.state === 'formation') sk.state = 'skating';
      sk.targetX = undefined;
      sk.targetY = undefined;
    }
    this.events.onFormationComplete?.(finished);
  }

  _breakFormation() {
    this.currentFormation = null;
    this.cooldown = FORMATION_COOLDOWN;
    for (const s of this.skaters) {
      if (s.state === 'formation') s.state = 'skating';
      s.targetX = undefined;
      s.targetY = undefined;
    }
  }
}
