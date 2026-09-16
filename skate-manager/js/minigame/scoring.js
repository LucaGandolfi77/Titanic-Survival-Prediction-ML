/* ===== Mini-game scoring — pure functions ===== */

import { ROUTINE_DURATION, PERFECT_BONUS, MUSIC_BONUS_SCALE, FALL_PENALTY } from '../config.js';

/**
 * Compute the final routine score from rounded components so the results
 * breakdown adds up exactly. Wobble penalties are already applied to
 * baseScore during the routine (-FALL_PENALTY per fall) and are NOT
 * subtracted again here.
 *
 * @param {Object} p
 * @param {number} p.baseScore          Live routine score (includes fall penalties)
 * @param {number} p.highTempoTime      Seconds spent above 1.3× tempo
 * @param {number} [p.duration]         Routine length in seconds
 * @param {number} p.cohesion           Team cohesion 0–100
 * @param {number} [p.syncBonusPercent] QuantumIce sponsor bonus (fraction)
 * @param {number} [p.tempoBonusPct]    CoolBreeze sponsor bonus (fraction)
 * @param {number} p.wobblesFailed      Number of falls
 * @param {boolean} p.perfectRoutine    True if no skater fell
 * @returns {{score:number, baseScore:number, musicBonus:number, tempoBoost:number,
 *           syncBonus:number, wobblePenalty:number, perfectBonus:number}}
 */
export function computeFinalScore({
  baseScore,
  highTempoTime,
  duration = ROUTINE_DURATION,
  cohesion,
  syncBonusPercent = 0,
  tempoBonusPct = 0,
  wobblesFailed,
  perfectRoutine
}) {
  const base = Math.round(baseScore);
  const musicBonus = Math.round(Math.round(highTempoTime / duration * 100) * MUSIC_BONUS_SCALE);
  const tempoBoost = Math.round(musicBonus * tempoBonusPct);
  const syncBonus = Math.round(cohesion * (0.5 + syncBonusPercent));
  const wobblePenalty = wobblesFailed * FALL_PENALTY; // informational: applied live
  const perfectBonus = perfectRoutine ? PERFECT_BONUS : 0;

  const score = Math.max(0, base + musicBonus + tempoBoost + syncBonus + perfectBonus);

  return { score, baseScore: base, musicBonus, tempoBoost, syncBonus, wobblePenalty, perfectBonus };
}
