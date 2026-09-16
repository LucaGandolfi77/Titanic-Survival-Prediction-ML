/* Unit tests for the mini-game scoring module */
import { computeFinalScore } from '../js/minigame/scoring.js';

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`  ✓ ${name}`);
  else { console.log(`  ✗ FAIL: ${name} ${detail}`); failures++; }
}

console.log('[scoring] computeFinalScore');

// Baseline: 30s of high tempo, cohesion 60, no sponsors, no falls
let r = computeFinalScore({
  baseScore: 1000, highTempoTime: 30, cohesion: 60,
  wobblesFailed: 0, perfectRoutine: true
});
// musicBonus = round(round(30/60*100)*1.3) = round(50*1.3) = 65
// syncBonus = round(60*0.5) = 30; perfect = 300
check('baseline total adds up', r.score === 1000 + 65 + 0 + 30 + 300, `(${JSON.stringify(r)})`);
check('breakdown sums to total',
  r.score === r.baseScore + r.musicBonus + r.tempoBoost + r.syncBonus + r.perfectBonus,
  `(${JSON.stringify(r)})`);
check('wobblePenalty informational only (0 falls)', r.wobblePenalty === 0);

// CoolBreeze boost
r = computeFinalScore({
  baseScore: 1000, highTempoTime: 30, cohesion: 60,
  syncBonusPercent: 0, tempoBonusPct: 0.25,
  wobblesFailed: 0, perfectRoutine: false
});
// musicBonus = 65; tempoBoost = round(65*0.25) = 16; syncBonus = 30; perfect = 0
check('tempoBoost applied (16)', r.tempoBoost === 16, `(${r.tempoBoost})`);
check('coolbreeze total adds up', r.score === 1000 + 65 + 16 + 30, `(${JSON.stringify(r)})`);

// QuantumIce sync bonus
r = computeFinalScore({
  baseScore: 1000, highTempoTime: 30, cohesion: 60,
  syncBonusPercent: 0.10,
  wobblesFailed: 0, perfectRoutine: false
});
// syncBonus = round(60*0.6) = 36
check('quantumice syncBonus (36)', r.syncBonus === 36, `(${r.syncBonus})`);

// Wobble penalties are informational (already applied to baseScore)
r = computeFinalScore({
  baseScore: 1000 - 150, highTempoTime: 30, cohesion: 60,
  wobblesFailed: 3, perfectRoutine: false
});
check('wobblePenalty = falls*50 (informational)', r.wobblePenalty === 150, `(${r.wobblePenalty})`);
check('penalty NOT subtracted again', r.score === 850 + 65 + 30, `(${JSON.stringify(r)})`);

// Clamped at zero
r = computeFinalScore({
  baseScore: -1000, highTempoTime: 0, cohesion: 0,
  wobblesFailed: 10, perfectRoutine: false
});
check('score clamped at 0', r.score === 0, `(${r.score})`);

// Rounding determinism: breakdown always sums exactly
for (let i = 0; i < 200; i++) {
  const t = computeFinalScore({
    baseScore: Math.random() * 3000 - 200,
    highTempoTime: Math.random() * 60,
    cohesion: Math.random() * 100,
    syncBonusPercent: Math.random() < 0.5 ? 0 : 0.10,
    tempoBonusPct: Math.random() < 0.5 ? 0 : 0.25,
    wobblesFailed: Math.floor(Math.random() * 6),
    perfectRoutine: Math.random() < 0.5
  });
  if (t.score > 0 && t.score !== t.baseScore + t.musicBonus + t.tempoBoost + t.syncBonus + t.perfectBonus) {
    check('breakdown sums (fuzz)', false, `(${JSON.stringify(t)})`);
    break;
  }
  if (t.score === 0 && t.baseScore + t.musicBonus + t.tempoBoost + t.syncBonus + t.perfectBonus > 0) {
    check('clamped only when raw sum <= 0', false, `(${JSON.stringify(t)})`);
    break;
  }
  if (i === 199) check('breakdown sums (fuzz, 200 cases)', true);
}

console.log(`\n${failures === 0 ? 'ALL PASSED ✓' : failures + ' FAILURES ✗'}`);
process.exit(failures === 0 ? 0 : 1);
