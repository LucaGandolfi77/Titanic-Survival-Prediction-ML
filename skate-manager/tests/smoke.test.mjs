/* Smoke test for skate-manager pure logic modules */
import { GameState, resetState, serializeState, migrateState } from '../js/state.js';
import { STARTING_MONEY, SAVE_VERSION } from '../js/config.js';
import { generateStartingSquad, getSquadAvgOverall } from '../js/skaters.js';
import { generateCalendar, generateRivals, generateRivalScores, canEnterCompetition,
         enterCompetition, withdrawCompetition, calculatePlacements, getThisWeekCompetition } from '../js/competitions.js';
import { getTotalWages } from '../js/skaters.js';
import { calcWage } from '../js/utils.js';
import { negotiateSponsor, getTempoBonus, getSyncBonus, processWeeklySponsors } from '../js/sponsors.js';
import { FORMATIONS } from '../js/formations.js';

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`  ✓ ${name}`);
  else { console.log(`  ✗ FAIL: ${name} ${detail}`); failures++; }
}

// ===== 1. Economy: wages vs starting money =====
console.log('\n[1] Economy');
resetState();
GameState.calendar = generateCalendar(1);
GameState.rivals = generateRivals();
for (const diff of ['amateur', 'semi-pro', 'elite']) {
  const { squad, reserve } = generateStartingSquad(diff);
  const wages = getTotalWages(squad, reserve);
  const money = STARTING_MONEY[diff];
  console.log(`  ${diff}: start ${money}, wages/wk ${wages}, runway ~${Math.floor(money / wages)} weeks, avg OV ${getSquadAvgOverall(squad)}`);
  check(`${diff} runway >= 6 weeks`, money / wages >= 6);
  check(`${diff} squad has 16 active + 4 reserves`, squad.length === 16 && reserve.length === 4);
}

// ===== 2. minOverall reachability =====
console.log('\n[2] Competition minOverall');
const t4min = 15 + 4 * 15;
check('tier 4 minOverall (75) <= 99', t4min <= 99);
resetState();
GameState.rivals = generateRivals();
// The random calendar may contain no tier-3 competition — regenerate until one exists
let tier3Comp;
for (let tries = 0; tries < 20 && !tier3Comp; tries++) {
  GameState.calendar = generateCalendar(1);
  tier3Comp = GameState.calendar.find(c => c.tier === 3);
}
check('calendar generated with a tier-3 competition', !!tier3Comp);
if (!tier3Comp) { console.log(`\n${failures + 1} FAILURES ✗`); process.exit(1); }
const eliteSquad = generateStartingSquad('elite').squad;
GameState.activeSquad = eliteSquad;
const avgOv = getSquadAvgOverall(eliteSquad);
check(`elite squad (avg ${avgOv}) can enter tier-3 comp (min ${tier3Comp.minOverall})`, avgOv >= tier3Comp.minOverall);
const eliteCheck = canEnterCompetition(GameState.calendar.indexOf(tier3Comp));
check(`canEnterCompetition tier-3 passes for elite`, eliteCheck.ok || eliteCheck.msg.includes('entry fee'), `(${eliteCheck.msg})`);

// ===== 3. Rival score ranges vs player scoring =====
console.log('\n[3] Rival scores');
for (const diff of ['amateur', 'semi-pro', 'elite']) {
  GameState.difficulty = diff;
  GameState.rivals = generateRivals();
  const all = [1, 2, 3, 4].flatMap(t => generateRivalScores(t).map(r => r.score));
  const min = Math.min(...all), max = Math.max(...all);
  console.log(`  ${diff}: rival scores ${min}–${max}`);
  check(`${diff} rivals within player range (<= 3000)`, max <= 3000, `(max ${max})`);
}

// ===== 4. Enter / withdraw competition =====
console.log('\n[4] Enter/withdraw');
resetState();
GameState.calendar = generateCalendar(1);
GameState.rivals = generateRivals();
const { squad } = generateStartingSquad('semi-pro');
GameState.activeSquad = squad;
GameState.money = STARTING_MONEY['semi-pro'] + 100000;
GameState.difficulty = 'semi-pro';
const comp = GameState.calendar[0];
check('week 1 has a competition', !!comp.name);
const enterRes = enterCompetition(0);
check('enter ok', enterRes.ok, JSON.stringify(enterRes));
check('money deducted', GameState.money === STARTING_MONEY['semi-pro'] + 100000 - comp.entryFee);
check('shows as entered', getThisWeekCompetition().entered === true);
const wd = withdrawCompetition(0);
check('withdraw ok', wd.ok, JSON.stringify(wd));
check('re-enter ok after withdraw', enterCompetition(0).ok);
const wdAfterCompeting = (() => {
  GameState.calendar[0].competition = { placement: 1 };
  return withdrawCompetition(0);
})();
check('withdraw blocked after competing', wdAfterCompeting.ok === false, JSON.stringify(wdAfterCompeting));

// ===== 5. Placements winnable =====
console.log('\n[5] Placements');
let wins = 0, podiums = 0;
for (let sim = 0; sim < 200; sim++) {
  GameState.enteredCompetitions[0] = true;
  const res = calculatePlacements(1600, 0); // strong player score
  GameState.calendar[0].competition = null; // reset for next sim
  if (res.placement === 1) wins++;
  if (res.placement <= 3) podiums++;
}
console.log(`  Player score 1600 (semi-pro): wins ${wins}/200, podiums ${podiums}/200`);
check('player can win with strong play', wins > 20, `(wins ${wins})`);
check('player podiums often with strong play', podiums > 60, `(podiums ${podiums})`);
let weakWins = 0;
for (let sim = 0; sim < 200; sim++) {
  GameState.enteredCompetitions[0] = true;
  const res = calculatePlacements(400, 0);
  GameState.calendar[0].competition = null;
  if (res.placement === 1) weakWins++;
}
check('weak play (400) rarely wins', weakWins < 20, `(wins ${weakWins})`);

// ===== 6. Sponsor perks =====
console.log('\n[6] Sponsors');
resetState();
GameState.fame = 100;
const cb = negotiateSponsor('coolbreeze');
const qt = negotiateSponsor('quantumice');
check('coolbreeze negotiable', cb.ok, JSON.stringify(cb));
check('quantumice negotiable', qt.ok, JSON.stringify(qt));
check('tempoBonus active = 0.25', getTempoBonus() === 0.25);
check('syncBonus active = 0.10', getSyncBonus() === 0.10);
const sr = processWeeklySponsors();
check('weekly sponsor income processed', sr.totalIncome === 2200 + 8000, `(${sr.totalIncome})`);

// ===== 7. Save serialize/migrate =====
console.log('\n[7] Save/load');
resetState();
GameState.activeSquad = generateStartingSquad('semi-pro').squad;
GameState.minigameActive = true;
GameState.currentCompetition = { name: 'X' };
const ser = serializeState();
check('transient fields not serialized', !('minigameActive' in ser) && !('currentCompetition' in ser));
check('saveVersion serialized', ser.saveVersion === SAVE_VERSION);
const v1 = JSON.parse(JSON.stringify(ser));
delete v1.saveVersion;
v1.marketRefreshWeek = 1;
v1.week = 5;
const migrated = migrateState(v1);
check('v1 migrated to v2', migrated.saveVersion === SAVE_VERSION);
check('marketRefreshWeek fixed', migrated.marketRefreshWeek >= 6, `(${migrated.marketRefreshWeek})`);
check('missing keys filled', 'autosave' in migrated);

// ===== 8. Formations: 16 positions each, in bounds =====
console.log('\n[8] Formations');
for (const f of FORMATIONS) {
  check(`${f.name}: 16 positions`, f.positions.length === 16, `(${f.positions.length})`);
  const inBounds = f.positions.every(p => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
  check(`${f.name}: positions in bounds`, inBounds);
}

// ===== 9. Wage function =====
console.log('\n[9] Wages');
check('calcWage(60) = 630', calcWage(60) === 630, `(${calcWage(60)})`);
check('calcWage(40) = 470', calcWage(40) === 470, `(${calcWage(40)})`);

console.log(`\n${failures === 0 ? 'ALL PASSED ✓' : failures + ' FAILURES ✗'}`);
process.exit(failures === 0 ? 0 : 1);
