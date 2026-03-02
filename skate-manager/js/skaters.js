/* ===== Skater data model and generation ===== */
import { uid, generateName, generateNationality, generateStats, calcOverall, calcValue, calcWage, randInt, clamp } from './utils.js';

const AVATARS = ['🧊', '⛸️', '❄️', '🌸', '💎', '✨', '🌟', '🦋'];

export function createSkater(tier = 2) {
  const stats = generateStats(tier);
  const overall = calcOverall(stats);
  const age = tier === 1 ? randInt(16, 20) :
              tier === 2 ? randInt(19, 28) :
              tier === 3 ? randInt(22, 30) :
              randInt(24, 32);
  const nat = generateNationality();
  const value = calcValue(overall, age);
  const wage = calcWage(overall);

  return {
    id: uid(),
    name: generateName(),
    age,
    nationality: nat,
    avatar: AVATARS[overall % AVATARS.length],
    stats: { ...stats },
    overall,
    value,
    wage,
    form: randInt(50, 90),
    morale: randInt(55, 85),
    status: 'active',
    injuryWeeks: 0,
    contract: {
      weeksRemaining: randInt(8, 36),
      wage
    }
  };
}

export function generateStartingSquad(difficulty) {
  const squad = [];
  const reserve = [];
  // Active squad: 16 skaters
  let tiers;
  if (difficulty === 'amateur')   tiers = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1];
  else if (difficulty === 'elite') tiers = [3,3,3,3,2,2,2,2,2,2,3,3,2,2,2,2];
  else                            tiers = [2,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2]; // semi-pro

  for (let i = 0; i < 16; i++) {
    const sk = createSkater(tiers[i]);
    sk.status = 'active';
    squad.push(sk);
  }
  // Reserve: 4 youth
  for (let i = 0; i < 4; i++) {
    const sk = createSkater(1);
    sk.status = 'reserve';
    reserve.push(sk);
  }
  return { squad, reserve };
}

export function generateMarketSkaters() {
  const skaters = [];
  // 3 per tier (1-4)
  for (let tier = 1; tier <= 4; tier++) {
    const count = tier === 4 ? 1 : 3;
    for (let i = 0; i < count; i++) {
      const sk = createSkater(tier);
      sk.status = 'market';
      // Asking price with random multiplier
      sk.askingPrice = Math.round(sk.value * (0.9 + Math.random() * 0.4));
      skaters.push(sk);
    }
  }
  return skaters;
}

export function recalcSkater(sk) {
  sk.overall = calcOverall(sk.stats);
  sk.value = calcValue(sk.overall, sk.age);
  sk.wage = calcWage(sk.overall);
  sk.contract.wage = sk.wage;
}

export function weeklyStatFluctuation(sk) {
  // Form fluctuates ±10
  sk.form = clamp(sk.form + randInt(-10, 10), 0, 100);
  // Morale fluctuates ±5
  sk.morale = clamp(sk.morale + randInt(-5, 5), 0, 100);
  // Heal injuries
  if (sk.injuryWeeks > 0) sk.injuryWeeks--;
  if (sk.injuryWeeks === 0 && sk.status === 'injured') sk.status = 'active';
  // Contract ticks
  if (sk.contract.weeksRemaining > 0) sk.contract.weeksRemaining--;
}

export function injureSkater(sk, weeks) {
  sk.status = 'injured';
  sk.injuryWeeks = weeks;
}

export function trainSkater(sk) {
  const stats = ['technique', 'stamina', 'rhythm', 'sync', 'charisma'];
  const stat = stats[randInt(0, 4)];
  sk.stats[stat] = clamp(sk.stats[stat] + 2, 1, 100);
  recalcSkater(sk);
  return stat;
}

export function getSquadAvgOverall(squad) {
  if (squad.length === 0) return 0;
  return Math.round(squad.reduce((s, sk) => s + sk.overall, 0) / squad.length);
}

export function getSquadAvgMorale(squad) {
  if (squad.length === 0) return 0;
  return Math.round(squad.reduce((s, sk) => s + sk.morale, 0) / squad.length);
}

export function getTeamCohesion(squad) {
  if (squad.length < 4) return 0;
  // Based on nationality diversity, age spread, morale average
  const nations = new Set(squad.map(s => s.nationality.country));
  const diversityBonus = Math.min(nations.size * 8, 40);
  const avgMorale = getSquadAvgMorale(squad);
  const moraleBonus = avgMorale * 0.4;
  const ages = squad.map(s => s.age);
  const ageSpread = Math.max(...ages) - Math.min(...ages);
  const agePenalty = Math.max(0, ageSpread - 10) * 2;
  return Math.round(clamp(diversityBonus + moraleBonus - agePenalty, 0, 100));
}

export function getTotalWages(activeSquad, reserveBench) {
  const all = [...activeSquad, ...reserveBench];
  return all.reduce((s, sk) => s + sk.wage, 0);
}
