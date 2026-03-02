/* ===== Competition system ===== */
import { GameState } from './state.js';
import { getSquadAvgOverall } from './skaters.js';
import { randInt, pick } from './utils.js';

const COMP_NAMES = {
  1: ['Regional Cup', 'City Open', 'Provincial Series'],
  2: ['National Open', 'National Classic', 'Federation Cup'],
  3: ['Grand Prix', 'International Trophy', 'Continental Championship'],
  4: ['World Championship', 'Olympic Qualifier', 'Diamond Series']
};

export function generateCalendar(season) {
  const calendar = [];
  for (let w = 0; w < 12; w++) {
    // Competition every 1-2 weeks, sometimes off weeks
    const hasComp = w === 0 || Math.random() < 0.7;
    if (hasComp) {
      // Tier progression: earlier weeks = lower tier
      let tier;
      if (w < 3) tier = 1;
      else if (w < 6) tier = randInt(1, 2);
      else if (w < 9) tier = randInt(2, 3);
      else tier = randInt(3, 4);

      // World Championship always at week 11 (last)
      if (w === 11) tier = 4;

      const name = pick(COMP_NAMES[tier]);
      const entryFee = tier * 2000;
      const prizeMultiplier = tier;

      calendar.push({
        week: w + 1,
        name: season > 1 && tier === 4 ? 'World Championship' : name,
        tier,
        entryFee,
        prizes: {
          1: 20000 * prizeMultiplier,
          2: 12000 * prizeMultiplier,
          3: 6000 * prizeMultiplier,
          4: 3000 * prizeMultiplier
        },
        pointsPool: {
          1: 100 * tier,
          2: 70 * tier,
          3: 40 * tier,
          4: 25 * tier,
          5: 15 * tier,
          6: 10 * tier
        },
        fameReward: {
          1: 8 * tier,
          2: 5 * tier,
          3: 3 * tier
        },
        minOverall: tier * 35,
        competition: null // Will hold result after playing
      });
    } else {
      calendar.push({
        week: w + 1,
        name: null,
        tier: 0,
        training: true
      });
    }
  }
  return calendar;
}

export function canEnterCompetition(weekIndex) {
  const comp = GameState.calendar[weekIndex];
  if (!comp || !comp.name) return { ok: false, msg: 'No competition this week' };
  if (GameState.enteredCompetitions[weekIndex]) return { ok: false, msg: 'Already entered' };
  if (GameState.money < comp.entryFee) return { ok: false, msg: `Need €${comp.entryFee.toLocaleString()} entry fee` };
  if (GameState.activeSquad.length < 16) return { ok: false, msg: 'Need 16 active skaters' };
  const avgOv = getSquadAvgOverall(GameState.activeSquad);
  if (avgOv < comp.minOverall) return { ok: false, msg: `Squad overall ${avgOv} below minimum ${comp.minOverall}` };
  return { ok: true };
}

export function enterCompetition(weekIndex) {
  const check = canEnterCompetition(weekIndex);
  if (!check.ok) return check;
  GameState.money -= GameState.calendar[weekIndex].entryFee;
  GameState.enteredCompetitions[weekIndex] = true;
  return { ok: true, msg: `Entered ${GameState.calendar[weekIndex].name}` };
}

export function generateRivalScores(tier) {
  // AI rival scores based on difficulty range per tier
  const scores = [];
  for (const rival of GameState.rivals) {
    const baseScore = rival.strength * 100;
    const variance = randInt(-2000, 2000);
    const tierBonus = tier * 1500;
    const score = Math.max(500, baseScore + tierBonus + variance);
    scores.push({ team: rival.name, score, isPlayer: false });
  }
  return scores;
}

export function calculatePlacements(playerScore, weekIndex) {
  const comp = GameState.calendar[weekIndex];
  const rivalScores = generateRivalScores(comp.tier);
  const allScores = [
    { team: GameState.teamName, score: playerScore, isPlayer: true },
    ...rivalScores
  ];
  allScores.sort((a, b) => b.score - a.score);

  // Assign placement
  allScores.forEach((entry, i) => entry.placement = i + 1);

  const playerEntry = allScores.find(e => e.isPlayer);
  const placement = playerEntry.placement;

  // Award prizes
  const prizeMoney = comp.prizes[placement] || 0;
  const pointsAwarded = comp.pointsPool[placement] || (placement <= 16 ? 5 * comp.tier : 0);
  const fameAwarded = comp.fameReward[placement] || 0;

  GameState.money += prizeMoney;
  GameState.points += pointsAwarded;
  GameState.fame += fameAwarded;

  // Award points to rivals
  allScores.forEach(entry => {
    if (!entry.isPlayer) {
      const rival = GameState.rivals.find(r => r.name === entry.team);
      if (rival) {
        rival.points += comp.pointsPool[entry.placement] || 5;
        rival.fame += comp.fameReward[entry.placement] || 0;
        if (entry.placement === 1) rival.wins++;
      }
    }
  });

  // Record result
  const result = {
    season: GameState.season,
    week: comp.week,
    competition: comp.name,
    tier: comp.tier,
    playerScore,
    placement,
    prizeMoney,
    pointsAwarded,
    fameAwarded,
    leaderboard: allScores
  };
  GameState.competitionResults.push(result);
  comp.competition = result;

  return result;
}

export function getThisWeekCompetition() {
  const idx = GameState.week - 1;
  if (idx < 0 || idx >= GameState.calendar.length) return null;
  const comp = GameState.calendar[idx];
  if (!comp.name) return null;
  return { comp, weekIndex: idx, entered: !!GameState.enteredCompetitions[idx] };
}

export function generateRivals() {
  const names = ['Arctic Blades', 'Frost Queens', 'Crystal Gliders', 'Polar Stars', 'Ice Phoenix'];
  const rivals = [];
  for (let i = 0; i < 3; i++) {
    rivals.push({
      name: names[i],
      strength: randInt(40, 75),
      points: 0,
      fame: randInt(5, 25),
      wins: 0,
      money: 50000
    });
  }
  return rivals;
}
