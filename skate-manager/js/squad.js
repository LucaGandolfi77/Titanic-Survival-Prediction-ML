/* ===== Squad management + market logic ===== */
import { GameState } from './state.js';
import { createSkater, generateMarketSkaters, trainSkater, getTotalWages, getTeamCohesion } from './skaters.js';
import { randInt, clamp } from './utils.js';
import { ACTIVE_SQUAD_SIZE, RESERVE_SIZE, MAX_ROSTER, TRAIN_TEAM_COST, SCOUT_COST, STAFF } from './config.js';

/**
 * Find a skater by id across every roster list (active, reserve, market, listed).
 * @param {string} id
 * @returns {import('./types.js').Skater|undefined}
 */
export function findSkater(id) {
  return GameState.activeSquad.find(s => s.id === id) ||
         GameState.reserveBench.find(s => s.id === id) ||
         GameState.marketSkaters.find(s => s.id === id) ||
         GameState.listedSkaters.find(s => s.id === id);
}

// ===== Squad actions =====
export function promoteToActive(skaterId) {
  if (GameState.activeSquad.length >= ACTIVE_SQUAD_SIZE) return false;
  const idx = GameState.reserveBench.findIndex(s => s.id === skaterId);
  if (idx === -1) return false;
  const sk = GameState.reserveBench.splice(idx, 1)[0];
  sk.status = 'active';
  GameState.activeSquad.push(sk);
  return true;
}

export function demoteToReserve(skaterId) {
  if (GameState.reserveBench.length >= RESERVE_SIZE) return false;
  const idx = GameState.activeSquad.findIndex(s => s.id === skaterId);
  if (idx === -1) return false;
  const sk = GameState.activeSquad.splice(idx, 1)[0];
  sk.status = 'reserve';
  GameState.reserveBench.push(sk);
  return true;
}

export function releaseSkater(skaterId) {
  let idx = GameState.activeSquad.findIndex(s => s.id === skaterId);
  if (idx !== -1) {
    GameState.activeSquad.splice(idx, 1);
    return true;
  }
  idx = GameState.reserveBench.findIndex(s => s.id === skaterId);
  if (idx !== -1) {
    GameState.reserveBench.splice(idx, 1);
    return true;
  }
  return false;
}

// ===== Market actions =====
export function buySkater(skaterId) {
  const totalRoster = GameState.activeSquad.length + GameState.reserveBench.length;
  if (totalRoster >= MAX_ROSTER) return { ok: false, msg: `Roster full (max ${MAX_ROSTER})` };

  const idx = GameState.marketSkaters.findIndex(s => s.id === skaterId);
  if (idx === -1) return { ok: false, msg: 'Skater no longer available' };

  const sk = GameState.marketSkaters[idx];
  const price = sk.askingPrice || sk.value;
  if (GameState.money < price) return { ok: false, msg: 'Insufficient funds' };

  GameState.money -= price;
  GameState.marketSkaters.splice(idx, 1);

  if (GameState.activeSquad.length < ACTIVE_SQUAD_SIZE) {
    sk.status = 'active';
    GameState.activeSquad.push(sk);
  } else {
    sk.status = 'reserve';
    GameState.reserveBench.push(sk);
  }

  return { ok: true, msg: `Signed ${sk.name} for €${price.toLocaleString()}` };
}

export function listForSale(skaterId, askingPrice) {
  // Find in active or reserve
  let sk = null;
  let idx = GameState.activeSquad.findIndex(s => s.id === skaterId);
  if (idx !== -1) {
    sk = GameState.activeSquad.splice(idx, 1)[0];
  } else {
    idx = GameState.reserveBench.findIndex(s => s.id === skaterId);
    if (idx !== -1) {
      sk = GameState.reserveBench.splice(idx, 1)[0];
    }
  }
  if (!sk) return false;

  sk.status = 'market';
  sk.askingPrice = askingPrice;
  GameState.listedSkaters.push(sk);
  return true;
}

export function cancelListing(skaterId) {
  const idx = GameState.listedSkaters.findIndex(s => s.id === skaterId);
  if (idx === -1) return false;
  const sk = GameState.listedSkaters.splice(idx, 1)[0];
  if (GameState.activeSquad.length < ACTIVE_SQUAD_SIZE) {
    sk.status = 'active';
    GameState.activeSquad.push(sk);
  } else if (GameState.reserveBench.length < RESERVE_SIZE) {
    sk.status = 'reserve';
    GameState.reserveBench.push(sk);
  } else {
    // No space, just put back on market
    GameState.marketSkaters.push(sk);
  }
  return true;
}

// AI buys from market/listed skaters
export function aiMarketActivity() {
  const messages = [];
  // Each rival has a chance to buy from market — signings raise their strength
  // for future competitions, so the AI genuinely improves over the season.
  for (const rival of GameState.rivals) {
    if (Math.random() < 0.3 && GameState.marketSkaters.length > 0) {
      const skIdx = randInt(0, GameState.marketSkaters.length - 1);
      const sk = GameState.marketSkaters.splice(skIdx, 1)[0];
      rival.strength = clamp(rival.strength + Math.round(sk.overall / 20), 30, 95);
      rival.fame += 1;
      messages.push(`${rival.name} signed ${sk.name} (their team got stronger)`);
    }
    // AI may buy your listed skater
    if (Math.random() < 0.25 && GameState.listedSkaters.length > 0) {
      const skIdx = randInt(0, GameState.listedSkaters.length - 1);
      const sk = GameState.listedSkaters.splice(skIdx, 1)[0];
      GameState.money += sk.askingPrice;
      messages.push(`${rival.name} bought ${sk.name} for €${sk.askingPrice.toLocaleString()}`);
    }
  }
  return messages;
}

export function refreshMarket() {
  GameState.marketSkaters = generateMarketSkaters();
  GameState.scoutedThisWeek = false;
}

export function scoutMarket() {
  if (GameState.scoutedThisWeek) return { ok: false, msg: 'Already scouted this week' };
  if (GameState.money < SCOUT_COST) return { ok: false, msg: `Need €${SCOUT_COST.toLocaleString()} for scouting` };
  GameState.money -= SCOUT_COST;
  GameState.scoutedThisWeek = true;
  // Head Scout: 4 hidden skaters with a better star chance
  const headScout = GameState.staff.includes('headscout');
  const hasStar = Math.random() < (headScout ? 0.35 : 0.25);
  const added = hasStar
    ? (headScout ? [4, 3, 2, 2] : [4, 3, 2])
    : (headScout ? [3, 2, 2, 2] : [3, 2, 2]);
  for (const tier of added) {
    const sk = createSkater(tier);
    sk.status = 'market';
    sk.askingPrice = Math.round(sk.value * (0.9 + Math.random() * 0.4));
    sk.scouted = true;
    GameState.marketSkaters.push(sk);
  }
  const count = added.length;
  return { ok: true, msg: hasStar ? `Scout found a STAR talent! 🌟 (+${count - 1} prospects)` : `Scout found ${count} new prospects` };
}

/**
 * Train all healthy active skaters. Focus picks which stat to train;
 * the Technique Coach raises the amount to +3.
 * @param {'balanced'|string} [focus]
 */
export function trainTeam(focus = 'balanced') {
  if (GameState.money < TRAIN_TEAM_COST) return { ok: false, msg: `Need €${TRAIN_TEAM_COST.toLocaleString()} for training` };
  GameState.money -= TRAIN_TEAM_COST;
  const amount = GameState.staff.includes('technique') ? 3 : 2;
  const results = [];
  for (const sk of GameState.activeSquad) {
    if (sk.status !== 'injured') {
      const stat = trainSkater(sk, focus, amount);
      results.push(`${sk.name}: +${amount} ${stat}`);
    }
  }
  return { ok: true, msg: `Trained ${results.length} skaters (${focus}${amount === 3 ? ', coached' : ''})`, details: results };
}

/**
 * Hire a coaching staff member (permanent, passive bonuses).
 * @param {string} staffId
 */
export function hireStaff(staffId) {
  const member = STAFF.find(s => s.id === staffId);
  if (!member) return { ok: false, msg: 'Unknown staff' };
  if (GameState.staff.includes(staffId)) return { ok: false, msg: 'Already hired' };
  if (GameState.money < member.cost) {
    return { ok: false, msg: `Need €${member.cost.toLocaleString()} to hire ${member.name}` };
  }
  GameState.money -= member.cost;
  GameState.staff.push(staffId);
  return { ok: true, msg: `${member.icon} Hired ${member.name}: ${member.description}` };
}

export function getCohesion() {
  return getTeamCohesion(GameState.activeSquad);
}

export function getWeeklyWages() {
  return getTotalWages(GameState.activeSquad, GameState.reserveBench);
}

/**
 * Renew a skater's contract for `weeks` more weeks at half their weekly wage
 * per week (loyalty discount). Syncs the contract wage with their current wage.
 * @param {string} skaterId
 * @param {number} [weeks]
 */
export function renewContract(skaterId, weeks = 12) {
  const sk = findSkater(skaterId);
  if (!sk) return { ok: false, msg: 'Skater not found' };
  if (GameState.activeSquad.includes(sk) === false && GameState.reserveBench.includes(sk) === false) {
    return { ok: false, msg: 'Only squad skaters can renew' };
  }
  const cost = Math.round(sk.wage * weeks / 2);
  if (GameState.money < cost) return { ok: false, msg: `Need €${cost.toLocaleString()} to renew` };
  GameState.money -= cost;
  sk.contract.weeksRemaining += weeks;
  sk.contract.wage = sk.wage;
  return { ok: true, msg: `${sk.name} signed a ${weeks}-week extension (−€${cost.toLocaleString()})` };
}
