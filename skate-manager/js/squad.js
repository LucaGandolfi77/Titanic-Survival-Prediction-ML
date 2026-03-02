/* ===== Squad management + market logic ===== */
import { GameState } from './state.js';
import { createSkater, generateMarketSkaters, recalcSkater, trainSkater, getTotalWages, getTeamCohesion } from './skaters.js';
import { randInt, clamp } from './utils.js';

// ===== Squad actions =====
export function promoteToActive(skaterId) {
  if (GameState.activeSquad.length >= 16) return false;
  const idx = GameState.reserveBench.findIndex(s => s.id === skaterId);
  if (idx === -1) return false;
  const sk = GameState.reserveBench.splice(idx, 1)[0];
  sk.status = 'active';
  GameState.activeSquad.push(sk);
  return true;
}

export function demoteToReserve(skaterId) {
  if (GameState.reserveBench.length >= 8) return false;
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

export function swapActivePositions(idA, idB) {
  const iA = GameState.activeSquad.findIndex(s => s.id === idA);
  const iB = GameState.activeSquad.findIndex(s => s.id === idB);
  if (iA === -1 || iB === -1) return false;
  [GameState.activeSquad[iA], GameState.activeSquad[iB]] =
    [GameState.activeSquad[iB], GameState.activeSquad[iA]];
  return true;
}

// ===== Market actions =====
export function buySkater(skaterId) {
  const totalRoster = GameState.activeSquad.length + GameState.reserveBench.length;
  if (totalRoster >= 24) return { ok: false, msg: 'Roster full (max 24)' };

  const idx = GameState.marketSkaters.findIndex(s => s.id === skaterId);
  if (idx === -1) return { ok: false, msg: 'Skater no longer available' };

  const sk = GameState.marketSkaters[idx];
  const price = sk.askingPrice || sk.value;
  if (GameState.money < price) return { ok: false, msg: 'Insufficient funds' };

  GameState.money -= price;
  GameState.marketSkaters.splice(idx, 1);

  if (GameState.activeSquad.length < 16) {
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
  let fromActive = false;
  let idx = GameState.activeSquad.findIndex(s => s.id === skaterId);
  if (idx !== -1) {
    sk = GameState.activeSquad.splice(idx, 1)[0];
    fromActive = true;
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
  if (GameState.activeSquad.length < 16) {
    sk.status = 'active';
    GameState.activeSquad.push(sk);
  } else if (GameState.reserveBench.length < 8) {
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
  // Each rival has a chance to buy from market
  for (const rival of GameState.rivals) {
    if (Math.random() < 0.3 && GameState.marketSkaters.length > 0) {
      const skIdx = randInt(0, GameState.marketSkaters.length - 1);
      const sk = GameState.marketSkaters.splice(skIdx, 1)[0];
      rival.points += Math.round(sk.overall * 0.5);
      messages.push(`${rival.name} signed ${sk.name}`);
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
  if (GameState.money < 2000) return { ok: false, msg: 'Need €2,000 for scouting' };
  GameState.money -= 2000;
  GameState.scoutedThisWeek = true;
  // Add 3 hidden skaters (one possibly tier 4)
  const hasStar = Math.random() < 0.25;
  const tiers = hasStar ? [4, 3, 2] : [3, 2, 2];
  for (const tier of tiers) {
    const sk = createSkater(tier);
    sk.status = 'market';
    sk.askingPrice = Math.round(sk.value * (0.9 + Math.random() * 0.4));
    sk.scouted = true;
    GameState.marketSkaters.push(sk);
  }
  return { ok: true, msg: hasStar ? 'Scout found a STAR talent! 🌟' : 'Scout found 3 new prospects' };
}

export function trainTeam() {
  if (GameState.money < 5000) return { ok: false, msg: 'Need €5,000 for training' };
  GameState.money -= 5000;
  const results = [];
  for (const sk of GameState.activeSquad) {
    if (sk.status !== 'injured') {
      const stat = trainSkater(sk);
      results.push(`${sk.name}: +2 ${stat}`);
    }
  }
  return { ok: true, msg: `Trained ${results.length} skaters`, details: results };
}

export function getCohesion() {
  return getTeamCohesion(GameState.activeSquad);
}

export function getWeeklyWages() {
  return getTotalWages(GameState.activeSquad, GameState.reserveBench);
}
