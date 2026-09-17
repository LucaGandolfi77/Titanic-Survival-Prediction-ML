const RARITY_WEIGHTS = { common: 55, rare: 30, epic: 12, legendary: 3 };

function rollRarity(pityRare, pityEpic, pityLegendary) {
  if (pityRare >= 10) return 'rare';
  if (pityEpic >= 10) return 'epic';
  if (pityLegendary >= 50) return 'legendary';

  const roll = Math.random() * 100;
  if (roll < RARITY_WEIGHTS.legendary) return 'legendary';
  if (roll < RARITY_WEIGHTS.legendary + RARITY_WEIGHTS.epic) return 'epic';
  if (roll < RARITY_WEIGHTS.legendary + RARITY_WEIGHTS.epic + RARITY_WEIGHTS.rare) return 'rare';
  return 'common';
}

function getRarityPool(rarity) {
  const pool = Object.entries(UNRAVELERS).filter(([, u]) => u.rarity === rarity);
  return pool[Math.floor(Math.random() * pool.length)];
}

function performPull(state) {
  const cost = state.hasFreePull ? 0 : RARITY_COST.common;
  if (cost > 0 && state.threads < cost) return null;

  if (cost > 0) {
    state.threads -= cost;
  }

  const rarity = rollRarity(state.pityRare, state.pityEpic, state.pityLegendary);
  const [id, unraveler] = getRarityPool(rarity);

  state.pullCount++;
  state.hasFreePull = false;

  if (rarity === 'rare') {
    state.pityRare = 0;
    state.pityEpic = 0;
    state.pityLegendary = 0;
  } else if (rarity === 'epic') {
    state.pityRare++;
    state.pityEpic = 0;
    state.pityLegendary = 0;
  } else if (rarity === 'legendary') {
    state.pityRare++;
    state.pityEpic++;
    state.pityLegendary = 0;
  } else {
    state.pityRare++;
    state.pityEpic++;
    state.pityLegendary++;
  }


  if (state.roster.includes(id)) {
    state.threadShards[id] = (state.threadShards[id] || 0) + 50;
  } else {
    state.roster.push(id);
    state.unravelerLevels[id] = 1;
    state.threadShards[id] = 0;
  }

  saveState(state);
  return { id, unraveler, rarity };
}
