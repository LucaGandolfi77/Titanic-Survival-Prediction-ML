/* ===== DATING SYSTEM ===== */
import { getState, getCharLoveLevel } from './state.js';
import { CHARACTERS, getDialogueForLevel } from './characters.js';
import { boostTeamHappiness } from './team.js';
import { clamp } from './utils.js';

export function canDateCharacter(state, charId) {
  const cs = state.characters[charId];
  const c = CHARACTERS[charId];
  if (!cs || !c) return false;
  if (!cs.met) return false;
  if (cs.love < 20) return false;
  if (c.unlockDay > state.day) return false;
  if (charId === 'captain' && cs.love < 40) return false;
  return true;
}

export function getDateScene(charId, state) {
  const c = CHARACTERS[charId];
  const cs = state.characters[charId];
  if (!c || !cs) return null;

  const dialogues = getDialogueForLevel(charId, cs.love);
  if (!dialogues || dialogues.length === 0) return null;

  const dialogue = dialogues[0];
  return {
    charId,
    character: c,
    dialogue,
    bg: c.dateBg,
    location: c.dateLocations[Math.min(Math.floor(cs.love / 35), c.dateLocations.length - 1)],
  };
}

export function applyDateChoice(state, charId, effect) {
  const cs = state.characters[charId];
  if (!cs) return 0;

  let loveChange = 0;
  switch (effect) {
    case 'best':
      loveChange = 15;
      break;
    case 'good':
      loveChange = 8;
      break;
    case 'bad':
      loveChange = -5;
      break;
  }

  cs.love = clamp(cs.love + loveChange, 0, 100);

  // Captain trust increases with positive interactions
  if (charId === 'captain' && loveChange > 0) {
    state.captainTrust = clamp(state.captainTrust + loveChange, 0, 100);
  }

  return loveChange;
}

export function checkGiftUnlock(state, charId) {
  const c = CHARACTERS[charId];
  const cs = state.characters[charId];
  if (!c || !cs) return null;
  if (cs.giftGiven) return null;
  if (cs.love < 70) return null;

  cs.giftGiven = true;

  if (c.giftFameBonus) {
    state.fame += c.giftFameBonus;
    state.dailyFame += c.giftFameBonus;
  }
  if (c.giftHappinessBonus) {
    boostTeamHappiness(state, c.giftHappinessBonus);
  }

  return {
    character: c.name,
    description: c.giftDescription,
    fameBonus: c.giftFameBonus || 0,
  };
}

export function getRandomEncounter(state) {
  const day = state.day;
  const encounters = [];

  for (const [id, c] of Object.entries(CHARACTERS)) {
    const cs = state.characters[id];
    if (!cs) continue;
    if (c.unlockDay > day) continue;

    // Auto-meet on unlock day
    if (c.unlockDay === day && !cs.met) {
      encounters.push({ type: 'meet', charId: id });
    }
  }

  return encounters.length > 0 ? encounters[0] : null;
}

export function meetCharacter(state, charId) {
  const cs = state.characters[charId];
  if (!cs) return;
  cs.met = true;
  state.dailyEvents.push(`🤝 Met ${CHARACTERS[charId].name}!`);
}

export function getHighestLoveCharacter(state) {
  let best = null;
  let bestLove = -1;
  for (const [id, cs] of Object.entries(state.characters)) {
    if (cs.love > bestLove) {
      bestLove = cs.love;
      best = id;
    }
  }
  return best ? { id: best, love: bestLove } : null;
}

export function getEndingForCharacter(charId) {
  const c = CHARACTERS[charId];
  return c ? c.dialogues.ending : "You made some friends along the way. Maybe next cruise...";
}

export function getEveningDateOptions(state) {
  const options = [];
  for (const [id, c] of Object.entries(CHARACTERS)) {
    if (canDateCharacter(state, id)) {
      options.push({ charId: id, name: c.name, emoji: c.emoji, love: state.characters[id].love });
    }
  }
  return options;
}
