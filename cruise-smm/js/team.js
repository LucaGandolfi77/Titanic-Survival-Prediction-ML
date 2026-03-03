/* ===== TEAM MANAGEMENT ===== */
import { getState } from './state.js';
import { clamp } from './utils.js';

export const TEAM_MEMBERS = {
  jessica: {
    id: 'jessica',
    name: 'JB',
    role: 'Lead Photographer',
    emoji: '📷',
    age: 27,
    nationality: 'Italian 🇮🇹',
    stats: { video: 60, photo: 98, editing: 75 },
    wage: 120,
    specialSkill: 'Photographer Eye',
    specialDesc: '+45% fame on photo tasks',
    catchphrase: '"Just one more angle — this will be perfect."',
  },
  marta: {
    id: 'marta',
    name: 'MDA',
    role: 'Food & Lifestyle Creator',
    emoji: '🥗',
    age: 26,
    nationality: 'Italian 🇮🇹',
    stats: { video: 60, photo: 70, editing: 65 },
    wage: 110,
    specialSkill: 'Vegan Guru',
    specialDesc: '+30% fame on plant-based / food content',
    catchphrase: '"Plants on my plate, heart on my sleeve."',
  },
  valentina: {
    id: 'valentina',
    name: 'VG',
    role: 'Videographer / Skater',
    emoji: '🛹',
    age: 24,
    nationality: 'Italian 🇮🇹',
    stats: { video: 90, photo: 65, editing: 70 },
    wage: 100,
    specialSkill: 'Skate Tricks',
    specialDesc: '+40% fame on dynamic action shots',
    catchphrase: '"Ready for a trick? Hold the camera steady!"',
  },
  chiara: {
    id: 'chiara',
    name: 'CB',
    role: 'Content Assistant / Botanist',
    emoji: '🌿',
    age: 23,
    nationality: 'Italian 🇮🇹',
    stats: { video: 50, photo: 55, editing: 60 },
    wage: 80,
    specialSkill: 'Plant Whisperer',
    specialDesc: 'When present, team mood +5 (green energy)',
    catchphrase: '"These plants give the best vibes for content."',
  },
};

export function getTotalWages() {
  return Object.values(TEAM_MEMBERS).reduce((sum, m) => sum + m.wage, 0);
}

export function payWages(state) {
  const wages = getTotalWages();
  state.money -= wages;
  state.dailyMoneySpent += wages;

  if (state.money < 0) {
    // Can't pay fully
    for (const m of Object.values(state.teamMembers)) {
      m.happiness = clamp(m.happiness - 20, 0, 100);
    }
    state.dailyEvents.push('😤 Team upset: not enough budget for wages!');
    return false;
  }
  return true;
}

export function applyDailyHappinessDecay(state) {
  for (const m of Object.values(state.teamMembers)) {
    m.happiness = clamp(m.happiness - 5, 0, 100);
  }
}

export function boostTeamHappiness(state, amount) {
  for (const m of Object.values(state.teamMembers)) {
    m.happiness = clamp(m.happiness + amount, 0, 100);
  }
}

export function boostMemberHappiness(state, memberId, amount) {
  const m = state.teamMembers[memberId];
  if (m) {
    m.happiness = clamp(m.happiness + amount, 0, 100);
  }
}

export function assignMemberToTask(state, memberId, taskId) {
  const m = state.teamMembers[memberId];
  if (m && m.status === 'available') {
    m.status = 'busy';
    m.assignedTo = taskId;
    return true;
  }
  return false;
}

export function releaseMember(state, memberId) {
  const m = state.teamMembers[memberId];
  if (m) {
    m.status = 'available';
    m.assignedTo = null;
  }
}

export function releaseAll(state) {
  for (const m of Object.values(state.teamMembers)) {
    m.status = 'available';
    m.assignedTo = null;
  }
}

export function getAvailableMembers(state) {
  return Object.entries(state.teamMembers)
    .filter(([id, m]) => m.status === 'available')
    .map(([id]) => id);
}

export function getMemberStatus(member) {
  if (member.happiness < 40) return 'unhappy';
  if (member.status === 'busy') return 'busy';
  return 'available';
}

export function getHappinessColor(happiness) {
  if (happiness >= 70) return '#22c55e';
  if (happiness >= 40) return '#f59e0b';
  return '#ef4444';
}

export function payTeamBonus(state, amount = 100) {
  if (state.money < amount) return false;
  state.money -= amount;
  state.dailyMoneySpent += amount;
  boostTeamHappiness(state, 15);
  state.dailyEvents.push('🎉 Team bonus paid! Everyone is happier.');
  return true;
}

export function checkHypeMember(state) {
  // If any team member with a 'Hype' specialSkill is assigned, boost team
  for (const [id, info] of Object.entries(TEAM_MEMBERS)) {
    if (!info.specialSkill) continue;
    if (info.specialSkill.toLowerCase().includes('hype')) {
      const member = state.teamMembers[id];
      if (member && member.assignedTo) {
        boostTeamHappiness(state, 5);
        return;
      }
    }
  }
}

export function getBestMemberForStat(state, stat) {
  let best = null;
  let bestVal = -1;
  for (const [id, info] of Object.entries(TEAM_MEMBERS)) {
    if (state.teamMembers[id].status !== 'available') continue;
    if (info.stats[stat] > bestVal) {
      bestVal = info.stats[stat];
      best = id;
    }
  }
  return best;
}
