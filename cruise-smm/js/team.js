/* ===== TEAM MANAGEMENT ===== */
import { getState } from './state.js';
import { clamp } from './utils.js';

export const TEAM_MEMBERS = {
  marco: {
    id: 'marco',
    name: 'Marco Ferrari',
    role: 'Senior Videographer',
    emoji: '🎥',
    age: 34,
    nationality: 'Italian 🇮🇹',
    stats: { video: 95, photo: 60, editing: 80 },
    wage: 120,
    specialSkill: 'Golden Hour Mode',
    specialDesc: '+50% fame on sunset shootings',
    catchphrase: '"This lighting is magnifico, but let me fix the angle."',
  },
  yuki: {
    id: 'yuki',
    name: 'Yuki Tanaka',
    role: 'Lead Photographer',
    emoji: '📷',
    age: 28,
    nationality: 'Japanese 🇯🇵',
    stats: { video: 55, photo: 97, editing: 70 },
    wage: 110,
    specialSkill: 'Macro Vision',
    specialDesc: '+40% fame on close-up shots',
    catchphrase: '"Wait... the shadow is 2mm off. Perfect now."',
  },
  sofia: {
    id: 'sofia',
    name: 'Sofia Andersen',
    role: 'Content Creator / Editor',
    emoji: '✏️',
    age: 25,
    nationality: 'Danish 🇩🇰',
    stats: { video: 65, photo: 70, editing: 92 },
    wage: 90,
    specialSkill: 'Viral Brain',
    specialDesc: '+60% fame on social media tasks',
    catchphrase: '"This is SO going to blow up. Trust me, I know trends."',
  },
  diego: {
    id: 'diego',
    name: 'Diego Morales',
    role: 'Junior Assistant / Sound',
    emoji: '🎙️',
    age: 22,
    nationality: 'Spanish 🇪🇸',
    stats: { video: 50, photo: 50, editing: 45 },
    wage: 70,
    specialSkill: 'Hype Man',
    specialDesc: 'When assigned, team happiness +5',
    catchphrase: '"I will carry ALL the equipment. All of it. I\'m ready."',
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

export function checkDiegoHypeMan(state) {
  const diego = state.teamMembers.diego;
  if (diego.assignedTo) {
    boostTeamHappiness(state, 5);
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
