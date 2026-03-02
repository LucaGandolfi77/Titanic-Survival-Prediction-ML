/* ===== ECONOMY SYSTEM ===== */
import { getState, getDifficultyMultiplier } from './state.js';
import { clamp, formatMoney } from './utils.js';

export const EQUIPMENT = [
  { id: 'proCamera',      name: 'Pro Camera Kit',     icon: '📷', price: 400, effect: 'Photo fame +30%',         effectDesc: 'photo fame +30%' },
  { id: 'cinemaLens',     name: 'Cinema Lens Set',    icon: '🎬', price: 600, effect: 'Video fame +40%',         effectDesc: 'video fame +40%' },
  { id: 'ledPanel',       name: 'LED Light Panel',    icon: '💡', price: 250, effect: 'Indoor shooting fame +25%', effectDesc: 'indoor fame +25%' },
  { id: 'audioInterface', name: 'Audio Interface',    icon: '🎚️', price: 300, effect: 'Interview fame +20%',     effectDesc: 'interview fame +20%' },
  { id: 'socialSuite',    name: 'Social Media Suite', icon: '📱', price: 350, effect: 'Social tasks fame +35%',  effectDesc: 'social fame +35%' },
  { id: 'drone',          name: 'Drone',              icon: '🚁', price: 800, effect: 'Aerial shots +80% fame, new location', effectDesc: 'aerial +80%, new location', unlockDay: 4 },
];

export function getAvailableEquipment(state) {
  return EQUIPMENT.filter(e => {
    if (state.equipment[e.id]) return false;
    if (e.unlockDay && state.day < e.unlockDay) return false;
    return true;
  });
}

export function buyEquipment(state, equipId) {
  const eq = EQUIPMENT.find(e => e.id === equipId);
  if (!eq) return false;
  if (state.equipment[equipId]) return false;
  if (state.money < eq.price) return false;
  if (eq.unlockDay && state.day < eq.unlockDay) return false;

  state.money -= eq.price;
  state.dailyMoneySpent += eq.price;
  state.equipment[equipId] = true;
  return true;
}

export function addFame(state, amount) {
  const rounded = Math.round(amount);
  state.fame += rounded;
  state.dailyFame += rounded;
  checkMilestones(state);
  return rounded;
}

export function addMoney(state, amount) {
  const rounded = Math.round(amount);
  state.money += rounded;
  state.dailyMoneyEarned += rounded;
  return rounded;
}

export function spendMoney(state, amount) {
  if (state.money < amount) return false;
  state.money -= amount;
  state.dailyMoneySpent += amount;
  return true;
}

export function checkMilestones(state) {
  const milestones = [
    { threshold: 10000,   key: '10000',   emoji: '⭐' },
    { threshold: 50000,   key: '50000',   emoji: '🌟' },
    { threshold: 100000,  key: '100000',  emoji: '💫' },
    { threshold: 500000,  key: '500000',  emoji: '🔥' },
    { threshold: 1000000, key: '1000000', emoji: '👑' },
  ];

  const newMilestones = [];
  for (const m of milestones) {
    if (state.fame >= m.threshold && !state.milestones[m.key]) {
      state.milestones[m.key] = true;
      newMilestones.push(m);
    }
  }
  return newMilestones;
}

export function getDayEndBonus(state) {
  const bonuses = [];

  if (state.day === 1 && state.fame > 5000) {
    bonuses.push({ type: 'money', amount: 200, text: 'Day 1 milestone bonus: +€200!' });
  }
  if (state.day === 3 && state.fame > 50000) {
    bonuses.push({ type: 'unlock', text: 'VIP Shoot location unlocked!' });
  }
  if (state.day === 5 && state.fame > 200000) {
    bonuses.push({ type: 'unlock', text: 'Captain interview unlocked early!' });
    state.captainTrust = Math.max(state.captainTrust, 80);
  }
  if (state.day === 7 && state.fame > 500000) {
    bonuses.push({ type: 'money', amount: 2000, text: 'Sponsor bonus: +€2,000!' });
  }

  for (const b of bonuses) {
    if (b.type === 'money') {
      state.money += b.amount;
      state.dailyMoneyEarned += b.amount;
    }
  }

  // Character daily gifts
  for (const [id, cs] of Object.entries(state.characters)) {
    if (cs.love >= 60 && cs.met) {
      const giftMoney = Math.round(30 + Math.random() * 50);
      state.money += giftMoney;
      state.dailyMoneyEarned += giftMoney;
      bonuses.push({ type: 'gift', text: `💝 ${id} sent you €${giftMoney}!` });
    }
    if (cs.love >= 81 && cs.met) {
      const giftFame = Math.round(100 + Math.random() * 200);
      state.fame += giftFame;
      state.dailyFame += giftFame;
      bonuses.push({ type: 'gift', text: `⭐ ${id}'s daily fame boost: +${giftFame}!` });
    }
  }

  return bonuses;
}

export function getEndingTier(fame) {
  if (fame >= 1000000) return { tier: 5, title: '👑 ONE MILLION!', text: "You're internet famous! The cruise line offers you a permanent position as head of global content." };
  if (fame >= 800000)  return { tier: 4, title: '🌟 AMAZING!', text: "You're the ship's new head of content. The Aurora Infinita has never looked better." };
  if (fame >= 500000)  return { tier: 3, title: '🚢 GREAT CRUISE!', text: "You're rehired for next season! The team celebrates with champagne." };
  if (fame >= 200000)  return { tier: 2, title: '📊 DECENT JOB', text: "The PR team was... grateful. You did okay, but there's room to grow." };
  return { tier: 1, title: '❌ FIRED', text: "The ship sailed without your content making any impact. Better luck next time." };
}

export function getRomanceEnding(charId, love) {
  if (love < 40) return { level: 'none', text: 'You made some friends. Nothing more. Maybe next cruise...' };
  if (love < 71) return { level: 'friend', text: 'A lovely friendship you\'ll always remember. The ship\'s crew waves goodbye as you disembark.' };
  return { level: 'romance', text: '' }; // Full romance ending from character data
}
