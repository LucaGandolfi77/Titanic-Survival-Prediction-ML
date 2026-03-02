/* ===== Sponsor system ===== */
import { GameState } from './state.js';
import { getSquadAvgOverall } from './skaters.js';

export const SPONSORS = [
  {
    id: 'icecraft',
    name: 'IceCraft Skates',
    icon: '⛸️',
    requiredFame: 10,
    weeklyIncome: 800,
    duration: 6,
    description: 'Logo on rink board',
    requirement: null
  },
  {
    id: 'frozenfit',
    name: 'FrozenFit Apparel',
    icon: '👗',
    requiredFame: 20,
    weeklyIncome: 1500,
    duration: 6,
    description: 'Skater uniforms',
    requirement: null
  },
  {
    id: 'coolbreeze',
    name: 'CoolBreeze Energy',
    icon: '🥤',
    requiredFame: 35,
    weeklyIncome: 2200,
    duration: 5,
    description: 'Bonus on high tempo routines',
    requirement: null,
    tempoBonus: true
  },
  {
    id: 'icypeak',
    name: 'IcyPeak Resorts',
    icon: '🏔️',
    requiredFame: 50,
    weeklyIncome: 3000,
    duration: 5,
    description: '+5 fame per competition',
    requirement: null,
    fameBonus: 5
  },
  {
    id: 'arcticgold',
    name: 'ArcticGold Jewelry',
    icon: '💍',
    requiredFame: 65,
    weeklyIncome: 4500,
    duration: 4,
    description: 'Requires avg charisma ≥ 70',
    requirement: { type: 'charisma', min: 70 }
  },
  {
    id: 'novasport',
    name: 'NovaSport Media',
    icon: '📺',
    requiredFame: 80,
    weeklyIncome: 6000,
    duration: 4,
    description: 'Extra points per win',
    requirement: null,
    winPointsBonus: 50
  },
  {
    id: 'quantumice',
    name: 'QuantumIce Tech',
    icon: '🔬',
    requiredFame: 90,
    weeklyIncome: 8000,
    duration: 4,
    description: 'Sync bonus +10%',
    requirement: null,
    syncBonus: 0.10
  },
  {
    id: 'olympicdream',
    name: 'OlympicDream Fund',
    icon: '🏅',
    requiredFame: 100,
    weeklyIncome: 12000,
    duration: 8,
    description: 'Legendary sponsor',
    requirement: { type: 'overall', min: 75 }
  }
];

export function getAvailableSponsors() {
  const activeIds = GameState.activeSponsors.map(a => a.sponsor.id);
  return SPONSORS.filter(sp => {
    if (activeIds.includes(sp.id)) return false;
    return true; // Show all, lock icon if fame too low
  });
}

export function canNegotiate(sponsorId) {
  if (GameState.activeSponsors.length >= GameState.maxSimultaneous) {
    return { ok: false, msg: 'Maximum 3 active sponsors' };
  }
  const sp = SPONSORS.find(s => s.id === sponsorId);
  if (!sp) return { ok: false, msg: 'Unknown sponsor' };
  if (GameState.fame < sp.requiredFame) {
    return { ok: false, msg: `Need fame ≥ ${sp.requiredFame} (current: ${GameState.fame})` };
  }
  const alreadyActive = GameState.activeSponsors.find(a => a.sponsor.id === sponsorId);
  if (alreadyActive) return { ok: false, msg: 'Already have this sponsor' };
  return { ok: true, sponsor: sp };
}

export function negotiateSponsor(sponsorId) {
  const check = canNegotiate(sponsorId);
  if (!check.ok) return check;
  const sp = check.sponsor;
  GameState.activeSponsors.push({
    sponsor: sp,
    weeksRemaining: sp.duration,
    breachCount: 0
  });
  return { ok: true, msg: `Signed deal with ${sp.name}: €${sp.weeklyIncome}/week for ${sp.duration} weeks` };
}

export function processWeeklySponsors() {
  let totalIncome = 0;
  const messages = [];
  const toRemove = [];

  for (let i = 0; i < GameState.activeSponsors.length; i++) {
    const deal = GameState.activeSponsors[i];
    const sp = deal.sponsor;

    // Check requirements
    let breached = false;
    if (GameState.fame < sp.requiredFame) {
      breached = true;
    }
    if (sp.requirement) {
      if (sp.requirement.type === 'charisma') {
        const avgCha = GameState.activeSquad.reduce((s, sk) => s + sk.stats.charisma, 0) / Math.max(1, GameState.activeSquad.length);
        if (avgCha < sp.requirement.min) breached = true;
      } else if (sp.requirement.type === 'overall') {
        const avgOv = getSquadAvgOverall(GameState.activeSquad);
        if (avgOv < sp.requirement.min) breached = true;
      }
    }

    if (breached) {
      deal.breachCount++;
      if (deal.breachCount >= 2) {
        messages.push(`📉 ${sp.name} cancelled deal (requirements not met for 2 weeks)`);
        toRemove.push(i);
        continue;
      } else {
        messages.push(`⚠️ ${sp.name} warning: requirements not met (1/2)`);
      }
    } else {
      deal.breachCount = 0;
    }

    // Pay income
    totalIncome += sp.weeklyIncome;
    deal.weeksRemaining--;

    if (deal.weeksRemaining <= 0) {
      messages.push(`💼 ${sp.name} deal expired`);
      toRemove.push(i);
    }
  }

  // Remove expired/cancelled (reverse order to preserve indices)
  for (let i = toRemove.length - 1; i >= 0; i--) {
    GameState.activeSponsors.splice(toRemove[i], 1);
  }

  GameState.money += totalIncome;
  return { totalIncome, messages };
}

export function getTotalSponsorIncome() {
  return GameState.activeSponsors.reduce((s, d) => s + d.sponsor.weeklyIncome, 0);
}

export function getSyncBonus() {
  const qt = GameState.activeSponsors.find(d => d.sponsor.id === 'quantumice');
  return qt ? qt.sponsor.syncBonus : 0;
}

export function getFameBonus() {
  let bonus = 0;
  for (const deal of GameState.activeSponsors) {
    if (deal.sponsor.fameBonus) bonus += deal.sponsor.fameBonus;
  }
  return bonus;
}

export function getWinPointsBonus() {
  let bonus = 0;
  for (const deal of GameState.activeSponsors) {
    if (deal.sponsor.winPointsBonus) bonus += deal.sponsor.winPointsBonus;
  }
  return bonus;
}
