import { resources } from './resource.js';
import { units, spawnUnit } from './unit.js';
import { mapState } from './map.js';
import { hexToPixelCenter } from './utils/hex-math.js';
import { FACTION_KEYS, FACTIONS } from './data/factions.js';
import { showToast, haptic } from './ui/notifications.js';
import { rand } from './utils/helpers.js';
import { particles } from './render/particles.js';

export const mercenary = {
  costMultiplier: 3,

  canAfford(unitId) {
    const cost = this.getCost(unitId);
    return resources.canAfford(cost);
  },

  getCost(unitId) {
    const mercCosts = {
      militia: { food: 90, wood: 0, gold: 60, stone: 0 },
      archer: { food: 100, wood: 45, gold: 90, stone: 0 },
      skirmisher: { food: 110, wood: 30, gold: 80, stone: 15 },
      knight: { food: 170, wood: 60, gold: 150, stone: 50 },
      cavalry_archer: { food: 160, wood: 75, gold: 130, stone: 40 },
      trebuchet: { food: 240, wood: 180, gold: 300, stone: 240 },
    };
    return mercCosts[unitId] || { food: 100, wood: 50, gold: 80, stone: 20 };
  },

  hire(unitId, factionId) {
    if (!this.canAfford(unitId)) {
      showToast('Not enough resources for mercenary');
      return null;
    }
    const cost = this.getCost(unitId);
    if (!resources.spendCost(cost)) return null;

    const fac = factionId || FACTION_KEYS[rand(0, FACTION_KEYS.length - 1)];
    const q = mapState.playerQ + rand(-8, 8);
    const r = mapState.playerR + rand(-8, 8);
    const tile = mapState.getTile(q, r);
    if (!tile) return null;

    const c = spawnUnit(unitId, q, r, fac);
    if (c) {
      showToast(`🔶 Mercenary ${c.name} hired!`);
      haptic(30);
      for (let i = 0; i < 5; i++) {
        particles.push({
          x: c.x, y: c.y,
          vx: (Math.random() - 0.5) * 4,
          vy: (Math.random() - 0.5) * 4 - 1,
          life: 1.0, decay: 0.02,
          size: 3 + Math.random() * 3,
          color: '#f59e0b',
        });
      }
    }
    return c;
  },
};
