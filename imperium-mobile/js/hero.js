import { rand } from '../utils/helpers.js';
import { resources } from './resource.js';
import { units, spawnUnit } from './unit.js';
import { hexToPixelCenter } from '../utils/hex-math.js';
import { FACTION_KEYS, FACTIONS } from '../data/factions.js';
import { showToast, haptic } from './ui/notifications.js';
import { mapState } from './map.js';

const HERO_COST = { food: 100, wood: 50, gold: 200, stone: 50 };
const HERO_BASE_STATS = { hp: 300, attack: 30, defense: 20, speed: 3.0 };
const XP_PER_LEVEL = 100;
const MAX_LEVEL = 10;
const HERO_ABILITIES = {
  3: { name: 'War Cry', desc: '+25% nearby ally damage', emoji: '🔊' },
  5: { name: 'Regenerate', desc: 'Heals 5% HP per second', emoji: '💚' },
  8: { name: 'Command', desc: '+50% nearby ally speed', emoji: '👑' },
  10: { name: 'Legendary', desc: 'Immune to debuffs, +100% all stats', emoji: '⭐' },
};

export const hero = {
  active: null,
  xp: 0,
  level: 1,
  selectedFaction: 'roma',

  canHire() {
    if (this.active) return false;
    return resources.canAfford(HERO_COST);
  },

  hire(factionId) {
    if (!this.canHire()) {
      showToast('Not enough gold for hero (200g)');
      return false;
    }
    resources.spendCost(HERO_COST);
    const fac = factionId || FACTION_KEYS[Math.floor(Math.random() * FACTION_KEYS.length)];
    const q = mapState.playerQ + rand(-3, 3);
    const r = mapState.playerR + rand(-3, 3);
    const unit = spawnUnit('militia', q, r, fac);
    if (!unit) return false;
    unit.isHero = true;
    unit.name = FACTIONS[fac].emoji + ' Hero';
    unit.level = 1;
    unit.xp = 0;
    unit.maxHp = HERO_BASE_STATS.hp;
    unit.hp = HERO_BASE_STATS.hp;
    unit.attack = HERO_BASE_STATS.attack;
    unit.defense = HERO_BASE_STATS.defense;
    unit.speed = HERO_BASE_STATS.speed;
    unit.abilities = [];
    this.active = unit;
    this.xp = 0;
    this.level = 1;
    this.selectedFaction = fac;
    showToast(`⚔️ Hero hired! Level ${this.level}`);
    haptic(50);
    return true;
  },

  gainXP(amount) {
    if (!this.active) return;
    this.xp += amount;
    while (this.xp >= XP_PER_LEVEL * this.level && this.level < MAX_LEVEL) {
      this.xp -= XP_PER_LEVEL * this.level;
      this.level++;
      this._levelUp();
    }
  },

  _levelUp() {
    const bonus = 1 + (this.level * 0.1);
    this.active.maxHp = Math.floor(HERO_BASE_STATS.hp * bonus);
    this.active.hp = this.active.maxHp;
    this.active.attack = Math.floor(HERO_BASE_STATS.attack * bonus);
    this.active.defense = Math.floor(HERO_BASE_STATS.defense * bonus);
    this.active.speed = HERO_BASE_STATS.speed * (1 + this.level * 0.05);
    this._updateAbilities();
    haptic(60);
    showToast(`⬆️ Hero leveled up to ${this.level}!`);
  },

  _updateAbilities() {
    this.active.abilities = [];
    for (const [lvl, ability] of Object.entries(HERO_ABILITIES)) {
      if (this.level >= parseInt(lvl)) {
        this.active.abilities.push(ability);
      }
    }
  },

  update(dt) {
    if (!this.active) return;
    if (this.active.hp <= 0) {
      showToast('💀 Hero fallen! Hire a new one.');
      this.active = null;
      this.level = 1;
      this.xp = 0;
    }
    const regenLevel = this.active.abilities.find(a => a.name === 'Regenerate');
    if (regenLevel && this.active.hp > 0 && this.active.hp < this.active.maxHp) {
      this.active.hp = Math.min(this.active.maxHp, this.active.hp + this.active.maxHp * 0.05 * dt);
    }
  },

  getStats() {
    if (!this.active) return null;
    return {
      unit: this.active,
      level: this.level,
      xp: this.xp,
      xpNext: XP_PER_LEVEL * this.level,
      abilities: this.active.abilities || [],
    };
  },
};
