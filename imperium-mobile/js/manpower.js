import { resources } from './resource.js';
import { units } from './unit.js';
import { citizens } from './citizen.js';
import { BUILDINGS } from '../data/buildings.js';
import { showToast, haptic } from './ui/notifications.js';

const AGE_MULTIPLIER = [1, 1.5, 2.5, 4];
const AGE_MAX_BASE = [100, 150, 250, 400];

export const manpower = {
  current: 0,
  max: 100,
  regenRate: 0,
  overdraft: false,
  lastWarning: 0,

  init(ageIndex) {
    this.current = 50;
    this.max = AGE_MAX_BASE[ageIndex] || 100;
    this.overdraft = false;
    this.lastWarning = 0;
    this.recalc();
  },

  recalc() {
    const pop = citizens.length;
    const ageMult = AGE_MULTIPLIER[0];
    let buildBonus = 0;
    for (const b of buildings) {
      if (b.type === 'barracks') buildBonus += 2;
      if (b.type === 'stable') buildBonus += 1;
      if (b.type === 'workshop') buildBonus += 1;
    }
    this.regenRate = 1 + pop * 0.1 + buildBonus;
    this.max = AGE_MAX_BASE[0] + citizens.length * 5;
  },

  canAfford(cost) {
    return this.current >= cost;
  },

  spend(cost) {
    if (!this.canAfford(cost)) return false;
    this.current -= cost;
    if (this.current < 0 && !this.overdraft) {
      this.overdraft = true;
      showToast('⚠️ Manpower overdraft! Penalties active');
      haptic(50);
    }
    return true;
  },

  regenerate(dt) {
    this.current = Math.min(this.max, this.current + this.regenRate * dt);
    if (this.current > 0) this.overdraft = false;

    if (this.current < 10 && Date.now() - this.lastWarning > 10000) {
      this.lastWarning = Date.now();
      showToast('⚠️ Low manpower!');
    }
  },

  getAvailable() {
    return Math.max(0, Math.floor(this.max - this.current));
  },

  getUsed() {
    return Math.floor(this.current);
  },

  getMax() {
    return this.max;
  },
};
