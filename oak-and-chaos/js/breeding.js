/* ── js/breeding.js ── Breeding system: offspring generation ── */

import { randomChance, randomFloat, randomInt, randomFrom, clamp, uid,
         generateOffspringName, generateBio, getOffspringEmoji } from './utils.js';

/* ── TRAIT POOL ── */
const TRAITS = {
  plant: [
    'Photosynthetic', 'Thorny', 'Fragrant', 'Deep-Rooted', 'Fast-Growing',
    'Bioluminescent', 'Venomous Sap', 'Flower Crown', 'Pollen Cloud',
    'Bark Armour', 'Rubber Skin', 'Spore Launcher', 'Symbiotic Fungi',
  ],
  animal: [
    'Fanged', 'Winged', 'Armoured',  'Venomous', 'Night Vision',
    'Echolocation', 'Pack Leader', 'Camouflage', 'Berserker',
    'Milk Producer', 'Webbed Feet', 'Thick Fur', 'Hibernation',
  ],
  taliban: [
    'Born Manager', 'Beard of Wisdom', 'Intimidating Stare', 'Accountant Mind',
    'Natural Mechanic', 'Commanding Presence', 'Turban of Power', 'Poker Face',
    'Haggling Expert', 'Night Watchman', 'Strategic Thinker', 'Loyal to Oak',
  ],
};

/* ── ROLES ── */
export const ROLES = [
  { id: 'dealer',    name: 'Casino Dealer',    desc: '+15% casino revenue',            icon: '🎰' },
  { id: 'bodyguard', name: 'Slot Bodyguard',   desc: '-30% machine breakdown chance',  icon: '💪' },
  { id: 'gardener',  name: 'Gardener',         desc: '+10% oak energy generation',     icon: '🌱' },
  { id: 'scout',     name: 'Scout',            desc: 'Reveals random events earlier',  icon: '🔭' },
  { id: 'breeder',   name: 'Breeder',          desc: 'Proxy breeding for oak',         icon: '💕' },
  { id: 'merchant',  name: 'Acorn Merchant',   desc: 'Passive DNA point income',       icon: '🌰' },
];

/* ══════════════════════════════════════════════════════════ */
export class BreedingLab {
  constructor() {
    this.breedCooldown = 0;
    this.breedSuccessBoost = 0;      // temporary boost (events)
    this._boostTimer = 0;
  }

  /* ── Attempt to breed oak with a partner ── */
  attemptBreed(oak, partner) {
    // Cost check
    if (oak.energy < 40) return { success: false, reason: 'Not enough energy (need 40⚡)' };
    if (oak.acorns < 1) return { success: false, reason: 'Need at least 1 acorn 🌰' };

    // Height lock check
    if (partner.category === 'animal' && !oak.canBreedAnimals()) {
      return { success: false, reason: 'Need height ≥ 5m to breed with animals' };
    }
    if (partner.category === 'taliban' && !oak.canBreedTaliban()) {
      return { success: false, reason: 'Need height ≥ 20m to breed with Taliban' };
    }

    // Spend
    oak.energy -= 40;
    oak.acorns -= 1;

    // Success roll
    const successRate = this.calculateSuccess(oak, partner);
    if (randomChance(successRate)) {
      const offspring = this.generateOffspring(oak, partner);
      return { success: true, offspring };
    }

    return { success: false, reason: 'Breeding failed! The acorn refused to germinate.' };
  }

  /* ── Calculate success chance ── */
  calculateSuccess(oak, partner) {
    let base = partner.compatibility;
    base += (oak.fertility / oak.maxFertility) * 20; // up to +20%
    base *= oak.breedSuccessMul;
    base += this.breedSuccessBoost;
    return clamp(base, 5, 98);
  }

  /* ── Generate offspring ── */
  generateOffspring(oak, partner) {
    const name = generateOffspringName('oak', partner.category);
    const emoji = getOffspringEmoji(partner.category, partner.emoji);
    const bio = generateBio();

    // Blend stats with random mutation ±20%
    const mutate = (base) => Math.round(base * randomFloat(0.8, 1.2));

    const stats = {
      health:    mutate(50 + oak.height * 0.5),
      energy:    mutate(40 + oak.leaves * 0.01),
      strength:  mutate(10 + oak.trunkGirth * 0.5),
      charisma:  mutate(oak.charisma * 0.5 + (partner.charismaBonus || 0)),
      speed:     mutate(partner.speedBonus || 10),
      luck:      mutate(partner.luckBonus || 5),
    };

    // Pick 1-3 random traits from the partner's pool
    const traitPool = TRAITS[partner.category] || TRAITS.plant;
    const traitCount = randomInt(1, 3);
    const traits = [];
    const usedIdx = new Set();
    for (let i = 0; i < traitCount; i++) {
      let idx;
      do { idx = randomInt(0, traitPool.length - 1); } while (usedIdx.has(idx) && usedIdx.size < traitPool.length);
      usedIdx.add(idx);
      traits.push(traitPool[idx]);
    }

    return {
      id: uid(),
      name,
      type: `Oak-${partner.name}`,
      category: partner.category,
      emoji,
      generation: oak.generation + 1,
      age: 0,
      stats,
      traits,
      role: null,
      description: bio,
      bornAt: Date.now(),
    };
  }

  /* ── Assign role ── */
  assignRole(offspring, roleId) {
    const role = ROLES.find(r => r.id === roleId);
    if (!role) return false;
    offspring.role = roleId;
    return true;
  }

  /* ── Update (timers) ── */
  update(delta) {
    if (this._boostTimer > 0) {
      this._boostTimer -= delta;
      if (this._boostTimer <= 0) {
        this.breedSuccessBoost = 0;
      }
    }
  }

  addSuccessBoost(amount, duration) {
    this.breedSuccessBoost = amount;
    this._boostTimer = duration;
  }
}
