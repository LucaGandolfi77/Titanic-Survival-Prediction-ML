import { randomChance, randomFloat, randomInt, randomFrom, clamp, uid, generateOffspringName, generateBio, getOffspringEmoji } from './utils';
import type { OakTree } from './oak';
import type { Partner, Offspring } from '../types/game';

export const ROLES = [
  { id: 'dealer', name: 'Casino Dealer', desc: '+15% casino revenue', icon: '🎰', bonus: 'Revenue +15%' },
  { id: 'bodyguard', name: 'Slot Bodyguard', desc: '-30% machine breakdown chance', icon: '💪', bonus: 'Breakdown -30%' },
  { id: 'gardener', name: 'Gardener', desc: '+10% oak energy generation', icon: '🌱', bonus: 'Energy +10%' },
  { id: 'scout', name: 'Scout', desc: 'Reveals random events earlier', icon: '🔭', bonus: 'Events earlier' },
  { id: 'breeder', name: 'Breeder', desc: 'Proxy breeding for oak', icon: '💕', bonus: 'Proxy breed' },
  { id: 'merchant', name: 'Acorn Merchant', desc: 'Passive DNA point income', icon: '🌰', bonus: 'DNA income' },
];

const TRAITS: Record<string, string[]> = {
  plant: [
    'Photosynthetic', 'Thorny', 'Fragrant', 'Deep-Rooted', 'Fast-Growing',
    'Bioluminescent', 'Venomous Sap', 'Flower Crown', 'Pollen Cloud',
    'Bark Armour', 'Rubber Skin', 'Spore Launcher', 'Symbiotic Fungi',
  ],
  animal: [
    'Fanged', 'Winged', 'Armoured', 'Venomous', 'Night Vision',
    'Echolocation', 'Pack Leader', 'Camouflage', 'Berserker',
    'Milk Producer', 'Webbed Feet', 'Thick Fur', 'Hibernation',
  ],
  taliban: [
    'Born Manager', 'Beard of Wisdom', 'Intimidating Stare', 'Accountant Mind',
    'Natural Mechanic', 'Commanding Presence', 'Turban of Power', 'Poker Face',
    'Haggling Expert', 'Night Watchman', 'Strategic Thinker', 'Loyal to Oak',
  ],
};

export interface BreedResult {
  success: boolean;
  offspring?: Offspring;
  reason?: string;
}

export class BreedingLab {
  breedCooldown: number = 0;
  breedSuccessBoost: number = 0;
  _boostTimer: number = 0;

  attemptBreed(oak: OakTree, partner: Partner): BreedResult {
    if (oak.energy < 40) {
      return { success: false, reason: 'Not enough energy (need 40⚡)' };
    }
    if (oak.acorns < 1) {
      return { success: false, reason: 'Need at least 1 acorn 🌰' };
    }
    if (partner.category === 'animal' && !oak.canBreedAnimals()) {
      return { success: false, reason: 'Need height ≥ 5m to breed with animals' };
    }
    if (partner.category === 'taliban' && !oak.canBreedTaliban()) {
      return { success: false, reason: 'Need height ≥ 20m to breed with Taliban' };
    }
    oak.energy -= 40;
    oak.acorns -= 1;
    const successRate = this.calculateSuccess(oak, partner);
    const roll = Math.random() * 100;
    if (randomChance(successRate)) {
      const offspring = this.generateOffspring(oak, partner);
      return { success: true, offspring };
    }
    return { success: false, reason: 'Breeding failed! The acorn refused to germinate.' };
  }

  calculateSuccess(oak: OakTree, partner: Partner): number {
    let base = partner.compatibility;
    base += (oak.fertility / oak.maxFertility) * 20;
    base *= oak.breedSuccessMul;
    base += this.breedSuccessBoost;
    return clamp(base, 5, 98);
  }

  generateOffspring(oak: OakTree, partner: Partner): Offspring {
    const name = generateOffspringName('oak', partner.category);
    const emoji = getOffspringEmoji(partner.category, partner.emoji);
    const bio = generateBio();
    const mutate = (base: number): number => Math.round(base * randomFloat(0.8, 1.2));
    const stats = {
      health: mutate(50 + oak.height * 0.5),
      energy: mutate(40 + oak.leaves * 0.01),
      strength: mutate(10 + oak.trunkGirth * 0.5),
      charisma: mutate(oak.charisma * 0.5 + (partner.charismaBonus || 0)),
      speed: mutate(partner.speedBonus || 10),
      luck: mutate(partner.luckBonus || 5),
    };
    const traitPool = TRAITS[partner.category] || TRAITS.plant;
    const traitCount = randomInt(1, 3);
    const traits: string[] = [];
    const usedIdx = new Set<number>();
    for (let i = 0; i < traitCount; i++) {
      let idx: number;
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

  assignRole(offspring: Offspring, roleId: string): boolean {
    const role = ROLES.find(r => r.id === roleId);
    if (!role) return false;
    offspring.role = roleId;
    return true;
  }

  update(delta: number): void {
    if (this._boostTimer > 0) {
      this._boostTimer -= delta;
      if (this._boostTimer <= 0) {
        this.breedSuccessBoost = 0;
      }
    }
  }

  addSuccessBoost(amount: number, duration: number): void {
    this.breedSuccessBoost = amount;
    this._boostTimer = duration;
  }
}
