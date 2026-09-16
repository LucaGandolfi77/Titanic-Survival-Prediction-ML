import { uid, randomInt } from './utils';
import type { Partner, Offspring } from '../types/game';

interface DefaultPartner extends Partner {
  id: string;
}

const DEFAULT_PARTNERS: Omit<DefaultPartner, 'id'>[] = [
  { name: 'Cactus', emoji: '🌵', category: 'plant', compatibility: 80, speedBonus: 5, charismaBonus: 2, luckBonus: 5, desc: 'Spiny but lovable' },
  { name: 'Sunflower', emoji: '🌻', category: 'plant', compatibility: 85, speedBonus: 8, charismaBonus: 10, luckBonus: 8, desc: 'Glows with warmth' },
  { name: 'Rosebush', emoji: '🌹', category: 'plant', compatibility: 80, speedBonus: 6, charismaBonus: 15, luckBonus: 6, desc: 'Thorny and charming' },
  { name: 'Bamboo', emoji: '🎍', category: 'plant', compatibility: 82, speedBonus: 15, charismaBonus: 3, luckBonus: 5, desc: 'Grows absurdly fast' },
  { name: 'Venus Flytrap', emoji: '🪴', category: 'plant', compatibility: 75, speedBonus: 7, charismaBonus: 1, luckBonus: 12, desc: 'Eats coins from machines' },
  { name: 'Poppy', emoji: '🌸', category: 'plant', compatibility: 78, speedBonus: 6, charismaBonus: 8, luckBonus: 15, desc: 'Produces mysterious resin' },
  { name: 'Bonsai', emoji: '🌳', category: 'plant', compatibility: 88, speedBonus: 3, charismaBonus: 20, luckBonus: 10, desc: 'Tiny but max charisma' },
  { name: 'Baobab', emoji: '🌴', category: 'plant', compatibility: 72, speedBonus: 2, charismaBonus: 5, luckBonus: 5, desc: 'Huge but incredibly slow' },
  { name: 'Goat', emoji: '🐐', category: 'animal', compatibility: 55, speedBonus: 12, charismaBonus: 3, luckBonus: 8, desc: 'Very common, gives milk+leaves' },
  { name: 'Eagle', emoji: '🦅', category: 'animal', compatibility: 50, speedBonus: 20, charismaBonus: 8, luckBonus: 6, desc: 'Offspring can fly and scout' },
  { name: 'Scorpion', emoji: '🦂', category: 'animal', compatibility: 45, speedBonus: 10, charismaBonus: 1, luckBonus: 5, desc: 'Venomous, excellent bouncer' },
  { name: 'Camel', emoji: '🐪', category: 'animal', compatibility: 52, speedBonus: 8, charismaBonus: 5, luckBonus: 7, desc: 'Stores 3× energy capacity' },
  { name: 'Snake', emoji: '🐍', category: 'animal', compatibility: 48, speedBonus: 14, charismaBonus: 12, luckBonus: 10, desc: 'Hypnotises gamblers' },
  { name: 'Donkey', emoji: '🫏', category: 'animal', compatibility: 53, speedBonus: 6, charismaBonus: 2, luckBonus: 4, desc: 'Stubborn but carries 5× loads' },
  { name: 'Pigeon', emoji: '🕊️', category: 'animal', compatibility: 50, speedBonus: 18, charismaBonus: 4, luckBonus: 6, desc: 'Delivers acorns remotely' },
  { name: 'Bear', emoji: '🐻', category: 'animal', compatibility: 42, speedBonus: 8, charismaBonus: 6, luckBonus: 3, desc: 'Huge and intimidating' },
  { name: 'Abdul', emoji: '👳', category: 'taliban', compatibility: 30, speedBonus: 5, charismaBonus: 10, luckBonus: 15, desc: 'Slot Supervisor — born wearing turban' },
  { name: 'Mahmoud', emoji: '🧔', category: 'taliban', compatibility: 32, speedBonus: 8, charismaBonus: 6, luckBonus: 10, desc: 'Mechanic — repairs slot machines' },
  { name: 'Omar', emoji: '👤', category: 'taliban', compatibility: 28, speedBonus: 3, charismaBonus: 8, luckBonus: 12, desc: 'Accountant — tracks finances naturally' },
  { name: 'Tariq', emoji: '💪', category: 'taliban', compatibility: 25, speedBonus: 10, charismaBonus: 3, luckBonus: 8, desc: 'Bodyguard — +intimidation stat' },
  { name: 'Commander Barkat', emoji: '⭐', category: 'taliban', compatibility: 1, speedBonus: 15, charismaBonus: 20, luckBonus: 25, desc: 'LEGENDARY — 1% chance! Offspring is a GENERAL' },
];

export class PopulationManager {
  partners: Partner[] = DEFAULT_PARTNERS.map(p => ({ ...p, id: uid() }));
  offspring: Offspring[] = [];

  getPartnersByType(type?: string): Partner[] {
    if (!type || type === 'all') return this.partners;
    return this.partners.filter(p => p.category === type);
  }

  getPartner(id: string): Partner | undefined {
    return this.partners.find(p => p.id === id);
  }

  assignRole(offspringId: string, roleId: string): void {
    const o = this.offspring.find(c => c.id === offspringId);
    if (o) o.role = roleId;
  }

  removeRole(offspringId: string): void {
    const o = this.offspring.find(c => c.id === offspringId);
    if (o) o.role = null;
  }

  addOffspring(child: Offspring): void {
    this.offspring.push(child);
  }

  removeOffspring(id: string): void {
    this.offspring = this.offspring.filter(o => o.id !== id);
  }

  getByRole(role: string | null): Offspring[] {
    return this.offspring.filter(o => o.role === role);
  }

  getUnassigned(): Offspring[] {
    return this.offspring.filter(o => !o.role);
  }

  updateAges(delta: number): void {
    for (const o of this.offspring) {
      o.age += delta / 10;
    }
  }

  get count(): number { return this.offspring.length; }

  toJSON(): { offspring: Offspring[] } {
    return { offspring: this.offspring };
  }

  loadJSON(data: { offspring?: Offspring[] } | null): void {
    if (!data) return;
    this.offspring = data.offspring || [];
  }
}
