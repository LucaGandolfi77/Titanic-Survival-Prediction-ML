/* ── src/oak.ts ── Oak Tree: growth, stats, evolution ── */

import { clamp, randomChance } from './utils';
import type { OakState, DNAUpgrade, Milestone, Buff } from '../types/game';

export const DNA_UPGRADES: DNAUpgrade[] = [
  { id: 'super_roots', name: 'Super Roots', cost: 5, desc: '+50% energy generation', purchased: false },
  { id: 'aphrodisiac_bark', name: 'Aphrodisiac Bark', cost: 10, desc: '+30% breeding success rate', purchased: false },
  { id: 'telepathic_leaves', name: 'Telepathic Leaves', cost: 15, desc: 'Unlock Taliban communication', purchased: false },
  { id: 'quantum_acorns', name: 'Quantum Acorns', cost: 20, desc: 'Cross-dimensional breeding', purchased: false },
  { id: 'beard_of_moss', name: 'Beard of Moss', cost: 8, desc: '+20 Charisma (Taliban love it)', purchased: false },
  { id: 'carnivore_mode', name: 'Carnivore Mode', cost: 25, desc: 'Eat failed offspring for energy', purchased: false },
];

const STAGES = [
  { name: 'Sapling', minH: 0, cssClass: 'stage-sapling' },
  { name: 'Young Oak', minH: 2, cssClass: 'stage-young' },
  { name: 'Mature Oak', minH: 8, cssClass: 'stage-mature' },
  { name: 'Ancient Oak', minH: 20, cssClass: 'stage-ancient' },
  { name: 'Cosmic Oak', minH: 50, cssClass: 'stage-cosmic' },
  { name: 'ZARGHUN ASCENDED', minH: 100, cssClass: 'stage-ascended' },
];

const MILESTONES: Milestone[] = [
  { height: 5, name: 'First Branching', desc: 'Can now breed with animals', triggered: false },
  { height: 10, name: 'Teenage Oak', desc: 'Casino revenue +20%', triggered: false },
  { height: 20, name: 'Mature Oak', desc: 'Unlock Machine 5, breed Taliban', triggered: false },
  { height: 30, name: 'Elder Oak', desc: 'Assign 3 offspring per role', triggered: false },
  { height: 50, name: 'Ancient Oak', desc: 'Unlock Machine 6, cosmic breeding', triggered: false },
  { height: 75, name: 'Zarghun Awakens', desc: 'All stats ×2, new abilities', triggered: false },
  { height: 100, name: '🏆 ZARGHUN ASCENDED', desc: 'YOU WIN!', triggered: false },
];

export class OakTree implements OakState {
  name: string = 'Zarghun';
  age: number = 0;
  height: number = 1.0;
  trunkGirth: number = 0.1;
  leaves: number = 10;
  energy: number = 100;
  maxEnergy: number = 100;
  fertility: number = 10;
  charisma: number = 5;
  dnaPoints: number = 5;
  acorns: number = 1;
  generation: number = 1;
  isMeditating: boolean = false;
  maxHeight: number = 100;
  maxLeaves: number = 100000;
  maxFertility: number = 100;
  isMeditating_: boolean = false;
  _meditateTimer: number = 0;
  upgrades: DNAUpgrade[] = DNA_UPGRADES.map(u => ({ ...u }));
  milestones: Milestone[] = MILESTONES.map(m => ({ ...m }));
  energyGenMul: number = 1.0;
  breedSuccessMul: number = 1.0;
  growthSpeedMul: number = 1.0;
  buffs: Buff[] = [];
  _prevStage: string = 'stage-sapling';

  getStage(): { name: string; minH: number; cssClass: string } {
    let stage = STAGES[0];
    for (const s of STAGES) {
      if (this.height >= s.minH) stage = s;
    }
    return stage;
  }

  getStageName(): string { return this.getStage().name; }
  getStageClass(): string { return this.getStage().cssClass; }

  canBreedAnimals(): boolean { return this.height >= 5; }
  canBreedTaliban(): boolean { return this.height >= 20; }
  hasWon(): boolean { return this.height >= 100; }

  update(delta: number, sunlight: number): Milestone[] {
    this.age += delta / 10;
    const baseGen = this.leaves * 0.01 * sunlight;
    const genMul = this.energyGenMul * this._buffMul('energyGen');
    this.energy = clamp(this.energy + baseGen * genMul * delta, 0, this.maxEnergy);

    if (this.isMeditating) {
      this._meditateTimer += delta;
      if (this._meditateTimer >= 30) {
        this._meditateTimer -= 30;
        this.dnaPoints += 1;
      }
    }

    this.buffs = this.buffs.filter(b => {
      b.remaining -= delta;
      return b.remaining > 0;
    });

    this.maxEnergy = clamp(100 + this.leaves * 0.05, 100, 9999);

    const newMilestones: Milestone[] = [];
    for (const m of this.milestones) {
      if (!m.triggered && this.height >= m.height) {
        m.triggered = true;
        newMilestones.push(m);
      }
    }
    return newMilestones;
  }

  grow(): boolean {
    const cost = 50;
    if (this.energy < cost) return false;
    if (this.height >= this.maxHeight) return false;
    this.energy -= cost;
    const mul = this.growthSpeedMul * this._buffMul('growthSpeed');
    this.height = clamp(this.height + 0.5 * mul, 0, this.maxHeight);
    this.trunkGirth = clamp(this.trunkGirth + 0.05 * mul, 0, 50);
    this.leaves = clamp(this.leaves + 20, 0, this.maxLeaves);
    if (randomChance(30)) this.fertility = clamp(this.fertility + 1, 0, this.maxFertility);
    if (randomChance(15)) this.charisma = clamp(this.charisma + 1, 0, 100);
    return true;
  }

  produceAcorn(): boolean {
    const cost = 30;
    if (this.energy < cost) return false;
    this.energy -= cost;
    this.acorns += 1;
    return true;
  }

  meditate(): void {
    this.isMeditating = !this.isMeditating;
    this._meditateTimer = 0;
  }

  purchaseUpgrade(id: string): boolean {
    const upg = this.upgrades.find(u => u.id === id);
    if (!upg || upg.purchased) return false;
    if (this.dnaPoints < upg.cost) return false;
    this.dnaPoints -= upg.cost;
    upg.purchased = true;
    switch (id) {
      case 'super_roots': this.energyGenMul *= 1.5; break;
      case 'aphrodisiac_bark': this.breedSuccessMul *= 1.3; break;
      case 'telepathic_leaves': break;
      case 'quantum_acorns': break;
      case 'beard_of_moss': this.charisma += 20; break;
      case 'carnivore_mode': break;
    }
    return true;
  }

  hasUpgrade(id: string): boolean {
    const u = this.upgrades.find(u => u.id === id);
    return u ? u.purchased : false;
  }

  addBuff(id: string, duration: number, mul: number = 1): void {
    this.buffs.push({ id, remaining: duration, multiplier: mul });
  }

  private _buffMul(id: string): number {
    let m = 1;
    for (const b of this.buffs) {
      if (b.id === id) m *= b.multiplier;
    }
    return m;
  }

  eatOffspring(): number {
    if (!this.hasUpgrade('carnivore_mode')) return 0;
    const gained = 80;
    this.energy = clamp(this.energy + gained, 0, this.maxEnergy);
    return gained;
  }

  toJSON(): Partial<OakState> {
    return {
      name: this.name, age: this.age, height: this.height,
      trunkGirth: this.trunkGirth, leaves: this.leaves,
      energy: this.energy, maxEnergy: this.maxEnergy,
      fertility: this.fertility, charisma: this.charisma,
      dnaPoints: this.dnaPoints, acorns: this.acorns,
      generation: this.generation, isMeditating: this.isMeditating,
      upgrades: this.upgrades, milestones: this.milestones,
      energyGenMul: this.energyGenMul,
      breedSuccessMul: this.breedSuccessMul,
      growthSpeedMul: this.growthSpeedMul,
    };
  }

  loadJSON(data: Partial<OakState>): void {
    if (!data) return;
    Object.assign(this, data);
    if (data.upgrades) this.upgrades = data.upgrades;
    if (data.milestones) this.milestones = data.milestones;
    this.buffs = [];
  }
}
