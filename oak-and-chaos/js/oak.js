/* ── js/oak.js ── Oak Tree: growth, stats, evolution ── */

import { clamp, randomChance } from './utils.js';

/* ── DNA UPGRADES DATABASE ── */
export const DNA_UPGRADES = [
  { id: 'super_roots',       name: 'Super Roots',        cost: 5,  desc: '+50% energy generation',          purchased: false },
  { id: 'aphrodisiac_bark',  name: 'Aphrodisiac Bark',   cost: 10, desc: '+30% breeding success rate',      purchased: false },
  { id: 'telepathic_leaves', name: 'Telepathic Leaves',  cost: 15, desc: 'Unlock Taliban communication',    purchased: false },
  { id: 'quantum_acorns',    name: 'Quantum Acorns',     cost: 20, desc: 'Cross-dimensional breeding',      purchased: false },
  { id: 'beard_of_moss',     name: 'Beard of Moss',      cost: 8,  desc: '+20 Charisma (Taliban love it)',   purchased: false },
  { id: 'carnivore_mode',    name: 'Carnivore Mode',     cost: 25, desc: 'Eat failed offspring for energy',  purchased: false },
];

/* ── GROWTH STAGES ── */
const STAGES = [
  { name: 'Sapling',           minH: 0,   cssClass: 'stage-sapling'  },
  { name: 'Young Oak',         minH: 2,   cssClass: 'stage-young'    },
  { name: 'Mature Oak',        minH: 8,   cssClass: 'stage-mature'   },
  { name: 'Ancient Oak',       minH: 20,  cssClass: 'stage-ancient'  },
  { name: 'Cosmic Oak',        minH: 50,  cssClass: 'stage-cosmic'   },
  { name: 'ZARGHUN ASCENDED',  minH: 100, cssClass: 'stage-ascended' },
];

/* ── MILESTONES ── */
const MILESTONES = [
  { height: 5,   name: 'First Branching',   desc: 'Can now breed with animals',       triggered: false },
  { height: 10,  name: 'Teenage Oak',       desc: 'Casino revenue +20%',              triggered: false },
  { height: 20,  name: 'Mature Oak',        desc: 'Unlock Machine 5, breed Taliban',  triggered: false },
  { height: 30,  name: 'Elder Oak',         desc: 'Assign 3 offspring per role',      triggered: false },
  { height: 50,  name: 'Ancient Oak',       desc: 'Unlock Machine 6, cosmic breeding',triggered: false },
  { height: 75,  name: 'Zarghun Awakens',   desc: 'All stats ×2, new abilities',      triggered: false },
  { height: 100, name: '🏆 ZARGHUN ASCENDED', desc: 'YOU WIN!',                       triggered: false },
];

export class OakTree {
  constructor() {
    this.name         = 'Zarghun';
    this.age          = 0;            // years (1 year = 10s real time)
    this.height       = 1.0;          // metres
    this.trunkGirth   = 0.1;
    this.leaves       = 10;
    this.energy       = 100;
    this.maxEnergy    = 100;
    this.fertility    = 10;
    this.charisma     = 5;
    this.dnaPoints    = 0;
    this.acorns       = 0;
    this.generation   = 1;

    this.maxHeight    = 100;
    this.maxLeaves    = 100000;
    this.maxFertility = 100;

    this.isMeditating   = false;
    this._meditateTimer = 0;

    this.upgrades  = DNA_UPGRADES.map(u => ({ ...u }));
    this.milestones = MILESTONES.map(m => ({ ...m }));

    // Multipliers (affected by upgrades / events)
    this.energyGenMul    = 1.0;
    this.breedSuccessMul = 1.0;
    this.growthSpeedMul  = 1.0;

    // Temporary buffs { id, remaining (sec), multiplier }
    this.buffs = [];

    this._prevStage = 'stage-sapling';
  }

  /* ── Getters ── */
  getStage() {
    let stage = STAGES[0];
    for (const s of STAGES) {
      if (this.height >= s.minH) stage = s;
    }
    return stage;
  }

  getStageName() { return this.getStage().name; }
  getStageClass() { return this.getStage().cssClass; }

  canBreedAnimals()  { return this.height >= 5; }
  canBreedTaliban()  { return this.height >= 20; }
  hasWon()           { return this.height >= 100; }

  /* ── Update (called every frame) ── */
  update(delta, sunlight) {
    // Age
    this.age += delta / 10;  // 10 real sec = 1 year

    // Energy generation from leaves
    const baseGen = this.leaves * 0.01 * sunlight;
    const genMul  = this.energyGenMul * this._buffMul('energyGen');
    this.energy = clamp(this.energy + baseGen * genMul * delta, 0, this.maxEnergy);

    // Meditation
    if (this.isMeditating) {
      this._meditateTimer += delta;
      if (this._meditateTimer >= 30) {
        this._meditateTimer -= 30;
        this.dnaPoints += 1;
      }
    }

    // Buff countdown
    this.buffs = this.buffs.filter(b => {
      b.remaining -= delta;
      return b.remaining > 0;
    });

    // Max energy scales with leaves
    this.maxEnergy = clamp(100 + this.leaves * 0.05, 100, 9999);

    // Check milestones
    const newMilestones = [];
    for (const m of this.milestones) {
      if (!m.triggered && this.height >= m.height) {
        m.triggered = true;
        newMilestones.push(m);
      }
    }
    return newMilestones;
  }

  /* ── Actions ── */
  grow() {
    const cost = 50;
    if (this.energy < cost) return false;
    if (this.height >= this.maxHeight) return false;
    this.energy -= cost;

    const mul = this.growthSpeedMul * this._buffMul('growthSpeed');
    this.height    = clamp(this.height + 0.5 * mul, 0, this.maxHeight);
    this.trunkGirth = clamp(this.trunkGirth + 0.05 * mul, 0, 50);
    this.leaves    = clamp(this.leaves + 20, 0, this.maxLeaves);

    // Fertility and charisma grow slowly
    if (randomChance(30)) this.fertility = clamp(this.fertility + 1, 0, this.maxFertility);
    if (randomChance(15)) this.charisma  = clamp(this.charisma + 1, 0, 100);

    return true;
  }

  produceAcorn() {
    const cost = 30;
    if (this.energy < cost) return false;
    this.energy -= cost;
    this.acorns += 1;
    return true;
  }

  meditate() {
    this.isMeditating = !this.isMeditating;
    this._meditateTimer = 0;
  }

  /* ── Upgrades ── */
  purchaseUpgrade(id) {
    const upg = this.upgrades.find(u => u.id === id);
    if (!upg || upg.purchased) return false;
    if (this.dnaPoints < upg.cost) return false;
    this.dnaPoints -= upg.cost;
    upg.purchased = true;

    // Apply effect
    switch (id) {
      case 'super_roots':       this.energyGenMul    *= 1.5; break;
      case 'aphrodisiac_bark':  this.breedSuccessMul *= 1.3; break;
      case 'telepathic_leaves': /* unlock handled elsewhere */ break;
      case 'quantum_acorns':    /* unlock handled elsewhere */ break;
      case 'beard_of_moss':     this.charisma += 20; break;
      case 'carnivore_mode':    /* unlock handled elsewhere */ break;
    }
    return true;
  }

  hasUpgrade(id) {
    const u = this.upgrades.find(u => u.id === id);
    return u ? u.purchased : false;
  }

  /* ── Buffs ── */
  addBuff(id, duration, mul = 1) {
    this.buffs.push({ id, remaining: duration, multiplier: mul });
  }

  _buffMul(id) {
    let m = 1;
    for (const b of this.buffs) {
      if (b.id === id) m *= b.multiplier;
    }
    return m;
  }

  /* ── Eat offspring (Carnivore Mode) ── */
  eatOffspring() {
    if (!this.hasUpgrade('carnivore_mode')) return 0;
    const gained = 80;
    this.energy = clamp(this.energy + gained, 0, this.maxEnergy);
    return gained;
  }

  /* ── Serialisation ── */
  toJSON() {
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

  loadJSON(data) {
    if (!data) return;
    Object.assign(this, data);
    // Restore upgrade objects
    if (data.upgrades) this.upgrades = data.upgrades;
    if (data.milestones) this.milestones = data.milestones;
    this.buffs = [];
  }
}
