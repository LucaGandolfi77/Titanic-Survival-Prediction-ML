import { FACTION_KEYS, FACTIONS } from './data/factions.js';
import { rand } from './utils/helpers.js';
import { hexToPixelCenter } from './utils/hex-math.js';
import { mapState } from './map.js';
import { units } from './unit.js';
import { resources } from './resource.js';
import { buildings, BUILDINGS } from './building.js';
import { UNITS } from './data/units.js';
import { showToast, haptic } from './ui/notifications.js';
import { hero } from './hero.js';
import { territory } from './territory.js';
import { weather } from './weather.js';

export const ai = {
  active: false,
  faction: null,
  difficulty: 'normal',
  baseQ: 0,
  baseR: 0,
  buildTimer: 0,
  trainTimer: 0,
  attackTimer: 0,
  resources: { food: 500, wood: 400, gold: 300, stone: 200 },
  mercenaryTimer: 0,

  start(factionId, difficulty) {
    this.active = true;
    this.faction = factionId || FACTION_KEYS[Math.floor(Math.random() * FACTION_KEYS.length)];
    this.difficulty = difficulty || 'normal';
    this.buildTimer = 0; this.trainTimer = 0; this.attackTimer = 0; this.mercenaryTimer = 0;
    this.resources = { food: 500, wood: 400, gold: 300, stone: 200 };
    this.baseQ = mapState.playerQ + 15 + rand(-5, 5);
    this.baseR = mapState.playerR - 12 + rand(-4, 4);
    showToast(FACTIONS[this.faction].emoji + ' ' + FACTIONS[this.faction].name + ' has appeared!');
    haptic(100);
  },

  update(dt, allUnits) {
    if (!this.active) return;
    const speedMult = { easy: 0.5, normal: 1, hard: 1.5, brutal: 2 }[this.difficulty] || 1;
    this.buildTimer += dt * speedMult;
    this.trainTimer += dt * speedMult;
    this.attackTimer += dt * speedMult;
    this.mercenaryTimer += dt * speedMult;
    if (this.buildTimer > 15) { this.buildTimer = 0; this.aiBuild(); }
    if (this.trainTimer > 10) { this.trainTimer = 0; this.aiTrain(); }
    if (this.attackTimer > 25 / speedMult) { this.attackTimer = 0; this.aiAttack(allUnits); }
    if (this.mercenaryTimer > 40 / speedMult) { this.mercenaryTimer = 0; this.aiHireMercenary(); }
    if (this.difficulty === 'brutal' && Math.random() < 0.01) { this.aiHireHero(); }
  },

  aiHireHero() {
    if (hero.active) return;
    if (hero.canHire()) {
      hero.hire(this.faction);
      showToast(FACTIONS[this.faction].emoji + ' hired a Hero!');
    }
  },

  aiBuild() {
    const types = ['house', 'lumber_camp', 'farm', 'wall', 'barracks'];
    const type = types[rand(0, types.length - 1)];
    const def = BUILDINGS[type];
    if (!def || !this.canAfford(def.cost)) return;
    for (let i = 0; i < 20; i++) {
      const bq = this.baseQ + rand(-4, 4);
      const br = this.baseR + rand(-4, 4);
      const tile = mapState.getTile(bq, br);
      if (tile && !tile.building && tile.terrain !== 'water') {
        this.spendCost(def.cost);
        const b = {
          id: Date.now() + '_ai', type, q: bq, r: br,
          hp: def.hp, maxHp: def.hp, faction: this.faction,
          constructed: true, productionTimer: 0,
          emoji: def.emoji, name: def.name,
        };
        buildings.push(b);
        tile.building = b;
        return;
      }
    }
  },

  aiTrain() {
    const available = ['militia', 'archer', 'skirmisher'];
    if (this.difficulty === 'hard' || this.difficulty === 'brutal') {
      available.push('knight', 'cavalry_archer');
    }
    if (this.difficulty === 'brutal') {
      available.push('trebuchet');
    }
    const unitId = available[rand(0, available.length - 1)];
    const def = UNITS[unitId];
    if (!def || !this.canAfford(def.cost)) return;
    if (this.difficulty === 'easy' && Math.random() < 0.3) return;
    this.spendCost(def.cost);
    spawnUnit(unitId, this.baseQ + rand(-1, 1), this.baseR + rand(-1, 1), this.faction);
  },

  aiHireMercenary() {
    if (this.difficulty === 'easy') return;
    const mercIds = ['militia', 'archer', 'skirmisher'];
    if (this.difficulty === 'hard' || this.difficulty === 'brutal') {
      mercIds.push('knight');
    }
    const unitId = mercIds[rand(0, mercIds.length - 1)];
    const mercCost = {
      militia: { food: 90, wood: 0, gold: 60, stone: 0 },
      archer: { food: 100, wood: 45, gold: 90, stone: 0 },
      skirmisher: { food: 110, wood: 30, gold: 80, stone: 15 },
      knight: { food: 170, wood: 60, gold: 150, stone: 50 },
    };
    const cost = mercCost[unitId];
    if (!cost || !this.canAfford(cost)) return;
    this.spendCost(cost);
    spawnUnit(unitId, this.baseQ + rand(-3, 3), this.baseR + rand(-3, 3), this.faction);
  },

  aiAttack(allUnits) {
    const myUnits = allUnits.filter(u => u.alive && u.faction === this.faction);
    const enemyUnits = allUnits.filter(u => u.alive && u.faction !== this.faction);
    if (myUnits.length === 0 || enemyUnits.length === 0) return;

    const targetFaction = this.chooseTargetFaction();
    const targets = enemyUnits.filter(u => u.faction === targetFaction);
    if (targets.length === 0) return;

    const attacker = myUnits[0];
    const target = targets.reduce((a, b) => a.hp < b.hp ? a : b);
    attacker.autoTarget = target;
    const path = mapState.findPath(attacker.q, attacker.r, target.q, target.r);
    if (path.length > 0) {
      attacker.path = path;
      attacker.pathIndex = 0;
      attacker.moving = true;
      const next = path[0];
      attacker.targetX = hexToPixelCenter(next.q, next.r).x;
      attacker.targetY = hexToPixelCenter(next.q, next.r).y;
    }
  },

  chooseTargetFaction() {
    const relations = {
      roma: { vikings: 'at_war', mongols: 'tense', egyptians: 'neutral', celts: 'friendly', japanese: 'neutral' },
      vikings: { roma: 'at_war', mongols: 'neutral', egyptians: 'friendly', celts: 'neutral', japanese: 'tense' },
    };
    const rel = relations[this.faction];
    if (!rel) return 'roma';

    const atWar = Object.entries(rel).filter(([, v]) => v === 'at_war').map(([k]) => k);
    const tense = Object.entries(rel).filter(([, v]) => v === 'tense').map(([k]) => k);

    if (atWar.length > 0 && Math.random() < 0.7) return atWar[0];
    if (tense.length > 0 && Math.random() < 0.4) return tense[0];

    const others = FACTION_KEYS.filter(k => k !== this.faction);
    return others[rand(0, others.length - 1)];
  },

  canAfford(cost) {
    for (const [type, amount] of Object.entries(cost)) {
      if ((this.resources[type] || 0) < amount) return false;
    }
    return true;
  },

  spendCost(cost) {
    for (const [type, amount] of Object.entries(cost)) {
      this.resources[type] -= amount;
    }
  },
};
