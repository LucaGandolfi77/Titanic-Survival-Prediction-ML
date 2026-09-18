import { units, spawnUnit, getEnemyUnitsNear, updateUnits } from './unit.js';
import { mapState } from './map.js';
import { resources } from './resource.js';
import { haptic } from './ui/notifications.js';
import { FACTIONS } from './data/factions.js';
import { hero } from './hero.js';

export const combat = {
  running: false,
  waveTimer: 0,
  aiAttackTimer: 0,

  update(dt) {
    this.waveTimer += dt;
    this.aiAttackTimer += dt;
    this.running = units.length > 0;
  },

  triggerCombat(attacker, target) {
    const damage = Math.max(1, attacker.attack - target.defense * 0.3);
    target.hp -= damage;
    haptic(20);
    if (target.hp <= 0) {
      target.alive = false;
      haptic(50);
      if (attacker.isHero) {
        hero.gainXP(15);
      }
      return true;
    }
    return false;
  },

  spawnEnemyWave(factionId) {
    const faction = FACTIONS[factionId || 'vikings'];
    const q = mapState.playerQ + 8 + Math.floor(Math.random() * 5);
    const r = mapState.playerR - 8 - Math.floor(Math.random() * 5);
    if (!mapState.getTile(q, r)) return;
    spawnUnit('militia', q, r, factionId || 'vikings');
    spawnUnit('militia', q + 1, r, factionId || 'vikings');
    spawnUnit('archer', q, r + 1, factionId || 'vikings');
    haptic(40);
  },

  spawnAIAttack(factionId) {
    const faction = FACTIONS[factionId || 'vikings'];
    const q = mapState.playerQ + 10;
    const r = mapState.playerR - 5;
    if (!mapState.getTile(q, r)) return;
    spawnUnit('militia', q, r, factionId || 'vikings');
    spawnUnit('archer', q + 1, r, factionId || 'vikings');
    spawnUnit('skirmisher', q, r - 1, factionId || 'vikings');
    haptic(60);
  },
};
