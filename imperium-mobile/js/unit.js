import { mapState } from './map.js';
import { hexToPixelCenter } from '../utils/hex-math.js';
import { rand } from '../utils/helpers.js';
import { UNITS } from '../data/units.js';
import { haptic } from './ui/notifications.js';

export const units = [];
export const selectedUnit = null;
export let nextUnitId = 0;

export function spawnUnit(unitId, q, r, faction) {
  const def = UNITS[unitId];
  if (!def) return null;

  const factionData = { roma: '#c4a035', vikings: '#4a7c59', mongols: '#8B6914', egyptians: '#d4a017', celts: '#3a7a3a', japanese: '#c43030' };
  const id = nextUnitId++;
  const unit = {
    id,
    unitId,
    name: def.name,
    emoji: def.emoji,
    q, r,
    x: hexToPixelCenter(q, r).x,
    y: hexToPixelCenter(q, r).y,
    targetX: 0, targetY: 0,
    moving: false, path: [], pathIndex: 0,
    speed: def.speed,
    hp: def.hp, maxHp: def.hp,
    attack: def.attack,
    defense: def.defense,
    range: def.range,
    type: def.type,
    role: def.role,
    faction: faction || 'roma',
    color: factionData[faction] || '#c4a035',
    lastAttackTime: 0,
    attackCooldown: 1.0,
    autoTarget: null,
    alive: true,
    killCount: 0,
  };
  units.push(unit);
  haptic(15);
  return unit;
}

export function killUnit(unit) {
  unit.alive = false;
  const idx = units.indexOf(unit);
  if (idx >= 0) units.splice(idx, 1);
}

export function getUnitsAtTile(q, r) {
  return units.filter(u => u.alive && u.q === q && u.r === r);
}

export function getEnemyUnitsNear(unit, range) {
  return units.filter(u => {
    if (!u.alive || u.faction === unit.faction) return false;
    const dist = Math.sqrt((u.x - unit.x) ** 2 + (u.y - unit.y) ** 2);
    return dist < range * 30;
  });
}

export function assignMove(unit, targetQ, targetR) {
  const path = mapState.findPath(unit.q, unit.r, targetQ, targetR);
  if (path.length > 0) {
    unit.path = path;
    unit.pathIndex = 0;
    unit.moving = true;
    updateUnitTarget(unit);
  }
}

function updateUnitTarget(unit) {
  if (unit.pathIndex < unit.path.length) {
    const next = unit.path[unit.pathIndex];
    unit.targetX = hexToPixelCenter(next.q, next.r).x;
    unit.targetY = hexToPixelCenter(next.q, next.r).y;
  } else {
    unit.moving = false;
  }
}

export function updateUnits(dt, timestamp) {
  for (const u of units) {
    if (!u.alive) continue;

    if (u.moving) {
      const dx = u.targetX - u.x;
      const dy = u.targetY - u.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 2) {
        u.pathIndex++;
        if (unit.pathIndex < unit.path.length) {
          updateUnitTarget(u);
        } else {
          u.moving = false;
          u.q = Math.round(u.x / 24);
          u.r = Math.round(u.y / 24);
        }
      } else {
        const moveSpeed = u.speed * dt * 60;
        u.x += (dx / dist) * moveSpeed;
        u.y += (dy / dist) * moveSpeed;
      }
    }

    if (!u.moving && timestamp - u.lastAttackTime > u.attackCooldown) {
      const enemies = getEnemyUnitsNear(u, u.range > 0 ? u.range : 2);
      if (enemies.length > 0) {
        const target = enemies.reduce((a, b) => a.hp < b.hp ? a : b);
        attackUnit(u, target, timestamp);
      }
    }

    const tile = mapState.getTile(u.q, u.r);
    if (tile) {
      tile.visible = true;
    }
  }

  const dead = units.filter(u => !u.alive);
  for (const d of dead) {
    const idx = units.indexOf(d);
    if (idx >= 0) units.splice(idx, 1);
  }
}

function attackUnit(attacker, target, timestamp) {
  attacker.lastAttackTime = timestamp;
  const damage = Math.max(1, attacker.attack - target.defense * 0.3);
  target.hp -= damage;

  if (target.hp <= 0) {
    killUnit(target);
    attacker.killCount++;
    haptic(30);
  }
}
