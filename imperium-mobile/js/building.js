import { mapState } from './map.js';
import { hexToPixelCenter } from '../utils/hex-math.js';
import { BUILDINGS } from '../data/buildings.js';
import { haptic } from './ui/notifications.js';
import { fog } from './fog.js';

export const buildings = [];

export function placeBuilding(type, q, r, faction) {
  const def = BUILDINGS[type];
  if (!def) return null;
  if (mapState.getTile(q, r).building) return null;

  const b = {
    id: Date.now() + '_' + Math.random().toString(36).substr(2, 6),
    type,
    q, r,
    x: hexToPixelCenter(q, r).x,
    y: hexToPixelCenter(q, r).y,
    hp: def.hp,
    maxHp: def.hp,
    faction: faction || 'roma',
    progress: 1,
    constructed: true,
    productionTimer: 0,
    emoji: def.emoji,
    name: def.name,
    color: def.color,
    size: def.size,
  };
  buildings.push(b);
  mapState.getTile(q, r).building = b;
  fog.reveal(q, r);
  haptic(20);
  return b;
}

export function removeBuilding(q, r) {
  const tile = mapState.getTile(q, r);
  if (!tile || !tile.building) return false;
  const idx = buildings.indexOf(tile.building);
  if (idx >= 0) buildings.splice(idx, 1);
  tile.building = null;
  return true;
}

export function getBuildingAt(q, r) {
  const tile = mapState.getTile(q, r);
  return tile ? tile.building : null;
}

export function updateBuildings(dt) {
  let productionResult = null;
  for (const b of buildings) {
    if (b.constructed) {
      const def = BUILDINGS[b.type];
      if (def && def.produces) {
        b.productionTimer += dt;
        const productionInterval = 4;
        if (b.productionTimer >= productionInterval) {
          b.productionTimer = 0;
          productionResult = { type: def.produces.type, amount: def.produces.rate, building: b };
        }
      }
    }
  }
  return productionResult;
}

export function getConstructedBuildings() {
  return buildings.filter(b => b.constructed);
}
