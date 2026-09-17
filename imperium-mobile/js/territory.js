import { mapState } from './map.js';
import { buildings, getConstructedBuildings } from './building.js';
import { showToast } from './ui/notifications.js';

const TERRITORY_RADIUS = 5;
const TERRITORY_BONUS = 0.1;

export const territory = {
  tiles: new Set(),
  lastClaimTime: 0,
  claimDelay: 2,

  init() {
    this.tiles.clear();
  },

  update(dt) {
    this.recalculate();
  },

  recalculate() {
    const newTiles = new Set();
    for (const b of getConstructedBuildings()) {
      if (!b.constructed) continue;
      for (let dq = -TERRITORY_RADIUS; dq <= TERRITORY_RADIUS; dq++) {
        for (let dr = -TERRITORY_RADIUS; dr <= TERRITORY_RADIUS; dr++) {
          if (dq * dq + dr * dr > TERRITORY_RADIUS * TERRITORY_RADIUS) continue;
          const q = b.q + dq;
          const r = b.r + dr;
          if (mapState.getTile(q, r) && mapState.getTile(q, r).terrain !== 'water') {
            newTiles.add(`${q},${r}`);
          }
        }
      }
    }
    if (newTiles.size > this.tiles.size) {
      this.tiles = newTiles;
    } else if (newTiles.size < this.tiles.size * 0.5) {
      this.tiles = newTiles;
      showToast('🏚️ Territory lost!');
    }
  },

  getBonus() {
    return 1 + (this.tiles.size / 100) * TERRITORY_BONUS;
  },

  getSize() {
    return this.tiles.size;
  },

  isClaimed(q, r) {
    return this.tiles.has(`${q},${r}`);
  },
};
