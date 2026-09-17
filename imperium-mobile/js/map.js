import { generateMap, MAP_SIZE, exploreMap } from './data/map-templates.js';
import { hexToPixelCenter, hexDistance, pixelToHex } from '../utils/hex-math.js';
import { Storage } from './utils/storage.js';
import { fog } from './fog.js';

export const mapState = {
  tiles: null,
  playerQ: 0,
  playerR: 0,
  selectedTiles: [],
  constructionSites: [],

  async init(seed) {
    this.tiles = generateMap(seed);
    this.playerQ = Math.floor(MAP_SIZE.width / 2);
    this.playerR = Math.floor(MAP_SIZE.height / 2);
    this.explore();
    await Storage.save('mapSeed', seed);
  },

  getTile(q, r) {
    if (r >= 0 && r < MAP_SIZE.height && q >= 0 && q < MAP_SIZE.width) {
      return this.tiles[r][q];
    }
    return null;
  },

  explore() {
    exploreMap(this.tiles, this.playerQ, this.playerR, 6);
    fog.revealArea(this.playerQ, this.playerR, 6);
  },

  isWalkable(q, r) {
    const tile = this.getTile(q, r);
    if (!tile) return false;
    if (tile.terrain === 'water') return false;
    return true;
  },

  findPath(startQ, startR, endQ, endR) {
    if (!this.isWalkable(endQ, endR)) return [];
    const open = new Map();
    const closed = new Set();
    const key = (q, r) => `${q},${r}`;
    const startK = key(startQ, startR);
    const endK = key(endQ, endR);
    open.set(startK, { q: startQ, r: startR, g: 0, f: hexDistance(startQ, startR, endQ, endR), parent: null });

    let iterations = 0;
    while (open.size > 0 && iterations < 500) {
      iterations++;
      let bestK = null;
      let bestF = Infinity;
      for (const [k, node] of open) {
        if (node.f < bestF) { bestF = node.f; bestK = k; }
      }
      const current = open.get(bestK);
      if (bestK === endK) {
        const path = [];
        let node = current;
        while (node) {
          path.unshift({ q: node.q, r: node.r });
          node = node.parent;
        }
        return path;
      }
      open.delete(bestK);
      closed.add(bestK);
      const neighbors = [
        [current.q + 1, current.r],
        [current.q - 1, current.r],
        [current.q, current.r + 1],
        [current.q, current.r - 1],
        [current.q + 1, current.r - 1],
        [current.q - 1, current.r + 1],
      ];
      for (const [nq, nr] of neighbors) {
        const nk = key(nq, nr);
        if (closed.has(nk)) continue;
        if (!this.isWalkable(nq, nr)) continue;
        const g = current.g + 1;
        const existing = open.get(nk);
        if (!existing || g < existing.g) {
          open.set(nk, { q: nq, r: nr, g, f: g + hexDistance(nq, nr, endQ, endR), parent: current });
        }
      }
    }
    return [];
  },
};
