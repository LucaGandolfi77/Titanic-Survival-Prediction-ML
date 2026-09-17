export const MAP_SIZE = { width: 40, height: 30 };

export const TERRAIN = {
  grass: { emoji: '🟩', color: '#4a8022', movementCost: 1, resource: null },
  forest: { emoji: '🌲', color: '#2d5016', movementCost: 2, resource: 'wood', amount: 80 },
  hill: { emoji: '⛰️', color: '#7a6a50', movementCost: 2, resource: 'stone', amount: 60 },
  water: { emoji: '🌊', color: '#1e5fa8', movementCost: 99, resource: null },
  sand: { emoji: '🏖️', color: '#c4a868', movementCost: 1, resource: null },
  swamp: { emoji: '🌿', color: '#3a6030', movementCost: 3, resource: 'food', amount: 40 },
  plain: { emoji: '🟩', color: '#5a9030', movementCost: 1, resource: 'food', amount: 50 },
  ruins: { emoji: '🏚️', color: '#6a6a6a', movementCost: 1, resource: 'gold', amount: 40 },
};

export function generateMap(seed) {
  const map = [];
  const rng = seededRandom(seed || Date.now());

  for (let r = 0; r < MAP_SIZE.height; r++) {
    map[r] = [];
    for (let q = 0; q < MAP_SIZE.width; q++) {
      const terrainKey = getTerrainType(q, r, rng);
      map[r][q] = {
        q,
        r,
        terrain: terrainKey,
        ...TERRAIN[terrainKey],
        building: null,
        resourceAmount: TERRAIN[terrainKey].resource ? TERRAIN[terrainKey].amount : 0,
        explored: false,
        visible: false,
        owner: null,
      };
    }
  }

  const centerQ = Math.floor(MAP_SIZE.width / 2);
  const centerR = Math.floor(MAP_SIZE.height / 2);

  for (let i = 0; i < 6; i++) {
    const [nq, nr] = hexNeighbor(centerQ, centerR, i);
    if (map[nr] && map[nr][nq]) {
      map[nr][nq].terrain = 'plain';
      map[nr][nq].color = '#5a9030';
      map[nr][nq].resource = 'food';
      map[nr][nq].resourceAmount = 60;
    }
  }

  return map;
}

function getTerrainType(q, r, rng) {
  const xRatio = q / MAP_SIZE.width;
  const yRatio = r / MAP_SIZE.height;
  const noise = seededRandom(q * 7 + r * 13 + seedOffset(q, r));

  if ((q <= 2 || q >= MAP_SIZE.width - 3) && (r <= 2 || r >= MAP_SIZE.height - 3)) {
    return 'water';
  }

  if (noise < 0.15) return 'water';
  if (noise < 0.25) return 'sand';
  if (noise < 0.40) return 'forest';
  if (noise < 0.48) return 'hill';
  if (noise < 0.58) return 'swamp';
  if (noise < 0.72) return 'plain';
  if (noise < 0.82) return 'ruins';
  return 'grass';
}

function seedOffset(q, r) {
  return Math.sin(q * 0.1 + r * 0.07) * 1000;
}

function seededRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export function exploreMap(map, playerQ, playerR, radius) {
  for (let dr = -radius; dr <= radius; dr++) {
    for (let dq = -radius; dq <= radius; dq++) {
      const dist = Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr);
      if (dist / 2 > radius) continue;
      const nq = playerQ + dq;
      const nr = playerR + dr;
      if (nr >= 0 && nr < MAP_SIZE.height && nq >= 0 && nq < MAP_SIZE.width) {
        map[nr][nq].explored = true;
      }
    }
  }
}
