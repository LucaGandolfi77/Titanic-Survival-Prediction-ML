import { FACTION_KEYS, FACTIONS } from './data/factions.js';
import { generateName, rand } from './utils/helpers.js';
import { hexToPixelCenter } from './utils/hex-math.js';
import { mapState } from './map.js';
import { fog } from './fog.js';
import { dna } from './dna-mutation.js';

const citizenNames = ['Bruno', 'Elena', 'Marcus', 'Aria', 'Lucan', 'Sera', 'Draven', 'Nora'];
const traits = ['Brave', 'Lazy', 'Clever', 'Strong', 'Quick', 'Resilient', 'Sharp', 'Steady'];
const skills = ['Gatherer', 'Builder', 'Warrior', 'Scout', 'Farmer', 'Lumberjack'];

export const citizens = [];
export let selectedCitizen = null;
export let faction = 'roma';

function generateCitizenAttributes() {
  const trait = traits[rand(0, traits.length - 1)];
  const skill = skills[rand(0, skills.length - 1)];
  return { trait, skill };
}

export function initCitizens(factionId) {
  faction = factionId || 'roma';
  citizens.length = 0;
  selectedCitizen = null;
  const centerTile = mapState.getTile(mapState.playerQ, mapState.playerR);
  for (let i = 0; i < 10; i++) {
    const c = createCitizen(
      citizenNames[i % citizenNames.length] + ' ' + (i + 1),
      mapState.playerQ + rand(-2, 2),
      mapState.playerR + rand(-2, 2),
      generateCitizenAttributes()
    );
    citizens.push(c);
  }
}

export function createCitizen(name, q, r, attrs) {
  const c = {
    id: Date.now() + '_' + Math.random().toString(36).substr(2, 6),
    name: name || generateName(),
    q, r,
    x: hexToPixelCenter(q, r).x,
    y: hexToPixelCenter(q, r).y,
    targetX: 0,
    targetY: 0,
    moving: false,
    path: [],
    pathIndex: 0,
    speed: 0.02 + Math.random() * 0.01,
    task: null,
    attributes: attrs || generateCitizenAttributes(),
    morale: 80 + rand(0, 20),
    experience: 0,
    level: 1,
    hp: 100,
    maxHp: 100,
    fighting: false,
    color: FACTIONS[faction]?.color || '#8B6914',
    gatherTarget: null,
    buildTarget: null,
    _progress: 0,
    mutations: [],
  };
  const mut = dna.apply(c);
  if (mut) {
    c.mutations.push(mut);
  }
  return c;
}

export function selectCitizen(citizen) {
  selectedCitizen = citizen;
}

export function deselectCitizen() {
  selectedCitizen = null;
}

export function assignTask(citizen, task, targetQ, targetR) {
  citizen.task = task;
  if (targetQ !== undefined && targetR !== undefined) {
    const { x, y } = hexToPixelCenter(targetQ, targetR);
    citizen.targetX = x;
    citizen.targetY = y;
    citizen.moving = true;
    citizen.path = [];
  }
}

export function updateCitizens(dt) {
  for (const c of citizens) {
    if (c.moving) {
      const dx = c.targetX - c.x;
      const dy = c.targetY - c.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 3) {
        c.moving = false;
        const { x: cx, y: cy } = hexToPixelCenter(c.q, c.r);
        if (Math.abs(c.x - cx) < 10 && Math.abs(c.y - cy) < 10) {
          // Position matches hex center
        }
        c.q = Math.round(c.x / 36);
        c.r = Math.round((c.y - c.q * 20.78) / 41.57);
        fog.reveal(c.q, c.r);

        if (c.task === 'gather' && c.gatherTarget) {
          const tile = mapState.getTile(c.gatherTarget.q, c.gatherTarget.r);
          if (tile && tile.resourceAmount > 0) {
            const resourceType = tile.terrain === 'forest' ? 'wood' : tile.terrain === 'hill' ? 'stone' : tile.terrain === 'ruins' ? 'gold' : 'food';
            const amount = rand(2, 5);
            tile.resourceAmount -= amount;
            resources_add(resourceType, amount);
          }
          c.task = null;
          c.gatherTarget = null;
        }
        if (c.task === 'build' && c.buildTarget) {
          c._progress = (c._progress || 0) + dt * 0.5;
          if (c._progress >= 1) {
            c._progress = 0;
            c.task = null;
            c.buildTarget = null;
          }
        }
        continue;
      }
      c.x += (dx / dist) * c.speed * dt * 60;
      c.y += (dy / dist) * c.speed * dt * 60;
    }

    if (c.morale < 100) {
      c.morale = Math.min(100, c.morale + dt * 0.1);
    }

    if (!c.task && !c.moving && Math.random() < 0.01) {
      const tile = mapState.getTile(c.q, c.r);
      if (tile && tile.resourceAmount > 0) {
        const resourceType = tile.terrain === 'forest' ? 'wood' : tile.terrain === 'hill' ? 'stone' : tile.terrain === 'ruins' ? 'gold' : 'food';
        const amount = rand(1, 2);
        tile.resourceAmount -= amount;
        resources_add(resourceType, amount);
        c.morale = Math.min(100, c.morale + 0.5);
      }
    }
  }
}

function resources_add(type, amount) {
  const r = globalResourcesRef;
  if (r && r[type] !== undefined) {
    r[type] += amount;
    if (r.gathered) r.gathered[type] = (r.gathered[type] || 0) + amount;
  }
}

export function autoGather(citizen) {
  const tile = mapState.getTile(citizen.q, citizen.r);
  if (!tile) return;
  if (tile.resourceAmount > 0 && !citizen.task) {
    const resourceType = tile.terrain === 'forest' ? 'wood' : tile.terrain === 'hill' ? 'stone' : tile.terrain === 'ruins' ? 'gold' : 'food';
    const amount = rand(1, 3);
    tile.resourceAmount -= amount;
    resources_add(resourceType, amount);
    citizen.morale = Math.min(100, citizen.morale + 0.5);
  }
}

export function getCitizensAtTile(q, r) {
  return citizens.filter(c => c.q === q && c.r === r);
}

export function getFactionInfo() {
  return FACTIONS[faction];
}

export function setGlobalResources(resRef) {
  globalResourcesRef = resRef;
}

let globalResourcesRef = null;
