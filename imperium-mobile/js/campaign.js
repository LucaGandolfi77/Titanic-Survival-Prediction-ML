import { resources } from './resource.js';
import { buildings } from './building.js';
import { units } from './unit.js';
import { citizens } from './citizen.js';
import { territory } from './territory.js';
import { ai } from './ai.js';
import { mapState } from './map.js';
import { FACTION_KEYS } from './data/factions.js';

export const CAMPAIGN_MISSIONS = [
  {
    id: 0,
    name: 'The Awakening',
    age: 0,
    description: 'Gather food and build houses to grow your settlement.',
    objectives: [
      { type: 'resource', resource: 'food', target: 500, label: 'Gather 500 food' },
      { type: 'building', building: 'house', target: 3, label: 'Build 3 houses' },
    ],
  },
  {
    id: 1,
    name: 'The Warband',
    age: 0,
    description: 'Train militia to defend your territory.',
    objectives: [
      { type: 'unit', unit: 'militia', target: 5, label: 'Train 5 militia' },
      { type: 'resource', resource: 'gold', target: 200, label: 'Gather 200 gold' },
    ],
  },
  {
    id: 2,
    name: 'The Siege',
    age: 1,
    description: 'Capture an enemy castle by force.',
    objectives: [
      { type: 'destroy_enemy_tc', target: 1, label: 'Destroy enemy town center' },
      { type: 'building', building: 'barracks', target: 1, label: 'Build a barracks' },
    ],
  },
  {
    id: 3,
    name: 'The Trade Route',
    age: 1,
    description: 'Build markets and amass gold through trade.',
    objectives: [
      { type: 'building', building: 'market', target: 2, label: 'Build 2 markets' },
      { type: 'resource', resource: 'gold', target: 1000, label: 'Gather 1000 gold' },
    ],
  },
  {
    id: 4,
    name: 'The Fortress',
    age: 2,
    description: 'Build towers and withstand enemy waves.',
    objectives: [
      { type: 'building', building: 'tower', target: 3, label: 'Build 3 towers' },
      { type: 'resource', resource: 'stone', target: 500, label: 'Gather 500 stone' },
    ],
  },
  {
    id: 5,
    name: 'The Crusade',
    age: 2,
    description: 'Lead a large battle with AI allies at your side.',
    objectives: [
      { type: 'unit', unit: 'knight', target: 5, label: 'Train 5 knights' },
      { type: 'resource', resource: 'food', target: 1500, label: 'Gather 1500 food' },
    ],
  },
  {
    id: 6,
    name: 'The Empire',
    age: 3,
    description: 'Expand and control the majority of the map.',
    objectives: [
      { type: 'territory', target: 60, label: 'Control 60% of map tiles' },
      { type: 'unit', unit: 'trebuchet', target: 2, label: 'Train 2 trebuchets' },
    ],
  },
  {
    id: 7,
    name: 'The Final War',
    age: 3,
    description: 'Defeat all rival factions to unite the land.',
    objectives: [
      { type: 'factions_remaining', target: 1, label: 'Be the last faction standing' },
      { type: 'resource', resource: 'gold', target: 5000, label: 'Amass 5000 gold' },
    ],
  },
];

export const CAMPAIGN_KEY = 'imperium_campaign_progress';

export function loadCampaignState() {
  try {
    const raw = localStorage.getItem(CAMPAIGN_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) { /* ignore */ }
  return { completed: [], currentMission: -1, unlocked: 1 };
}

export function saveCampaignState(state) {
  try { localStorage.setItem(CAMPAIGN_KEY, JSON.stringify(state)); } catch (e) { /* ignore */ }
}

export function getMissionObjectives(missionId) {
  const mission = CAMPAIGN_MISSIONS.find(m => m.id === missionId);
  return mission ? mission.objectives : [];
}

export function getMissionProgress(missionId) {
  const objectives = getMissionObjectives(missionId);
  if (!objectives.length) return [];
  return objectives.map(obj => {
    let current = 0;
    if (obj.type === 'resource') {
      current = Math.floor(resources[obj.resource] || 0);
    } else if (obj.type === 'building') {
      current = buildings.filter(b => b.type === obj.building && b.constructed).length;
    } else if (obj.type === 'unit') {
      current = units.filter(u => u.type === obj.unit && u.alive !== false).length;
    } else if (obj.type === 'territory') {
      current = territory.getSize();
    } else if (obj.type === 'factions_remaining') {
      current = getFactionsRemaining();
    } else if (obj.type === 'destroy_enemy_tc') {
      current = countEnemyTownCenters();
    }
    const complete = current >= obj.target;
    return { ...obj, current, complete };
  });
}

function getFactionsRemaining() {
  const alive = new Set();
  for (const u of units) {
    if (u.alive && u.faction !== 'roma') alive.add(u.faction);
  }
  for (const b of buildings) {
    if (b.constructed && b.faction !== 'roma') alive.add(b.faction);
  }
  if (ai.active && ai.faction) alive.add(ai.faction);
  return alive.size + 1;
}

function countEnemyTownCenters() {
  let count = 0;
  for (const b of buildings) {
    if (b.type === 'town_center' && b.faction !== 'roma' && b.constructed) count++;
  }
  return count;
}

export function checkCampaignWin(missionId) {
  const progress = getMissionProgress(missionId);
  return progress.length > 0 && progress.every(p => p.complete);
}

export function checkCampaignLose() {
  const aliveCitizens = citizens.length;
  const aliveUnits = units.filter(u => u.alive).length;
  if (aliveCitizens <= 0 && aliveUnits <= 0) return true;
  const totalRes = resources.food + resources.wood + resources.gold + resources.stone;
  if (totalRes <= 0 && aliveUnits <= 0) return true;
  return false;
}

export function completeMission(missionId) {
  const state = loadCampaignState();
  if (!state.completed.includes(missionId)) {
    state.completed.push(missionId);
  }
  state.currentMission = missionId;
  if (missionId + 1 >= state.unlocked && missionId + 1 < CAMPAIGN_MISSIONS.length) {
    state.unlocked = missionId + 2;
  }
  saveCampaignState(state);
}
