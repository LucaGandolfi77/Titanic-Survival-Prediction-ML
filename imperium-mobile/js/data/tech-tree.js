export const AGES = [
  { id: 0, name: 'Dark Age', emoji: '🌑', requiredPopulation: 0, requiredTechs: [] },
  { id: 1, name: 'Feudal Age', emoji: '🌓', requiredPopulation: 30, requiredTechs: ['architecture'] },
  { id: 2, name: 'Castle Age', emoji: '🌕', requiredPopulation: 60, requiredTechs: ['chemistry'] },
  { id: 3, name: 'Imperial Age', emoji: '👑', requiredPopulation: 100, requiredTechs: ['civilization'] },
];

export const TECHS = {
  architecture: { name: 'Architecture', emoji: '🏗️', ageRequired: 1, effects: { buildingHp: 1.15 }, required: [], cost: { food: 150, wood: 100, gold: 80, stone: 50 } },
  masonry: { name: 'Masonry', emoji: '🧱', ageRequired: 1, effects: { wallHp: 1.2 }, required: ['architecture'], cost: { food: 100, wood: 80, gold: 60, stone: 80 } },
  chemistry: { name: 'Chemistry', emoji: '⚗️', ageRequired: 2, effects: { unitAttack: 1.1 }, required: ['architecture'], cost: { food: 200, wood: 150, gold: 120, stone: 80 } },
  metallurgy: { name: 'Metallurgy', emoji: '🔨', ageRequired: 2, effects: { unitDefense: 1.1 }, required: ['architecture'], cost: { food: 180, wood: 120, gold: 100, stone: 90 } },
  civilization: { name: 'Civilization', emoji: '🏛️', ageRequired: 3, effects: { allBonus: 1.15 }, required: ['chemistry', 'metallurgy'], cost: { food: 400, wood: 300, gold: 250, stone: 200 } },
  farming: { name: 'Advanced Farming', emoji: '🌾', ageRequired: 1, effects: { foodRate: 1.25 }, required: [], cost: { food: 80, wood: 60, gold: 40, stone: 20 } },
  lumber_harvest: { name: 'Lumber Harvest', emoji: '🪵', ageRequired: 1, effects: { woodRate: 1.25 }, required: [], cost: { food: 60, wood: 80, gold: 30, stone: 20 } },
  scale_barding: { name: 'Scale Barding Armor', emoji: '🛡️', ageRequired: 2, effects: { cavalryDefense: 1.2 }, required: ['metallurgy'], cost: { food: 200, wood: 100, gold: 150, stone: 80 } },
  iron_casting: { name: 'Iron Casting', emoji: '⚔️', ageRequired: 2, effects: { infantryAttack: 1.2 }, required: ['chemistry'], cost: { food: 180, wood: 120, gold: 120, stone: 100 } },
  ballistics: { name: 'Ballistics', emoji: '🎯', ageRequired: 3, effects: { rangedAccuracy: 1.2 }, required: ['chemistry', 'metallurgy'], cost: { food: 300, wood: 200, gold: 150, stone: 120 } },
};
