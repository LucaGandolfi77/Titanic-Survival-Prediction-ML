// Level configurations — JSON-driven level definitions
// Edit this file or use the Level Editor to create custom levels

export const DEFAULT_LEVELS = [
  {
    level: 1,
    name: "First Delivery",
    description: "Learn the basics. Deliver 2 packages.",
    numCells: 3,
    deliveriesTarget: 2,
    timeLimit: 90,
    pointsMultiplier: 1,
    availableCells: [0, 1, 2],
    portalSpeed: 1.0,
    fogDensity: 0.015,
  },
  {
    level: 2,
    name: "Getting Hot",
    description: "3 deliveries across 4 cells. Time pressure rising.",
    numCells: 4,
    deliveriesTarget: 3,
    timeLimit: 80,
    pointsMultiplier: 1.2,
    availableCells: [0, 1, 2, 3],
    portalSpeed: 1.0,
    fogDensity: 0.018,
  },
  {
    level: 3,
    name: "Maze Runner",
    description: "5 deliveries in 5 cells. The hypercube gets larger.",
    numCells: 5,
    deliveriesTarget: 5,
    timeLimit: 75,
    pointsMultiplier: 1.5,
    availableCells: [0, 1, 2, 3, 4],
    portalSpeed: 1.1,
    fogDensity: 0.02,
  },
  {
    level: 4,
    name: "Portal Dash",
    description: "4 deliveries, faster portals, thicker fog.",
    numCells: 6,
    deliveriesTarget: 4,
    timeLimit: 70,
    pointsMultiplier: 1.8,
    availableCells: [0, 1, 2, 3, 4, 5],
    portalSpeed: 1.2,
    fogDensity: 0.025,
  },
  {
    level: 5,
    name: "Tesseract Madness",
    description: "6 deliveries in 6 cells. Master the 4D.",
    numCells: 6,
    deliveriesTarget: 6,
    timeLimit: 65,
    pointsMultiplier: 2.0,
    availableCells: [0, 1, 2, 3, 4, 5],
    portalSpeed: 1.3,
    fogDensity: 0.03,
  },
  {
    level: 6,
    name: "Void Walker",
    description: "3 deliveries in 7 cells. The Void awaits.",
    numCells: 7,
    deliveriesTarget: 3,
    timeLimit: 60,
    pointsMultiplier: 2.5,
    availableCells: [0, 1, 2, 3, 4, 5, 6],
    portalSpeed: 1.3,
    fogDensity: 0.035,
  },
  {
    level: 7,
    name: "Full Hypercube",
    description: "All 8 cells unlocked. Deliver 8 packages.",
    numCells: 8,
    deliveriesTarget: 8,
    timeLimit: 60,
    pointsMultiplier: 3.0,
    availableCells: [0, 1, 2, 3, 4, 5, 6, 7],
    portalSpeed: 1.4,
    fogDensity: 0.04,
  },
  {
    level: 8,
    name: "The Void Protocol",
    description: "Maximum difficulty. Only the best survive.",
    numCells: 8,
    deliveriesTarget: 10,
    timeLimit: 50,
    pointsMultiplier: 4.0,
    availableCells: [0, 1, 2, 3, 4, 5, 6, 7],
    portalSpeed: 1.5,
    fogDensity: 0.05,
  },
];

export function getLevelConfig(level) {
  const custom = loadCustomLevel(level);
  if (custom) return custom;
  const def = DEFAULT_LEVELS.find(l => l.level === level);
  if (def) return def;
  return generateDynamicLevel(level);
}

function generateDynamicLevel(level) {
  const numCells = Math.min(8, 2 + level * 2);
  return {
    level,
    name: `Dimension ${level}`,
    description: `Procedural challenge — ${numCells} cells`,
    numCells,
    deliveriesTarget: Math.min(10, 2 + Math.floor(level / 2)),
    timeLimit: Math.max(30, 90 - (level * 5)),
    pointsMultiplier: 1 + (level * 0.15),
    availableCells: Array.from({length: numCells}, (_, i) => i),
    portalSpeed: 1 + (level * 0.05),
    fogDensity: 0.015 + (level * 0.003),
  };
}

function loadCustomLevel(level) {
  try {
    const customs = JSON.parse(localStorage.getItem('hds_custom_levels') || '[]');
    return customs.find(l => l.level === level) || null;
  } catch {
    return null;
  }
}

export function saveCustomLevel(config) {
  try {
    const customs = JSON.parse(localStorage.getItem('hds_custom_levels') || '[]');
    const idx = customs.findIndex(l => l.level === config.level);
    if (idx >= 0) customs[idx] = config;
    else customs.push(config);
    localStorage.setItem('hds_custom_levels', JSON.stringify(customs));
    return true;
  } catch {
    return false;
  }
}

export function deleteCustomLevel(level) {
  try {
    const customs = JSON.parse(localStorage.getItem('hds_custom_levels') || '[]');
    const filtered = customs.filter(l => l.level !== level);
    localStorage.setItem('hds_custom_levels', JSON.stringify(filtered));
    return true;
  } catch {
    return false;
  }
}

export function exportLevelConfig(config) {
  return JSON.stringify(config, null, 2);
}

export function importLevelConfig(jsonStr) {
  try {
    const config = JSON.parse(jsonStr);
    if (!config.level || !config.numCells) return null;
    return config;
  } catch {
    return null;
  }
}
