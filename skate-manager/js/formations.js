/* ===== Formation definitions ===== */

export const FORMATIONS = [
  {
    id: 'star',
    name: 'STAR',
    emoji: '⭐',
    difficulty: 1.0,
    unlockFame: 0,
    positions: (() => {
      const pts = [];
      for (let i = 0; i < 16; i++) {
        const angle = (i / 16) * Math.PI * 2 - Math.PI / 2;
        const r = i % 2 === 0 ? 0.40 : 0.18;
        pts.push({ x: 0.5 + r * Math.cos(angle), y: 0.5 + r * Math.sin(angle) });
      }
      return pts;
    })()
  },
  {
    id: 'circle',
    name: 'CIRCLE',
    emoji: '🔵',
    difficulty: 1.2,
    unlockFame: 0,
    positions: Array.from({ length: 16 }, (_, i) => ({
      x: 0.5 + 0.38 * Math.cos(i * Math.PI * 2 / 16),
      y: 0.5 + 0.38 * Math.sin(i * Math.PI * 2 / 16)
    }))
  },
  {
    id: 'cross',
    name: 'CROSS',
    emoji: '➕',
    difficulty: 1.4,
    unlockFame: 5,
    positions: (() => {
      const pts = [];
      // Vertical line (8 skaters)
      for (let i = 0; i < 8; i++) {
        pts.push({ x: 0.5, y: 0.1 + i * 0.1 });
      }
      // Horizontal line (8 skaters, skip center)
      const hPositions = [0.12, 0.22, 0.32, 0.42, 0.58, 0.68, 0.78, 0.88];
      for (let i = 0; i < 8; i++) {
        pts.push({ x: hPositions[i], y: 0.5 });
      }
      return pts.slice(0, 16);
    })()
  },
  {
    id: 'snowflake',
    name: 'SNOWFLAKE',
    emoji: '❄️',
    difficulty: 1.5,
    unlockFame: 10,
    positions: (() => {
      const pts = [];
      // 8 outer points
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2 - Math.PI / 2;
        pts.push({ x: 0.5 + 0.40 * Math.cos(angle), y: 0.5 + 0.40 * Math.sin(angle) });
      }
      // 8 inner points (offset by half step)
      for (let i = 0; i < 8; i++) {
        const angle = ((i + 0.5) / 8) * Math.PI * 2 - Math.PI / 2;
        pts.push({ x: 0.5 + 0.20 * Math.cos(angle), y: 0.5 + 0.20 * Math.sin(angle) });
      }
      return pts;
    })()
  },
  {
    id: 'pyramid',
    name: 'PYRAMID',
    emoji: '🏠',
    difficulty: 1.6,
    unlockFame: 15,
    positions: (() => {
      // Rows: 1, 3, 5, 7 = 16
      const pts = [];
      const rows = [1, 3, 5, 7];
      let yPos = 0.15;
      for (const count of rows) {
        const startX = 0.5 - (count - 1) * 0.08;
        for (let i = 0; i < count; i++) {
          pts.push({ x: startX + i * 0.16, y: yPos });
        }
        yPos += 0.22;
      }
      return pts;
    })()
  },
  {
    id: 'diamond',
    name: 'DIAMOND GRID',
    emoji: '💎',
    difficulty: 1.8,
    unlockFame: 25,
    positions: (() => {
      const pts = [];
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
          const x = 0.25 + col * 0.17;
          const y = 0.20 + row * 0.18;
          // Offset odd rows for diamond look
          const offset = row % 2 === 0 ? 0 : 0.085;
          pts.push({ x: x + offset, y });
        }
      }
      return pts;
    })()
  },
  {
    id: 'spiral',
    name: 'SPIRAL',
    emoji: '🌀',
    difficulty: 2.0,
    unlockFame: 35,
    positions: Array.from({ length: 16 }, (_, i) => {
      const angle = (i / 16) * Math.PI * 4; // 2 full revolutions
      const r = 0.08 + i * 0.022;
      return { x: 0.5 + r * Math.cos(angle), y: 0.5 + r * Math.sin(angle) };
    })
  },
  {
    id: 'butterfly',
    name: 'BUTTERFLY',
    emoji: '🦋',
    difficulty: 2.2,
    unlockFame: 45,
    positions: (() => {
      const pts = [];
      // Left wing (8 skaters)
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI - Math.PI / 2;
        const r = 0.25 + 0.1 * Math.sin(angle * 2);
        pts.push({ x: 0.3 + r * Math.cos(angle) * 0.8, y: 0.5 + r * Math.sin(angle) });
      }
      // Right wing (8 skaters, mirrored)
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI - Math.PI / 2;
        const r = 0.25 + 0.1 * Math.sin(angle * 2);
        pts.push({ x: 0.7 - r * Math.cos(angle) * 0.8, y: 0.5 + r * Math.sin(angle) });
      }
      return pts;
    })()
  },
  {
    id: 'wave',
    name: 'WAVE',
    emoji: '🌊',
    difficulty: 2.5,
    unlockFame: 55,
    positions: Array.from({ length: 16 }, (_, i) => ({
      x: 0.08 + i * 0.055,
      y: 0.5 + 0.25 * Math.sin((i / 16) * Math.PI * 3)
    }))
  },
  {
    id: 'crown',
    name: 'CROWN ROYAL',
    emoji: '👑',
    difficulty: 3.0,
    unlockFame: 60,
    positions: (() => {
      const pts = [];
      // Base (6 skaters, horizontal line)
      for (let i = 0; i < 6; i++) {
        pts.push({ x: 0.2 + i * 0.12, y: 0.75 });
      }
      // Crown peaks (5 points)
      const peakX = [0.2, 0.35, 0.5, 0.65, 0.8];
      const peakY = [0.35, 0.20, 0.10, 0.20, 0.35];
      for (let i = 0; i < 5; i++) {
        pts.push({ x: peakX[i], y: peakY[i] });
      }
      // Crown valleys (5 skaters between peaks)
      for (let i = 0; i < 5; i++) {
        pts.push({ x: 0.275 + i * 0.12, y: 0.45 });
      }
      return pts.slice(0, 16);
    })()
  }
];

export function getFormation(id) {
  return FORMATIONS.find(f => f.id === id) || FORMATIONS[0];
}

export function getAvailableFormations(fame) {
  return FORMATIONS.filter(f => fame >= f.unlockFame);
}

export function getLockedFormations(fame) {
  return FORMATIONS.filter(f => fame < f.unlockFame);
}
