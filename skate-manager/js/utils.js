/* ===== Utility helpers ===== */

let _idCounter = 0;
export function uid() { return 'sk_' + (++_idCounter) + '_' + Math.random().toString(36).slice(2, 7); }

export function randInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
export function randFloat(min, max) { return Math.random() * (max - min) + min; }
export function pick(arr) { return arr[randInt(0, arr.length - 1)]; }
export function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = randInt(0, i);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
export function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
export function lerp(a, b, t) { return a + (b - a) * t; }
export function formatMoney(n) {
  if (n >= 1_000_000) return '€' + (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return '€' + (n / 1_000).toFixed(1) + 'k';
  return '€' + Math.round(n).toLocaleString();
}
export function formatMoneyFull(n) { return '€' + Math.round(n).toLocaleString(); }

// ===== Name generation =====
const FIRST_NAMES = [
  'Elena', 'Yuki', 'Sofia', 'Anna', 'Mei', 'Anya', 'Lucia', 'Hana',
  'Mia', 'Kaito', 'Yuna', 'Sara', 'Irina', 'Chiara', 'Nadia', 'Aiko',
  'Valentina', 'Sakura', 'Olga', 'Mariko', 'Lena', 'Haruka', 'Bianca', 'Nina',
  'Katerina', 'Emilia', 'Suki', 'Rosa', 'Daria', 'Rina', 'Giulia', 'Mika'
];
const LAST_NAMES = [
  'Rossi', 'Ivanova', 'Tanaka', 'Müller', 'Park', 'Kovacs',
  'Ferrari', 'Nakamura', 'Petrov', 'Andersen', 'Kim', 'Bianchi',
  'Sato', 'Volkov', 'Moretti', 'Johansson', 'Suzuki', 'Romano',
  'Kovalenko', 'Yamamoto', 'Colombo', 'Larsson', 'Hayashi', 'Conti'
];
const NATIONALITIES = [
  { flag: '🇮🇹', country: 'Italy' },
  { flag: '🇷🇺', country: 'Russia' },
  { flag: '🇯🇵', country: 'Japan' },
  { flag: '🇩🇪', country: 'Germany' },
  { flag: '🇰🇷', country: 'Korea' },
  { flag: '🇭🇺', country: 'Hungary' },
  { flag: '🇸🇪', country: 'Sweden' },
  { flag: '🇫🇮', country: 'Finland' },
  { flag: '🇨🇦', country: 'Canada' },
  { flag: '🇺🇸', country: 'USA' },
  { flag: '🇫🇷', country: 'France' },
  { flag: '🇨🇳', country: 'China' }
];

export function generateName() {
  return pick(FIRST_NAMES) + ' ' + pick(LAST_NAMES);
}
export function generateNationality() {
  return pick(NATIONALITIES);
}

// ===== Stat generation for tiers =====
export function generateStats(tier) {
  const ranges = {
    1: [30, 50],
    2: [50, 70],
    3: [70, 90],
    4: [85, 99]
  };
  const [lo, hi] = ranges[tier] || ranges[2];
  return {
    technique: randInt(lo, hi),
    stamina:   randInt(lo, hi),
    rhythm:    randInt(lo, hi),
    sync:      randInt(lo, hi),
    charisma:  randInt(lo, hi)
  };
}

export function calcOverall(stats) {
  return Math.round(
    stats.technique * 0.25 +
    stats.stamina   * 0.20 +
    stats.rhythm    * 0.20 +
    stats.sync      * 0.20 +
    stats.charisma  * 0.15
  );
}

export function calcValue(overall, age) {
  const peakBonus = age >= 20 && age <= 28 ? 1.3 : age < 20 ? 1.1 : 0.8;
  return Math.round(overall * overall * 12 * peakBonus);
}

export function calcWage(overall) {
  return Math.round(overall * 22 + 200);
}

export function overallColor(ov) {
  if (ov >= 85) return 'overall-gold';
  if (ov >= 70) return 'overall-green';
  if (ov >= 50) return 'overall-yellow';
  return 'overall-red';
}

export function formDotClass(form) {
  if (form >= 65) return 'high';
  if (form >= 35) return 'mid';
  return 'low';
}

// ===== Weighted random =====
export function weightedRandom(weights) {
  const total = weights.reduce((s, w) => s + w, 0);
  let r = Math.random() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

// ===== Deep clone =====
export function deepClone(obj) { return JSON.parse(JSON.stringify(obj)); }

// ===== Delay helper =====
export function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
