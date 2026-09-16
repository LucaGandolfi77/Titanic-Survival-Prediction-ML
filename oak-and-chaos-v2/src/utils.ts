/* ── src/utils.ts ── RNG, helpers, name generator ── */

/* ── Seeded RNG (Mulberry32) — reproducible sequences ── */
export class SeededRNG {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed | 0;
  }

  next(): number {
    let t = (this.seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(min: number, max: number): number {
    return this.next() * (max - min) + min;
  }

  int(min: number, max: number): number {
    return Math.floor(this.range(min, max + 1));
  }

  chance(percent: number): boolean {
    return this.next() * 100 < percent;
  }

  from<T>(arr: T[]): T {
    return arr[this.int(0, arr.length - 1)];
  }

  static fromDate(): SeededRNG {
    return new SeededRNG(Date.now() | 0);
  }

  static fromGameState(oak: { height: number; age: number }, casino: { totalCoins: number }): SeededRNG {
    const seed = (oak.height * 1000 + oak.age * 100 + casino.totalCoins) | 0;
    return new SeededRNG(seed);
  }
}

export function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function randomChance(percent: number): boolean {
  return Math.random() * 100 < percent;
}

export function randomFrom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function weightedRandom<T>(items: T[], weights: number[]): T {
  const total = weights.reduce((s, w) => s + w, 0);
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

export function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e4) return (n / 1e3).toFixed(1) + 'K';
  return Math.floor(n).toLocaleString();
}

/* ── Unique ID generator ── */
let _idCounter = 0;
export function uid(): string {
  return `id_${Date.now().toString(36)}_${(++_idCounter).toString(36)}`;
}

/* ── Name Generator ── */
const AFGHAN_SYLLABLES: string[] = [
  'Zar', 'Gul', 'Shah', 'Khan', 'Din', 'War', 'Mor',
  'Bak', 'Tar', 'Nur', 'Jam', 'Raf', 'Hal', 'Sar', 'Nas',
  'Abb', 'Mah', 'Yas', 'Fer', 'Qas', 'Sal', 'Rez', 'Hom'
];

const PLANT_SUFFIXES: string[] = [
  'oak', 'leaf', 'root', 'bark', 'branch', 'thorn', 'bloom',
  'seed', 'moss', 'vine', 'fern', 'bud', 'stump', 'petal'
];

const TITLES: string[] = [
  'The Thorny', 'Al-Photosynth', 'ibn Acorn', 'The Rooted',
  'The Branching', 'Al-Verdant', 'The Mossy', 'ibn Zarghun',
  'The Evergreen', 'Al-Sappy', 'The Leafy', 'The Barky',
  'The Cosmic', 'Al-Canopy', 'The Sprout', 'ibn Chlorophyll',
  'The Ancient', 'The Twisted', 'Al-Humus', 'The Pollinated'
];

const ANIMAL_NAMES: string[] = [
  'Hooves', 'Claws', 'Fangs', 'Wings', 'Scales', 'Horns',
  'Talons', 'Tusks', 'Quills', 'Feathers', 'Paws', 'Jaws'
];

export function generateOffspringName(parent1Type: string, parent2Type: string): string {
  const syl = randomFrom(AFGHAN_SYLLABLES);
  let suffix: string;
  if (parent2Type === 'animal') {
    suffix = randomFrom(ANIMAL_NAMES).toLowerCase();
  } else {
    suffix = randomFrom(PLANT_SUFFIXES);
  }
  const title = randomFrom(TITLES);
  return `${syl}${suffix} ${title}`;
}

const BIO_TEMPLATES: string[] = [
  'Born screaming photosynthesis into the void.',
  'Has an inexplicable talent for counting casino chips.',
  'Smells faintly of acorns and regret.',
  'Once tried to grow a beard. Failed beautifully.',
  'Believes deeply in the power of roots.',
  'Was offered a slot machine, chose a sunbeam instead.',
  'Can recite the Quran AND the periodic table. In bark.',
  'Their leaves whisper secrets to the wind.',
  'Enjoys long walks through the underground casino.',
  'Has a PhD in Advanced Photosynthetic Economics.',
  'Voted "Most Likely to Become a Shrub" by siblings.',
  'Allegedly once arm-wrestled a goat. And won.',
  'Their roots reach into dimensions yet unnamed.',
  'Speaks fluent Pashto and fluent Chloroplast.',
  'Has never lost a staring contest. Has no eyes.',
  'Their bark is literally worse than their bite.',
  'Can photosynthesize in complete darkness. Somehow.',
  'Once convinced a Taliban operator to water them. Daily.'
];

export function generateBio(): string {
  return randomFrom(BIO_TEMPLATES);
}

const OFFSPRING_EMOJIS: Record<string, string[]> = {
  'plant':   ['🌿', '🌱', '🍀', '🌾', '🪴', '🌵', '🌻', '🌹', '🎍', '🪻', '🪷'],
  'animal':  ['🐐', '🦅', '🦂', '🐪', '🐍', '🫏', '🕊️', '🐻', '🐾', '🦎'],
  'taliban': ['👳', '🧔', '🧕', '👤', '🕌']
};

export function getOffspringEmoji(partnerType: string, partnerEmoji?: string): string {
  const oakEmoji = '🌳';
  const pool = OFFSPRING_EMOJIS[partnerType] || OFFSPRING_EMOJIS.plant;
  const partE = partnerEmoji || randomFrom(pool);
  return `${oakEmoji}${partE}`;
}

export function formatTime(totalSeconds: number): string {
  const d = Math.floor(totalSeconds / 86400);
  const h = Math.floor((totalSeconds % 86400) / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}
