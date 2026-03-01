/* ── js/utils.js ── RNG, helpers, name generator ── */

export function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function randomFloat(min, max) {
  return Math.random() * (max - min) + min;
}

export function randomChance(percent) {
  return Math.random() * 100 < percent;
}

export function randomFrom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function weightedRandom(items, weights) {
  const total = weights.reduce((s, w) => s + w, 0);
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

export function formatNumber(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e4) return (n / 1e3).toFixed(1) + 'K';
  return Math.floor(n).toLocaleString();
}

/* ── Unique ID generator ── */
let _idCounter = 0;
export function uid() {
  return `id_${Date.now().toString(36)}_${(++_idCounter).toString(36)}`;
}

/* ── Name Generator ── */
const AFGHAN_SYLLABLES = [
  'Zar', 'Gul', 'Shah', 'Khan', 'Din', 'War', 'Mor',
  'Bak', 'Tar', 'Nur', 'Jam', 'Raf', 'Hal', 'Sar', 'Nas',
  'Abb', 'Mah', 'Yas', 'Fer', 'Qas', 'Sal', 'Rez', 'Hom'
];

const PLANT_SUFFIXES = [
  'oak', 'leaf', 'root', 'bark', 'branch', 'thorn', 'bloom',
  'seed', 'moss', 'vine', 'fern', 'bud', 'stump', 'petal'
];

const TITLES = [
  'The Thorny', 'Al-Photosynth', 'ibn Acorn', 'The Rooted',
  'The Branching', 'Al-Verdant', 'The Mossy', 'ibn Zarghun',
  'The Evergreen', 'Al-Sappy', 'The Leafy', 'The Barky',
  'The Cosmic', 'Al-Canopy', 'The Sprout', 'ibn Chlorophyll',
  'The Ancient', 'The Twisted', 'Al-Humus', 'The Pollinated'
];

const ANIMAL_NAMES = [
  'Hooves', 'Claws', 'Fangs', 'Wings', 'Scales', 'Horns',
  'Talons', 'Tusks', 'Quills', 'Feathers', 'Paws', 'Jaws'
];

export function generateOffspringName(parent1Type, parent2Type) {
  const syl = randomFrom(AFGHAN_SYLLABLES);
  let suffix;
  if (parent2Type === 'animal') {
    suffix = randomFrom(ANIMAL_NAMES).toLowerCase();
  } else {
    suffix = randomFrom(PLANT_SUFFIXES);
  }
  const title = randomFrom(TITLES);
  return `${syl}${suffix} ${title}`;
}

/* ── Funny bio generator ── */
const BIO_TEMPLATES = [
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

export function generateBio() {
  return randomFrom(BIO_TEMPLATES);
}

/* ── Emoji combiner for offspring ── */
const OFFSPRING_EMOJIS = {
  'plant':   ['🌿', '🌱', '🍀', '🌾', '🪴', '🌵', '🌻', '🌹', '🎍', '🪻', '🪷'],
  'animal':  ['🐐', '🦅', '🦂', '🐪', '🐍', '🫏', '🕊️', '🐻', '🐾', '🦎'],
  'taliban': ['👳', '🧔', '🧕', '👤', '🕌']
};

export function getOffspringEmoji(partnerType, partnerEmoji) {
  const oakEmoji = '🌳';
  const pool = OFFSPRING_EMOJIS[partnerType] || OFFSPRING_EMOJIS.plant;
  const partE = partnerEmoji || randomFrom(pool);
  return `${oakEmoji}${partE}`;
}

/* ── Time formatter ── */
export function formatTime(totalSeconds) {
  const d = Math.floor(totalSeconds / 86400);
  const h = Math.floor((totalSeconds % 86400) / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}
