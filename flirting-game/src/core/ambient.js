// "Mood Match" ambient engine — adapts the narrative tone to the player's
// context. Zero-permission core (local hour + IANA timezone); opt-in sensors
// (Ambient Light, Battery) are feature-detected and fail silently.
// Phase 3: phase labels + flavors come from the i18n layer.

import { t } from './i18n.js';

const PHASES = [
  { id: 'late-night', from: 0, to: 5, tone: 'intimate' },
  { id: 'morning', from: 5, to: 11, tone: 'fresh' },
  { id: 'afternoon', from: 11, to: 17, tone: 'bright' },
  { id: 'evening', from: 17, to: 22, tone: 'warm' },
  { id: 'night', from: 22, to: 24, tone: 'intimate' }
];

const REGION_LABELS = {
  Europe: 'European',
  America: 'American',
  Asia: 'Asian',
  Africa: 'African',
  Australia: 'Australian',
  Atlantic: 'Atlantic',
  Pacific: 'Pacific',
  Arctic: 'Arctic',
  Indian: 'Indian Ocean',
  UTC: 'UTC'
};

function phaseFor(hour) {
  return PHASES.find((p) => hour >= p.from && hour < p.to) || PHASES[PHASES.length - 1];
}

function regionFor(timezone) {
  const area = (timezone || '').split('/')[0];
  return REGION_LABELS[area] || 'Local';
}

async function detectLighting() {
  if (typeof window === 'undefined' || typeof window.AmbientLightSensor !== 'function') return null;
  return new Promise((resolve) => {
    try {
      const sensor = new window.AmbientLightSensor();
      const finish = (value) => {
        sensor.removeEventListener('reading', onReading);
        sensor.stop?.();
        resolve(value);
      };
      const onReading = () => finish(sensor.illuminance > 150 ? 'bright' : 'dim');
      sensor.addEventListener('reading', onReading);
      sensor.start();
      setTimeout(() => finish(null), 400); // never block the boot on a slow sensor
    } catch {
      resolve(null);
    }
  });
}

async function detectBattery() {
  try {
    if (typeof navigator === 'undefined' || typeof navigator.getBattery !== 'function') return null;
    const battery = await navigator.getBattery();
    return { level: Math.round(battery.level * 100), charging: battery.charging };
  } catch {
    return null;
  }
}

let cached = null;

export async function collectAmbient({ force = false } = {}) {
  if (cached && !force) return cached;
  const hour = new Date().getHours();
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
  const [lighting, battery] = await Promise.all([detectLighting(), detectBattery()]);
  cached = {
    hour,
    phase: phaseFor(hour),
    timezone,
    region: regionFor(timezone),
    lighting,
    battery
  };
  return cached;
}

export function moodChip(ambient) {
  const icons = { 'late-night': '🌙', morning: '🌅', afternoon: '☀️', evening: '🌆', night: '🌃' };
  const parts = [`${icons[ambient.phase.id] || '✨'} ${t(`mood.${ambient.phase.id}`)}`, ambient.region];
  if (ambient.lighting) parts.push(ambient.lighting === 'bright' ? 'bright room' : 'dim room');
  if (ambient.battery && ambient.battery.level <= 20 && !ambient.battery.charging) {
    parts.push(`low battery ${ambient.battery.level}%`);
  }
  return parts.join(' · ');
}

export function moodLine(ambient) {
  return `${moodChip(ambient)} — ${t(`mood.${ambient.phase.id}.flavor`)}.`;
}
