import { rand } from '../utils/helpers.js';
import { showToast, haptic } from './ui/notifications.js';

const WEATHER_TYPES = ['clear', 'rain', 'storm', 'fog', 'snow'];
const WEATHER_WEIGHTS = [40, 25, 10, 15, 10];

const WEATHER_EFFECTS = {
  clear: { name: 'Clear', speedMod: 1, gatherMod: 1, rangeMod: 1, color: 'rgba(200,200,200,0)', icon: '☀️' },
  rain: { name: 'Rain', speedMod: 0.95, gatherMod: 1.05, rangeMod: 0.9, color: 'rgba(40,60,80,0.12)', icon: '🌧️' },
  storm: { name: 'Storm', speedMod: 0.8, gatherMod: 0.9, rangeMod: 0.85, color: 'rgba(20,30,50,0.2)', icon: '⛈️' },
  fog: { name: 'Fog', speedMod: 0.9, gatherMod: 0.95, rangeMod: 0.8, color: 'rgba(180,180,180,0.15)', icon: '🌫️' },
  snow: { name: 'Snow', speedMod: 0.85, gatherMod: 0.9, rangeMod: 0.95, color: 'rgba(200,210,230,0.1)', icon: '❄️' },
};

export const weather = {
  type: 'clear',
  timer: 0,
  minDuration: 60,
  maxDuration: 120,
  nextDuration: 90,

  init() {
    this.type = 'clear';
    this.timer = 0;
    this.nextDuration = this.minDuration + rand(0, this.maxDuration - this.minDuration);
  },

  update(dt) {
    this.timer += dt;
    if (this.timer >= this.nextDuration) {
      this.change();
    }
  },

  change() {
    let total = WEATHER_WEIGHTS.reduce((a, b) => a + b, 0);
    let r = rand(0, total);
    let newType = 'clear';
    for (let i = 0; i < WEATHER_TYPES.length; i++) {
      r -= WEATHER_WEIGHTS[i];
      if (r <= 0) { newType = WEATHER_TYPES[i]; break; }
    }
    this.type = newType;
    this.timer = 0;
    this.nextDuration = this.minDuration + rand(0, this.maxDuration - this.minDuration);
    const fx = WEATHER_EFFECTS[newType];
    showToast(`${fx.icon} Weather: ${fx.name}`);
    haptic(15);
  },

  getEffects() {
    return WEATHER_EFFECTS[this.type] || WEATHER_EFFECTS.clear;
  },

  getSpeedModifier() { return this.getEffects().speedMod; },
  getGatherModifier() { return this.getEffects().gatherMod; },
  getRangeModifier() { return this.getEffects().rangeMod; },
  getOverlayColor() { return this.getEffects().color; },
  getIcon() { return this.getEffects().icon; },
};
