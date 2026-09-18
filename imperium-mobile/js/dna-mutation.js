import { rand } from './utils/helpers.js';
import { showToast } from './ui/notifications.js';

const MUTATION_CHANCE = 0.05;

const POSITIVE_MUTATIONS = [
  { id: 'swift', name: 'Swift', desc: '+15% speed', stat: 'speed', mod: 1.15, emoji: '💨' },
  { id: 'strong', name: 'Strong', desc: '+15% attack', stat: 'attack', mod: 1.15, emoji: '💪' },
  { id: 'tough', name: 'Tough', desc: '+15% defense', stat: 'defense', mod: 1.15, emoji: '🛡️' },
  { id: 'vitality', name: 'Vitality', desc: '+20% HP', stat: 'hp', mod: 1.2, emoji: '❤️' },
  { id: 'gatherer', name: 'Gatherer', desc: '+15% gather', stat: 'gather', mod: 1.15, emoji: '🌾' },
  { id: 'focused', name: 'Focused', desc: '+10% all', stat: 'all', mod: 1.1, emoji: '🎯' },
];

const NEGATIVE_MUTATIONS = [
  { id: 'sluggish', name: 'Sluggish', desc: '-10% speed', stat: 'speed', mod: 0.9, emoji: '🐌' },
  { id: 'weak', name: 'Weak', desc: '-10% attack', stat: 'attack', mod: 0.9, emoji: '🦴' },
  { id: 'fragile', name: 'Fragile', desc: '-10% defense', stat: 'defense', mod: 0.9, emoji: '💔' },
];

export const dna = {
  init() {},

  roll() {
    if (Math.random() > MUTATION_CHANCE) return null;
    const isPositive = Math.random() < 0.7;
    const pool = isPositive ? POSITIVE_MUTATIONS : NEGATIVE_MUTATIONS;
    const mutation = pool[rand(0, pool.length - 1)];
    return mutation;
  },

  apply(citizen) {
    const mutation = this.roll();
    if (!mutation) return false;
    if (citizen.mutations.some(m => m.id === mutation.id)) return false;
    citizen.mutations.push(mutation);
    if (mutation.stat === 'speed') citizen.speed = (citizen.speed || 2) * mutation.mod;
    if (mutation.stat === 'attack') citizen.attack = (citizen.attack || 0) * mutation.mod;
    if (mutation.stat === 'defense') citizen.defense = (citizen.defense || 0) * mutation.mod;
    if (mutation.stat === 'hp') citizen.maxHp = Math.floor((citizen.maxHp || 50) * mutation.mod);
    if (mutation.stat === 'gather') citizen.gatherRate = (citizen.gatherRate || 1) * mutation.mod;
    if (mutation.stat === 'all') {
      citizen.speed = (citizen.speed || 2) * mutation.mod;
      citizen.attack = (citizen.attack || 0) * mutation.mod;
      citizen.defense = (citizen.defense || 0) * mutation.mod;
    }
    return true;
  },
};
