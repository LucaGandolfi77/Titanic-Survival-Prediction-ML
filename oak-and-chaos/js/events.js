/* ── js/events.js ── Random Events System ── */

import { randomFrom, randomChance, randomFloat, randomInt } from './utils.js';

/* ── EVENT DEFINITIONS ── */
const POSITIVE_EVENTS = [
  {
    id: 'rainstorm', name: '🌧️ Rainstorm!', type: 'positive',
    desc: 'A sudden rainstorm drenches the oak. +200 energy!',
    apply: (gs) => { gs.oak.energy = Math.min(gs.oak.energy + 200, gs.oak.maxEnergy); },
  },
  {
    id: 'tourist', name: '🧳 Tourist Found Casino!', type: 'positive',
    desc: 'A lost tourist stumbles into the underground casino. +500 coins!',
    apply: (gs) => { gs.casino.totalCoins += 500; },
  },
  {
    id: 'gold_acorn', name: '🌰 Acorn Falls in Gold Mine!', type: 'positive',
    desc: 'An acorn rolled into a forgotten gold vein. +50 DNA points!',
    apply: (gs) => { gs.oak.dnaPoints += 50; },
  },
  {
    id: 'goat_wanders', name: '🐐 Goat Wanders In!', type: 'positive',
    desc: 'A friendly goat appeared! Breeding boost +40% for 30s.',
    apply: (gs) => { gs.breeding.addSuccessBoost(40, 30); },
  },
  {
    id: 'eagle_news', name: '🦅 Eagle Brings News!', type: 'positive',
    desc: 'An eagle reveals incoming events. Scouts activated!',
    apply: () => {},
  },
  {
    id: 'oasis', name: '🏝️ Taliban Finds Oasis!', type: 'positive',
    desc: 'All NPC operators found water! Mood +40, revenue +30% for 60s.',
    apply: (gs) => {
      for (const n of gs.casino.npcs) n.mood = Math.min(n.mood + 40, 100);
      gs.casino.revenueMul *= 1.3;
      setTimeout(() => { gs.casino.revenueMul = Math.max(1.0, gs.casino.revenueMul / 1.3); }, 60000);
    },
  },
];

const NEGATIVE_EVENTS = [
  {
    id: 'drone_spotted', name: '✈️ Drone Spotted!', type: 'negative',
    desc: 'Everyone hide! Casino closes for 30 seconds!',
    apply: (gs) => { gs.casino.closeCasino(30); },
  },
  {
    id: 'machine_glitch', name: '⚡ Machine Glitch!', type: 'negative',
    desc: 'A random slot machine just broke down!',
    apply: (gs) => {
      const active = gs.casino.machines.filter(m => !m.isLocked && !m.isBroken);
      if (active.length) randomFrom(active).isBroken = true;
    },
  },
  {
    id: 'omar_sleep', name: '😴 Omar Fell Asleep!', type: 'negative',
    desc: 'Machine 3 stops for 45 seconds. Classic Omar.',
    apply: (gs) => {
      const omar = gs.casino.npcs.find(n => n.id === 'omar');
      if (omar) { omar.isSleeping = true; omar._sleepTimer = 45; }
    },
  },
  {
    id: 'goat_wires', name: '🐐 Goat Ate the Wiring!', type: 'negative',
    desc: 'A goat chewed through power cables. -20% energy gen for 60s.',
    apply: (gs) => {
      gs.oak.addBuff('energyGen', 60, 0.8);
    },
  },
  {
    id: 'tax_inspector', name: '🕴️ Tax Inspector!', type: 'negative',
    desc: 'Lose 20% of coins unless you bribe him (300🪙)!',
    apply: (gs) => {
      const loss = Math.floor(gs.casino.totalCoins * 0.2);
      gs.casino.totalCoins -= loss;
    },
  },
  {
    id: 'sandstorm', name: '🌪️ Sandstorm!', type: 'negative',
    desc: 'Sunlight -50% for 90 seconds!',
    apply: (gs) => {
      gs.oak.addBuff('energyGen', 90, 0.5);
    },
  },
];

const WEIRD_EVENTS = [
  {
    id: 'zarghun_speaks', name: '🌳 Zarghun Speaks!', type: 'weird',
    desc: '',   // filled dynamically
    apply: (gs) => {
      const QUOTES = [
        'I have seen empires rise and fall from within this pot.',
        'My roots touch the dreams of sleeping Taliban.',
        'To be an oak is to understand that patience IS violence.',
        'I once knew a cactus. It was prickly. But honest.',
        'Coins are just round acorns. Think about it.',
        'Every leaf is a solar panel of the soul.',
        'The casino is my garden. The gamblers: my flowers.',
        'I do not need legs. The world comes to MY roots.',
      ];
      gs._oakQuote = randomFrom(QUOTES);
    },
  },
  {
    id: 'cosmic_alignment', name: '🌌 Cosmic Alignment!', type: 'weird',
    desc: 'The stars aligned! All breeding +100% success for 30s!',
    apply: (gs) => { gs.breeding.addSuccessBoost(100, 30); },
  },
  {
    id: 'poetry_slam', name: '📜 Taliban Poetry Slam!', type: 'weird',
    desc: 'All NPCs stop to recite poetry for 20s. Mood +50 afterward.',
    apply: (gs) => {
      gs.casino.closeCasino(20);
      setTimeout(() => {
        for (const n of gs.casino.npcs) n.mood = Math.min(n.mood + 50, 100);
      }, 20000);
    },
  },
  {
    id: 'sentient_acorn', name: '🧠 Acorn Becomes Sentient!', type: 'weird',
    desc: 'A random offspring gains +30 to ALL stats!',
    apply: (gs) => {
      if (gs.population.offspring.length > 0) {
        const child = randomFrom(gs.population.offspring);
        for (const key in child.stats) child.stats[key] += 30;
      }
    },
  },
  {
    id: 'portal', name: '🌀 Machine 4 Creates Portal!', type: 'weird',
    desc: 'Interdimensional portal! Machine 4 output ×2 for 60s.',
    apply: (gs) => {
      const m4 = gs.casino.machines.find(m => m.id === 4);
      if (m4) {
        m4.spinsPerMin *= 2;
        setTimeout(() => { m4.spinsPerMin /= 2; }, 60000);
      }
    },
  },
  {
    id: 'abdul_wins', name: '🎉 Abdul Wins His Own Jackpot!', type: 'weird',
    desc: 'Abdul celebrates, works 2× faster... then gets suspicious.',
    apply: (gs) => {
      const m1 = gs.casino.machines.find(m => m.id === 1);
      if (m1) {
        m1.spinsPerMin *= 2;
        setTimeout(() => { m1.spinsPerMin /= 2; }, 30000);
      }
      const abdul = gs.casino.npcs.find(n => n.id === 'abdul');
      if (abdul) abdul.mood = 100;
    },
  },
];

/* ══════════════════════════════════════════════════════════ */
export class EventSystem {
  constructor() {
    this.history = [];      // { event, timestamp }
    this.pending = [];
    this._timer = randomFloat(30, 60);
    this._eventListeners = [];
    this.maxHistory = 50;
  }

  onEvent(fn) { this._eventListeners.push(fn); }

  /* ── Update ── */
  update(delta, gameState) {
    // Tick blocked event timers
    if (this._blockedEvents) {
      for (const id of Object.keys(this._blockedEvents)) {
        this._blockedEvents[id] -= delta;
        if (this._blockedEvents[id] <= 0) delete this._blockedEvents[id];
      }
    }

    this._timer -= delta;
    if (this._timer <= 0) {
      let evt = this.getRandomEvent();
      // Reroll if blocked
      if (this._blockedEvents && this._blockedEvents[evt.id]) {
        evt = this.getRandomEvent();
      }
      this.triggerEvent(evt, gameState);
      this._timer = randomFloat(30, 60);
    }
  }

  /* ── Get weighted random event ── */
  getRandomEvent() {
    const r = Math.random() * 100;
    let pool;
    if (r < 40)      pool = POSITIVE_EVENTS;
    else if (r < 80) pool = NEGATIVE_EVENTS;
    else              pool = WEIRD_EVENTS;
    return randomFrom(pool);
  }

  /* ── Trigger ── */
  triggerEvent(evt, gameState) {
    // Handle special desc for zarghun_speaks
    if (evt.id === 'zarghun_speaks') {
      evt.apply(gameState);
      evt = { ...evt, desc: `"${gameState._oakQuote}"` };
    } else {
      evt.apply(gameState);
    }

    const record = { event: evt, timestamp: Date.now() };
    this.history.unshift(record);
    if (this.history.length > this.maxHistory) this.history.pop();

    for (const fn of this._eventListeners) fn(evt);
  }

  /* ── Temporarily block a specific event ── */
  temporarilyBlock(eventId, duration) {
    this._blockedEvents = this._blockedEvents || {};
    this._blockedEvents[eventId] = duration;
  }

  /* ── Serialisation ── */
  toJSON() { return { history: this.history.slice(0, 20) }; }
  loadJSON(data) {
    if (!data) return;
    this.history = data.history || [];
  }
}
