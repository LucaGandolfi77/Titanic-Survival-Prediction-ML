/* ── js/casino.js ── Slot Machine Mechanics + Taliban NPCs ── */

import { randomInt, randomFloat, randomFrom, randomChance, clamp, uid } from './utils.js';

/* ── REEL SYMBOL DATABASES ── */
const MACHINE_DEFS = [
  {
    id: 1, name: 'Holy Wheel of Fortune', unlockHeight: 0,
    symbols: [
      { s: '🌙', mult: 2,  label: 'Crescent' },
      { s: '⭐', mult: 3,  label: 'Star' },
      { s: '📖', mult: 5,  label: 'Holy Book' },
      { s: '🕌', mult: 8,  label: 'Mosque' },
      { s: '🌳', mult: 25, label: 'JACKPOT' },
      { s: '💣', mult: 0,  label: 'Dud' },
    ],
    betRange: [5, 10], spinsRange: [8, 12], payoutRate: 0.75,
  },
  {
    id: 2, name: 'Desert Storm', unlockHeight: 0,
    symbols: [
      { s: '🐪', mult: 2 }, { s: '🏜️', mult: 3 }, { s: '🦂', mult: 5 },
      { s: '🌪️', mult: 8 }, { s: '🌳', mult: 25 }, { s: '💥', mult: 0 },
    ],
    betRange: [5, 12], spinsRange: [7, 11], payoutRate: 0.72,
  },
  {
    id: 3, name: 'Mountain Glory', unlockHeight: 0,
    symbols: [
      { s: '🏔️', mult: 2 }, { s: '🦅', mult: 3 }, { s: '❄️', mult: 4 },
      { s: '🗡️', mult: 8 }, { s: '🌳', mult: 30 }, { s: '🪨', mult: 0 },
    ],
    betRange: [5, 10], spinsRange: [6, 10], payoutRate: 0.70,
  },
  {
    id: 4, name: 'Opium Dreams', unlockHeight: 0,
    symbols: [
      { s: '🌸', mult: 2 }, { s: '🦋', mult: 4 }, { s: '✨', mult: 6 },
      { s: '🌈', mult: 10 }, { s: '🌳', mult: 50 }, { s: '😵', mult: 0 },
    ],
    betRange: [8, 15], spinsRange: [6, 10], payoutRate: 0.68,
  },
  {
    id: 5, name: "Zarghun's Revenge", unlockHeight: 20,
    symbols: [
      { s: '🌳', mult: 3 }, { s: '🍂', mult: 5 }, { s: '🪵', mult: 8 },
      { s: '🌰', mult: 12 }, { s: '👑', mult: 100 }, { s: '🔥', mult: 0 },
    ],
    betRange: [10, 20], spinsRange: [8, 12], payoutRate: 0.65,
  },
  {
    id: 6, name: 'The Ascension', unlockHeight: 50,
    symbols: [
      { s: '🌌', mult: 5 }, { s: '🌀', mult: 8 }, { s: '⚡', mult: 12 },
      { s: '💎', mult: 20 }, { s: '🌳✨', mult: 500 }, { s: '☠️', mult: -2 },
    ],
    betRange: [15, 20], spinsRange: [10, 12], payoutRate: 0.60,
  },
];

/* ── TALIBAN NPC OPERATORS ── */
const NPC_DEFS = [
  {
    id: 'abdul', name: 'Abdul the Devoted', personality: 'devout',
    quotes: [
      'Bismillah... and press SPIN.',
      'Every spin is a prayer.',
      'Allah guides the reels.',
      'I fast between jackpots.',
      'The crescent moon favours me today.',
    ],
    special: 'prayerLuck', // +10% luck during prayer animation
    machineId: 1,
  },
  {
    id: 'mahmoud', name: 'Mahmoud the Sweaty', personality: 'nervous',
    quotes: [
      'Is this haram? Probably. SPIN ANYWAY.',
      '*wipes brow* One more spin...',
      'I should not be here. *spins*',
      "Don't tell my mother about this.",
      'The sweat is from... the heat. Yes.',
    ],
    special: 'careful', // -50% breakdown
    machineId: 2,
  },
  {
    id: 'omar', name: 'Omar the Sleepy', personality: 'sleepy',
    quotes: [
      'Zzzz... oh sorry... *yawn* ...SPIN',
      'I dream of jackpots...',
      "Five more minutes... then I'll spin.",
      '*snore* JACKPOT! Oh wait, still dreaming.',
      'The reels lull me to sleep...',
    ],
    special: 'sleepy', // occasionally sleeps 30s
    machineId: 3,
  },
  {
    id: 'tariq', name: 'Tariq the Lucky', personality: 'greedy',
    quotes: [
      'The house always wins. I AM the house.',
      'Every coin is MINE... I mean, ours.',
      'I count coins in my sleep.',
      "What's yours is mine. What's mine is also mine.",
      'Profit margin looking EXCELLENT.',
    ],
    special: 'greedy', // +20% rev but steals 10%
    machineId: 4,
  },
  {
    id: 'barakat', name: 'Barakat the Paranoid', personality: 'paranoid',
    quotes: [
      'Are you a spy? ... SPIN QUICKLY.',
      '*looks behind shoulder* All clear.',
      'I installed 47 cameras. Just in case.',
      "Trust no one. Especially the oak tree.",
      'Who moved my machine 2mm to the left?!',
    ],
    special: 'vigilant', // -80% breakdown, detect cheaters
    machineId: 5,
  },
  {
    id: 'yusuf', name: 'Commander Yusuf', personality: 'authoritative',
    quotes: [
      'By my authority, MAXIMUM BET.',
      'I command you: SPIN.',
      'This casino is my jurisdiction.',
      'All profits serve the cause... of oak growth.',
      'Discipline. Order. Jackpots.',
    ],
    special: 'commander', // all bets ×2
    machineId: 6,
  },
];

/* ── Machine runtime state ── */
function createMachineState(def) {
  return {
    id: def.id,
    name: def.name,
    unlockHeight: def.unlockHeight,
    symbols: def.symbols,
    bet: randomInt(def.betRange[0], def.betRange[1]),
    spinsPerMin: randomFloat(def.spinsRange[0], def.spinsRange[1]),
    payoutRate: def.payoutRate,
    breakdownChance: 2,
    isBroken: false,
    isLocked: def.unlockHeight > 0,
    totalRevenue: 0,
    revenuePerMinute: 0,
    currentReels: ['🌙', '🌙', '🌙'],
    spinning: false,
    lastResult: null,  // { win, mult, jackpot }
    _spinTimer: randomFloat(2, 6),
    _sleepTimer: 0,
    dealers: [],        // assigned offspring
  };
}

function createNPCState(def) {
  return {
    ...def,
    mood: randomInt(50, 80),
    beardLength: randomInt(2, 10),
    currentQuote: def.quotes[0],
    _quoteTimer: randomFloat(5, 15),
    _prayTimer: 0,
    _sleepTimer: 0,
    isSleeping: false,
    isPraying: false,
  };
}

/* ══════════════════════════════════════════════════════════ */
export class Casino {
  constructor() {
    this.machines = MACHINE_DEFS.map(d => createMachineState(d));
    this.npcs     = NPC_DEFS.map(d => createNPCState(d));
    this.totalCoins    = 0;
    this.todayRevenue  = 0;
    this.dailyHistory  = [0, 0, 0, 0, 0, 0, 0];  // last 7 days
    this.jackpotsHit   = 0;
    this.isClosed      = false;
    this._closedTimer  = 0;
    this._dayAccum     = 0;
    this.revenueMul    = 1.0;  // can be buffed by events

    // Spin result callback (set by main to trigger UI effects)
    this.onSpinResult = null;
  }

  /* ── Update (per frame) ── */
  update(delta, oakHeight) {
    // Casino closure countdown
    if (this.isClosed) {
      this._closedTimer -= delta;
      if (this._closedTimer <= 0) this.isClosed = false;
      return;
    }

    // Day tracking for revenue history
    this._dayAccum += delta;
    if (this._dayAccum >= 600) {  // 600s real = 1 in-game "day" for revenue chart
      this._dayAccum = 0;
      this.dailyHistory.push(this.todayRevenue);
      if (this.dailyHistory.length > 7) this.dailyHistory.shift();
      this.todayRevenue = 0;
    }

    // Unlock machines
    for (const m of this.machines) {
      if (m.isLocked && oakHeight >= m.unlockHeight) {
        m.isLocked = false;
      }
    }

    // Process each machine
    for (const m of this.machines) {
      if (m.isLocked || m.isBroken) continue;

      const npc = this.npcs.find(n => n.machineId === m.id);

      // NPC sleeping?
      if (npc && npc.special === 'sleepy' && !npc.isSleeping) {
        npc._sleepTimer -= delta;
        if (npc._sleepTimer <= 0 && randomChance(1)) {
          npc.isSleeping = true;
          npc._sleepTimer = 30;
        }
      }
      if (npc && npc.isSleeping) {
        npc._sleepTimer -= delta;
        if (npc._sleepTimer <= 0) {
          npc.isSleeping = false;
          npc._sleepTimer = randomFloat(30, 60);
        }
        continue; // skip this machine's spin
      }

      // NPC praying
      if (npc && npc.special === 'prayerLuck') {
        npc._prayTimer -= delta;
        if (npc._prayTimer <= 0 && !npc.isPraying && randomChance(5)) {
          npc.isPraying = true;
          npc._prayTimer = 5;
        }
        if (npc.isPraying) {
          npc._prayTimer -= delta;
          if (npc._prayTimer <= 0) npc.isPraying = false;
        }
      }

      // Quote cycling
      if (npc) {
        npc._quoteTimer -= delta;
        if (npc._quoteTimer <= 0) {
          npc.currentQuote = randomFrom(npc.quotes);
          npc._quoteTimer = randomFloat(8, 20);
        }
      }

      // Auto-spin timer
      m._spinTimer -= delta;
      if (m._spinTimer <= 0) {
        this._executeSpin(m, npc);
        m._spinTimer = 60 / m.spinsPerMin + randomFloat(-0.5, 0.5);
      }
    }
  }

  /* ── Execute a single spin ── */
  _executeSpin(machine, npc) {
    const syms = machine.symbols;

    // Pick 3 random symbols
    const reels = [randomFrom(syms), randomFrom(syms), randomFrom(syms)];
    machine.currentReels = reels.map(r => r.s);
    machine.spinning = true;
    setTimeout(() => { machine.spinning = false; }, 1800);

    // Evaluate result
    const result = this._evaluateResult(reels, machine, npc);
    machine.lastResult = result;

    // Revenue
    const bet = npc && npc.special === 'commander' ? machine.bet * 2 : machine.bet;
    let revenue;
    if (result.win) {
      // Player wins (house takes from NPC gamblers — we represent net casino income)
      revenue = bet * result.mult * 0.1 * this.revenueMul;
    } else {
      // House keeps the bet
      revenue = bet * this.revenueMul;
    }

    // Dealer bonus
    const dealerBonus = machine.dealers.length * 0.15;
    revenue *= (1 + dealerBonus);

    // NPC greedy: steals 10% but +20% base rev
    if (npc && npc.special === 'greedy') {
      revenue *= 1.2;
      revenue *= 0.9; // net +8%
    }

    revenue = Math.round(revenue);
    machine.totalRevenue += revenue;
    machine.revenuePerMinute = Math.round(machine.spinsPerMin * bet * (1 - machine.payoutRate) * this.revenueMul);
    this.totalCoins   += revenue;
    this.todayRevenue += revenue;

    if (result.jackpot) this.jackpotsHit++;

    // Breakdown check
    let breakChance = machine.breakdownChance;
    if (npc && npc.special === 'careful')   breakChance *= 0.5;
    if (npc && npc.special === 'vigilant')  breakChance *= 0.2;
    // Bodyguard offspring reduce breakdown
    // (handled externally by checking dealers with bodyguard trait)
    if (randomChance(breakChance)) {
      machine.isBroken = true;
    }

    // NPC mood
    if (npc) {
      npc.mood = clamp(npc.mood + (result.win ? 5 : -1), 0, 100);
    }

    // Callback
    if (this.onSpinResult) {
      this.onSpinResult(machine, result, revenue);
    }
  }

  /* ── Evaluate spin result ── */
  _evaluateResult(reels, machine, npc) {
    const [a, b, c] = reels;
    const jackpot = a.mult > 20 && a.s === b.s && b.s === c.s;
    const threeMatch = a.s === b.s && b.s === c.s;
    const twoMatch = a.s === b.s || b.s === c.s || a.s === c.s;

    let win = false;
    let mult = 0;

    if (threeMatch) {
      win = true;
      mult = a.mult;
      if (a.mult === 0) { win = false; mult = 0; } // three duds — still lose
      if (a.mult < 0) { win = false; mult = a.mult; } // Oblivion — lose ×2
    } else if (twoMatch) {
      // Two match — small win
      const matched = a.s === b.s ? a : b.s === c.s ? b : a;
      if (matched.mult > 0) {
        win = true;
        mult = Math.ceil(matched.mult * 0.4);
      }
    }

    // Prayer luck bonus
    if (npc && npc.isPraying && !win && randomChance(10)) {
      win = true;
      mult = 2;
    }

    return { win, mult, jackpot };
  }

  /* ── Repair machine ── */
  repairMachine(machineId) {
    const cost = 150;
    if (this.totalCoins < cost) return false;
    const m = this.machines.find(m => m.id === machineId);
    if (!m || !m.isBroken) return false;
    this.totalCoins -= cost;
    m.isBroken = false;
    return true;
  }

  /* ── Unlock machine (purchase) ── */
  unlockMachine(machineId) {
    const cost = 500;
    if (this.totalCoins < cost) return false;
    const m = this.machines.find(m => m.id === machineId);
    if (!m || !m.isLocked) return false;
    this.totalCoins -= cost;
    m.isLocked = false;
    return true;
  }

  /* ── Hire offspring as dealer ── */
  hireDealer(offspringId, machineId) {
    const cost = 100;
    if (this.totalCoins < cost) return false;
    const m = this.machines.find(m => m.id === machineId);
    if (!m) return false;
    this.totalCoins -= cost;
    m.dealers.push(offspringId);
    return true;
  }

  /* ── Spend coins ── */
  spendCoins(amount) {
    if (this.totalCoins < amount) return false;
    this.totalCoins -= amount;
    return true;
  }

  /* ── Get total revenue per minute ── */
  getTotalRPM() {
    return this.machines.reduce((sum, m) => sum + (m.isLocked || m.isBroken ? 0 : m.revenuePerMinute), 0);
  }

  /* ── Close casino temporarily ── */
  closeCasino(duration) {
    this.isClosed = true;
    this._closedTimer = duration;
  }

  /* ── Serialisation ── */
  toJSON() {
    return {
      totalCoins: this.totalCoins,
      todayRevenue: this.todayRevenue,
      dailyHistory: this.dailyHistory,
      jackpotsHit: this.jackpotsHit,
      machines: this.machines.map(m => ({
        id: m.id, isBroken: m.isBroken, isLocked: m.isLocked,
        totalRevenue: m.totalRevenue, dealers: m.dealers,
      })),
      npcs: this.npcs.map(n => ({ id: n.id, mood: n.mood })),
    };
  }

  loadJSON(data) {
    if (!data) return;
    this.totalCoins   = data.totalCoins || 0;
    this.todayRevenue = data.todayRevenue || 0;
    this.dailyHistory = data.dailyHistory || [0,0,0,0,0,0,0];
    this.jackpotsHit  = data.jackpotsHit || 0;
    if (data.machines) {
      for (const saved of data.machines) {
        const m = this.machines.find(m => m.id === saved.id);
        if (m) {
          m.isBroken     = saved.isBroken;
          m.isLocked     = saved.isLocked;
          m.totalRevenue = saved.totalRevenue;
          m.dealers      = saved.dealers || [];
        }
      }
    }
    if (data.npcs) {
      for (const saved of data.npcs) {
        const n = this.npcs.find(n => n.id === saved.id);
        if (n) n.mood = saved.mood;
      }
    }
  }
}
