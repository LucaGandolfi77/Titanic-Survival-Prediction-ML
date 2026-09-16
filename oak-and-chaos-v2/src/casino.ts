/* ── src/casino.ts ── Slot Machine Mechanics + Taliban NPCs ── */

import { randomInt, randomFloat, randomFrom, randomChance, clamp, uid } from './utils';
import type { MachineState, NPCState, SpinResult, CasinoState } from '../types/game';

interface SymbolDef {
  s: string;
  mult: number;
  label?: string;
}

interface MachineDef {
  id: number;
  name: string;
  unlockHeight: number;
  symbols: SymbolDef[];
  betRange: [number, number];
  spinsRange: [number, number];
  payoutRate: number;
}

interface NPCDef {
  id: string;
  name: string;
  personality: string;
  quotes: string[];
  special: string;
  machineId: number;
}

const MACHINE_DEFS: MachineDef[] = [
  {
    id: 1, name: 'Holy Wheel of Fortune', unlockHeight: 0,
    symbols: [
      { s: '🌙', mult: 2, label: 'Crescent' }, { s: '⭐', mult: 3, label: 'Star' },
      { s: '📖', mult: 5, label: 'Holy Book' }, { s: '🕌', mult: 8, label: 'Mosque' },
      { s: '🌳', mult: 25, label: 'JACKPOT' }, { s: '💣', mult: 0, label: 'Dud' },
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

function createMachineState(def: MachineDef): MachineState {
  return {
    id: def.id, name: def.name, unlockHeight: def.unlockHeight,
    symbols: def.symbols, bet: randomInt(def.betRange[0], def.betRange[1]),
    spinsPerMin: randomFloat(def.spinsRange[0], def.spinsRange[1]),
    payoutRate: def.payoutRate, breakdownChance: 2, isBroken: false,
    isLocked: def.unlockHeight > 0, totalRevenue: 0, revenuePerMinute: 0,
    currentReels: ['🌙', '🌙', '🌙'], spinning: false, lastResult: null,
    _spinTimer: randomFloat(2, 6), _sleepTimer: 0, dealers: [],
  };
}

function createNPCState(def: NPCDef): NPCState {
  return {
    ...def, mood: randomInt(50, 80), beardLength: randomInt(2, 10),
    currentQuote: def.quotes[0], _quoteTimer: randomFloat(5, 15),
    _prayTimer: 0, _sleepTimer: 0, isSleeping: false, isPraying: false,
  };
}

export class Casino {
  machines: MachineState[] = [];
  npcs: NPCState[] = [];
  totalCoins: number = 0;
  todayRevenue: number = 0;
  dailyHistory: number[] = [0, 0, 0, 0, 0, 0, 0];
  jackpotsHit: number = 0;
  isClosed: boolean = false;
  _closedTimer: number = 0;
  _dayAccum: number = 0;
  revenueMul: number = 1.0;
  onSpinResult: ((machine: MachineState, result: SpinResult, revenue: number) => void) | null = null;

  constructor() {
    this.machines = MACHINE_DEFS.map(d => createMachineState(d));
    this.npcs = [
      { id: 'abdul', name: 'Abdul the Devoted', personality: 'devout', quotes: ['Bismillah... and press SPIN.', 'Every spin is a prayer.', 'Allah guides the reels.', 'I fast between jackpots.', 'The crescent moon favours me today.'], special: 'prayerLuck', machineId: 1 },
      { id: 'mahmoud', name: 'Mahmoud the Sweaty', personality: 'nervous', quotes: ['Is this haram? Probably. SPIN ANYWAY.', '*wipes brow* One more spin...', 'I should not be here. *spins*', "Don't tell my mother about this.", 'The sweat is from... the heat. Yes.'], special: 'careful', machineId: 2 },
      { id: 'omar', name: 'Omar the Sleepy', personality: 'sleepy', quotes: ['Zzzz... oh sorry... *yawn* ...SPIN', 'I dream of jackpots...', "Five more minutes... then I'll spin.", '*snore* JACKPOT! Oh wait, still dreaming.', 'The reels lull me to sleep...'], special: 'sleepy', machineId: 3 },
      { id: 'tariq', name: 'Tariq the Lucky', personality: 'greedy', quotes: ['The house always wins. I AM the house.', 'Every coin is MINE... I mean, ours.', 'I count coins in my sleep.', "What's yours is mine. What's mine is also mine.", 'Profit margin looking EXCELLENT.'], special: 'greedy', machineId: 4 },
      { id: 'barakat', name: 'Barakat the Paranoid', personality: 'paranoid', quotes: ['Are you a spy? ... SPIN QUICKLY.', '*looks behind shoulder* All clear.', 'I installed 47 cameras. Just in case.', "Trust no one. Especially the oak tree.", 'Who moved my machine 2mm to the left?!'], special: 'vigilant', machineId: 5 },
      { id: 'yusuf', name: 'Commander Yusuf', personality: 'authoritative', quotes: ['By my authority, MAXIMUM BET.', 'I command you: SPIN.', 'This casino is my jurisdiction.', 'All profits serve the cause... of oak growth.', 'Discipline. Order. Jackpots.'], special: 'commander', machineId: 6 },
    ].map(d => createNPCState(d));
  }

  update(delta: number, oakHeight: number): void {
    if (this.isClosed) {
      this._closedTimer -= delta;
      if (this._closedTimer <= 0) this.isClosed = false;
      return;
    }
    this._dayAccum += delta;
    if (this._dayAccum >= 600) {
      this._dayAccum = 0;
      this.dailyHistory.push(this.todayRevenue);
      if (this.dailyHistory.length > 7) this.dailyHistory.shift();
      this.todayRevenue = 0;
    }
    for (const m of this.machines) {
      if (m.isLocked && oakHeight >= m.unlockHeight) {
        m.isLocked = false;
      }
    }
    for (const m of this.machines) {
      if (m.isLocked || m.isBroken) continue;
      const npc = this.npcs.find(n => n.machineId === m.id);
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
        continue;
      }
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
      if (npc) {
        npc._quoteTimer -= delta;
        if (npc._quoteTimer <= 0) {
          npc.currentQuote = randomFrom(npc.quotes);
          npc._quoteTimer = randomFloat(8, 20);
        }
      }
      m._spinTimer -= delta;
      if (m._spinTimer <= 0) {
        this._executeSpin(m, npc);
        m._spinTimer = 60 / m.spinsPerMin + randomFloat(-0.5, 0.5);
      }
    }
  }

  private _executeSpin(machine: MachineState, npc: NPCState | undefined): void {
    const syms = machine.symbols;
    const reels = [randomFrom(syms), randomFrom(syms), randomFrom(syms)];
    machine.currentReels = reels.map(r => r.s);
    machine.spinning = true;
    const spinTimeoutId = window.setTimeout(() => { machine.spinning = false; }, 1800);
    machine._spinTimeoutId = spinTimeoutId;
    const result = this._evaluateResult(reels, machine, npc);
    machine.lastResult = result;
    const bet = npc && npc.special === 'commander' ? machine.bet * 2 : machine.bet;
    let revenue: number;
    if (result.win) {
      revenue = bet * result.mult * 0.1 * this.revenueMul;
    } else {
      revenue = bet * this.revenueMul;
    }
    const dealerBonus = machine.dealers.length * 0.15;
    revenue *= (1 + dealerBonus);
    if (npc && npc.special === 'greedy') {
      revenue *= 1.2;
      revenue *= 0.9;
    }
    revenue = Math.round(revenue);
    machine.totalRevenue += revenue;
    machine.revenuePerMinute = Math.round(machine.spinsPerMin * bet * (1 - machine.payoutRate) * this.revenueMul);
    this.totalCoins += revenue;
    this.todayRevenue += revenue;
    if (result.jackpot) this.jackpotsHit++;
    let breakChance = machine.breakdownChance;
    if (npc && npc.special === 'careful') breakChance *= 0.5;
    if (npc && npc.special === 'vigilant') breakChance *= 0.2;
    if (randomChance(breakChance)) {
      machine.isBroken = true;
    }
    if (npc) {
      npc.mood = clamp(npc.mood + (result.win ? 5 : -1), 0, 100);
    }
    if (this.onSpinResult) {
      this.onSpinResult(machine, result, revenue);
    }
  }

  private _evaluateResult(reels: SymbolDef[], machine: MachineState, npc: NPCState | undefined): SpinResult {
    const [a, b, c] = reels;
    const jackpot = a.mult > 20 && a.s === b.s && b.s === c.s;
    const threeMatch = a.s === b.s && b.s === c.s;
    const twoMatch = a.s === b.s || b.s === c.s || a.s === c.s;
    let win = false;
    let mult = 0;
    if (threeMatch) {
      win = true;
      mult = a.mult;
      if (a.mult === 0) { win = false; mult = 0; }
      if (a.mult < 0) { win = false; mult = a.mult; }
    } else if (twoMatch) {
      const matched = a.s === b.s ? a : b.s === c.s ? b : a;
      if (matched.mult > 0) {
        win = true;
        mult = Math.ceil(matched.mult * 0.4);
      }
    }
    if (npc && npc.isPraying && !win && randomChance(10)) {
      win = true;
      mult = 2;
    }
    return { win, mult, jackpot };
  }

  repairMachine(machineId: number): boolean {
    const cost = 150;
    if (this.totalCoins < cost) return false;
    const m = this.machines.find(m => m.id === machineId);
    if (!m || !m.isBroken) return false;
    this.totalCoins -= cost;
    m.isBroken = false;
    return true;
  }

  unlockMachine(machineId: number): boolean {
    const cost = 500;
    if (this.totalCoins < cost) return false;
    const m = this.machines.find(m => m.id === machineId);
    if (!m || !m.isLocked) return false;
    this.totalCoins -= cost;
    m.isLocked = false;
    return true;
  }

  hireDealer(offspringId: string, machineId: number): boolean {
    const cost = 100;
    if (this.totalCoins < cost) return false;
    const m = this.machines.find(m => m.id === machineId);
    if (!m) return false;
    this.totalCoins -= cost;
    m.dealers.push(offspringId);
    return true;
  }

  spendCoins(amount: number): boolean {
    if (this.totalCoins < amount) return false;
    this.totalCoins -= amount;
    return true;
  }

  getTotalRPM(): number {
    return this.machines.reduce((sum, m) => sum + (m.isLocked || m.isBroken ? 0 : m.revenuePerMinute), 0);
  }

  closeCasino(duration: number): void {
    this.isClosed = true;
    this._closedTimer = duration;
  }

  toJSON(): { totalCoins: number; todayRevenue: number; dailyHistory: number[]; jackpotsHit: number; machines: Array<{id: number; isBroken: boolean; isLocked: boolean; totalRevenue: number; dealers: string[]}>; npcs: Array<{id: string; mood: number}>; } {
    return {
      totalCoins: this.totalCoins, todayRevenue: this.todayRevenue,
      dailyHistory: this.dailyHistory, jackpotsHit: this.jackpotsHit,
      machines: this.machines.map(m => ({
        id: m.id, isBroken: m.isBroken, isLocked: m.isLocked,
        totalRevenue: m.totalRevenue, dealers: m.dealers,
      })),
      npcs: this.npcs.map(n => ({ id: n.id, mood: n.mood })),
    };
  }

  loadJSON(data: { totalCoins?: number; todayRevenue?: number; dailyHistory?: number[]; jackpotsHit?: number; machines?: Array<{id: number; isBroken: boolean; isLocked: boolean; totalRevenue: number; dealers: string[]}>; npcs?: Array<{id: string; mood: number}>; } | null): void {
    if (!data) return;
    this.totalCoins = data.totalCoins || 0;
    this.todayRevenue = data.todayRevenue || 0;
    this.dailyHistory = data.dailyHistory || [0,0,0,0,0,0,0];
    this.jackpotsHit = data.jackpotsHit || 0;
    if (data.machines) {
      for (const saved of data.machines) {
        const m = this.machines.find(m => m.id === saved.id);
        if (m) {
          m.isBroken = saved.isBroken;
          m.isLocked = saved.isLocked;
          m.totalRevenue = saved.totalRevenue;
          m.dealers = saved.dealers || [];
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
