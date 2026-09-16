export interface OakState {
  name: string;
  age: number;
  height: number;
  trunkGirth: number;
  leaves: number;
  energy: number;
  maxEnergy: number;
  fertility: number;
  charisma: number;
  dnaPoints: number;
  acorns: number;
  generation: number;
  isMeditating: boolean;
  upgrades: DNAUpgrade[];
  milestones: Milestone[];
  energyGenMul: number;
  breedSuccessMul: number;
  growthSpeedMul: number;
  buffs: Buff[];
  canBreedAnimals(): boolean;
  canBreedTaliban(): boolean;
  addBuff(id: string, duration: number, mul: number): void;
}

export interface DNAUpgrade {
  id: string;
  name: string;
  cost: number;
  desc: string;
  purchased: boolean;
}

export interface Milestone {
  height: number;
  name: string;
  desc: string;
  triggered: boolean;
}

export interface Buff {
  id: string;
  remaining: number;
  multiplier: number;
}

export interface CasinoState {
  totalCoins: number;
  todayRevenue: number;
  dailyHistory: number[];
  jackpotsHit: number;
  isClosed: boolean;
  machines: MachineState[];
  npcs: NPCState[];
  revenueMul: number;
  closeCasino(duration: number): void;
  repairMachine(machineId: number): boolean;
  unlockMachine(machineId: number): boolean;
  hireDealer(offspringId: string, machineId: number): boolean;
  spendCoins(amount: number): boolean;
  getTotalRPM(): number;
}

export interface MachineState {
  id: number;
  name: string;
  unlockHeight: number;
  symbols: SymbolDef[];
  bet: number;
  spinsPerMin: number;
  payoutRate: number;
  breakdownChance: number;
  isBroken: boolean;
  isLocked: boolean;
  totalRevenue: number;
  revenuePerMinute: number;
  currentReels: string[];
  spinning: boolean;
  lastResult: SpinResult | null;
  _spinTimer: number;
  _sleepTimer: number;
  dealers: string[];
  _spinTimeoutId?: number;
}

export interface NPCState {
  id: string;
  name: string;
  personality: string;
  quotes: string[];
  special: string;
  machineId: number;
  mood: number;
  beardLength: number;
  currentQuote: string;
  _quoteTimer: number;
  _prayTimer: number;
  _sleepTimer: number;
  isSleeping: boolean;
  isPraying: boolean;
}

export interface SymbolDef {
  s: string;
  mult: number;
  label?: string;
}

export interface SpinResult {
  win: boolean;
  mult: number;
  jackpot: boolean;
}

export interface Offspring {
  id: string;
  name: string;
  type: string;
  category: 'plant' | 'animal' | 'taliban';
  emoji: string;
  generation: number;
  age: number;
  stats: OffspringStats;
  traits: string[];
  role: string | null;
  description: string;
  bornAt: number;
}

export interface OffspringStats {
  health: number;
  energy: number;
  strength: number;
  charisma: number;
  speed: number;
  luck: number;
  [key: string]: number;
}

export interface Partner {
  name: string;
  emoji: string;
  category: 'plant' | 'animal' | 'taliban';
  compatibility: number;
  speedBonus: number;
  charismaBonus: number;
  luckBonus: number;
  desc: string;
  id: string;
}

export interface BreedingState {
  breedCooldown: number;
  breedSuccessBoost: number;
  _boostTimer: number;
  addSuccessBoost(amount: number, duration: number): void;
}

export interface GameState {
  oak: OakState;
  casino: CasinoState;
  breeding: BreedingState;
  population: PopulationState;
  _oakQuote: string;
}

export interface EventDef {
  id: string;
  name: string;
  type: 'positive' | 'negative' | 'weird';
  desc: string;
  apply: (gs: GameState) => void;
}

export interface PopulationState {
  partners: Partner[];
  offspring: Offspring[];
}
