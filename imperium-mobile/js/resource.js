export const resources = {
  food: 500,
  wood: 400,
  gold: 300,
  stone: 200,
  population: { current: 0, max: 20 },
  income: { food: 0, wood: 0, gold: 0, stone: 0 },
  gathered: { food: 0, wood: 0, gold: 0, stone: 0 },

  add(type, amount) {
    if (this[type] !== undefined) {
      this[type] += amount;
      this.gathered[type] = (this.gathered[type] || 0) + amount;
    }
  },

  spend(type, amount) {
    if ((this[type] || 0) >= amount) {
      this[type] -= amount;
      return true;
    }
    return false;
  },

  canAfford(cost) {
    for (const [type, amount] of Object.entries(cost)) {
      if ((this[type] || 0) < amount) return false;
    }
    return true;
  },

  spendCost(cost) {
    if (!this.canAfford(cost)) return false;
    for (const [type, amount] of Object.entries(cost)) {
      this[type] -= amount;
    }
    return true;
  },

  addIncome(type, amount) {
    this.income[type] = (this.income[type] || 0) + amount;
  },

  updateIncome(dt) {
    for (const type of ['food', 'wood', 'gold', 'stone']) {
      this[type] += this.income[type] * dt;
    }
  },

  updatePopulation(current, max) {
    this.population = { current, max };
  },

  reset() {
    this.food = 500;
    this.wood = 400;
    this.gold = 300;
    this.stone = 200;
    this.income = { food: 0, wood: 0, gold: 0, stone: 0 };
    this.gathered = { food: 0, wood: 0, gold: 0, stone: 0 };
    this.population = { current: 0, max: 20 };
  },
};
