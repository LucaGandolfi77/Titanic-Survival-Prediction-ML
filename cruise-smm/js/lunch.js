/* ===== LUNCH BREAK SYSTEM ===== */
import { getState } from './state.js';
import { clamp } from './utils.js';
import { boostTeamHappiness, boostMemberHappiness } from './team.js';

export const FOOD_MENU = [
  { id: 'salad',     emoji: '🥗', name: 'Caesar Salad',      price: 8,  energy: 10, happiness: 5,  bonusMember: null, bonusAmount: 0 },
  { id: 'carbonara', emoji: '🍝', name: 'Pasta Carbonara',   price: 12, energy: 20, happiness: 15, bonusMember: 'marco', bonusAmount: 5 },
  { id: 'sushi',     emoji: '🍣', name: 'Sushi Platter',     price: 18, energy: 25, happiness: 20, bonusMember: 'yuki', bonusAmount: 8 },
  { id: 'steak',     emoji: '🥩', name: 'Grilled Steak',     price: 22, energy: 30, happiness: 18, bonusMember: null, bonusAmount: 0 },
  { id: 'pizza',     emoji: '🍕', name: 'Margherita Pizza',   price: 10, energy: 15, happiness: 12, bonusMember: null, bonusAmount: 0 },
  { id: 'tacos',     emoji: '🌮', name: 'Tacos del Mar',     price: 14, energy: 18, happiness: 14, bonusMember: 'diego', bonusAmount: 6 },
  { id: 'bento',     emoji: '🍱', name: 'Bento Box',         price: 16, energy: 22, happiness: 16, bonusMember: null, bonusAmount: 0 },
  { id: 'croissant', emoji: '🥐', name: 'Croissant & Fruit', price: 6,  energy: 8,  happiness: 8,  bonusMember: 'sofia', bonusAmount: 4 },
];

export const DRINKS_MENU = [
  { id: 'espresso',  emoji: '☕', name: 'Espresso',          price: 3,  energy: 0,   happiness: 0,  focus: 10, effect: 'Next task fame +5%' },
  { id: 'orange',    emoji: '🧃', name: 'Fresh Orange',      price: 4,  energy: 8,   happiness: 0,  focus: 0,  effect: '+8 energy' },
  { id: 'prosecco',  emoji: '🥂', name: 'Prosecco Toast',    price: 8,  energy: 0,   happiness: 10, focus: 0,  effect: 'Team happiness +10', teamHappiness: 10 },
  { id: 'cocktail',  emoji: '🍹', name: 'Tropical Cocktail', price: 10, energy: -5,  happiness: 15, focus: 0,  effect: '+15 happiness, -5 energy' },
  { id: 'tea',       emoji: '🫖', name: 'Herbal Tea',        price: 3,  energy: 0,   happiness: 0,  focus: 5,  effect: 'Reduces task failure chance' },
  { id: 'smoothie',  emoji: '🥛', name: 'Smoothie Bowl',     price: 6,  energy: 12,  happiness: 8,  focus: 0,  effect: '+12 energy, +8 happiness' },
  { id: 'beer',      emoji: '🍺', name: 'Craft Beer',        price: 6,  energy: 0,   happiness: 12, focus: -8, effect: '+12 happiness, -8 focus' },
];

export const TEAM_TREATS = [
  { id: 'coffees',    emoji: '☕', name: 'Round of Coffees',  price: 12, teamHappiness: 8,  fame: 0,  description: 'Coffee for everyone!' },
  { id: 'pizza_party',emoji: '🍕', name: 'Pizza Party',       price: 40, teamHappiness: 20, fame: 0,  description: 'Team pizza party!' },
  { id: 'champagne',  emoji: '🥂', name: 'Champagne Toast',   price: 60, teamHappiness: 30, fame: 50, description: 'Luxury toast + social post!' },
];

export function applyFood(state, food) {
  if (state.money < food.price) return false;
  state.money -= food.price;
  state.dailyMoneySpent += food.price;
  state.energy = clamp(state.energy + food.energy, 0, 100);
  boostTeamHappiness(state, food.happiness);

  if (food.bonusMember && state.teamMembers[food.bonusMember]) {
    boostMemberHappiness(state, food.bonusMember, food.bonusAmount);
  }
  return true;
}

export function applyDrink(state, drink) {
  if (state.money < drink.price) return false;
  state.money -= drink.price;
  state.dailyMoneySpent += drink.price;
  state.energy = clamp(state.energy + (drink.energy || 0), 0, 100);
  state.focus = clamp(state.focus + (drink.focus || 0), -20, 30);

  if (drink.happiness) {
    boostTeamHappiness(state, drink.happiness);
  }
  if (drink.teamHappiness) {
    boostTeamHappiness(state, drink.teamHappiness);
  }
  return true;
}

export function applyTeamTreat(state, treat) {
  if (state.money < treat.price) return false;
  state.money -= treat.price;
  state.dailyMoneySpent += treat.price;
  boostTeamHappiness(state, treat.teamHappiness);
  if (treat.fame > 0) {
    state.fame += treat.fame;
    state.dailyFame += treat.fame;
  }
  state.dailyEvents.push(`${treat.emoji} ${treat.name}!`);
  return true;
}

export function inviteCharacterToLunch(state, charId) {
  const cs = state.characters[charId];
  if (!cs || cs.love < 20) return false; // changed from 30 to 20 for easier access
  cs.love = clamp(cs.love + 15, 0, 100);
  state.dailyEvents.push(`💕 Lunch with ${charId}! Love +15`);
  return true;
}
