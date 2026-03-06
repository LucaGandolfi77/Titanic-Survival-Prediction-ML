// js/state.js — global reactive state
import { saveGame, loadGame } from './utils.js';

export const state = {
  day: 1,
  score: 0,
  selectedCharId: null,
  selectedOutfitId: null,
  activeSlot: 'top',
  // character id -> { top, bottom, shoes, accessory, outerwear, locked: Set }
  outfits: {},
  // pairKey -> pairState object (see relationships.js)
  pairs: {},
  // character id -> { mood, events[] }
  charData: {},
  // event queue for the current day
  eventQueue: [],
  phase: 'play', // 'play' | 'event' | 'wedding' | 'divorce' | 'gameover'
  gameOver: false,
};

export function hydrate(saved){
  Object.assign(state, saved);
  // Re-create Sets for locked items
  for(const cid in state.outfits){
    if(Array.isArray(state.outfits[cid].locked)){
      state.outfits[cid].locked = new Set(state.outfits[cid].locked);
    }
  }
}

export function persist(){
  // Serialise Sets to arrays before saving
  const ser = JSON.parse(JSON.stringify(state, (_k,v)=>v instanceof Set?[...v]:v));
  saveGame(ser);
}

export function tryLoad(){
  const saved = loadGame();
  if(saved){ hydrate(saved); return true; }
  return false;
}

