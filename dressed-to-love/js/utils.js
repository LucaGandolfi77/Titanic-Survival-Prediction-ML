// js/utils.js — helpers: random, lerp, UUID, save/load

export const rand     = (min=0,max=1)=>Math.random()*(max-min)+min;
export const randInt  = (min,max)=>Math.floor(rand(min,max+1));
export const lerp     = (a,b,t)=>a+(b-a)*t;
export const clamp    = (v,lo,hi)=>Math.max(lo,Math.min(hi,v));
export const uid      = ()=>Math.random().toString(36).slice(2,9);
export const pick     = arr=>arr[randInt(0,arr.length-1)];
export const shuffle  = arr=>[...arr].sort(()=>Math.random()-.5);

export function hexToRgb(hex){
  const r=parseInt(hex.slice(1,3),16);
  const g=parseInt(hex.slice(3,5),16);
  const b=parseInt(hex.slice(5,7),16);
  return {r,g,b};
}

// Save full game state to localStorage
export function saveGame(state){
  try{
    localStorage.setItem('dtl_save', JSON.stringify(state));
  }catch(e){console.warn('Save failed',e)}
}

// Load game state; returns null if nothing saved
export function loadGame(){
  try{
    const s=localStorage.getItem('dtl_save');
    return s?JSON.parse(s):null;
  }catch(e){return null}
}

export function clearSave(){
  localStorage.removeItem('dtl_save');
}

// Format number with + sign
export const fmtDelta=(v)=>(v>0?`+${v}`:String(v));

// Two-character pair key (sorted)
export const pairKey=(a,b)=>[a,b].sort().join(':');

