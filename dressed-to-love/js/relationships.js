// js/relationships.js — pair states, compatibility engine, status progression
import { CHARACTERS, CHAR_MAP, PERSONALITY_MATCH } from './characters.js';
import { state } from './state.js';
import { pairKey, rand, clamp } from './utils.js';

export const STATUSES     = ['strangers','acquaintances','friends','best_friends','crush','dating','engaged','married'];
export const BAD_STATUSES = ['rivals','enemies','affair','divorced','widowed'];
export const ALL_STATUSES = [...STATUSES, ...BAD_STATUSES];

export function makePair(idA, idB){
  return { a:idA, b:idB, status:'strangers', affinity:0, trust:50, tension:0,
           sharedEvents:0, daysKnown:0, affairWith:null, divorceRound:0 };
}

export function initPairs(){
  CHARACTERS.forEach((cA,i)=>{
    CHARACTERS.slice(i+1).forEach(cB=>{
      const key = pairKey(cA.id,cB.id);
      if(!state.pairs[key]){
        state.pairs[key] = makePair(cA.id,cB.id);
        const base = cA.baseAffinity?.[cB.id] ?? cB.baseAffinity?.[cA.id] ?? 20;
        state.pairs[key].affinity = base;
      }
    });
  });
}

export function getPair(aId,bId){ return state.pairs[pairKey(aId,bId)]; }

export function calcStyleSynergy(charAId, charBId){
  const cA = CHAR_MAP[charAId], cB = CHAR_MAP[charBId];
  if(!cA||!cB) return 50;
  const outA = Object.values(state.outfits[charAId]||{});
  const outB = Object.values(state.outfits[charBId]||{});
  const tagsA = outA.flatMap(o=>o?.compatTags||[]);
  const tagsB = outB.flatMap(o=>o?.compatTags||[]);
  let score = 50;
  tagsA.forEach(t=>{ if(tagsB.includes(t)) score += 8; });
  const sA = outA.map(o=>o?.style), sB = outB.map(o=>o?.style);
  sA.forEach(s=>{
    if(!s) return;
    if(cA.compatStyles?.includes(s) && cB.compatStyles?.includes(s)) score += 5;
    if(cA.incompatStyles?.includes(s) || cB.incompatStyles?.includes(s)) score -= 5;
  });
  return clamp(score, 0, 100);
}

export function calcPersonalityMatch(aId, bId){
  return PERSONALITY_MATCH[aId]?.[bId] ?? PERSONALITY_MATCH[bId]?.[aId] ?? 50;
}

export function calcCompat(aId, bId){
  const pair = getPair(aId,bId);
  if(!pair) return 0;
  const styleSynergy     = calcStyleSynergy(aId,bId);
  const personalityMatch = calcPersonalityMatch(aId,bId);
  const trustLevel       = pair.trust;
  const sharedBond       = clamp(pair.sharedEvents*3, 0, 100);
  return Math.round(styleSynergy*0.30 + personalityMatch*0.25 + trustLevel*0.25 + sharedBond*0.20);
}

function _autoStatus(pair, compat){
  if(BAD_STATUSES.includes(pair.status)) return;
  if(compat>=72 && pair.status==='dating'  && pair.daysKnown>=20) pair.status='engaged';
  else if(compat>=60 && pair.status==='crush' && pair.daysKnown>=10) pair.status='dating';
  else if(compat>=48 && pair.status==='best_friends' && pair.daysKnown>=8) pair.status='crush';
  else if(compat>=38 && pair.status==='friends' && pair.daysKnown>=6) pair.status='best_friends';
  else if(compat>=28 && pair.status==='acquaintances') pair.status='friends';
  else if(compat>=15 && pair.status==='strangers') pair.status='acquaintances';
  if(compat<20 && pair.trust<30 && pair.status==='friends') pair.status='rivals';
}

export function advancePairDay(aId, bId){
  const pair = getPair(aId,bId);
  if(!pair) return;
  pair.daysKnown++;
  const compat = calcCompat(aId,bId);
  if(pair.affinity < compat) pair.affinity = clamp(pair.affinity+rand(0,1.5), 0, 100);
  else pair.affinity = clamp(pair.affinity-rand(0,0.5), 0, 100);
  _autoStatus(pair, compat);
  if(['dating','engaged','married'].includes(pair.status)){
    if(pair.trust<40) pair.tension = clamp(pair.tension+rand(0,2), 0, 100);
    else pair.tension = clamp(pair.tension-rand(0,1), 0, 100);
  }
}

export function boostAffinity(aId, bId, amount){
  const pair = getPair(aId,bId); if(!pair) return;
  pair.affinity = clamp(pair.affinity+amount, 0, 100);
  pair.sharedEvents++;
  pair.trust = clamp(pair.trust+amount*0.4, 0, 100);
  _autoStatus(pair, calcCompat(aId,bId));
}

export function damageAffinity(aId, bId, amount){
  const pair = getPair(aId,bId); if(!pair) return;
  pair.affinity = clamp(pair.affinity-amount, 0, 100);
  pair.trust    = clamp(pair.trust-amount*0.6, 0, 100);
  pair.tension  = clamp(pair.tension+amount*0.5, 0, 100);
}

export function setPairStatus(aId, bId, status){
  const pair = getPair(aId,bId); if(pair) pair.status = status;
}

export function getAllPairs(){ return Object.values(state.pairs); }
export function getPairsForChar(charId){
  return getAllPairs().filter(p=>p.a===charId||p.b===charId);
}
export function getPartner(charId){
  return getAllPairs().find(p=>(p.a===charId||p.b===charId)&&['engaged','married','dating'].includes(p.status));
}

