// js/divorce.js — 3-step divorce: confirm, asset split, social fallout
import { state, persist } from './state.js';
import { getPair, setPairStatus, damageAffinity, getAllPairs } from './relationships.js';
import { showModal, showToast, renderRelMap } from './ui.js';
import { CHAR_MAP } from './characters.js';
import { setCharAnimation, updateOutfitColor } from './scene3d.js';
import { playSound } from './audio.js';

// Entry point—open divorce wizard step 1
export function openDivorce(aId, bId){
  const pair = getPair(aId, bId);
  if(!pair || !['married','engaged'].includes(pair.status)){
    showToast('They\'re not married or engaged.','danger');
    return;
  }
  const cA = CHAR_MAP[aId], cB = CHAR_MAP[bId];
  _step1Confirm(aId, bId, pair, cA, cB);
}

function _step1Confirm(aId, bId, pair, cA, cB){
  showModal(
    `💔 Divorce: Step 1 of 3 — Confirm`,
    `<p>Are you sure you want to end ${cA.name} & ${cB.name}'s relationship?</p>
     <p>Bond Strength: <strong>${Math.round(pair.affinity)}%</strong></p>
     <p class="warning-text">⚠️ This will cause significant social fallout.</p>`,
    [
      { text:'📋 Proceed to asset split', fn:()=>_step2AssetSplit(aId, bId, pair, cA, cB) },
      { text:'💗 Reconcile instead',      fn:()=>{ showToast('Glad you reconsidered!','love'); } }
    ]
  );
}

function _step2AssetSplit(aId, bId, pair, cA, cB){
  // Each outfit slot the pair share is assigned
  const slotsA = Object.keys(state.outfits[aId]||{});
  const slotsB = Object.keys(state.outfits[bId]||{});
  const sharedSlots = slotsA.filter(s=>slotsB.includes(s));

  const splitHTML = sharedSlots.length
    ? `<p>Shared wardrobe items to split:</p><ul>${sharedSlots.map(s=>`<li>${s} outfit</li>`).join('')}</ul>`
    : '<p>No shared wardrobe items. Clean split.</p>';

  showModal(
    `⚖️ Divorce: Step 2 of 3 — Asset Split`,
    `${splitHTML}<p>Who keeps the shared style points?</p>`,
    [
      { text:`👗 ${cA.name} keeps them`, fn:()=>{ state.score = Math.max(0, state.score - 20); _step3Fallout(aId,bId,pair,cA,cB,'A'); } },
      { text:`👗 ${cB.name} keeps them`, fn:()=>{ state.score = Math.max(0, state.score - 20); _step3Fallout(aId,bId,pair,cA,cB,'B'); } },
      { text:'Split evenly ✌️',           fn:()=>{ state.score = Math.max(0, state.score - 10); _step3Fallout(aId,bId,pair,cA,cB,'split'); } }
    ]
  );
}

function _step3Fallout(aId, bId, pair, cA, cB, keepSide){
  // Apply divorce
  setPairStatus(aId, bId, 'divorced');
  damageAffinity(aId, bId, 40);
  pair.trust    = 0;
  pair.tension  = 80;

  // Social fallout: damage all close pairs of both
  const affected = getAllPairs().filter(p=>
    (p.a===aId||p.b===aId||p.a===bId||p.b===bId) &&
    ['friends','best_friends'].includes(p.status)
  );
  affected.forEach(p=>{ p.trust = Math.max(0,p.trust-10); });

  // Post-divorce outfit glow-up (reset to primary colors)
  ['top','bottom','shoes'].forEach(slot=>{
    updateOutfitColor(aId, slot, '#e91e63');
    updateOutfitColor(bId, slot, '#673ab7');
  });

  setCharAnimation(aId, 'sad');
  setCharAnimation(bId, 'sad');
  playSound('dramatic');
  state.score -= 30;
  persist();
  renderRelMap();

  showModal(
    `💔 Divorce: Step 3 of 3 — Fallout`,
    `<p>${cA.name} & ${cB.name} are now divorced.</p>
     <p>Both characters will go through a <em>glow-up</em> phase with new outfit colors.</p>
     <p>Score impact: <strong>-30</strong></p>
     <p class="compat-value">Social fallout affected ${affected.length} relationships.</p>`,
    [
      { text:'OK 😢', fn:()=>{ setTimeout(()=>{ setCharAnimation(aId,'idle'); setCharAnimation(bId,'idle'); }, 2000); } }
    ]
  );
}

