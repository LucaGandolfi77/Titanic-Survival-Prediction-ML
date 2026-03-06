// js/social.js — BFF Style Pact, fashion duels, enemy reconciliation
import { state, persist } from './state.js';
import { getPair, setPairStatus, boostAffinity, damageAffinity } from './relationships.js';
import { showModal, showToast, renderRelMap } from './ui.js';
import { CHAR_MAP } from './characters.js';
import { calcStyleSynergy } from './relationships.js';
import { setCharAnimation, spawnHeartParticles } from './scene3d.js';
import { playSound } from './audio.js';

// BFF Style Pact — available for best_friends
export function proposeStylePact(aId, bId){
  const pair = getPair(aId, bId);
  if(!pair || pair.status !== 'best_friends'){
    showToast('You need to be Best Friends first!','danger'); return;
  }
  const cA = CHAR_MAP[aId], cB = CHAR_MAP[bId];
  const synergy = calcStyleSynergy(aId, bId);
  showModal(
    `💫 BFF Style Pact`,
    `<p>${cA.name}${cA.emoji} & ${cB.name}${cB.emoji} want to form a Style Pact!</p>
     <p>Style Synergy: <strong>${synergy}%</strong></p>
     <p>A Style Pact boosts both characters' compat with everyone by +8.</p>`,
    [
      { text:'🤝 Seal the pact!', fn:()=>{
          pair.sharedEvents += 5;
          boostAffinity(aId, bId, 20);
          state.score += 35;
          // BFF badge
          if(!state.stylePacts) state.stylePacts = [];
          state.stylePacts.push({a:aId, b:bId, day:state.day});
          setCharAnimation(aId,'happy'); setCharAnimation(bId,'happy');
          spawnHeartParticles(aId); spawnHeartParticles(bId);
          showToast(`✨ ${cA.name} & ${cB.name} have a Style Pact!`,'love');
          playSound('chime');
          persist(); renderRelMap();
      }},
      { text:'🤷 Not today', fn:()=>showToast('Maybe another time.','info') }
    ]
  );
}

// Fashion Duel — available for rivals
export function challengeDuel(aId, bId){
  const pair = getPair(aId, bId);
  if(!pair || !['rivals','enemies'].includes(pair.status)){
    showToast('Fashion duels are for rivals only!','danger'); return;
  }
  const cA = CHAR_MAP[aId], cB = CHAR_MAP[bId];
  const synA = calcStyleSynergy(aId, bId);
  // Simulate duel winner based on style synergy + random
  const scoreA = synA + Math.random()*30;
  const scoreB = 60   + Math.random()*30;
  const winner  = scoreA > scoreB ? cA : cB;
  const loser   = winner === cA ? cB : cA;
  showModal(
    `⚔️ Fashion Duel: ${cA.name} vs ${cB.name}`,
    `<p>Both characters strut their best outfits on the runway.</p>
     <p>${cA.name}: <strong>${Math.round(scoreA)}</strong> pts &nbsp;|&nbsp; ${cB.name}: <strong>${Math.round(scoreB)}</strong> pts</p>
     <p>🏆 Winner: <strong>${winner.name}</strong></p>`,
    [
      { text:'🎉 Accept result', fn:()=>{
          boostAffinity(winner.id, loser.id, 8);
          damageAffinity(loser.id, winner.id, 4);
          if(pair.affinity > 40) setPairStatus(aId, bId, 'acquaintances');
          state.score += winner===cA ? 20 : 5;
          showToast(`${winner.name} wins the duel!`, winner===cA?'love':'info');
          setCharAnimation(winner.id,'happy'); setCharAnimation(loser.id,'sad');
          playSound('fanfare'); persist(); renderRelMap();
          setTimeout(()=>{ setCharAnimation(winner.id,'idle'); setCharAnimation(loser.id,'idle'); }, 2500);
      }},
      { text:'😡 Demand rematch', fn:()=>{ damageAffinity(aId,bId,5); showToast('They\'re not done!','danger'); } }
    ]
  );
}

// Enemy reconciliation attempt
export function tryReconcile(aId, bId){
  const pair = getPair(aId, bId);
  if(!pair || pair.status !== 'enemies'){
    showToast('They are not enemies.','info'); return;
  }
  const cA = CHAR_MAP[aId], cB = CHAR_MAP[bId];
  // Chance based on affinity remnant
  const chance = pair.affinity > 25 ? 'good' : 'low';
  showModal(
    `🤝 Reconciliation Attempt`,
    `<p>${cA.name} reaches out to ${cB.name} with an olive branch.</p>
     <p>Chance of success: <em>${chance}</em></p>`,
    [
      { text:'🌿 Offer peace', fn:()=>{
          const success = pair.affinity > 25 || Math.random() > 0.55;
          if(success){
            setPairStatus(aId, bId, 'acquaintances');
            boostAffinity(aId, bId, 15);
            showToast(`${cA.name} & ${cB.name} are no longer enemies!`,'love');
            playSound('chime'); renderRelMap();
          } else {
            damageAffinity(aId, bId, 5);
            showToast(`${cB.name} rejected the olive branch.`,'danger');
          }
          persist();
      }},
      { text:'🚪 Back off', fn:()=>{} }
    ]
  );
}

