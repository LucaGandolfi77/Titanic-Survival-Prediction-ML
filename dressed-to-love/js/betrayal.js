// js/betrayal.js — affair system, discovery roll, confrontation
import { state, persist } from './state.js';
import { getPair, setPairStatus, damageAffinity, boostAffinity, getAllPairs } from './relationships.js';
import { showModal, showToast, renderRelMap } from './ui.js';
import { CHAR_MAP } from './characters.js';
import { rand, pairKey } from './utils.js';
import { spawnHeartParticles, setCharAnimation } from './scene3d.js';
import { playSound } from './audio.js';

// Start a secret affair between two characters
export function startAffair(actorId, targetId){
  const pair = getPair(actorId, targetId);
  if(!pair) return;

  // Find actor’s current partner (if any)
  const allPairs = getAllPairs();
  const partnerPair = allPairs.find(p=>
    (p.a===actorId||p.b===actorId) &&
    ['married','engaged','dating'].includes(p.status) &&
    pairKey(p.a,p.b) !== pairKey(actorId,targetId)
  );

  pair.affairWith = targetId;
  pair.status     = 'affair';
  if(partnerPair) partnerPair.tension += 25;

  const actor  = CHAR_MAP[actorId];
  const target = CHAR_MAP[targetId];
  showToast(`🔥 ${actor.name} is having a secret affair with ${target.name}!`,'danger');
  persist();
}

// Daily discovery roll: called each day for affair pairs
export function rollDiscovery(affairPair){
  if(affairPair.status !== 'affair') return;
  const actorId  = affairPair.a;
  const targetId = affairPair.b;

  // Find the cheated-on partner pair
  const allPairs = getAllPairs();
  const victimPair = allPairs.find(p=>
    (p.a===actorId||p.b===actorId) &&
    ['married','engaged','dating'].includes(p.status) &&
    pairKey(p.a,p.b) !== pairKey(actorId,targetId)
  );
  if(!victimPair) return;

  const tension = victimPair.tension || 0;
  const choices = affairPair.sharedEvents || 0;
  const roll = tension*0.4 + choices*0.3 + rand(0,30);

  if(roll >= 60){
    const cheater = CHAR_MAP[actorId];
    const victim  = CHAR_MAP[victimPair.a===actorId ? victimPair.b : victimPair.a];
    const sidepiece = CHAR_MAP[targetId];
    showModal(
      `🔍 Affair Discovered!`,
      `<p><strong>${victim.name}</strong> has discovered ${cheater.name}'s affair with ${sidepiece.name}!</p>
       <p>Trust: <strong>${Math.round(victimPair.trust)}</strong> &bull; Tension: <strong>${Math.round(tension)}</strong></p>`,
      [
        {
          text:'💔 Expose & End it',
          fn:()=>_expose(actorId, targetId, victimPair, affairPair)
        },
        {
          text:'🤡 Forgive (secretly)',
          fn:()=>_forgive(actorId, victimPair, affairPair)
        },
        {
          text:'⚔️ Confront publicly',
          fn:()=>_publicConfront(actorId, victimPair, affairPair, cheater, victim, sidepiece)
        }
      ]
    );
    playSound('dramatic');
  }
}

function _expose(actorId, targetId, victimPair, affairPair){
  affairPair.status = 'acquaintances';
  affairPair.affairWith = null;
  damageAffinity(victimPair.a, victimPair.b, 30);
  victimPair.trust = Math.max(0, victimPair.trust - 40);
  victimPair.tension = Math.max(0, victimPair.tension - 10);
  const actor  = CHAR_MAP[actorId];
  showToast(`${actor.name}'s affair is exposed! 💔`,'danger');
  setCharAnimation(actorId,'sad');
  persist();
  renderRelMap();
}

function _forgive(actorId, victimPair, affairPair){
  affairPair.status = 'acquaintances';
  affairPair.affairWith = null;
  damageAffinity(victimPair.a, victimPair.b, 15);
  victimPair.trust = Math.max(0, victimPair.trust - 20);
  victimPair.tension = Math.max(0, victimPair.tension - 20);
  showToast('They decided to quietly forgive… for now.','info');
  persist();
}

function _publicConfront(actorId, victimPair, affairPair, cheater, victim, sidepiece){
  affairPair.status = 'enemies';
  affairPair.affairWith = null;
  damageAffinity(victimPair.a, victimPair.b, 50);
  victimPair.trust = 0;
  victimPair.tension = 100;
  const noCheaterPartner = victimPair.a===actorId ? victimPair.b : victimPair.a;
  damageAffinity(actorId, noCheaterPartner, 20);
  state.score -= 20;
  showToast(`🗣️ ${victim.name} confronts ${cheater.name} in front of everyone!`,'danger');
  setCharAnimation(actorId,'fight');
  setCharAnimation(victimPair.a===actorId ? victimPair.b : victimPair.a,'fight');
  setTimeout(()=>{ setCharAnimation(actorId,'idle'); setCharAnimation(victimPair.a===actorId?victimPair.b:victimPair.a,'idle'); }, 3000);
  persist();
  renderRelMap();
}

