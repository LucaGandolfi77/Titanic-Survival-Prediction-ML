// js/romance.js — flirt, dating, engagement, wedding ceremony
import { state } from './state.js';
import { getPair, setPairStatus, boostAffinity } from './relationships.js';
import { showModal, showToast, renderRelMap } from './ui.js';
import { spawnHeartParticles, setCharAnimation, showWeddingDecor } from './scene3d.js';
import { CHAR_MAP } from './characters.js';
import { persist } from './state.js';
import { playSound } from './audio.js';

export function tryFlirt(actorId, targetId){
  const pair = getPair(actorId, targetId);
  if(!pair) return;
  if(['enemies','rivals'].includes(pair.status)){
    showToast('They want nothing to do with you right now.','danger');
    return;
  }
  const actor  = CHAR_MAP[actorId];
  const target = CHAR_MAP[targetId];
  showModal(
    `❤️ ${actor.name} flirts with ${target.name}${target.emoji}`,
    `<p>"${_flirtLine(actor, target)}"</p><p>Affinity: <strong>${Math.round(pair.affinity)}</strong></p>`,
    [
      { text:'💋 Go for it!',    fn:()=>{ boostAffinity(actorId,targetId,8); setCharAnimation(actorId,'flirt'); setCharAnimation(targetId,'flirt'); spawnHeartParticles(actorId); showToast(`${actor.name} flirted with ${target.name}!`,'love'); persist(); } },
      { text:'🙈 Play it cool', fn:()=>{ boostAffinity(actorId,targetId,3); showToast(`${actor.name} played it cool.`,'info'); } },
      { text:'❌ Cancel',      fn:()=>{} }
    ]
  );
}

export function proposeDate(actorId, targetId){
  const pair = getPair(actorId, targetId);
  if(!pair || pair.affinity < 40){ showToast('They\'re not interested yet.','danger'); return; }
  const actor  = CHAR_MAP[actorId];
  const target = CHAR_MAP[targetId];
  showModal(
    `☕ Date Proposal!`,
    `<p>${actor.name} asks ${target.name} on a date.</p>`,
    [
      { text:'💗 Yes please!', fn:()=>{ setPairStatus(actorId,targetId,'dating'); boostAffinity(actorId,targetId,15); showToast(`${actor.name} & ${target.name} are now dating!`,'love'); playSound('chime'); persist(); renderRelMap(); } },
      { text:'💔 No thanks',  fn:()=>{ showToast(`${target.name} politely declined.`,'info'); } }
    ]
  );
}

export function proposeEngagement(actorId, targetId){
  const pair = getPair(actorId, targetId);
  if(!pair || pair.status !== 'dating'){ showToast('You need to be dating first!','danger'); return; }
  if(pair.affinity < 70){ showToast('The time isn\'t right yet…','danger'); return; }
  const actor  = CHAR_MAP[actorId];
  const target = CHAR_MAP[targetId];
  showModal(
    `💍 Engagement Proposal!`,
    `<p>${actor.name} gets down on one knee for ${target.name}${target.emoji}!</p>
     <p class="compat-value">Love Meter: ${Math.round(pair.affinity)}%</p>`,
    [
      { text:'💍 Yes! Absolutely!', fn:()=>{ setPairStatus(actorId,targetId,'engaged'); boostAffinity(actorId,targetId,25); state.score+=60; showToast(`${actor.name} & ${target.name} are ENGAGED! 💍`,'love'); playSound('fanfare'); persist(); renderRelMap(); } },
      { text:'⏳ I need more time…', fn:()=>showToast(`${target.name} asked for more time.`,'info') },
      { text:'💔 No.',              fn:()=>{ showToast(`${target.name} said no. 💔`,'danger'); } }
    ]
  );
}

export function holdWedding(actorId, targetId){
  const pair = getPair(actorId, targetId);
  if(!pair || pair.status !== 'engaged'){
    showToast('They need to be engaged first!','danger'); return;
  }
  const actor  = CHAR_MAP[actorId];
  const target = CHAR_MAP[targetId];
  showModal(
    `💍 The Wedding of ${actor.name} & ${target.name}`,
    `<div class="wedding-stage">
       <p>✨ ${actor.emoji} marries ${target.emoji} ✨</p>
       <p>The runway clears. Guests gather. Music fills the air.</p>
       <p class="compat-value">Bond Strength: ${Math.round(pair.affinity)}%</p>
     </div>`,
    [
      { text:'💍 I do ❤️', fn:()=>{
          setPairStatus(actorId,targetId,'married');
          boostAffinity(actorId,targetId,30);
          state.score += 100;
          showWeddingDecor(true);
          setCharAnimation(actorId,'dance');
          setCharAnimation(targetId,'dance');
          spawnHeartParticles(actorId);
          spawnHeartParticles(targetId);
          playSound('wedding');
          showToast(`💍 ${actor.name} & ${target.name} are MARRIED!`,'love');
          persist();
          renderRelMap();
          setTimeout(()=>{ setCharAnimation(actorId,'idle'); setCharAnimation(targetId,'idle'); showWeddingDecor(false); }, 6000);
      }},
      { text:'🚪 Leave at the altar', fn:()=>{
          setPairStatus(actorId,targetId,'divorced');
          showToast(`${actor.name} left ${target.name} at the altar. 💔`,'danger');
          persist(); renderRelMap();
      }}
    ]
  );
}

function _flirtLine(actor, target){
  const lines = [
    `Your outfit looks stunning today, ${target.name}.`,
    `I can’t stop thinking about your style, ${target.name}.`,
    `Did it hurt when you fell from the runway? Because you’re gorgeous.`,
    `You and I would make the most fashionable couple.`,
    `I love how ${actor.style} looks on you.`,
    `Every time I see you, my heart skips a beat.`,
  ];
  return lines[Math.floor(Math.random()*lines.length)];
}

