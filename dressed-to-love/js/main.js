// js/main.js — game loop, state machine, day progression, input wiring
import { state, tryLoad, persist } from './state.js';
import { CHARACTERS, CHAR_MAP } from './characters.js';
import { initScene, setCharAnimation, focusCharacter, resetCamera } from './scene3d.js';
import { initPairs, advancePairDay, getAllPairs } from './relationships.js';
import { initWardrobe, refreshWardrobe } from './wardrobe.js';
import { rollEvent, resolveEvent } from './events.js';
import { initAudio, playTheme, playSound, setMuted, isMuted } from './audio.js';
import {
  updateHUD, renderCharAvatars, renderPairPills, showModal,
  showToast, toggleRelMap, toggleStats, renderCharInfo
} from './ui.js';
import { rollDiscovery } from './betrayal.js';

// ─── Boot ─────────────────────────────────────────────────────────────────────
function init(){
  // Try restoring saved game
  tryLoad();

  // Ensure outfits map exists for all chars
  CHARACTERS.forEach(c=>{ if(!state.outfits[c.id]) state.outfits[c.id] = {}; });
  if(!state.selectedCharId) state.selectedCharId = CHARACTERS[0].id;
  if(!state.activeSlot) state.activeSlot = 'top';

  // Init relationship pairs
  initPairs();

  // Init 3D scene
  const canvas = document.getElementById('stage');
  if(canvas) initScene(canvas);

  // Init wardrobe panel
  initWardrobe();

  // Init audio (deferred to first click)
  initAudio();

  // Wire buttons
  _wireButtons();

  // First render
  renderCharAvatars();
  renderPairPills();
  updateHUD();
  renderCharInfo(CHAR_MAP[state.selectedCharId]);

  // Start music
  playTheme('morning');

  // Show welcome
  showToast('💍 Welcome to Dressed To Love! Select a character and equip an outfit.', 'love');
}

// ─── Button wiring ────────────────────────────────────────────────────────────
function _wireButtons(){
  // Primary controls (IDs in index.html)
  _btn('btn-next-day',   ()=>nextDay());
  _btn('btn-relationships', ()=>toggleRelMap());
  _btn('btn-stats',      ()=>toggleStats());
  _btn('btn-save',       ()=>{ persist(); showToast('Game saved! 💾','info'); playSound('chime'); });
  _btn('btn-reset',      ()=>_confirmReset());
  _btn('btn-mute',       ()=>_toggleMute());

  // Wire any element with data-close attribute to hide the target overlay
  document.querySelectorAll('[data-close]').forEach(el=>{
    el.addEventListener('click', ()=>{
      const target = el.getAttribute('data-close');
      if(!target) return;
      const node = document.getElementById(target);
      if(node) node.classList.add('hidden');
    });
  });
}

function _btn(id, fn){
  const b = document.getElementById(id);
  if(b) b.addEventListener('click', fn);
}

// ─── Next Day ─────────────────────────────────────────────────────────────────
export function nextDay(){
  state.day++;

  // Advance every pair
  const pairs = getAllPairs();
  pairs.forEach(p=>{ advancePairDay(p.a, p.b); });

  // Discovery rolls for affair pairs
  pairs.filter(p=>p.status==='affair').forEach(p=>rollDiscovery(p));

  // Daily event roll
  const result = rollEvent();
  if(result){
    const {event, pair} = result;
    const cA = CHAR_MAP[pair.a], cB = CHAR_MAP[pair.b];
    showModal(
      `📅 Day ${state.day} — ${event.label}`,
      `<p>Between <strong>${cA?.name}${cA?.emoji}</strong> and <strong>${cB?.name}${cB?.emoji}</strong></p>
       <p class="compat-value">Current status: <em>${pair.status}</em></p>`,
      event.choices.map((c,i)=>({
        text: c.text,
        fn: ()=>_onEventChoice(event, pair, i)
      }))
    );
  }

  // Score: +2 per existing relationship
  state.score += pairs.filter(p=>p.status!=='strangers').length * 2;

  // Music shift
  if(state.day % 10 === 0) playTheme(state.day % 20 === 0 ? 'wedding' : 'evening');
  else if(state.day % 5  === 0) playTheme('romance');

  // Check game-over (day 60)
  if(state.day >= 60) return _gameOver();

  // Update all UI
  updateHUD();
  renderPairPills();
  renderCharAvatars();
  persist();
}

function _onEventChoice(event, pair, i){
  resolveEvent(event, pair, i);
  updateHUD();
  renderPairPills();
  renderCharAvatars();
  refreshWardrobe();
  persist();
}

// ─── Game Over ────────────────────────────────────────────────────────────────
function _gameOver(){
  state.gameOver = true;
  const pairs    = getAllPairs();
  const married  = pairs.filter(p=>p.status==='married').length;
  const divorced = pairs.filter(p=>p.status==='divorced').length;
  playTheme('wedding');
  showModal(
    '🎉 Season Finale!',
    `<p>60 days on the runway are over!</p>
     <div class="stat-card">💍 Married: <strong>${married}</strong></div>
     <div class="stat-card">💔 Divorced: <strong>${divorced}</strong></div>
     <div class="stat-card">⭐ Final Score: <strong>${state.score}</strong></div>
     <p>${state.score>=300?'Fashion royalty! 👑':state.score>=150?'Pretty stylish! ✨':'Room to grow… 👗'}</p>`,
    [
      { text:'🔄 Play Again', fn:()=>_hardReset() },
      { text:'📊 View Stats', fn:()=>toggleStats() }
    ]
  );
}

// ─── Reset helpers ────────────────────────────────────────────────────────────
function _confirmReset(){
  showModal('⚠️ Reset Game?', '<p>All progress will be lost.</p>', [
    { text:'Yes, reset', fn:()=>_hardReset() },
    { text:'Cancel',     fn:()=>{} }
  ]);
}

function _hardReset(){
  import('./utils.js').then(({clearSave})=>{ clearSave(); location.reload(); });
}

function _toggleMute(){
  const now = isMuted();
  setMuted(!now);
  const btn = document.getElementById('btn-mute');
  if(btn) btn.textContent = now ? '🔊' : '🔇';
  showToast(now ? 'Sound on' : 'Muted', 'info');
}

// ─── Start ────────────────────────────────────────────────────────────────────
init();

