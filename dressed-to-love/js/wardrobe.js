// js/wardrobe.js — wardrobe panel UI: tabs, cards, equip, compat preview
import { state, persist } from './state.js';
import { OUTFIT_MAP, bySlot, moodSummary } from './outfits.js';
import { CHAR_MAP } from './characters.js';
import { calcCompat } from './relationships.js';
import { updateOutfitColor } from './scene3d.js';

const SLOTS = ['top','bottom','shoes','accessory','outerwear'];
const SLOT_LABELS = {top:'Tops',bottom:'Bottoms',shoes:'Shoes',accessory:'Accessories',outerwear:'Outerwear'};

export function initWardrobe(){
  _buildTabs();
  _ensureWardrobeShell();
  _buildCards(state.activeSlot||'top');
  _buildColorPicker();
  _buildCompatBar();
}

function _ensureWardrobeShell(){
  const container = document.getElementById('wardrobe-cards');
  if(!container) return;
  // If already initialized, skip
  if(container.dataset.initialized) return;
  container.innerHTML = `
    <div class="wardrobe-inner">
      <div id="compat-preview" class="compat-preview" style="opacity:0;transition:opacity .2s"></div>
      <div id="outfit-grid" class="outfit-grid"></div>
      <div id="color-picker-wrap" class="color-picker"></div>
      <div id="wardrobe-compat" class="wardrobe-compat"></div>
    </div>`;
  container.dataset.initialized = '1';
}

function _buildTabs(){
  const tabs = document.getElementById('wardrobe-tabs');
  if(!tabs) return;
  tabs.innerHTML = SLOTS.map(s=>`
    <button class="tab-btn${s===(state.activeSlot||'top')?' active':''}" data-slot="${s}">
      ${SLOT_LABELS[s]}
    </button>`).join('');
  tabs.querySelectorAll('.tab-btn').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      state.activeSlot = btn.dataset.slot;
      tabs.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      _buildCards(state.activeSlot);
    });
  });
}

function _buildCards(slot){
  const grid = document.getElementById('outfit-grid');
  if(!grid) return;
  const items = bySlot(slot);
  grid.innerHTML = items.map(item=>{
    const equipped = _isEquipped(item.id);
    const locked   = state.lockedOutfits?.[item.id];
    return `<div class="outfit-card${equipped?' equipped':''}${locked?' locked':''}" data-id="${item.id}">
      <div class="card-swatch" style="background:${item.colors.primary}"></div>
      <div class="card-info">
        <span class="card-name">${item.name}</span>
        <span class="style-badge style-${item.style}">${item.style}</span>
      </div>
      ${locked?'<span class="lock-icon">🔒</span>':''}
    </div>`;
  }).join('');
  grid.querySelectorAll('.outfit-card').forEach(card=>{
    card.addEventListener('click', ()=>_equipItem(card.dataset.id));
    card.addEventListener('mouseenter', ()=>_showCompat(card.dataset.id));
    card.addEventListener('mouseleave', ()=>_clearCompat());
    card.addEventListener('contextmenu', e=>{ e.preventDefault(); _toggleLock(card.dataset.id); });
  });
}

function _isEquipped(outfitId){
  const cId = state.selectedCharId; if(!cId) return false;
  const item = OUTFIT_MAP[outfitId]; if(!item) return false;
  return state.outfits[cId]?.[item.slot]?.id === outfitId;
}

function _equipItem(outfitId){
  const cId = state.selectedCharId; if(!cId) return;
  if(state.lockedOutfits?.[outfitId]) return;
  const item = OUTFIT_MAP[outfitId]; if(!item) return;
  if(!state.outfits[cId]) state.outfits[cId] = {};
  state.outfits[cId][item.slot] = item;
  updateOutfitColor(cId, item.slot, item.colors.primary);
  _buildCards(state.activeSlot);
  _buildCompatBar();
  persist();
}

function _showCompat(outfitId){
  const item = OUTFIT_MAP[outfitId]; if(!item) return;
  const preview = document.getElementById('compat-preview'); if(!preview) return;
  const mood = moodSummary(item);
  const moodText = Array.isArray(mood) ? mood.join(' · ') : (mood || '');
  preview.textContent = `${item.name} — ${moodText}`;
  preview.style.opacity = '1';
}
function _clearCompat(){
  const preview = document.getElementById('compat-preview');
  if(preview) preview.style.opacity = '0';
}

function _toggleLock(outfitId){
  if(!state.lockedOutfits) state.lockedOutfits = {};
  state.lockedOutfits[outfitId] = !state.lockedOutfits[outfitId];
  _buildCards(state.activeSlot);
  persist();
}

function _buildColorPicker(){
  const wrap = document.getElementById('color-picker-wrap'); if(!wrap) return;
  wrap.innerHTML = `<label>Hue</label><input id="outfit-hue" type="range" min="0" max="360" value="0">`;
  wrap.querySelector('#outfit-hue').addEventListener('input', e=>{
    const cId = state.selectedCharId; if(!cId) return;
    const item = state.outfits[cId]?.[state.activeSlot]; if(!item) return;
    updateOutfitColor(cId, state.activeSlot, `hsl(${e.target.value},70%,55%)`);
  });
}

function _buildCompatBar(){
  const bar = document.getElementById('wardrobe-compat'); if(!bar) return;
  const cId = state.selectedCharId; if(!CHAR_MAP[cId]){ bar.innerHTML=''; return; }
  const scores = Object.values(CHAR_MAP).filter(c=>c.id!==cId).map(other=>{
    const s = calcCompat(cId, other.id);
    return `<span class="compat-chip">${other.emoji} ${s}</span>`;
  }).join('');
  bar.innerHTML = `<span class="compat-label">Style compat:</span>${scores}`;
}

export function refreshWardrobe(){
  _buildTabs();
  _buildCards(state.activeSlot||'top');
  _buildCompatBar();
}

