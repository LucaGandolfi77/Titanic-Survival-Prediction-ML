// js/ui.js — all UI rendering: HUD, modals, toasts, rel-map, char info, overlays
import { state } from './state.js';
import { CHARACTERS, CHAR_MAP } from './characters.js';
import { getAllPairs, getPair, calcCompat, ALL_STATUSES } from './relationships.js';
import { focusCharacter, resetCamera } from './scene3d.js';
import { playSound } from './audio.js';

// ──────────── HUD ───────────────────────────────────────────────────────────────────
export function updateHUD(){
  el('hud-day',   `Day ${state.day}`);
  el('hud-score', `⭐ ${state.score}`);
  el('hud-char',  state.selectedCharId ? (CHAR_MAP[state.selectedCharId]?.emoji||'') : '❤️');
}

// ──────────── Character selector avatars ─────────────────────────────────────────────────
export function renderCharAvatars(){
  const wrap = document.getElementById('char-selector');
  if(!wrap) return;
  wrap.innerHTML = CHARACTERS.map(c=>{
    const sel = state.selectedCharId === c.id ? ' selected' : '';
    const status = _charStatus(c.id);
    return `<button class="char-avatar-circle ring-${status}${sel}" data-cid="${c.id}" title="${c.name}">
      <span>${c.emoji}</span>
    </button>`;
  }).join('');
  wrap.querySelectorAll('.char-avatar-circle').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      state.selectedCharId = btn.dataset.cid;
      renderCharAvatars();
      renderCharInfo(CHAR_MAP[btn.dataset.cid]);
      focusCharacter(btn.dataset.cid);
      import('./wardrobe.js').then(m=>m.refreshWardrobe());
    });
  });
}

function _charStatus(charId){
  const active = getAllPairs().find(p=>
    (p.a===charId||p.b===charId)&&
    ['married','dating','engaged','affair','divorced','widowed'].includes(p.status)
  );
  return active ? active.status : 'strangers';
}

// ──────────── Character info bar ──────────────────────────────────────────────────────
export function renderCharInfo(char){
  const bar = document.getElementById('char-info-bar');
  if(!bar||!char) return;
  const pairs = getAllPairs().filter(p=>p.a===char.id||p.b===char.id);
  const pills = pairs.filter(p=>p.status!=='strangers').map(p=>{
    const other = p.a===char.id ? p.b : p.a;
    const oc = CHAR_MAP[other];
    return `<span class="pair-pill pp-${p.status}">${oc?.emoji||other} ${p.status}</span>`;
  }).join('');
  bar.innerHTML = `
    <span class="char-info-name">${char.emoji} ${char.name}</span>
    <span class="char-info-style style-badge style-${char.style}">${char.style}</span>
    <span class="char-info-trait">${char.specialTrait}</span>
    <div class="char-info-pairs">${pills||'<em>No relationships yet</em>'}</div>`;
  bar.classList.remove('hidden');
}

// ──────────── Pair pills in relationship bar ───────────────────────────────────────────────
export function renderPairPills(){
  const bar = document.getElementById('relationship-bar');
  if(!bar) return;
  const active = getAllPairs().filter(p=>p.status!=='strangers');
  if(!active.length){ bar.innerHTML='<em class="bar-empty">No relationships yet. Select a character and equip styles!</em>'; return; }
  bar.innerHTML = active.map(p=>{
    const cA  = CHAR_MAP[p.a], cB = CHAR_MAP[p.b];
    const compat = calcCompat(p.a,p.b);
    return `<span class="pair-pill pp-${p.status}" data-a="${p.a}" data-b="${p.b}">
      ${cA?.emoji}${cB?.emoji} <em>${p.status}</em> — ${compat}%
    </span>`;
  }).join('');
  bar.querySelectorAll('.pair-pill').forEach(pill=>{
    pill.addEventListener('click', ()=>{
      const {a,b} = pill.dataset;
      renderRelDetail(a, b);
    });
  });
}

// ──────────── Relationship detail modal ──────────────────────────────────────────────────
export function renderRelDetail(aId, bId){
  const pair = getPair(aId,bId);
  if(!pair) return;
  const cA = CHAR_MAP[aId], cB = CHAR_MAP[bId];
  const compat = calcCompat(aId,bId);
  const body = `
    <div class="rel-stats">
      <div class="rel-bar-row"><label>Affinity</label><progress max="100" value="${Math.round(pair.affinity)}"></progress><span>${Math.round(pair.affinity)}%</span></div>
      <div class="rel-bar-row"><label>Trust</label><progress max="100" value="${Math.round(pair.trust)}"></progress><span>${Math.round(pair.trust)}%</span></div>
      <div class="rel-bar-row"><label>Tension</label><progress max="100" value="${Math.round(pair.tension)}"></progress><span>${Math.round(pair.tension)}%</span></div>
      <div class="rel-bar-row"><label>Compat</label><progress max="100" value="${compat}"></progress><span>${compat}%</span></div>
    </div>
    <p>Days Known: <strong>${pair.daysKnown}</strong> &bull; Shared Events: <strong>${pair.sharedEvents}</strong></p>
    <p>Status: <span class="status-badge badge-${pair.status}">${pair.status}</span></p>`;

  showModal(`${cA?.emoji} ${cA?.name} & ${cB?.name} ${cB?.emoji}`, body, [
    { text:'❌ Close', fn:()=>{} }
  ]);
}

// ──────────── SVG Relationship Map ───────────────────────────────────────────────────────
export function renderRelMap(){
  const overlay = document.getElementById('relationship-map-overlay');
  const svg     = document.getElementById('relationship-svg');
  if(!overlay||!svg) return;

  const W=480, H=480, CX=240, CY=240, R=160;
  const n = CHARACTERS.length;
  const positions = CHARACTERS.map((_,i)=>({
    x: CX + Math.cos((2*Math.PI*i/n) - Math.PI/2)*R,
    y: CY + Math.sin((2*Math.PI*i/n) - Math.PI/2)*R
  }));

  const statusColor = {
    strangers:'#444', acquaintances:'#666', friends:'#4caf50', best_friends:'#00bcd4',
    crush:'#ff80ab', dating:'#e91e63', engaged:'#ff6b9d', married:'#ffd700',
    rivals:'#ff5722', enemies:'#f44336', affair:'#9c27b0', divorced:'#795548', widowed:'#607d8b'
  };

  // Lines
  const lines = getAllPairs().filter(p=>p.status!=='strangers').map(p=>{
    const iA = CHARACTERS.findIndex(c=>c.id===p.a);
    const iB = CHARACTERS.findIndex(c=>c.id===p.b);
    const posA = positions[iA], posB = positions[iB];
    const color = statusColor[p.status]||'#666';
    const compat = calcCompat(p.a,p.b);
    const stroke = Math.max(1, compat/25);
    return `<line x1="${posA.x}" y1="${posA.y}" x2="${posB.x}" y2="${posB.y}" stroke="${color}" stroke-width="${stroke}" opacity="0.7"/>`;
  }).join('');

  // Nodes
  const nodes = CHARACTERS.map((c,i)=>{
    const pos = positions[i];
    const sel = state.selectedCharId===c.id?' stroke="#fff" stroke-width="3"':'';
    // compute simple stat: average compat with others (non-strangers)
    const myPairs = getAllPairs().filter(p=>p.a===c.id||p.b===c.id);
    const avgCompat = myPairs.length ? Math.round(myPairs.reduce((s,p)=>s + calcCompat(p.a,p.b), 0) / myPairs.length) : 0;
    return `
      <circle cx="${pos.x}" cy="${pos.y}" r="22" fill="${c.eyes}"${sel} class="svg-char-node" data-cid="${c.id}"/>
      <text x="${pos.x}" y="${pos.y+6}" text-anchor="middle" font-size="16">${c.emoji}</text>
      <text x="${pos.x}" y="${pos.y+36}" text-anchor="middle" fill="#ddd" font-size="11">${c.name}</text>
      <text x="${pos.x}" y="${pos.y+52}" text-anchor="middle" fill="#cfcfcf" font-size="10">Avg:${avgCompat}%</text>`;
  }).join('');

  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.innerHTML = lines + nodes;

  svg.querySelectorAll('.svg-char-node').forEach(n=>{
    n.style.cursor='pointer';
    n.addEventListener('click',()=>{
      state.selectedCharId = n.dataset.cid;
      renderCharAvatars();
      renderCharInfo(CHAR_MAP[n.dataset.cid]);
      focusCharacter(n.dataset.cid);
    });
  });
}

// ──────────── Stats overlay ───────────────────────────────────────────────────────────────────
export function renderStatsOverlay(){
  const content = document.getElementById('stats-content');
  if(!content) return;
  const pairs = getAllPairs().filter(p=>p.status!=='strangers');
  const married  = pairs.filter(p=>p.status==='married').length;
  const dating   = pairs.filter(p=>['dating','engaged'].includes(p.status)).length;
  const divorced = pairs.filter(p=>p.status==='divorced').length;
  const enemies  = pairs.filter(p=>p.status==='enemies').length;
  const topPair  = pairs.sort((a,b)=>calcCompat(b.a,b.b)-calcCompat(a.a,a.b))[0];
  const top = topPair ? `${CHAR_MAP[topPair.a]?.emoji}${CHAR_MAP[topPair.b]?.emoji} (${calcCompat(topPair.a,topPair.b)}%)` : 'N/A';
  content.innerHTML = `
    <div class="stat-card">❤️ Married: <strong>${married}</strong></div>
    <div class="stat-card">👨‍❤️‍👩 Dating: <strong>${dating}</strong></div>
    <div class="stat-card">💔 Divorced: <strong>${divorced}</strong></div>
    <div class="stat-card">⚔️ Enemies: <strong>${enemies}</strong></div>
    <div class="stat-card">🏆 Top Pair: <strong>${top}</strong></div>
    <div class="stat-card">⭐ Score: <strong>${state.score}</strong></div>
    <div class="stat-card">🗓️ Day: <strong>${state.day}</strong></div>`;
}

// ──────────── Modal ─────────────────────────────────────────────────────────────────────────
export function showModal(header, body, choices){
  const layer = document.getElementById('modal-layer');
  const box   = document.getElementById('modal-box');
  if(!layer||!box) return;
  box.innerHTML = `
    <h2 class="modal-header">${header}</h2>
    <div class="modal-body">${body}</div>
    <div class="modal-choices">${choices.map((c,i)=>`
      <button class="modal-choice-btn" data-i="${i}">${c.text}</button>`).join('')}
    </div>`;
  layer.classList.remove('hidden');
  layer.classList.add('visible');
  playSound('click');
  box.querySelectorAll('.modal-choice-btn').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const choice = choices[+btn.dataset.i];
      _closeModal();
      if(choice?.fn) choice.fn();
      playSound('click');
    });
  });
  // Close on backdrop click
  layer.addEventListener('click', e=>{ if(e.target===layer) _closeModal(); }, {once:true});
}

function _closeModal(){
  const layer = document.getElementById('modal-layer');
  if(layer){ layer.classList.remove('visible'); layer.classList.add('hidden'); }
}

// ──────────── Toasts ────────────────────────────────────────────────────────────────────────
export function showToast(msg, type='info'){
  const container = document.getElementById('toasts');
  if(!container) return;
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(()=>t.remove(), 3500);
}

// ──────────── Overlay toggles ──────────────────────────────────────────────────────────────
export function toggleRelMap(){
  const ov = document.getElementById('relationship-map-overlay');
  if(!ov) return;
  const visible = !ov.classList.contains('hidden');
  if(visible){ ov.classList.add('hidden'); }
  else { renderRelMap(); ov.classList.remove('hidden'); }
}

export function toggleStats(){
  const ov = document.getElementById('stats-overlay');
  if(!ov) return;
  const visible = !ov.classList.contains('hidden');
  if(visible){ ov.classList.add('hidden'); }
  else { renderStatsOverlay(); ov.classList.remove('hidden'); }
}

// ──────────── Helpers ────────────────────────────────────────────────────────────────────────
function el(id, text){
  const e = document.getElementById(id);
  if(e) e.textContent = text;
}

