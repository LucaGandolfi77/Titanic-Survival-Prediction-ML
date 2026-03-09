
"use strict";

// ── State ────────────────────────────────────────────────────────────────────
const S = {
  user: null, pet: null, friends: [], friendReqs: [],
  challenges: {incoming:[],outgoing:[],completed:[]},
  users: [], refreshTimer: null,
};

// ── API ──────────────────────────────────────────────────────────────────────
const api = {
  async req(method, path, body=null){
    const opts = {method, credentials:'include', headers:{'Content-Type':'application/json'}};
    if(body) opts.body = JSON.stringify(body);
    const r = await fetch('/api'+path, opts);
    return r.json();
  },
  get : (p)    => api.req('GET',  p),
  post: (p, b) => api.req('POST', p, b),
};

// ── Toast ────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, isError=false){
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = 'toast'+(isError?' error':'');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>el.classList.add('hidden'),3200);
}

// ── Auth ─────────────────────────────────────────────────────────────────────
function showTab(t){
  ['login','register'].forEach(n=>{
    document.getElementById('tab-'+n).classList.toggle('hidden', n!==t);
    document.querySelectorAll('.tab-btn').forEach((b,i)=>{
      b.classList.toggle('active', (i===0&&t==='login')||(i===1&&t==='register'));
    });
  });
  document.getElementById('auth-msg').classList.add('hidden');
}

function showMsg(msg, isError=true){
  const el = document.getElementById('auth-msg');
  el.textContent = msg;
  el.className   = 'msg-box'+(isError?' error':' success');
}

async function doLogin(){
  const u = document.getElementById('login-username').value.trim();
  const p = document.getElementById('login-password').value;
  if(!u||!p){showMsg('Please fill all fields');return}
  const d = await api.post('/login',{username:u,password:p});
  if(d.error){showMsg(d.error);return}
  S.user = d.user;
  enterApp();
}

async function doRegister(){
  const u = document.getElementById('reg-username').value.trim();
  const p = document.getElementById('reg-password').value;
  if(!u||!p){showMsg('Please fill all fields');return}
  const d = await api.post('/register',{username:u,password:p});
  if(d.error){showMsg(d.error);return}
  S.user = d.user;
  enterApp();
}

async function doLogout(){
  await api.post('/logout');
  clearInterval(S.refreshTimer);
  S.user=null; S.pet=null;
  document.getElementById('auth-screen').classList.remove('hidden');
  document.getElementById('app-screen').classList.add('hidden');
}

// ── App entry ────────────────────────────────────────────────────────────────
async function enterApp(){
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('app-screen').classList.remove('hidden');
  document.getElementById('hdr-username').textContent = S.user.username;
  await loadPet();
  await loadSocial();
  clearInterval(S.refreshTimer);
  S.refreshTimer = setInterval(async()=>{
    await loadPet();
    if(!document.getElementById('tab-social').classList.contains('hidden')) await loadSocial();
    if(!document.getElementById('tab-leaderboard').classList.contains('hidden')) await loadLeaderboard();
  }, 30000);
}

// ── Check session on page load ───────────────────────────────────────────────
async function checkSession(){
  try{
    const d = await api.get('/me');
    if(d.id){ S.user=d; enterApp(); }
  }catch(e){}
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name, btn){
  document.querySelectorAll('.main-tab').forEach(t=>t.classList.add('hidden'));
  document.getElementById('tab-'+name).classList.remove('hidden');
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  if(name==='leaderboard') loadLeaderboard();
  if(name==='social')      loadSocial();
}

// ── Pet ──────────────────────────────────────────────────────────────────────
async function loadPet(){
  const d = await api.get('/pet');
  S.pet = d.pet;
  if(!S.pet){
    document.getElementById('create-pet-panel').classList.remove('hidden');
    document.getElementById('pet-dashboard').classList.add('hidden');
  } else {
    document.getElementById('create-pet-panel').classList.add('hidden');
    document.getElementById('pet-dashboard').classList.remove('hidden');
    renderPet();
  }
}

let selectedSpecies = 'blob';
function selectSpecies(el){
  document.querySelectorAll('.species-card').forEach(c=>c.classList.remove('selected'));
  el.classList.add('selected');
  selectedSpecies = el.dataset.species;
}

async function createPet(){
  const name = document.getElementById('new-pet-name').value.trim();
  if(!name){toast('Give your pet a name!',true);return}
  const d = await api.post('/pet/create',{name,species:selectedSpecies});
  if(d.error){toast(d.error,true);return}
  S.pet = d.pet;
  toast(d.message);
  await loadPet();
}

// Species → body colour class & emoji
const SPECIES_EMOJI = {
  blob:   ['🫧','💚','💙','👾','🤖'],
  cat:    ['🥚','🐱','😺','🐈','🦁'],
  dragon: ['🥚','🦕','🐲','🐉','🔥'],
  bunny:  ['🥚','🐣','🐰','🐇','🦊'],
};
const MOOD_EMOJI = {happy:'😄',content:'🙂',sad:'😢',sleeping:'😴',sick:'🤒',critical:'😰',dead:'💀'};
const MOOD_FACES = {
  happy:    {eye:'circle',mouth:'smile',cheeks:true},
  content:  {eye:'circle',mouth:'flat', cheeks:false},
  sad:      {eye:'circle',mouth:'frown',cheeks:false},
  sleeping: {eye:'line',  mouth:'open', cheeks:false},
  sick:     {eye:'x',    mouth:'flat', cheeks:false},
  critical: {eye:'circle',mouth:'frown',cheeks:true},
  dead:     {eye:'x',    mouth:'flat', cheeks:false},
};

function renderPet(){
  const p = S.pet;
  if(!p) return;

  // dead check
  if(!p.is_alive){
    document.getElementById('dead-panel').classList.remove('hidden');
    document.querySelector('.pet-grid').classList.add('hidden');
    return;
  }
  document.getElementById('dead-panel').classList.add('hidden');
  document.querySelector('.pet-grid').classList.remove('hidden');

  // Pet character
  const char = document.getElementById('pet-character');
  char.className = `pet-character mood-${p.mood} species-${p.species} stage-${p.evolution_stage}`;

  // Eyes
  updateEyes(p.mood);
  updateMouth(p.mood);
  document.getElementById('pet-cheeks').style.display =
    MOOD_FACES[p.mood]?.cheeks ? 'block' : 'none';

  // Particles
  updateParticles(p.mood);

  // Info
  document.getElementById('pet-name-disp').textContent = p.name;
  document.getElementById('pet-evo-badge').textContent = p.evolution_name;
  document.getElementById('pet-level').textContent    = p.level;
  document.getElementById('pet-age').textContent      = p.age_days;
  document.getElementById('pet-wins').textContent     = p.wins;
  document.getElementById('pet-losses').textContent   = p.losses;
  document.getElementById('xp-bar').style.width       = p.experience+'%';
  document.getElementById('xp-val').textContent       = Math.round(p.experience);
  document.getElementById('pet-mood-label').textContent = MOOD_EMOJI[p.mood]||'😐';

  // Sleep button label
  document.getElementById('sleep-lbl').textContent = p.is_sleeping ? 'Wake' : 'Sleep';

  // Stats
  setBar('bar-hunger', p.hunger);  document.getElementById('val-hunger').textContent = Math.round(p.hunger);
  setBar('bar-happy',  p.happiness);document.getElementById('val-happy').textContent  = Math.round(p.happiness);
  setBar('bar-energy', p.energy);   document.getElementById('val-energy').textContent = Math.round(p.energy);
  setBar('bar-health', p.health);   document.getElementById('val-health').textContent = Math.round(p.health);
}

function setBar(id, val){
  document.getElementById(id).style.width = Math.max(0,Math.min(100,val))+'%';
}

function updateEyes(mood){
  ['eye-l','eye-r'].forEach(id=>{
    const el = document.getElementById(id);
    el.style.cssText='';
    if(mood==='sleeping'){el.style.height='4px';el.style.borderRadius='2px';el.style.top='36px';}
    else if(mood==='sick'||mood==='dead'){el.innerHTML='<span style="position:absolute;top:-2px;left:2px;font-size:12px;color:#0d0d1a">✕</span>';}
    else el.innerHTML='';
  });
}

function updateMouth(mood){
  const m = document.getElementById('pet-mouth');
  m.style.cssText='';
  if(mood==='sad'||mood==='sick'){
    m.style.borderRadius='20px 20px 0 0';
    m.style.borderTop='3px solid #0d0d1a';
    m.style.borderBottom='none';
    m.style.bottom='18px';
  } else if(mood==='sleeping'){
    m.style.width='18px';m.style.height='8px';
    m.style.borderRadius='0 0 10px 10px';
    m.style.borderTop='none';
    m.style.bottom='22px';
  }
}

let particleInterval;
function updateParticles(mood){
  clearInterval(particleInterval);
  const zzz  = document.getElementById('zzz-container');
  const hearts = document.getElementById('hearts-container');
  const sick   = document.getElementById('sick-particles');
  zzz.innerHTML=''; hearts.innerHTML=''; sick.innerHTML='';

  if(mood==='sleeping'){
    let i=0;
    particleInterval=setInterval(()=>{
      const z=document.createElement('span');
      z.className='zzz'; z.textContent='z';
      z.style.left=(50+Math.random()*30)+'%';
      z.style.bottom=(60+Math.random()*20)+'%';
      z.style.animationDelay=(i%3*.8)+'s';
      z.style.fontSize=(14+i%3*4)+'px';
      zzz.appendChild(z);
      if(zzz.children.length>6) zzz.removeChild(zzz.firstChild);
      i++;
    },900);
  } else if(mood==='happy'){
    particleInterval=setInterval(()=>{
      const h=document.createElement('span');
      h.className='heart'; h.textContent='❤️';
      h.style.left=(20+Math.random()*60)+'%';
      h.style.bottom='60%';
      h.style.setProperty('--dx',(Math.random()*60-30)+'px');
      h.style.setProperty('--dy',(-40-Math.random()*30)+'px');
      hearts.appendChild(h);
      setTimeout(()=>h.remove(),1500);
    },1200);
  } else if(mood==='sick'){
    for(let i=0;i<5;i++){
      const s=document.createElement('div');
      s.className='sick-dot';
      s.style.left=(20+Math.random()*60)+'%';
      s.style.bottom=(30+Math.random()*30)+'%';
      s.style.animationDelay=(Math.random()*2)+'s';
      s.style.setProperty('--dx',(Math.random()*20-10)+'px');
      sick.appendChild(s);
    }
  }
}

// ── Pet actions ───────────────────────────────────────────────────────────────
const ACTIONS = {feed:'/pet/feed', play:'/pet/play', sleep:'/pet/sleep', heal:'/pet/heal'};
async function petAction(action){
  const d = await api.post(ACTIONS[action]);
  if(d.error){toast(d.error,true);return}
  S.pet = d.pet;
  toast(d.message);
  renderPet();
  addLog(d.message);
}

function addLog(msg){
  const ul = document.getElementById('activity-log');
  const li = document.createElement('li');
  const time = new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  li.textContent = `[${time}] ${msg}`;
  ul.prepend(li);
  while(ul.children.length>15) ul.removeChild(ul.lastChild);
}

async function revivePet(){
  await api.post('/pet/revive');
  S.pet=null;
  await loadPet();
}

// ── Social ────────────────────────────────────────────────────────────────────
async function loadSocial(){
  const [users, friends, reqs, chs] = await Promise.all([
    api.get('/social/users'),
    api.get('/social/friends'),
    api.get('/social/friend/requests'),
    api.get('/social/challenges'),
  ]);
  S.users       = users.users||[];
  S.friends     = friends.friends||[];
  S.friendReqs  = reqs.requests||[];
  S.challenges  = chs;
  renderSocial();
}

function speciesAvatar(sp){
  const m={'blob':'🫧','cat':'🐱','dragon':'🐲','bunny':'🐰'};
  return m[sp]||'🐾';
}

function renderSocial(){
  // Friends
  const fl = document.getElementById('friends-list');
  fl.innerHTML = S.friends.length===0 ? '<p style="color:var(--text-dim);font-size:13px">No friends yet. Discover trainers below!</p>' : '';
  S.friends.forEach(f=>{
    const sp = f.pet?.species||'blob';
    fl.innerHTML += `
    <div class="friend-row">
      <div class="user-avatar" style="background:rgba(176,102,255,.2)">${speciesAvatar(sp)}</div>
      <div class="user-info">
        <strong>${f.user.username}</strong>
        <small>${f.pet?`${f.pet.name} · Lv.${f.pet.level} ${f.pet.evolution_name}`:'No pet'}</small>
      </div>
      <div class="user-actions">
        <button class="btn-icon" onclick="interactFriend(${f.user.id})" title="Play together">🎉</button>
        <button class="btn-icon btn-challenge" onclick="challengeUser(${f.user.id})" title="Challenge">⚔️</button>
      </div>
    </div>`;
  });

  // Requests
  const rl = document.getElementById('friend-requests-list');
  rl.innerHTML = S.friendReqs.length===0 ? '<p style="color:var(--text-dim);font-size:13px">No pending requests.</p>' : '';
  S.friendReqs.forEach(r=>{
    rl.innerHTML += `
    <div class="user-row">
      <div class="user-avatar" style="background:rgba(0,212,255,.2)">👤</div>
      <div class="user-info">
        <strong>${r.requester.username}</strong>
        <small>${r.pet?`${r.pet.name} · Lv.${r.pet.level}`:''}</small>
      </div>
      <div class="user-actions">
        <button class="btn-icon btn-accept"  onclick="respondFriend(${r.id},'accept')">✓</button>
        <button class="btn-icon btn-decline" onclick="respondFriend(${r.id},'decline')">✗</button>
      </div>
    </div>`;
  });

  // Discover
  const ul = document.getElementById('users-list');
  const friendIds = new Set(S.friends.map(f=>f.user.id));
  const reqIds    = new Set(S.friendReqs.map(r=>r.requester.id));
  ul.innerHTML = '';
  S.users.forEach(u=>{
    const isFriend = friendIds.has(u.id);
    const sp = u.pet?.species||'blob';
    ul.innerHTML += `
    <div class="user-row">
      <div class="user-avatar" style="background:rgba(0,255,136,.15)">${speciesAvatar(sp)}</div>
      <div class="user-info">
        <strong>${u.username}</strong>
        <small>${u.pet.name} · Lv.${u.pet.level} ${u.pet.evolution_name} · ${u.pet.mood}</small>
      </div>
      <div class="user-actions">
        ${isFriend
          ? `<button class="btn-icon btn-challenge" onclick="challengeUser(${u.id})">⚔️</button>`
          : `<button class="btn-icon" onclick="addFriend(${u.id})">➕</button>`
        }
      </div>
    </div>`;
  });
  if(S.users.length===0) ul.innerHTML='<p style="color:var(--text-dim);font-size:13px">No other trainers online yet.</p>';

  // Challenges
  const ci = document.getElementById('ch-incoming');
  ci.innerHTML = S.challenges.incoming.length===0?'<p style="color:var(--text-dim);font-size:12px">None</p>':'';
  S.challenges.incoming.forEach(c=>{
    ci.innerHTML += `
    <div class="challenge-row">
      ⚔️ <strong>${c.challenger.username}</strong> challenges you!
      <div style="display:flex;gap:6px;margin-top:6px">
        <button class="btn-icon btn-accept"  onclick="respondChallenge(${c.id},'accept')">Accept ⚔️</button>
        <button class="btn-icon btn-decline" onclick="respondChallenge(${c.id},'decline')">Decline</button>
      </div>
    </div>`;
  });

  const co = document.getElementById('ch-outgoing');
  co.innerHTML = S.challenges.outgoing.length===0?'<p style="color:var(--text-dim);font-size:12px">None</p>':'';
  S.challenges.outgoing.forEach(c=>{
    co.innerHTML += `<div class="challenge-row">⏳ Waiting for <strong>${c.challenged.username}</strong>…</div>`;
  });

  const cc = document.getElementById('ch-completed');
  cc.innerHTML = S.challenges.completed.length===0?'<p style="color:var(--text-dim);font-size:12px">No battles yet</p>':'';
  S.challenges.completed.forEach(c=>{
    const won = c.winner?.username===S.user.username;
    cc.innerHTML += `
    <div class="challenge-row">
      ${won?'🏆':'💀'} <strong>${c.challenger.username}</strong> vs <strong>${c.challenged.username}</strong>
      → ${c.winner?c.winner.username+' wins':'Draw'}
      ${c.battle_log?`<div class="challenge-log">${c.battle_log}</div>`:''}
    </div>`;
  });
}

async function addFriend(uid){
  const d = await api.post('/social/friend/request',{user_id:uid});
  toast(d.error||d.message, !!d.error);
  if(!d.error) await loadSocial();
}

async function respondFriend(fid, action){
  const d = await api.post('/social/friend/respond',{friendship_id:fid,action});
  toast(d.error||d.message, !!d.error);
  await loadSocial();
}

async function challengeUser(uid){
  const d = await api.post('/social/challenge',{user_id:uid});
  toast(d.error||d.message, !!d.error);
  await loadSocial();
}

async function respondChallenge(cid, action){
  const d = await api.post('/social/challenge/respond',{challenge_id:cid,action});
  if(d.error){toast(d.error,true);return}
  toast(d.message||'Battle done!');
  if(d.pet){S.pet=d.pet;renderPet();}
  if(d.battle_log) addLog(d.battle_log.join(' | '));
  await loadSocial();
}

async function interactFriend(uid){
  const d = await api.post('/social/interact',{user_id:uid});
  if(d.error){toast(d.error,true);return}
  S.pet=d.pet; renderPet();
  toast(d.message);
  addLog(d.message);
}

// ── Leaderboard ───────────────────────────────────────────────────────────────
async function loadLeaderboard(){
  const d = await api.get('/leaderboard');
  const tbody = document.getElementById('lb-body');
  const sp_emoji = {blob:'🫧',cat:'🐱',dragon:'🐲',bunny:'🐰'};
  tbody.innerHTML = (d.leaderboard||[]).map(r=>`
    <tr class="rank-${r.rank}">
      <td>${r.rank}</td>
      <td>${r.owner}</td>
      <td>${r.name}</td>
      <td>${sp_emoji[r.species]||'🐾'} ${r.species}</td>
      <td>${r.evolution_name}</td>
      <td>${r.level}</td>
      <td>${r.wins}</td>
    </tr>`).join('');
}

// ── Init ──────────────────────────────────────────────────────────────────────
checkSession();
