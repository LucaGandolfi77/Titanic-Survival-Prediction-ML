import { mapState } from './map.js';
import { citizens, faction, initCitizens, selectedCitizen, deselectCitizen, assignTask, updateCitizens, autoGather, createCitizen, setGlobalResources } from './citizen.js';
import { placeBuilding, removeBuilding, buildings, updateBuildings, getConstructedBuildings } from './building.js';
import { units, spawnUnit, updateUnits } from './unit.js';
import { combat } from './combat.js';
import { ai } from './ai.js';
import { particles, spawnParticle, updateParticles } from './render/particles.js';
import { fog } from './fog.js';
import { resources } from './resource.js';
import { manpower } from './manpower.js';
import { diplomacy } from './diplomacy.js';
import { mercenary } from './mercenary.js';
import { hud } from './ui/hud.js';
import { Screens } from './ui/screens.js';
import { showToast, haptic } from './ui/notifications.js';
import { Camera, setCameraCanvas } from './render/camera.js';
import { Renderer } from './render/renderer.js';
import { AGES, TECHS } from './data/tech-tree.js';
import { FACTION_KEYS, FACTIONS } from './data/factions.js';
import { BUILDINGS } from './data/buildings.js';
import { UNITS } from './data/units.js';
import { timeRipple } from './time-ripple.js';
import { weather } from './weather.js';
import { dna } from './dna-mutation.js';
import { hero } from './hero.js';
import { territory } from './territory.js';
import { CAMPAIGN_MISSIONS, loadCampaignState, completeMission, checkCampaignWin, checkCampaignLose, getMissionProgress } from './campaign.js';

export { manpower, diplomacy, mercenary };
export let camera, renderer;
export let isRunning = false;
export let isPaused = false;
export let gameSpeed = 1;
export let gameMode = 'menu';
export let currentAge = 0;
export let researchedTechs = new Set();
export let selectedUnit = null;
export let selectedFaction = 'roma';
export let selectedDifficulty = 'normal';
export let p1Faction = 'roma';
export let p2Faction = 'vikings';
export let activePlayer = 1;
export let isTwoPlayer = false;
export let turnIncome = { food: 0, wood: 0, gold: 0, stone: 0 };
export { CAMPAIGN_MISSIONS, loadCampaignState, completeMission, checkCampaignWin, checkCampaignLose, getMissionProgress };

export function refreshCampaignList() {
  const state = loadCampaignState();
  for (let i = 0; i < CAMPAIGN_MISSIONS.length; i++) {
    const el = document.getElementById('ms-' + i);
    if (!el) continue;
    if (state.completed.includes(i)) {
      el.textContent = '✅';
      el.style.color = '#357a38';
    } else if (i < state.unlocked) {
      el.textContent = '▶';
      el.style.color = '#d4a017';
    } else {
      el.textContent = '🔒';
      el.style.color = '#666';
    }
    const card = document.querySelector(`[data-mission="${i}"]`);
    if (card) {
      card.style.opacity = (i < state.unlocked || state.completed.includes(i)) ? '1' : '0.5';
      if (i === state.currentMission && !state.completed.includes(i)) {
        card.style.border = '2px solid var(--primary)';
      } else {
        card.style.border = '1px solid var(--primary-dark)';
      }
    }
  }
}
let lastTime = 0;
let incomeTimer = 0;
let autoGatherTimer = 0;
let renderCounter = 0;
let animFrameId = null;
let canvas;

setGlobalResources(resources);
diplomacy.init();

export async function initGame(seed) {
  gameMode = 'skirmish';
  resources.reset();
  buildings.length = 0;
  citizens.length = 0;
  units.length = 0;
  deselectCitizen();
  selectedUnit = null;
  currentAge = 0;
  researchedTechs.clear();
  fog.clear();
  isPaused = false;
  gameSpeed = 1;
  ai.active = false;
  isTwoPlayer = false;
  activePlayer = 1;
  p1Faction = 'roma';
  p2Faction = 'vikings';
  turnIncome = { food: 0, wood: 0, gold: 0, stone: 0 };
  timeRipple.state = 'normal';
  timeRipple.timer = 0;
  timeRipple.isActive = false;
  timeRipple.cooldownTimer = 0;
  weather.init();
  territory.init();
  hero.active = null;
  hero.xp = 0;
  hero.level = 1;
  dna.init();

  canvas = document.getElementById('game-canvas');
  if (!camera) camera = new Camera();
  setCameraCanvas(canvas);
  if (!renderer) renderer = new Renderer(canvas);
  else renderer.resize();

  await mapState.init(seed || Date.now());
  initCitizens('roma');
  fog.revealArea(mapState.playerQ, mapState.playerR, 6);

  placeBuilding('town_center', mapState.playerQ, mapState.playerR, 'roma');
  placeBuilding('house', mapState.playerQ + 1, mapState.playerR, 'roma');
  placeBuilding('house', mapState.playerQ - 1, mapState.playerR, 'roma');

  manpower.init(0);
  diplomacy.init();
  hud.updatePopulation(10, 20);
  hud.setManpower(manpower.current, manpower.max, manpower.overdraft);

  resources.updatePopulation(10, 20);
  hud.updateResources(resources);
  hud.updateAge(0);
  hud.setSpeed(1);

  Screens.show('screen-menu');
  setupTouchInput(canvas);
  window.addEventListener('resize', () => renderer && renderer.resize());
  startLoop();
  setTimeout(() => showToast('⚔️ Imperium — Tap a citizen to select'), 500);
}

export function startTwoPlayer(p1, p2) {
  isTwoPlayer = true;
  p1Faction = p1;
  p2Faction = p2;
  gameMode = '2p';
  gameMode = 'skirmish';
  resources.reset();
  buildings.length = 0;
  citizens.length = 0;
  units.length = 0;
  deselectCitizen();
  selectedUnit = null;
  currentAge = 0;
  researchedTechs.clear();
  fog.clear();
  isPaused = false;
  gameSpeed = 1;
  ai.active = false;
  activePlayer = 1;
  turnIncome = { food: 0, wood: 0, gold: 0, stone: 0 };
  timeRipple.state = 'normal';
  timeRipple.timer = 0;
  timeRipple.isActive = false;
  timeRipple.cooldownTimer = 0;
  weather.init();
  territory.init();
  hero.active = null;
  hero.xp = 0;
  hero.level = 1;
  dna.init();

  canvas = document.getElementById('game-canvas');
  if (!camera) camera = new Camera();
  setCameraCanvas(canvas);
  if (!renderer) renderer = new Renderer(canvas);
  else renderer.resize();

  mapState.init(Date.now()).then(() => {
    initCitizensFor2P(p1, p2);
    fog.revealArea(mapState.playerQ, mapState.playerR, 6);
    placeBuilding('town_center', mapState.playerQ, mapState.playerR, p1);
    placeBuilding('house', mapState.playerQ + 1, mapState.playerR, p1);
    placeBuilding('house', mapState.playerQ - 1, mapState.playerR, p1);
    placeBuilding('town_center', mapState.playerQ + 15, mapState.playerR - 10, p2);
    placeBuilding('house', mapState.playerQ + 16, mapState.playerR - 10, p2);
    placeBuilding('house', mapState.playerQ + 14, mapState.playerR - 10, p2);
    manpower.init(0);
    diplomacy.init();
    hud.updatePopulation(10, 20);
    hud.setManpower(manpower.current, manpower.max, manpower.overdraft);
    resources.updatePopulation(10, 20);
    hud.updateResources(resources);
    hud.updateAge(0);
    hud.setSpeed(1);
    hud.setActivePlayer(activePlayer);
    Screens.show('screen-menu');
    setupTouchInput(canvas);
    window.addEventListener('resize', () => renderer && renderer.resize());
    startLoop();
    showToast('👥 2P MODE — Player ' + activePlayer + "'s turn");
  });
}

export function switchPlayer() {
  if (!isTwoPlayer) return;
  activePlayer = activePlayer === 1 ? 2 : 1;
  hud.setActivePlayer(activePlayer);
  showToast('Player ' + activePlayer + "'s turn");
}

export function check2PWin() {
  const p1Alive = units.some(u => u.alive && u.faction === p1Faction) || citizens.some(c => c.faction === p1Faction);
  const p2Alive = units.some(u => u.alive && u.faction === p2Faction) || citizens.some(c => c.faction === p2Faction);
  if (!p1Alive || !p2Alive) {
    const winner = p1Alive ? 1 : 2;
    showToast('👑 Player ' + winner + ' wins!');
    Screens.show('screen-victory');
    isPaused = true;
  }
}

function initCitizensFor2P(p1, p2) {
  citizens.length = 0;
  deselectCitizen();
  for (let i = 0; i < 5; i++) {
    const c = createCitizen(
      ['Bruno', 'Elena', 'Marcus', 'Aria', 'Lucan'][i] + ' P1',
      mapState.playerQ + Math.floor(Math.random() * 3) - 1,
      mapState.playerR + Math.floor(Math.random() * 3) - 1,
      { trait: 'Brave', skill: 'Gatherer' }
    );
    c.faction = p1;
    c.color = FACTIONS[p1]?.color || '#c4a035';
    citizens.push(c);
  }
  for (let i = 0; i < 5; i++) {
    const c = createCitizen(
      ['Nora', 'Draven', 'Sera', 'Titan', 'Rex'][i] + ' P2',
      mapState.playerQ + 15 + Math.floor(Math.random() * 3) - 1,
      mapState.playerR - 10 + Math.floor(Math.random() * 3) - 1,
      { trait: 'Strong', skill: 'Builder' }
    );
    c.faction = p2;
    c.color = FACTIONS[p2]?.color || '#c43030';
    citizens.push(c);
  }
}

export function startCampaign(missionIndex) {
  showToast('Mission ' + (missionIndex + 1) + ': ' + CAMPAIGN_MISSIONS[missionIndex]?.name);
  initGame(Date.now() + missionIndex * 1000).then(() => {
    gameMode = 'campaign';
    const state = loadCampaignState();
    state.currentMission = missionIndex;
    saveCampaignState(state);
    const mission = CAMPAIGN_MISSIONS[missionIndex];
    if (mission && mission.age > currentAge) {
      for (let i = currentAge; i < mission.age; i++) {
        currentAge++;
        hud.updateAge(currentAge);
      }
    }
    fog.revealArea(mapState.playerQ, mapState.playerR, 15);
    placeBuilding('town_center', mapState.playerQ, mapState.playerR, 'roma');
    placeBuilding('house', mapState.playerQ + 1, mapState.playerR, 'roma');
    placeBuilding('house', mapState.playerQ - 1, mapState.playerR, 'roma');
    placeBuilding('house', mapState.playerQ + 2, mapState.playerR, 'roma');
    hud.updateResources(resources);
    Screens.show('screen-menu');
  });
}

function saveCampaignState(state) {
  try { localStorage.setItem('imperium_campaign_progress', JSON.stringify(state)); } catch (e) { /* ignore */ }
}

export function togglePause() {
  isPaused = !isPaused;
  if (isPaused) Screens.show('screen-pause');
  else { Screens.show('screen-menu'); }
}

export function ageUp() {
  if (currentAge >= 3) { showToast('Already at Imperial Age!'); return; }
  const cost = { food: 300, wood: 250, gold: 200, stone: 150 };
  if (!resources.spendCost(cost)) { showToast('Need more resources'); return; }
  currentAge++;
  manpower.init(currentAge);
  territory.tiles.clear();
  hud.updateAge(currentAge);
  haptic(50);
  showToast('Aging up to ' + ['Dark', 'Feudal', 'Castle', 'Imperial'][currentAge] + '!');
}

export function researchTech(techId) {
  const tech = TECHS[techId];
  if (!tech) return;
  if (researchedTechs.has(techId)) { showToast('Already researched'); return; }
  if (tech.ageRequired > currentAge) { showToast('Requires ' + ['Dark', 'Feudal', 'Castle', 'Imperial'][tech.ageRequired] + ' Age'); return; }
  if (!resources.spendCost(tech.cost)) { showToast('Not enough resources'); return; }
  researchedTechs.add(techId);
  haptic(30);
  showToast('Researched: ' + tech.name);
}

export function buildSpecific(type) {
  const def = BUILDINGS[type];
  if (!def) return;
  if (currentAge < def.ageRequired) { showToast('Requires ' + ['Dark', 'Feudal', 'Castle', 'Imperial'][def.ageRequired] + ' Age'); return; }
  if (!resources.spendCost(def.cost)) { showToast('Not enough resources'); return; }
  for (let i = 0; i < 30; i++) {
    const bq = mapState.playerQ + Math.floor(Math.random() * 11) - 5;
    const br = mapState.playerR + Math.floor(Math.random() * 11) - 5;
    const tile = mapState.getTile(bq, br);
    if (tile && !tile.building && tile.terrain !== 'water') {
      placeBuilding(type, bq, br);
      spawnParticle(tile.q * 36 + 12, tile.r * 41.57 + 12, 'build');
      haptic(20);
      showToast('Built ' + def.name);
      return;
    }
  }
  showToast('No space');
}

export function trainSpecific(unitId) {
  const def = UNITS[unitId];
  if (!def) return;
  if (currentAge < def.ageRequired) { showToast('Requires ' + ['Dark', 'Feudal', 'Castle', 'Imperial'][def.ageRequired] + ' Age'); return; }
  if (!resources.spendCost(def.cost)) { showToast('Not enough resources'); return; }
  if (!manpower.canAfford(def.manpowerCost)) {
    showToast('Not enough manpower! (' + manpower.getAvailable() + ' available)');
    haptic(50);
    return;
  }
  manpower.spend(def.manpowerCost);
  const c = spawnUnit(unitId, mapState.playerQ + Math.floor(Math.random() * 3) - 1, mapState.playerR + Math.floor(Math.random() * 3) - 1, 'roma');
  if (c) { showToast('Trained ' + def.name); spawnParticle(c.x, c.y, 'build'); haptic(20); }
  hud.setManpower(manpower.current, manpower.max, manpower.overdraft);
}

export function hireMercenary(unitId) {
  const c = mercenary.hire(unitId);
  if (c) hud.setManpower(manpower.current, manpower.max, manpower.overdraft);
}

export function hireHero() {
  return hero.hire(selectedFaction);
}

export function useTimeRipple(type) {
  return timeRipple.use(type);
}

export function openHeroPanel() {
  const panel = document.getElementById('hero-panel');
  if (!panel) return;
  panel.classList.remove('hidden');
  renderHeroUI();
}

export function updateRippleUI() {
  const accel = document.getElementById('btn-ripple-accel');
  const slow = document.getElementById('btn-ripple-slow');
  const freeze = document.getElementById('btn-ripple-freeze');
  if (accel) {
    accel.disabled = !timeRipple.canUse() || resources.gold < 50;
    accel.style.opacity = accel.disabled ? '0.5' : '1';
  }
  if (slow) {
    slow.disabled = !timeRipple.canUse() || resources.gold < 75;
    slow.style.opacity = slow.disabled ? '0.5' : '1';
  }
  if (freeze) {
    freeze.disabled = !timeRipple.canUse() || resources.gold < 75;
    freeze.style.opacity = freeze.disabled ? '0.5' : '1';
  }
}

function renderHeroUI() {
  const info = document.getElementById('hero-info');
  if (!info) return;
  const stats = hero.getStats();
  if (!stats) {
    info.innerHTML = '<p style="color:var(--text-muted);">No hero hired. Spend 200g to hire one.</p>';
    return;
  }
  const u = stats.unit;
  info.innerHTML = `
    <div style="background:var(--bg-card);padding:10px;border-radius:8px;border:1px solid var(--primary-dark);margin-bottom:8px;">
      <div style="font-weight:700;font-size:16px;">⭐ ${u.name}</div>
      <div style="font-size:12px;color:var(--text-muted);">Level ${stats.level} — ${u.faction}</div>
      <div style="margin-top:4px;">HP: ${Math.floor(u.hp)}/${u.maxHp} | ATK: ${u.attack} | DEF: ${u.defense}</div>
      <div style="margin-top:4px;">XP: ${stats.xp}/${stats.xpNext}</div>
      <div style="height:4px;background:var(--bg-dark);border-radius:2px;margin-top:2px;">
        <div style="height:100%;width:${(stats.xp/stats.xpNext)*100}%;background:var(--primary);border-radius:2px;"></div>
      </div>
    </div>
    ${stats.abilities.length > 0 ? `
      <div style="display:flex;flex-wrap:wrap;gap:4px;">
        ${stats.abilities.map(a => `<span style="background:var(--bg-card);padding:4px 8px;border-radius:4px;font-size:11px;border:1px solid var(--primary-dark);">${a.emoji} ${a.name}</span>`).join('')}
      </div>
    ` : '<p style="color:var(--text-muted);font-size:11px;">No abilities yet (level up to unlock)</p>'}
  `;
}

export function startSkirmishWithAI(factionId, difficulty) {
  selectedFaction = factionId;
  selectedDifficulty = difficulty;
  initGame(Date.now()).then(() => {
    ai.start(factionId, difficulty);
  });
}

export function openDiplomacyPanel() {
  const panel = document.getElementById('diplomacy-panel');
  if (panel) {
    panel.classList.remove('hidden');
    renderDiplomacyUI();
  }
}

export function closeDiplomacyPanel() {
  document.getElementById('diplomacy-panel')?.classList.add('hidden');
}

export function diplomacyAction(f2, action) {
  const f1 = selectedFaction;
  if (diplomacy.performAction(f1, f2, action)) {
    renderDiplomacyUI();
    haptic(10);
  }
}

function renderDiplomacyUI() {
  const list = document.getElementById('diplomacy-list');
  if (!list) return;
  list.innerHTML = '';
  const others = FACTION_KEYS.filter(f => f !== selectedFaction);
  for (const f2 of others) {
    const rel = diplomacy.getRelation(selectedFaction, f2);
    const relColors = { allied: '#357a38', friendly: '#1e5fa8', neutral: '#a8a8a8', tense: '#d4a017', at_war: '#9b2226' };
    const relLabels = { allied: 'Allied', friendly: 'Friendly', neutral: 'Neutral', tense: 'Tense', at_war: 'At War' };
    const canAct = diplomacy.canAction(selectedFaction, f2, 'declare_war');

    const div = document.createElement('div');
    div.style.cssText = 'background:var(--bg-card);padding:8px 10px;border-radius:8px;border:1px solid var(--primary-dark);margin-bottom:4px;';
    div.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-weight:600;">${FACTIONS[f2].emoji} ${FACTIONS[f2].name}</span>
        <span style="color:${relColors[rel]};font-weight:700;font-size:11px;">● ${relLabels[rel]}</span>
      </div>
      <div style="display:flex;gap:4px;margin-top:4px;flex-wrap:wrap;">
        <button class="diplomacy-btn menu-btn" style="padding:3px;font-size:9px;flex:1;" data-faction="${f2}" data-action="peace">🤝 Peace</button>
        <button class="diplomacy-btn menu-btn" style="padding:3px;font-size:9px;flex:1;" data-faction="${f2}" data-action="ally">⭐ Ally</button>
        <button class="diplomacy-btn menu-btn" style="padding:3px;font-size:9px;flex:1;" data-faction="${f2}" data-action="trade">💰 Trade</button>
        <button class="diplomacy-btn menu-btn" style="padding:3px;font-size:9px;flex:1;" data-faction="${f2}" data-action="nap">📜 NAP</button>
      </div>
    `;
    list.appendChild(div);
  }

  document.querySelectorAll('.diplomacy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      diplomacyAction(btn.dataset.faction, btn.dataset.action);
    });
  });
}

function startLoop() {
  isRunning = true;
  lastTime = performance.now();
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = requestAnimationFrame(gameLoop);
}

function gameLoop(timestamp) {
  if (!isRunning) return;
  if (isPaused) { animFrameId = requestAnimationFrame(gameLoop); return; }

  timeRipple.update(1);
  weather.update(1);
  hero.update(1);
  territory.update(1);

  const timeSpeed = timeRipple.getSpeedMultiplier();
  const weatherSpeedMod = weather.getSpeedModifier();
  const effectiveSpeed = timeSpeed * weatherSpeedMod;
  const dt = Math.min((timestamp - lastTime) / 1000, 0.1) * effectiveSpeed;
  lastTime = timestamp;

  camera.update();
  updateBuildings(dt);
  updateCitizens(dt);
  updateUnits(dt, timestamp);
  combat.update(dt);
  ai.update(dt, units);
  resources.updateIncome(dt);
  manpower.regenerate(dt);
  diplomacy.update(dt);
  updateParticles(dt);
  fog.revealArea(mapState.playerQ, mapState.playerR, 6);

  incomeTimer += dt;
  if (incomeTimer >= 4) {
    incomeTimer = 0;
    const ageBonus = 1 + currentAge * 0.5;
    const territoryBonus = territory.getBonus();
    const weatherIncomeMod = 1 + (weather.type === 'rain' ? 0.05 : 0);
    resources.add('food', Math.floor(2 * ageBonus * territoryBonus * weatherIncomeMod));
    resources.add('wood', Math.floor(2 * ageBonus * territoryBonus * weatherIncomeMod));
    hud.updateResources(resources);
  }

  autoGatherTimer += dt;
  if (autoGatherTimer >= 2) {
    autoGatherTimer = 0;
    for (const c of citizens) {
      if (!c.task && !c.moving) autoGather(c);
    }
  }

  renderCounter++;
  if (renderCounter % 2 === 0 && renderer && canvas) {
    renderer.resize();
    renderer.render(camera, citizens, units, getConstructedBuildings(), particles, weather);
  }

  hud.setManpower(manpower.current, manpower.max, manpower.overdraft);
  hud.setWeather(weather.getIcon(), weather.getEffects());
  hud.setTimeRipple(timeRipple.isActive, timeRipple.state, timeRipple.getCooldownPercent());
  if (isTwoPlayer) hud.setActivePlayer(activePlayer);
  const switchBtn = document.getElementById('btn-switch-player');
  if (switchBtn) switchBtn.style.display = isTwoPlayer ? '' : 'none';
  updateMiniMap();
  updateHUD();

  if (gameMode === 'campaign') {
    checkCampaignState();
  }
  if (isTwoPlayer) {
    check2PWin();
  }

  animFrameId = requestAnimationFrame(gameLoop);
}

let lastCampaignCheck = 0;
function checkCampaignState() {
  const now = performance.now();
  if (now - lastCampaignCheck < 5000) return;
  lastCampaignCheck = now;
  const state = loadCampaignState();
  const missionId = state.currentMission;
  if (missionId < 0) return;
  if (checkCampaignWin(missionId)) {
    completeMission(missionId);
    showCampaignResult(true);
  } else if (checkCampaignLose()) {
    showCampaignResult(false);
  }
}

function showCampaignResult(won) {
  isPaused = true;
  const titleEl = document.getElementById('campaign-result-title');
  const progressEl = document.getElementById('campaign-result-progress');
  const btnEl = document.getElementById('campaign-result-btn');
  const menuEl = document.getElementById('campaign-result-menu');

  if (won) {
    const state = loadCampaignState();
    if (titleEl) titleEl.innerHTML = '<span style="color:#357a38;">🏆 Mission Complete!</span>';
    const progress = getMissionProgress(state.currentMission);
    if (progressEl) {
      progressEl.innerHTML = progress.map(p => `
        <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--bg-dark);font-size:12px;">
          <span>${p.label}</span>
          <span style="color:${p.complete ? '#357a38' : '#9b2226'};">${Math.min(p.current, p.target)}/${p.target}</span>
        </div>
      `).join('');
    }
    if (btnEl) {
      btnEl.textContent = state.currentMission >= CAMPAIGN_MISSIONS.length - 1 ? '👑 VICTORY!' : 'NEXT MISSION';
      btnEl.onclick = () => {
        if (state.currentMission >= CAMPAIGN_MISSIONS.length - 1) {
          Screens.show('screen-victory');
        } else {
          Screens.show('screen-campaign');
        }
      };
    }
    Screens.show('screen-campaign-result');
  } else {
    if (titleEl) titleEl.innerHTML = '<span style="color:#9b2226;">💀 Mission Failed</span>';
    if (progressEl) progressEl.innerHTML = '<p style="color:var(--text-muted);text-align:center;font-size:13px;">Too few resources or units. Try again!</p>';
    if (btnEl) { btnEl.textContent = 'RETRY'; btnEl.onclick = () => { const s = loadCampaignState(); startCampaign(s.currentMission); }; }
    Screens.show('screen-campaign-result');
  }
}

function updateMiniMap() {
  const mc = document.getElementById('mini-map-canvas');
  if (!mc) return;
  const mctx = mc.getContext('2d');
  const mw = 80, mh = 60;
  mctx.fillStyle = '#0d0a04';
  mctx.fillRect(0, 0, mw, mh);
  if (!mapState.tiles) return;
  const sq = mw / 40;
  const sr = mh / 30;
  for (let r = 0; r < Math.min(30, mapState.tiles.length); r++) {
    if (!mapState.tiles[r]) continue;
    for (let q = 0; q < Math.min(40, mapState.tiles[r].length); q++) {
      const tile = mapState.tiles[r][q];
      if (!fog.isRevealed(q, r)) { mctx.fillStyle = '#0d0a04'; mctx.fillRect(q * sq, r * sr, sq, sr); continue; }
      mctx.fillStyle = tile.terrain === 'water' ? '#1e5fa8' : tile.color || '#4a8022';
      mctx.globalAlpha = tile.explored ? 1 : 0.5;
      mctx.fillRect(q * sq, r * sr, sq + 0.5, sr + 0.5);
      mctx.globalAlpha = 1;
      if (tile.building) { mctx.fillStyle = '#faf7f0'; mctx.fillRect(q * sq - 1, r * sr - 1, 3, 3); }
    }
  }
  mctx.fillStyle = '#faf7f0';
  mctx.beginPath();
  mctx.arc(mapState.playerQ * sq, mapState.playerR * sr, 3, 0, Math.PI * 2);
  mctx.fill();
  if (ai.active && ai.baseQ) { mctx.fillStyle = '#9b2226'; mctx.fillRect(ai.baseQ * sq - 2, ai.baseR * sr - 2, 4, 4); }
}

function updateHUD() {
  const el = document.getElementById('unit-info');
  if (!el) return;
  const military = units.filter(u => u.alive && u.faction === 'roma');
  const citizenCount = citizens.length;
  if (military.length > 0 || citizenCount > 0) {
    el.innerHTML = `
      <span class="unit-name">👥${citizenCount}</span>
      <span class="unit-stat">⚔️${military.length}</span>
      <span class="unit-stat">🌾${Math.floor(resources.food)}</span>
      <span class="unit-stat">🪵${Math.floor(resources.wood)}</span>
      <span class="unit-stat">🪙${Math.floor(resources.gold)}</span>
      <span class="unit-stat">🪨${Math.floor(resources.stone)}</span>
    `;
  } else {
    el.innerHTML = '<span style="color:var(--text-muted);font-size:11px;">Tap a citizen to select</span>';
  }
}
