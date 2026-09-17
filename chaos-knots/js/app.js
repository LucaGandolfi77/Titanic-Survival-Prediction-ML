let state = getState();
let board = [];
let selectedUnraveler = null;
let selectedCell = null;
let isAnimating = false;
let cooldownTimers = {};
let currentLevelConfig = null;
let gameScore = 0;
let levelThreadReward = 0;

const LEVELS = [
  { region: 'Tangled Meadow', title: 'Soft Beginnings', desc: 'Learn to untangle' },
  { region: 'Dusk Garden', title: 'Fading Blooms', desc: 'Gold threads appear' },
  { region: 'Shadow Hollow', title: 'Dark Webs', desc: 'Shadow threads emerge' },
  { region: 'Chaos Void', title: 'Endless Tangle', desc: 'All thread types' },
];

function getLevelRegion(level) {
  const idx = Math.min(Math.floor((level - 1) / 10), LEVELS.length - 1);
  return LEVELS[idx];
}

function showScreen(id) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.remove('active'));
  const screen = document.getElementById(id);
  if (screen) {
    screen.classList.add('active');
    screen.style.display = 'flex';
  }
  updateMenuThreads();
}

function updateMenuThreads() {
  const el = document.getElementById('menuThreads');
  if (el) el.textContent = state.threads;
  const idle = document.getElementById('idleNotice');
  if (idle) {
    const elapsed = state.lastPlayDate ? (Date.now() - state.lastPlayDate) / 1000 : 0;
    if (elapsed > 120) {
      const gain = Math.floor(elapsed / 60) * 2;
      if (gain > 0) {
        idle.textContent = 'Unravelers earned ' + gain + ' threads while you were away';
        idle.style.display = '';
      }
    }
  }
}

function stopAllCooldowns() {
  Object.values(cooldownTimers).forEach((t) => clearInterval(t));
  cooldownTimers = {};
}

function startCooldownTimer(unravelerId) {
  const u = UNRAVELERS[unravelerId];
  if (!u) return;
  if (cooldownTimers[unravelerId]) clearInterval(cooldownTimers[unravelerId]);
  cooldownTimers[unravelerId] = setInterval(() => {
    state.cooldowns[unravelerId] = Math.max(0, (state.cooldowns[unravelerId] || 0) - 0.1);
    updateHotbar();
    if (state.cooldowns[unravelerId] <= 0) {
      clearInterval(cooldownTimers[unravelerId]);
      delete cooldownTimers[unravelerId];
      state.cooldowns[unravelerId] = 0;
    }
  }, 100);
}

function renderBoard() {
  const container = document.getElementById('boardContainer');
  container.innerHTML = '';
  const boardEl = document.createElement('div');
  boardEl.className = 'board';

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.r = r;
      cell.dataset.c = c;
      const data = board[r][c];
      if (data) {
        cell.textContent = THREAD_EMOJI[data.type] || '🧵';
        if (data.isAnchor) cell.classList.add('anchor');
        if (data.isBoss) cell.classList.add('boss');
        if (data.isAnchor) {
          const timeLeft = state.anchorTimers?.[`${r},${c}`];
          if (timeLeft !== undefined && timeLeft < 3000 && timeLeft > 0) {
            cell.classList.add('warning');
          }
        }
      }
      cell.addEventListener('click', () => handleCellClick(r, c));
      boardEl.appendChild(cell);
    }
  }
  container.appendChild(boardEl);
}

function updateCell(r, c) {
  const cells = document.querySelectorAll('.cell');
  const idx = r * COLS + c;
  const cell = cells[idx];
  if (!cell) return;
  const data = board[r][c];
  if (data) {
    cell.textContent = THREAD_EMOJI[data.type] || '🧵';
    cell.className = 'cell';
    if (data.isAnchor) cell.classList.add('anchor');
    if (data.isBoss) cell.classList.add('boss');
    if (data.isAnchor) {
      const timeLeft = state.anchorTimers?.[`${r},${c}`];
      if (timeLeft !== undefined && timeLeft < 3000 && timeLeft > 0) cell.classList.add('warning');
    }
  } else {
    cell.textContent = '';
    cell.className = 'cell';
  }
}

function handleCellClick(r, c) {
  if (isAnimating) return;

  if (selectedUnraveler) {
    const u = UNRAVELERS[selectedUnraveler];
    if ((state.cooldowns[selectedUnraveler] || 0) > 0) {
      playErrorSound();
      return;
    }
    if (!canDeploy(state, selectedUnraveler)) return;
    deployOnCell(r, c);
    return;
  }

  const cell = board[r][c];
  if (!cell) return;

  if (!selectedCell) {
    selectedCell = { r, c };
    highlightCell(r, c, true);
  } else if (selectedCell.r === r && selectedCell.c === c) {
    selectedCell = null;
    highlightCell(r, c, false);
  } else {
    const dr = Math.abs(selectedCell.r - r);
    const dc = Math.abs(selectedCell.c - c);
    if ((dr === 1 && dc === 0) || (dr === 0 && dc === 1)) {
      trySwap(selectedCell.r, selectedCell.c, r, c);
      selectedCell = null;
    } else {
      highlightCell(selectedCell.r, selectedCell.c, false);
      selectedCell = { r, c };
      highlightCell(r, c, true);
    }
  }
}

function highlightCell(r, c, on) {
  const cells = document.querySelectorAll('.cell');
  const idx = r * COLS + c;
  const cell = cells[idx];
  if (cell) {
    if (on) cell.classList.add('selected');
    else cell.classList.remove('selected');
  }
}

function trySwap(r1, c1, r2, c2) {
  if (isAnimating) return;
  const temp = board[r1][c1];
  board[r1][c1] = board[r2][c2];
  board[r2][c2] = temp;

  const matches = findMatches(board);
  if (matches.length > 0) {
    playCutSound();
    selectedCell = null;
    isAnimating = true;
    state.cooldowns = {};
    processMove(matches);
  } else {
    playErrorSound();
    const temp2 = board[r1][c1];
    board[r1][c1] = board[r2][c2];
    board[r2][c2] = temp2;
    renderBoard();
    selectedCell = null;
  }
}

function processMove(matches) {
  const positions = matches;
  playCutSound();

  positions.forEach(({ r, c }) => {
    const cells = document.querySelectorAll('.cell');
    const cell = cells[r * COLS + c];
    if (cell) cell.classList.add('cutting');
  });

  setTimeout(() => {
    bossDamage(board, positions);

    const results = processBoard(board, (cascadeLevel) => {
      playCascadeSound(cascadeLevel);
    });

    gameScore += results.matches * 10 + results.cascades * 50;
    document.getElementById('gameScore').textContent = gameScore;

    const threadGain = results.matches;
    if (threadGain > 0) {
      addThread(threadGain);
      levelThreadReward += threadGain;
      updateHUD();
      showThreadGain(threadGain);
    }

    renderBoard();
    updateBossBar();
    updateHotbar();
    updateAnchorWarnings();

    if (checkBossDefeated()) {
      isAnimating = false;
      setTimeout(() => endLevel(true), 500);
      return;
    }

    setTimeout(() => {
      isAnimating = false;
      checkAnchorActivation();
      checkLevelDefeat();
    }, 400);
  }, 350);
}

function checkBossDefeated() {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c] && board[r][c].isBoss && board[r][c].hp > 0) return false;
    }
  }
  return true;
}

function updateBossBar() {
  const bossBar = document.getElementById('bossHealth');
  let hasBoss = false;
  let totalHp = 0;
  let currentHp = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c] && board[r][c].isBoss) {
        hasBoss = true;
        totalHp += board[r][c].hp;
        currentHp += board[r][c].hp;
      }
    }
  }
  if (!hasBoss) { bossBar.style.display = 'none'; return; }
  bossBar.style.display = 'block';
  bossBar.innerHTML = `
    <div class="boss-bar-container">
      <div class="boss-bar-label">👹 Boss Knot</div>
      <div class="boss-bar-track"><div class="boss-bar-fill" style="width:${(currentHp / Math.max(totalHp, 1)) * 100}%"></div></div>
    </div>
  `;
}

function checkAnchorActivation() {
  let activated = false;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c] && board[r][c].isAnchor) {
        const key = `${r},${c}`;
        if (!state.anchorTimers) state.anchorTimers = {};
        if (state.anchorTimers[key] === undefined) {
          state.anchorTimers[key] = 6000;
        }
        state.anchorTimers[key] -= 400;
        if (state.anchorTimers[key] <= 0) {
          const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
          for (const [dr, dc] of dirs) {
            const nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && !board[nr][nc]) {
              const types = ['red', 'blue', 'gold', 'shadow'];
              const maxT = Math.min(2 + Math.floor((state.currentLevel || 1) / 3), types.length);
              board[nr][nc] = createCell(types[Math.floor(Math.random() * maxT)], false, false);
            }
          }
          delete state.anchorTimers[key];
          playAnchorSound();
          activated = true;
        }
      }
    }
  }
  if (activated) renderBoard();
}

function updateAnchorWarnings() {
  const cells = document.querySelectorAll('.cell.anchor');
  cells.forEach((cell) => {
    const r = parseInt(cell.dataset.r);
    const c = parseInt(cell.dataset.c);
    const key = `${r},${c}`;
    const timeLeft = state.anchorTimers?.[key];
    if (timeLeft !== undefined && timeLeft < 3000 && timeLeft > 0) {
      cell.classList.add('warning');
    } else {
      cell.classList.remove('warning');
    }
  });
}

function checkLevelDefeat() {
  if (isAnimating) return;
  let hasThreads = false;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c]) { hasThreads = true; break; }
    }
    if (hasThreads) break;
  }
  let hasMoves = hasValidMoves(board);
  if (!hasMoves) {
    shuffleBoard(board);
    renderBoard();
  }
  let threadCount = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c]) threadCount++;
    }
  }
  if (threadCount <= 2 && !checkBossDefeated()) {
    setTimeout(() => endLevel(false), 500);
  }
}

function deployOnCell(r, c) {
  const result = deployUnraveler(state, selectedUnraveler, board, r, c);
  if (!result) return;
  const { cellsToRemove, unraveler } = result;

  if (unravelerId === 'shade') {
    const ids = Object.keys(UNRAVELERS);
    const shuffled = [...ids].sort(() => Math.random() - 0.5);
    shuffled.forEach((id, i) => {
      const rr = Math.floor(i / COLS);
      const cc = i % COLS;
      if (rr < ROWS && board[rr][cc] && UNRAVELERS[id]) {
        board[rr][cc].type = THREAD_TYPES[Math.floor(Math.random() * THREAD_TYPES.length)];
      }
    });
    shuffleBoard(board);
    playDeploySound();
    state.cooldowns[selectedUnraveler] = UNRAVELERS[selectedUnraveler].cooldown;
    startCooldownTimer(selectedUnraveler);
    selectedUnraveler = null;
    renderBoard();
    updateHotbar();
    return;
  }

  if (cellsToRemove.length === 0) {
    playErrorSound();
    return;
  }

  playDeploySound();
  cellsToRemove.forEach(({ r: cr, c: cc }) => {
    const cells = document.querySelectorAll('.cell');
    const cell = cells[cr * COLS + cc];
    if (cell) cell.classList.add('cutting');
  });

  setTimeout(() => {
    removeCells(board, cellsToRemove);
    bossDamage(board, cellsToRemove);
    const results = processBoard(board, (lv) => playCascadeSound(lv));
    gameScore += results.matches * 10 + results.cascades * 50;
    document.getElementById('gameScore').textContent = gameScore;
    const gain = results.matches;
    if (gain > 0) { addThread(gain); levelThreadReward += gain; updateHUD(); }

    state.cooldowns[selectedUnraveler] = UNRAVELERS[selectedUnraveler].cooldown;
    startCooldownTimer(selectedUnraveler);

    renderBoard();
    updateBossBar();
    updateHotbar();
    selectedUnraveler = null;

    if (checkBossDefeated()) {
      isAnimating = false;
      setTimeout(() => endLevel(true), 500);
      return;
    }
    isAnimating = false;
  }, 350);
}

function renderHotbar() {
  const hotbar = document.getElementById('hotbar');
  hotbar.innerHTML = '';
  const roster = (state.roster || []).slice(0, 4);
  while (roster.length < 4) roster.push(null);

  roster.forEach((id) => {
    if (!id) {
      const empty = document.createElement('div');
      empty.className = 'hotbar-slot';
      empty.style.opacity = '0.2';
      hotbar.appendChild(empty);
      return;
    }
    const u = UNRAVELERS[id];
    const slot = document.createElement('div');
    slot.className = 'hotbar-slot' + (selectedUnraveler === id ? ' selected' : '');
    slot.innerHTML = `<span>${u.icon}</span>`;
    slot.dataset.id = id;
    slot.addEventListener('click', () => {
      if ((state.cooldowns[id] || 0) > 0) { playErrorSound(); return; }
      if (selectedUnraveler === id) {
        selectedUnraveler = null;
        clearHighlights();
      } else {
        selectedUnraveler = id;
        selectedCell = null;
        clearHighlights();
      }
      updateHotbar();
    });
    hotbar.appendChild(slot);
  });
  updateHotbar();
}

function updateHotbar() {
  const slots = document.querySelectorAll('.hotbar-slot');
  slots.forEach((slot) => {
    const id = slot.dataset.id;
    if (!id) return;
    const cd = state.cooldowns[id] || 0;
    const u = UNRAVELERS[id];
    if (cd > 0) {
      slot.classList.add('on-cooldown');
      const cdText = slot.querySelector('.cd-text');
      if (cdText) cdText.textContent = Math.ceil(cd) + 's';
    } else {
      slot.classList.remove('on-cooldown');
      const cdText = slot.querySelector('.cd-text');
      if (cdText) cdText.textContent = '';
    }
  });
}

function clearHighlights() {
  document.querySelectorAll('.cell.selected, .cell.target-valid').forEach((c) => {
    c.classList.remove('selected', 'target-valid');
  });
}

function updateHUD() {
  const ge = document.getElementById('gameThreads');
  if (ge) ge.textContent = state.threads;
  updateMenuThreads();
}

function startLevel(levelNum) {
  stopAllCooldowns();
  state.currentLevel = levelNum;
  state.anchorTimers = {};
  state.cooldowns = {};
  gameScore = 0;
  levelThreadReward = 0;
  selectedUnraveler = null;
  selectedCell = null;
  isAnimating = false;

  currentLevelConfig = getLevelRegion(levelNum);
  board = generateBoard(levelNum);

  while (!hasValidMoves(board)) {
    shuffleBoard(board);
  }

  const regionInfo = getLevelRegion(levelNum);
  document.getElementById('levelInfo').textContent = `Lv.${levelNum} — ${regionInfo.region}`;
  document.getElementById('gameScore').textContent = '0';

  showScreen('screenGame');
  renderBoard();
  renderHotbar();
  updateHUD();
  updateBossBar();
}

function endLevel(won) {
  stopAllCooldowns();
  if (won) {
    playVictorySound();
    const stars = 1 + (levelThreadReward > 10 ? 1 : 0) + (levelThreadReward > 20 ? 1 : 0);
    const prev = state.levelStars[state.currentLevel] || 0;
    if (stars > prev) {
      state.levelStars[state.currentLevel] = stars;
    }
    saveState(state);
    showResult(true, stars);
  } else {
    playDefeatSound();
    showResult(false, 0);
  }
}

function showResult(won, stars) {
  const screen = document.getElementById('screenResult');
  document.getElementById('resultIcon').textContent = won ? '🎉' : '😔';
  document.getElementById('resultTitle').textContent = won ? 'Untangled!' : 'Too Tangled';
  document.getElementById('resultSubtitle').textContent = won
    ? `Level ${state.currentLevel} cleared with ${stars} star${stars > 1 ? 's' : ''}!`
    : 'Not enough threads were cut. Try again!';

  const reward = won ? Math.max(20, levelThreadReward * 2) : 5;
  document.getElementById('resultRewards').innerHTML = `
    <div class="threads-display">+${reward} 🧶</div>
    ${won && stars > 1 ? `<div style="margin-top:0.5rem;color:var(--accent);font-size:0.85rem;">⭐ ${'⭐'.repeat(stars)}</div>` : ''}
  `;

  addThread(reward);

  document.getElementById('btnNextLevel').style.display = won ? '' : 'none';

  showScreen('screenResult');
  setTimeout(() => {
    document.getElementById('screenResult').style.display = 'flex';
    document.querySelector('#screenResult > div').classList.add('result-anim');
  }, 50);
}

function goToLevelSelect() {
  showScreen('screenLevels');
  renderLevelList();
}

function renderLevelList() {
  const list = document.getElementById('levelList');
  list.innerHTML = '';
  for (let i = 1; i <= 30; i++) {
    const region = getLevelRegion(i);
    const stars = state.levelStars[i] || 0;
    const unlocked = i === 1 || (state.levelStars[i - 1] && state.levelStars[i - 1] > 0);
    const isCurrentRegion = getLevelRegion(i).region === region.region;
    const card = document.createElement('div');
    card.className = 'level-card' + (unlocked ? '' : ' locked');
    card.innerHTML = `
      <div class="level-num">${i}</div>
      <div class="level-info">
        <div class="level-title">${region.title}</div>
        <div class="level-desc">${unlocked ? region.desc : 'Complete previous region'}</div>
      </div>
      <div class="level-stars">${stars > 0 ? '⭐'.repeat(stars) : '☆☆☆'}</div>
    `;
    if (unlocked) {
      card.addEventListener('click', () => startLevel(i));
    }
    list.appendChild(card);
  }
}

function openGacha() {
  state = getState();
  showScreen('screenGacha');
  renderGacha();
}

function renderGacha() {
  const content = document.getElementById('gachaContent');
  const cost = state.hasFreePull ? 0 : 50;
  content.innerHTML = `
    <div class="gacha-pull">
      <div style="font-size:2.5rem;margin-bottom:0.5rem;">🧵</div>
      <h3 style="font-size:1.2rem;font-weight:700;color:var(--accent);margin-bottom:0.5rem;">Void Pull</h3>
      <p style="font-size:0.85rem;color:var(--text-secondary);line-height:1.5;margin-bottom:1rem;">Pull an Unraveler to add to your roster. Duplicates become Thread Shards.</p>
      <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.5rem;">
        Common 55% · Rare 30% · Epic 12% · Legendary 3%
      </div>
      <div class="pull-cost">${state.hasFreePull ? 'Free pull available!' : `Costs ${cost} 🧶`}</div>
      <div class="shards-display">Pity: Rare ${state.pityRare}/10 · Epic ${state.pityEpic}/10</div>
      <button id="btnDoPull" class="btn-primary pull-btn" ${state.threads < cost && !state.hasFreePull ? 'disabled style="opacity:0.5"' : ''}>
        ${state.hasFreePull ? 'Free Pull' : 'Pull'}
      </button>
    </div>
    <div id="gachaResult"></div>
  `;
  document.getElementById('btnDoPull').addEventListener('click', doPull);
}

function doPull() {
  state = getState();
  const cost = state.hasFreePull ? 0 : 50;
  if (state.threads < cost) {
    playErrorSound();
    renderGacha();
    return;
  }

  const result = performPull(state);
  if (!result) {
    playErrorSound();
    renderGacha();
    return;
  }

  playPullSound();

  state = getState();

  const resultEl = document.getElementById('gachaResult');
  const spin = document.createElement('div');
  spin.style.cssText = 'text-align:center;margin-top:2rem;animation:gachaSpin 1.5s ease-out forwards;';
  spin.innerHTML = `<div style="font-size:3rem;animation:gachaSpinIcon 1s linear infinite;">🧵</div><div style="font-size:0.9rem;color:var(--text-secondary);margin-top:0.5rem;">Unraveling...</div>`;
  resultEl.innerHTML = '';
  resultEl.appendChild(spin);

  setTimeout(() => {
    playRevealSound(result.rarity);
    const rarityClass = `gacha-rarity-${result.rarity}`;
    resultEl.innerHTML = `
      <div class="gacha-card ${rarityClass}" style="margin-top:1.5rem;animation:gachaReveal 0.6s ease-out;">
        <div style="font-size:3rem;margin-bottom:0.5rem;">${result.unraveler.icon}</div>
        <div style="font-size:1.2rem;font-weight:700;color:var(--accent);">${result.unraveler.name}</div>
        <div class="roster-rarity rarity-${result.rarity}" style="margin-top:0.25rem;">${result.rarity}</div>
        <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.5rem;">${result.unraveler.desc}</div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.5rem;">
          ${state.roster.filter((id) => id === result.id).length > 1 ? '+50 Thread Shards' : 'Added to roster!'}
        </div>
      </div>
    `;
  }, 1500);
}

function openRoster() {
  state = getState();
  showScreen('screenRoster');
  renderRoster();
}

function renderRoster() {
  const content = document.getElementById('rosterContent');
  content.innerHTML = '';
  if (!state.roster || state.roster.length === 0) {
    content.innerHTML = '<p style="text-align:center;color:var(--text-muted);">No Unravelers yet. Try a Void Pull!</p>';
    return;
  }

  state.roster.forEach((id) => {
    const u = UNRAVELERS[id];
    if (!u) return;
    const lvl = state.unravelerLevels[id] || 1;
    const shards = state.threadShards[id] || 0;
    const card = document.createElement('div');
    card.className = 'roster-card';
    card.innerHTML = `
      <div class="roster-icon">${u.icon}</div>
      <div class="roster-info">
        <div class="roster-name">${u.name}</div>
        <div class="roster-desc">${u.desc}</div>
        <div class="roster-rarity rarity-${u.rarity}">${u.rarity} · Lv.${lvl}</div>
        <div class="shards-display">${shards > 0 ? `${shards} Thread Shards` : ''}</div>
      </div>
    `;
    content.appendChild(card);
  });
}

function quitLevel() {
  stopAllCooldowns();
  endLevel(false);
  goToLevelSelect();
}

function init() {
  state = getState();
  state.lastPlayDate = Date.now();
  saveState(state);

  document.getElementById('btnContinue').addEventListener('click', () => {
    startLevel(state.currentLevel);
  });
  document.getElementById('btnPlay').addEventListener('click', goToLevelSelect);
  document.getElementById('btnGacha').addEventListener('click', openGacha);
  document.getElementById('btnRoster').addEventListener('click', openRoster);
  document.getElementById('btnBackFromLevels').addEventListener('click', () => showScreen('screenMenu'));
  document.getElementById('btnBackFromGacha').addEventListener('click', () => showScreen('screenMenu'));
  document.getElementById('btnBackFromRoster').addEventListener('click', () => showScreen('screenMenu'));
  document.getElementById('btnQuit').addEventListener('click', quitLevel);
  document.getElementById('btnResultMenu').addEventListener('click', () => {
    showScreen('screenMenu');
    renderLevelList();
  });
  document.getElementById('btnNextLevel').addEventListener('click', () => {
    const next = state.currentLevel + 1;
    if (next <= 30) {
      startLevel(next);
    } else {
      showScreen('screenMenu');
    }
  });

  updateMenuThreads();
  renderLevelList();

  // Show Continue button if there's an active level
  const continueBtn = document.getElementById('btnContinue');
  if (state.currentLevel > 1) {
    continueBtn.style.display = '';
    document.getElementById('continueLevel').textContent = state.currentLevel;
  }

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }

  let lastTenants = 0;
  function idleProgress() {
    const now = Date.now();
    if (state.lastPlayDate) {
      const elapsed = (now - state.lastPlayDate) / 1000;
      if (elapsed > 60) {
        const gain = Math.floor(elapsed / 60) * 2;
        if (gain > 0) {
          addThread(gain);
          if (gain > lastTenants) {
            lastTenants = gain;
            showIdleNotice(gain);
          }
        }
      }
    }
    state.lastPlayDate = now;
    saveState(state);
    updateMenuThreads();
    setTimeout(idleProgress, 30000);
  }
  setTimeout(idleProgress, 10000);
}

function showIdleNotice(gain) {
  const idle = document.getElementById('idleNotice');
  if (idle) {
    idle.textContent = 'Unravelers earned ' + gain + ' threads while you were away';
    idle.style.display = '';
  }
}

function showThreadGain(amount) {
  const boardEl = document.getElementById('boardContainer');
  if (!boardEl) return;
  const toast = document.createElement('div');
  toast.className = 'thread-toast';
  toast.textContent = '+' + amount + ' 🧶';
  boardEl.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.5s'; setTimeout(() => toast.remove(), 500); }, 800);
}

init();
