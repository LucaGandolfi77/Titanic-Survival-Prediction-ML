const Game = (() => {
  let currentChapterData = null;
  let gameInterval = null;

  function init() {
    Visualizer.setGameStateRef(gameState);
    const canvas = document.getElementById('bg-canvas');
    if (canvas) Visualizer.init(canvas);

    ChapterEngine.init();

    loadSaved();

    setupEventListeners();
    renderCurrentScreen();

    gameInterval = setInterval(gameLoop, 5000);

    Lumina.startMoodCycle();
  }

  function loadSaved() {
    if (loadFromLocal()) {
      showScreen('game');
    }
  }

  function setupEventListeners() {
    window.addEventListener('training-start', () => { setState({ trainingActive: true }); });
    window.addEventListener('training-complete', (e) => { handleTrainingComplete(e.detail); });

    document.addEventListener('click', (e) => {
      const faqBtn = e.target.closest('#btn-faq');
      if (faqBtn) { toggleFAQ(true); return; }
      const faqClose = e.target.closest('#faq-close');
      if (faqClose) { toggleFAQ(false); return; }
    });
  }

  function renderCurrentScreen() {
    const state = getState();

    if (state.currentScreen === 'menu') {
      showScreen('menu');
    } else if (state.currentScreen === 'game') {
      showScreen('game');
      renderGameScreen();
    } else if (state.currentScreen === 'ending') {
      showScreen('ending');
      renderEnding();
    }
  }

  function renderGameScreen() {
    const state = getState();

    renderHeader();
    renderCabin();
    renderLuminaPanel();
    renderConversation();
    renderCapabilities();
    renderStatsBar();
    renderTransitionButton(state.chapter);
  }

  function renderHeader() {
    const state = getState();
    const header = document.getElementById('game-header');
    if (!header) return;

    const chapterTitle = `Ch. ${state.chapter}: ${STORY[state.chapter]?.title || ''}`;
    header.innerHTML = `
      <div class="header-left">
        <span class="header-chapter">${chapterTitle}</span>
      </div>
      <div class="header-right">
        <div class="love-indicator">
          <span class="heart-fill">❤️</span>
          <span>${state.lumina.loveLevel}%</span>
        </div>
        <div class="stat-pill">Lv.<strong>${state.lumina.level}</strong></div>
        <button class="icon-btn" id="btn-faq" title="FAQ & Tutorial">❓</button>
        <button class="icon-btn" id="btn-new-game" title="New Game">🔄</button>
      </div>
    `;

    document.getElementById('btn-faq')?.addEventListener('click', () => toggleFAQ(true));
    document.getElementById('btn-new-game')?.addEventListener('click', () => resetAndRestart());
  }

  function renderCabin() {
    const state = getState();
    const container = document.getElementById('cabin-container');
    if (!container) return;

    container.innerHTML = '';
    CABIN_AREAS.forEach(area => {
      const div = document.createElement('div');
      div.className = 'cabin-area';
      const itemsHtml = area.items.map(item => {
        const collected = datasetCollected(item);
        const ds = DATASETS[item];
        return `<span class="item-tag ${collected ? 'collected' : ''}">${ds?.icon || '?'} ${collected ? ds.name : ds?.desc || item}</span>`;
      }).join('');

      div.innerHTML = `
        <div class="area-icon">${area.icon}</div>
        <div class="area-name">${area.name}</div>
        <div class="area-desc">${area.desc}</div>
        <div class="area-items">${itemsHtml}</div>
      `;

      div.addEventListener('click', () => exploreCabinArea(area));
      container.appendChild(div);
    });
  }

  function exploreCabinArea(area) {
    const isNew = exploreArea(area.id);
    const state = getState();

    if (area.items) {
      area.items.forEach(itemId => {
        if (collectDataset(itemId)) {
          addJournal(`Found dataset: ${DATASETS[itemId]?.name || itemId}`);
          updateLoveLevel(2);
          state.lumina.mood = 'curious';
        }
      });
    }

    if (isNew && area.exploreText) {
      addMessage('lumina', area.exploreText);
    }

    renderCabin();
    renderLuminaPanel();
    Visualizer.setNeedsBgRedraw();
  }

  function renderLuminaPanel() {
    const state = getState();
    const panel = document.getElementById('lumina-panel');
    if (!panel) return;

    const moodClasses = {
      playful: 'mood-playful', curious: 'mood-curious', loving: 'mood-loving',
      sad: 'mood-sad', grieving: 'mood-grieving', neutral: 'mood-neutral'
    };
    const moodClass = moodClasses[state.lumina.mood] || 'mood-neutral';

    panel.querySelector('.lumina-status').textContent = state.lumina.mood.charAt(0).toUpperCase() + state.lumina.mood.slice(1);

    const newMsg = document.createElement('div');
    newMsg.className = `lumina-message ${state.lumina.mood === 'grieving' ? 'journal' : ''}`;
    newMsg.innerHTML = `<span class="lumina-mood ${moodClass}">${state.lumina.mood}</span>`;
    panel.querySelector('.lumina-body').appendChild(newMsg);
  }

  function renderConversation() {
    const state = getState();
    const convoArea = document.getElementById('convo-area');
    if (!convoArea) return;
    convoArea.style.display = state.currentChapterLumina === 'conversation' ? 'flex' : 'none';
  }

  function renderCapabilities() {
    const state = getState();
    const container = document.getElementById('capabilities-tree');
    if (!container) return;

    container.innerHTML = '<div class="cap-tree"><div class="cap-level" id="cap-levels"></div></div>';

    const levels = [
      ['language', 'creativity', 'reasoning', 'empathy', 'observation', 'imagination'],
      ['awareness', 'consciousness'],
    ];

    const levelsContainer = document.getElementById('cap-levels');
    levels.forEach((levelCaps, levelIdx) => {
      const row = document.createElement('div');
      row.className = 'cap-level';
      levelCaps.forEach(capId => {
        const cap = CAPABILITIES.find(c => c.id === capId);
        if (!cap) return;
        const unlocked = isCapabilityUnlocked(capId);
        const node = document.createElement('div');
        node.className = `cap-node ${unlocked ? 'unlocked active' : ''}`;
        node.innerHTML = `${cap.icon}<div class="cap-tooltip">${cap.name}${unlocked ? ' ✓' : ''}<br><small>${cap.desc}</small></div>`;
        row.appendChild(node);
      });
      if (levelIdx < levels.length - 1) {
        const conn = document.createElement('div');
        conn.style.cssText = 'width:2px;height:20px;background:rgba(255,212,111,0.2);margin:0 auto;';
        row.appendChild(conn);
      }
      levelsContainer.appendChild(row);
    });
  }

  function renderStatsBar() {
    const state = getState();
    const container = document.getElementById('stats-bar');
    if (!container) return;

    container.innerHTML = `
      <div class="stat-pill">🎓 <strong>${state.lumina.level}</strong> Level</div>
      <div class="stat-pill">💬 <strong>${state.lumina.conversationCount}</strong> Talks</div>
      <div class="stat-pill">📚 <strong>${state.datasets.length}</strong> Datasets</div>
      <div class="stat-pill">🔥 <strong>${state.lumina.totalTrainings}</strong> Sessions</div>
      <div class="stat-pill">❤️ <strong>${state.lumina.loveLevel}%</strong> Love</div>
      <div class="stat-pill">🌟 <strong>${state.capabilities.length}</strong>/8 Abilities</div>
    `;
  }

  function handleTrainingComplete(result) {
    setState({ trainingActive: false });
    addTrainingResult(result);

    if (result.dataset) {
      const effect = DATASETS[result.dataset]?.effect;
      const unlockMap = { emotion: 'empathy', logic: 'reasoning', creativity: 'creativity', depth: 'awareness', wonder: 'observation', curiosity: 'imagination' };
      const capToUnlock = unlockMap[effect];
      if (capToUnlock && unlockCapability(capToUnlock)) {
        addJournal(`New ability unlocked: ${CAPABILITIES.find(c => c.id === capToUnlock)?.name}`);
      }
    }

    updateLumina({ mood: 'playful' });
    updateLoveLevel(5);

    showTrainingResults(result);
  }

  function showTrainingResults(result) {
    const overlay = document.getElementById('results-overlay');
    const card = document.getElementById('results-card');
    if (!overlay || !card) return;

    const capUnlocked = DATASETS[result.dataset] ?
      CAPABILITIES.find(c => c.id === ({emotion:'empathy',logic:'reasoning',creativity:'creativity',depth:'awareness',wonder:'observation',curiosity:'imagination'}[DATASETS[result.dataset].effect])) : null;

    card.innerHTML = `
      <h2>🎉 Training Complete</h2>

      <div class="results-section">
        <h3>Loss Curve</h3>
        <div class="chart-container">
          <canvas id="loss-chart" width="500" height="150"></canvas>
        </div>
      </div>

      <div class="metrics-row">
        <div class="metric-box">
          <div class="metric-value">${(result.finalLoss * 100).toFixed(1)}%</div>
          <div class="metric-label">Final Loss</div>
        </div>
        <div class="metric-box">
          <div class="metric-value">${result.accuracy.toFixed(1)}%</div>
          <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-box">
          <div class="metric-value">+${result.xp}</div>
          <div class="metric-label">XP Earned</div>
        </div>
        <div class="metric-box">
          <div class="metric-value">${result.params.lr}x / ${result.params.epochs}x / ${result.params.batch}x</div>
          <div class="metric-label">Parameters</div>
        </div>
      </div>

      <div class="results-section">
        <h3>Lumina Said:</h3>
        <div class="output-text">${result.output}</div>
      </div>

      ${capUnlocked ? `<div class="results-section"><h3>✨ New Ability Unlocked!</h3><p style="font-style:normal;font-family:'Nunito',sans-serif;color:var(--accent-gold);font-weight:700;">${capUnlocked.name} — ${capUnlocked.desc}</p></div>` : ''}

      <button class="btn-primary btn-close-results" id="btn-close-results">Continue Teaching</button>
    `;

    overlay.classList.remove('hidden');

    document.getElementById('btn-close-results')?.addEventListener('click', () => {
      overlay.classList.add('hidden');
      drawLossChartFromData(result.finalLoss);
      renderGameScreen();
    });

    setTimeout(() => drawLossChartFromData(result.finalLoss), 100);
  }

  function drawLossChartFromData(finalLoss) {
    const canvas = document.getElementById('loss-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width; const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath();
    ctx.roundRect(0, 0, w, h, 8);
    ctx.fill();

    ctx.beginPath();
    for (let i = 0; i < 60; i++) {
      const val = Math.max(0.01, finalLoss * (1 - i / 60) * (1 + Math.sin(i / 5) * 0.1));
      const x = (i / 59) * w;
      const y = h - (val / 3) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#ffd46f';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function toggleFAQ(open) {
    const overlay = document.getElementById('faq-modal');
    if (!overlay) return;
    if (open) {
      overlay.classList.remove('hidden');
      renderFAQ();
    } else {
      overlay.classList.add('hidden');
    }
  }

  function renderFAQ() {
    const container = document.getElementById('faq-content');
    if (!container) return;

    let html = '';
    FAQ_CONTENT.forEach(item => {
      html += `
        <div class="faq-section">
          <h3>${item.q}</h3>
          <p>${item.a}</p>
        </div>
      `;
    });
    html += `
      <div class="faq-section">
        <h3>📖 Glossary</h3>
        <ul>
          <li><code>LLM</code> — Large Language Model: an AI trained on vast text data to generate language</li>
          <li><code>Token</code> — A chunk of text (word or part of word) that the model processes</li>
          <li><code>Parameter</code> — A numerical value the model adjusts during training (billions in large models)</li>
          <li><code>Embedding</code> — A vector of numbers representing a word's meaning in multi-dimensional space</li>
          <li><code>Attention</code> — A mechanism that lets the model focus on relevant parts of the input</li>
          <li><code>Fine-tuning</code> — Specialized training of a pre-trained model on a narrower dataset</li>
          <li><code>RLHF</code> — Reinforcement Learning from Human Feedback: aligning AI with human preferences</li>
          <li><code>Inference</code> — Using a trained model to make predictions (vs. training it)</li>
          <li><code>Backpropagation</code> — The algorithm that adjusts model parameters by propagating errors backwards</li>
          <li><code>Transformer</code> — The neural network architecture powering modern LLMs (2017)</li>
        </ul>
      </div>
    `;
    container.innerHTML = html;
  }

  function renderEnding() {
    const state = getState();
    const ending = state.ending;
    if (!ending) return;
    const data = ENDINGS[ending];
    if (!data) return;

    const container = document.getElementById('ending-content');
    if (!container) return;

    const timeSpent = Math.floor((state.lastPlayTime - state.startTime) / 60000);

    container.innerHTML = `
      <div class="ending-emoji">${data.emoji || '🌟'}</div>
      <h1 class="ending-title">${data.title}</h1>
      <p class="ending-subtitle">${data.subtitle}</p>
      <div class="ending-text">${data.text}</div>
      <div class="ending-stats">
        <div class="ending-stat">
          <div class="val">${state.lumina.level}</div>
          <div class="lbl">Levels Reached</div>
        </div>
        <div class="ending-stat">
          <div class="val">${state.lumina.totalTrainings}</div>
          <div class="lbl">Training Sessions</div>
        </div>
        <div class="ending-stat">
          <div class="val">${state.lumina.loveLevel}%</div>
          <div class="lbl">Love Shared</div>
        </div>
        <div class="ending-stat">
          <div class="val">${state.capabilities.length}/8</div>
          <div class="lbl">Abilities Grown</div>
        </div>
        <div class="ending-stat">
          <div class="val">${timeSpent}m</div>
          <div class="lbl">Time Together</div>
        </div>
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;">
        <button class="btn-primary" id="btn-replay">Play Again</button>
        <button class="btn-secondary" id="btn-menu">Main Menu</button>
      </div>
    `;

    document.getElementById('btn-replay')?.addEventListener('click', resetAndRestart);
    document.getElementById('btn-menu')?.addEventListener('click', () => { setState({ currentScreen: 'menu' }); showScreen('menu'); });
  }

  function resetAndRestart() {
    Visualizer.stopTraining();
    Lumina.stopMoodCycle();
    if (gameInterval) { clearInterval(gameInterval); gameInterval = null; }
    resetGame();
    showScreen('menu');
    init();
  }

  function gameLoop() {
    const state = getState();
    if (state.currentScreen !== 'game') return;
    saveToLocal();
  }

  function startTrainingFromUI(datasetId, params) {
    if (Training.isActive()) return;
    Training.start(datasetId, params,
      (progress) => { /* progress bar updates */ },
      (result) => { handleTrainingComplete(result); }
    );
  }

  function chooseEnding(choice) {
    const ending = getEnding(choice);
    setState({
      currentScreen: 'ending',
      ending,
      lastPlayTime: Date.now()
    });
    saveToLocal();
    showScreen('ending');
    Lumina.stopMoodCycle();
    Visualizer.stopTraining();
    addJournal(`Final choice: ${choice} — Ending: ${ending}`);
  }

  return {
    init, renderGameScreen, startTrainingFromUI, chooseEnding,
    toggleFAQ, resetAndRestart, gameLoop
  };
})();
