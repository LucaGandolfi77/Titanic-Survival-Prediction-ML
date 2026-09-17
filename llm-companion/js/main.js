const Game = (() => {
  let initialized = false;

  function init() {
    if (initialized) return;
    initialized = true;

    const canvas = document.getElementById('bg-canvas');
    if (canvas) Visualizer.init(canvas);
    if (document.getElementById('convo-input')) Conversation.init();

    document.getElementById('btn-start')?.addEventListener('click', startNewGame);
    document.getElementById('btn-continue')?.addEventListener('click', resumeGame);
    document.getElementById('faq-close')?.addEventListener('click', () => toggleFAQ(false));

    Lumina.startMoodCycle();
    renderCurrentScreen();
    setInterval(gameLoop, 5000);
  }

  function loadSaved() {
    if (loadFromLocal()) {
      setState({ currentScreen: 'game' });
      showScreen('game');
      renderGameScreen();
    }
  }

  function startNewGame() {
    resetGame();
    setState({ currentScreen: 'game' });
    showScreen('game');
    renderGameScreen();
    Lumina.startMoodCycle();
  }

  function resumeGame() {
    setState({ currentScreen: 'game' });
    showScreen('game');
    renderGameScreen();
    Lumina.startMoodCycle();
  }

  function showScreen(name) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    const screen = document.getElementById('screen-' + name);
    if (screen) screen.classList.add('active');
  }

  function renderCurrentScreen() {
    const state = getState();
    if (state.currentScreen === 'menu') showScreen('menu');
    else if (state.currentScreen === 'ending') { showScreen('ending'); renderEnding(); }
    else renderGameScreen();
  }

  function renderGameScreen() {
    const state = getState();
    renderHeader();
    renderCabin();
    renderTrainingPanel();
    renderLuminaPanel();
    renderStatsBar();
    renderCapabilitiesTree();
    ChapterEngine.renderTransitionButton();
  }

  function renderHeader() {
    const state = getState();
    const header = document.getElementById('game-header');
    if (!header) return;

    header.innerHTML = `
      <div class="header-left">
        <span class="header-chapter" id="chapter-title">Ch. ${state.chapter}: ${STORY[state.chapter]?.title || ''}</span>
      </div>
      <div class="header-right">
        <div class="love-indicator"><span class="heart-fill">❤️</span><span>${state.lumina.loveLevel}%</span></div>
        <div class="stat-pill">Lv.<strong>${state.lumina.level}</strong></div>
        <button class="icon-btn" id="btn-faq" title="FAQ & Tutorial">❓</button>
        <button class="icon-btn" id="btn-new-game" title="New Game">🔄</button>
      </div>
    `;

    document.getElementById('btn-faq')?.addEventListener('click', () => toggleFAQ(true));
    document.getElementById('btn-new-game')?.addEventListener('click', resetAndRestart);

    document.getElementById('chapter-title') && updateLoveDisplay();
  }

  function updateLoveDisplay() {
    const state = getState();
    const indicator = document.querySelector('.love-indicator');
    if (indicator) indicator.innerHTML = `<span class="heart-fill">❤️</span><span>${state.lumina.loveLevel}%</span>`;
    const title = document.getElementById('chapter-title');
    if (title) title.textContent = `Ch. ${state.chapter}: ${STORY[state.chapter]?.title || ''}`;
  }

  function renderCabin() {
    const state = getState();
    const container = document.getElementById('cabin-container');
    if (!container) return;

    let html = '<h2 style="font-family:\'Cormorant Garamond\',serif;color:var(--accent-gold);font-size:1.3rem;margin-bottom:12px;width:100%;max-width:900px;">Explore the Cabin</h2>';
    html += '<div class="cabin-view">';
    CABIN_AREAS.forEach(area => {
      const explored = state.cabin.exploredAreas.includes(area.id);
      let itemsHtml = '';
      if (area.items) {
        itemsHtml = area.items.map(itemId => {
          const collected = datasetCollected(itemId);
          const ds = DATASETS[itemId];
          return `<span class="item-tag ${collected ? 'collected' : ''}">${ds?.icon || '?'} ${collected ? ds.name : (ds?.desc || item).substring(0, 25)}</span>`;
        }).join('');
      }
      html += `<div class="cabin-area" data-area="${area.id}">
        <div class="area-icon">${area.icon}</div>
        <div class="area-name">${area.name}</div>
        <div class="area-desc">${area.desc}</div>
        <div class="area-items">${itemsHtml}</div>
      </div>`;
    });
    html += '</div>';
    container.innerHTML = html;

    container.querySelectorAll('.cabin-area').forEach(el => {
      el.addEventListener('click', () => exploreCabinArea(el.dataset.area));
    });
  }

  function exploreCabinArea(areaId) {
    const area = CABIN_AREAS.find(a => a.id === areaId);
    if (!area) return;

    const isNew = exploreArea(areaId);

    if (area.items) {
      area.items.forEach(itemId => {
        if (collectDataset(itemId)) {
          addJournal('Found dataset: ' + DATASETS[itemId]?.name);
          updateLoveLevel(2);
          setMood('curious');
        }
      });
    }

    addLuminaMessage(area.exploreText || 'You explored the area.', 'journal');
    renderCabin();
    renderLuminaPanel();
    renderStatsBar();
  }

  function addLuminaMessage(text, type) {
    const body = document.getElementById('lumina-messages');
    if (!body) return;
    const msg = document.createElement('div');
    msg.className = 'lumina-message' + (type === 'player' ? ' player' : '');
    if (type === 'journal') {
      msg.style.cssText = 'border-left:3px solid var(--accent-warm);background:rgba(233,69,96,0.04);font-style:normal;font-family:"Cormorant Garamond",serif;font-size:1rem;color:var(--accent-warm);';
    }
    msg.textContent = text;
    msg.style.animation = 'fadeIn 0.5s ease';
    body.appendChild(msg);
    body.scrollTop = body.scrollHeight;
  }

  function renderLuminaPanel() {
    const body = document.getElementById('lumina-messages');
    if (!body) return;
    const panel = document.getElementById('lumina-panel');
    if (!panel) return;
    const status = panel.querySelector('.lumina-status');
    if (status) status.textContent = getState().lumina.mood.charAt(0).toUpperCase() + getState().lumina.mood.slice(1);
  }

  function renderTrainingPanel() {
    const state = getState();
    const container = document.getElementById('training-panel');
    if (!container) return;

    const collected = state.cabin.collectedItems;
    if (collected.length === 0) {
      container.innerHTML = `<div style="width:100%;max-width:900px;margin-top:10px;padding:20px 24px;background:var(--bg-card);border-radius:var(--radius-lg);border:1px solid rgba(255,255,255,0.06);text-align:center;">
        <div style="font-size:2rem;margin-bottom:8px;">🔬</div>
        <h3 style="font-family:'Cormorant Garamond',serif;color:var(--accent-gold);margin-bottom:6px;font-size:1.2rem;">No datasets yet</h3>
        <p style="color:var(--text-secondary);font-size:0.9rem;">Explore the cabin to find datasets you can use to train Lumina.</p>
      </div>`;
      return;
    }

    let datasetsHtml = '';
    collected.forEach(dsId => {
      const ds = DATASETS[dsId];
      if (!ds) return;
      datasetsHtml += `<div class="dataset-card" data-dataset="${dsId}">
        <div class="ds-icon">${ds.icon}</div>
        <div class="ds-name">${ds.name}</div>
        <div class="ds-desc">${ds.desc.substring(0, 40)}...</div>
      </div>`;
    });

    container.innerHTML = `<div style="width:100%;max-width:900px;margin-top:10px;">
      <h3 style="font-family:'Cormorant Garamond',serif;color:var(--accent-gold);font-size:1.2rem;margin-bottom:12px;">🔬 Train Lumina</h3>
      <div class="dataset-grid" style="margin-bottom:20px;">${datasetsHtml}</div>
      <div style="background:var(--bg-card);border-radius:var(--radius-lg);padding:20px 24px;border:1px solid rgba(255,255,255,0.08);">
        <p style="font-weight:700;margin-bottom:10px;font-size:0.95rem;">Choose learning parameters:</p>
        <div class="param-row">
          <span class="param-label">Learning Rate</span>
          <div class="param-options">
            <span class="param-opt" data-param="lr" data-val="0.5">Gentle</span>
            <span class="param-opt selected" data-param="lr" data-val="1">Steady</span>
            <span class="param-opt" data-param="lr" data-val="2">Intense</span>
          </div>
        </div>
        <div class="param-row">
          <span class="param-label">Epochs</span>
          <div class="param-options">
            <span class="param-opt" data-param="epochs" data-val="1">Shallow</span>
            <span class="param-opt selected" data-param="epochs" data-val="2">Deep</span>
            <span class="param-opt" data-param="epochs" data-val="3">Immersive</span>
          </div>
        </div>
        <div class="param-row">
          <span class="param-label">Batch Size</span>
          <div class="param-options">
            <span class="param-opt selected" data-param="batch" data-val="1">Small</span>
            <span class="param-opt" data-param="batch" data-val="2">Medium</span>
            <span class="param-opt" data-param="batch" data-val="4">Large</span>
          </div>
        </div>
        <button class="btn-train" id="btn-train" style="width:100%;margin-top:16px;padding:16px;">✨ Train Lumina</button>
      </div>
    </div>`;

    // Wire up dataset selection
    container.querySelectorAll('.dataset-card').forEach(card => {
      card.addEventListener('click', () => {
        container.querySelectorAll('.dataset-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
      });
    });
    container.querySelector('.dataset-card')?.classList.add('selected');

    // Wire up parameter toggles
    container.querySelectorAll('.param-opt').forEach(opt => {
      opt.addEventListener('click', () => {
        const param = opt.dataset.param;
        container.querySelectorAll(`.param-opt[data-param="${param}"]`).forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
      });
    });

    document.getElementById('btn-train')?.addEventListener('click', () => {
      const selected = container.querySelector('.dataset-card.selected');
      if (!selected) return;
      const dsId = selected.dataset.dataset;
      const lr = parseFloat(container.querySelector('.param-opt.selected[data-param="lr"]')?.dataset.val || '1');
      const epochs = parseInt(container.querySelector('.param-opt.selected[data-param="epochs"]')?.dataset.val || '2');
      const batch = parseInt(container.querySelector('.param-opt.selected[data-param="batch"]')?.dataset.val || '1');
      startTraining(dsId, { lr, epochs, batch });
    });
  }

  function startTraining(dsId, params) {
    if (Training.isActive()) return;
    const state = getState();
    state.trainingActive = true;

    // Update UI to show training in progress
    const panel = document.getElementById('training-panel');
    if (panel) {
      panel.innerHTML = `<div style="width:100%;max-width:900px;margin-top:10px;">
        <div class="training-active-panel">
          <div class="training-active-header">
            <h3>🔬 Training on ${DATASETS[dsId]?.name || 'Unknown'}</h3>
            <p style="color:var(--text-secondary);font-size:0.9rem;margin:4px 0;">LR: ${params.lr}x · Epochs: ${params.epochs}x · Batch: ${params.batch}x</p>
            <div class="training-progress-bar" style="margin:12px 0;"><div class="training-progress-fill" id="training-progress-fill" style="width:0%"></div></div>
            <p style="color:var(--accent-gold);font-weight:700;font-size:1.2rem;" id="training-percent">0%</p>
          </div>
        </div>
      </div>`;
    }

    Training.start(dsId, params, (result) => {
      state.trainingActive = false;
      handleTrainingComplete(result);
    });

    // Poll for progress
    const pollInterval = setInterval(() => {
      if (!Training.isActive()) { clearInterval(pollInterval); return; }
      // Progress is shown via training.js internally
    }, 100);
  }

  function handleTrainingComplete(result) {
    const state = getState();
    addTrainingResult(result);

    const ds = DATASETS[result.dataset];
    if (ds) {
      const effectToCap = { emotion: 'empathy', logic: 'reasoning', creativity: 'creativity', depth: 'awareness', wonder: 'observation', curiosity: 'imagination' };
      const capToUnlock = effectToCap[ds.effect];
      if (capToUnlock && unlockCapability(capToUnlock)) {
        addJournal('New ability unlocked: ' + CAPABILITIES.find(c => c.id === capToUnlock)?.name);
      }
    }

    setMood('playful');
    updateLoveLevel(5);
    showTrainingResults(result);
    renderStatsBar();
    renderLuminaPanel();
  }

  function showTrainingResults(result) {
    const overlay = document.getElementById('results-overlay');
    const card = document.getElementById('results-card');
    if (!overlay || !card) return;

    const ds = DATASETS[result.dataset];
    const effectToCap = { emotion: 'empathy', logic: 'reasoning', creativity: 'creativity', depth: 'awareness', wonder: 'observation', curiosity: 'imagination' };
    const capInfo = effectToCap[ds?.effect] ? CAPABILITIES.find(c => c.id === effectToCap[ds.effect]) : null;

    card.innerHTML = `
      <h2 style="font-family:'Cormorant Garamond',serif;color:var(--accent-gold);font-size:1.8rem;margin-bottom:16px;">🎉 Training Complete</h2>
      <div class="results-section"><h3>Loss Curve</h3><div class="chart-container"><canvas id="loss-chart" width="550" height="150"></canvas></div></div>
      <div class="metrics-row">
        <div class="metric-box"><div class="metric-value">${(result.finalLoss * 100).toFixed(1)}%</div><div class="metric-label">Final Loss</div></div>
        <div class="metric-box"><div class="metric-value">${result.accuracy.toFixed(1)}%</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-box"><div class="metric-value">+${result.xp}</div><div class="metric-label">XP Earned</div></div>
        <div class="metric-box"><div class="metric-value" style="font-size:0.9rem;font-family:Nunito,sans-serif;">${result.params.lr}x · ${result.params.epochs}x · ${result.params.batch}x</div><div class="metric-label">Parameters</div></div>
      </div>
      <div class="results-section"><h3 style="font-family:Nunito,sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:1px;font-size:0.95rem;color:var(--text-secondary);margin-bottom:8px;">Lumina's Output</h3><div class="output-text">${result.output}</div></div>
      ${capInfo ? `<div class="results-section"><h3 style="font-family:Nunito,sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:1px;font-size:0.95rem;color:var(--accent-gold);margin-bottom:8px;">✨ New Ability Unlocked</h3><p style="font-style:normal;font-family:Nunito,sans-serif;color:var(--accent-gold);font-weight:700;">${capInfo.name} — ${capInfo.desc}</p></div>` : ''}
      <button class="btn-primary btn-close-results" id="btn-close-results" style="width:100%;">Continue Teaching</button>
    `;

    overlay.classList.remove('hidden');

    document.getElementById('btn-close-results')?.addEventListener('click', () => {
      overlay.classList.add('hidden');
      renderGameScreen();
      renderTrainingPanel();
    });

    setTimeout(() => drawLossChart(result.finalLoss), 200);
  }

  function drawLossChart(finalLoss) {
    const canvas = document.getElementById('loss-chart');
    if (!canvas) return;
    canvas.width = 550; canvas.height = 150;
    const ctx = canvas.getContext('2d');
    const w = 550, h = 150;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(0,0,0,0.3)'; ctx.beginPath(); ctx.roundRect(0, 0, w, h, 8); ctx.fill();
    ctx.beginPath();
    for (let i = 0; i < 60; i++) {
      const val = Math.max(0.01, finalLoss * (1 - i / 60) * (1 + Math.sin(i / 5) * 0.1));
      const x = (i / 59) * w, y = h - (val / 3) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#ffd46f'; ctx.lineWidth = 2; ctx.stroke();
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = 'rgba(255, 212, 111, 0.08)'; ctx.fill();
  }

  function toggleFAQ(open) {
    const overlay = document.getElementById('faq-modal');
    if (!overlay) return;
    if (open) { overlay.classList.remove('hidden'); renderFAQ(); }
    else { overlay.classList.add('hidden'); }
  }

  function renderFAQ() {
    const container = document.getElementById('faq-content');
    if (!container) return;
    let html = '';
    FAQ_CONTENT.forEach(item => {
      html += `<div class="faq-section"><h3>${item.q}</h3><p>${item.a}</p></div>`;
    });
    html += `<div class="faq-section"><h3>📖 Glossary</h3><ul style="list-style:disc;padding-left:20px;">
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">LLM</code> — Large Language Model: an AI trained on vast text data to generate language</li>
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">Token</code> — A chunk of text (word or part of word) that the model processes</li>
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">Parameter</code> — A numerical value the model adjusts during training</li>
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">Embedding</code> — A vector representing a word's meaning in multi-dimensional space</li>
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">Attention</code> — A mechanism that lets the model focus on relevant input parts</li>
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">Fine-tuning</code> — Specialized training of a pre-trained model</li>
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">RLHF</code> — Reinforcement Learning from Human Feedback</li>
      <li style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;margin-bottom:4px;"><code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;color:var(--accent-gold);">Inference</code> — Using a trained model to make predictions</li>
    </ul></div>`;
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
      <div style="text-align:center;padding:40px;max-width:650px;margin:0 auto;">
        <div style="font-size:60px;margin-bottom:16px;">${data.emoji || '🌟'}</div>
        <h1 style="font-family:'Cormorant Garamond',serif;font-size:2.5rem;color:var(--accent-gold);margin-bottom:8px;">${data.title}</h1>
        <p style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:1.2rem;color:var(--text-secondary);margin-bottom:24px;">${data.subtitle}</p>
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.15rem;line-height:1.9;margin-bottom:28px;text-align:left;">${data.text}</div>
        <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-bottom:28px;">
          <div style="background:var(--bg-card);border-radius:var(--radius-md);padding:14px 20px;min-width:80px;">
            <div style="font-size:1.5rem;font-weight:700;color:var(--accent-gold);font-family:'Cormorant Garamond',serif;">${state.lumina.level}</div>
            <div style="font-size:0.8rem;color:var(--text-secondary);">Levels</div>
          </div>
          <div style="background:var(--bg-card);border-radius:var(--radius-md);padding:14px 20px;min-width:80px;">
            <div style="font-size:1.5rem;font-weight:700;color:var(--accent-gold);font-family:'Cormorant Garamond',serif;">${state.lumina.totalTrainings}</div>
            <div style="font-size:0.8rem;color:var(--text-secondary);">Sessions</div>
          </div>
          <div style="background:var(--bg-card);border-radius:var(--radius-md);padding:14px 20px;min-width:80px;">
            <div style="font-size:1.5rem;font-weight:700;color:var(--accent-gold);font-family:'Cormorant Garamond',serif;">${state.lumina.loveLevel}%</div>
            <div style="font-size:0.8rem;color:var(--text-secondary);">Love</div>
          </div>
          <div style="background:var(--bg-card);border-radius:var(--radius-md);padding:14px 20px;min-width:80px;">
            <div style="font-size:1.5rem;font-weight:700;color:var(--accent-gold);font-family:'Cormorant Garamond',serif;">${state.capabilities.length}/8</div>
            <div style="font-size:0.8rem;color:var(--text-secondary);">Abilities</div>
          </div>
          <div style="background:var(--bg-card);border-radius:var(--radius-md);padding:14px 20px;min-width:80px;">
            <div style="font-size:1.5rem;font-weight:700;color:var(--accent-gold);font-family:'Cormorant Garamond',serif;">${timeSpent}m</div>
            <div style="font-size:0.8rem;color:var(--text-secondary);">Together</div>
          </div>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;">
          <button class="btn-primary" id="btn-replay">Play Again</button>
          <button class="btn-secondary" id="btn-menu">Main Menu</button>
        </div>
      </div>
    `;

    document.getElementById('btn-replay')?.addEventListener('click', resetAndRestart);
    document.getElementById('btn-menu')?.addEventListener('click', () => { setState({ currentScreen: 'menu' }); showScreen('menu'); });
  }

  function chooseEnding(choice) {
    const ending = getEnding(choice);
    setState({ currentScreen: 'ending', ending, lastPlayTime: Date.now() });
    saveToLocal();
    Lumina.stopMoodCycle();
    Visualizer.stopTraining();
    addJournal('Final choice: ' + choice + ' — Ending: ' + ending);
    showScreen('ending');
    renderEnding();
  }

  function resetAndRestart() {
    Visualizer.stopTraining();
    Lumina.stopMoodCycle();
    resetGame();
    setState({ currentScreen: 'menu' });
    showScreen('menu');
    document.getElementById('btn-continue').style.display = 'none';
    init();
  }

  function gameLoop() {
    const state = getState();
    if (state.currentScreen === 'game') {
      saveToLocal();
      const indicator = document.querySelector('.love-indicator');
      if (indicator) {
        const s = getState();
        indicator.innerHTML = `<span class="heart-fill">❤️</span><span>${s.lumina.loveLevel}%</span>`;
      }
    }
  }

  return { init, chooseEnding, toggleFAQ, resetAndRestart };
})();
