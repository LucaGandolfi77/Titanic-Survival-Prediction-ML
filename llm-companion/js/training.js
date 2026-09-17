const Training = (() => {
  let active = false, dataset = null, params = { lr: 1, epochs: 1, batch: 1 };
  let progress = 0, intervalId = null;

  function start(ds, p, onComplete) {
    if (active) return;
    active = true; dataset = ds; params = p; progress = 0;
    Visualizer.startTraining();
    document.dispatchEvent(new CustomEvent('training-start'));
    intervalId = setInterval(tick, 50);
  }

  function tick() {
    const speedFactor = params.lr * (1 + params.epochs * 0.3);
    progress = Math.min(1, progress + 0.002 * speedFactor * (1 + Math.random() * 0.1));
    Visualizer.setTrainingProgress(progress);
    const state = getState();
    state.trainingActive = true;
    const panel = document.getElementById('training-active');
    if (panel) {
      const name = DATASETS[dataset]?.name || 'Unknown';
      panel.classList.remove('hidden');
      panel.innerHTML = `<div class="training-active-panel"><div class="training-active-header"><h3>🔬 Training on ${name}</h3><p>Parameters: LR=${params.lr}x · Epochs=${params.epochs}x · Batch=${params.batch}x</p><div class="training-progress-bar"><div class="training-progress-fill" style="width:${Math.round(progress * 100)}%"></div></div><p style="margin-top:6px;color:var(--accent-gold);font-weight:700;">${Math.round(progress * 100)}%</p></div></div>`;
    }
    if (progress >= 1) finish();
  }

  function finish() {
    active = false;
    if (intervalId) { clearInterval(intervalId); intervalId = null; }
    Visualizer.stopTraining();
    document.dispatchEvent(new CustomEvent('training-complete'));
  }

  function cancel() { if (intervalId) { clearInterval(intervalId); intervalId = null; } active = false; Visualizer.stopTraining(); }
  function isActive() { return active; }
  function getDataset() { return dataset; }
  function getParams() { return params; }

  return { start, cancel, isActive, getDataset, getParams };
})();
