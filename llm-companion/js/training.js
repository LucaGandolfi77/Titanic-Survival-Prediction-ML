const Training = (() => {
  let active = false;
  let dataset = null;
  let params = { lr: 1, epochs: 1, batch: 1 };
  let progress = 0;
  let intervalId = null;
  let onUpdate = null;
  let onComplete = null;

  function start(ds, p, onProgress, onFinish) {
    if (active) return;
    active = true;
    dataset = ds;
    params = p;
    progress = 0;
    onUpdate = onProgress;
    onComplete = onFinish;
    Visualizer.startTraining();
    if (window.dispatchEvent) {
      window.dispatchEvent(new CustomEvent('training-start'));
    }
    const interval = 50;
    intervalId = setInterval(tick, interval);
  }

  function tick() {
    const speedFactor = params.lr * (1 + params.epochs * 0.3);
    const advance = 0.002 * speedFactor * (1 + Math.random() * 0.1);
    progress = Math.min(1, progress + advance);

    if (onUpdate) onUpdate(progress, dataset, params);
    Visualizer.setTrainingProgress(progress);

    if (progress >= 1) {
      finish();
    }
  }

  function finish() {
    active = false;
    if (intervalId) { clearInterval(intervalId); intervalId = null; }
    Visualizer.stopTraining();

    const ds = DATASETS[dataset];
    const lr = params.lr;
    const epochs = params.epochs;
    const batch = params.batch;

    const baseLoss = 2.5;
    const lossReduction = Math.min(0.98, 0.3 * lr * (epochs / 2) * (batch / 2) + Math.random() * 0.05);
    const finalLoss = Math.max(0.01, baseLoss * (1 - lossReduction));
    const accuracy = Math.min(99.9, 40 + 50 * lossReduction + Math.random() * 10);
    const xp = Math.floor(20 * lossReduction + 10 * epochs + 5);

    const outputType = getOutputType(ds.effect);
    const output = getOutputByType(outputType);

    const result = {
      dataset: dataset,
      datasetName: ds ? ds.name : 'Unknown',
      lossReduction, finalLoss, accuracy, xp,
      outputType, output,
      params: { ...params },
      timestamp: Date.now(),
    };

    if (onComplete) onComplete(result);
    window.dispatchEvent(new CustomEvent('training-complete', { detail: result }));
  }

  function getOutputType(effect) {
    switch (effect) {
      case 'emotion': case 'depth': return 'letter';
      case 'creativity': return 'poem';
      case 'logic': case 'reasoning': return 'calculation';
      case 'wonder': case 'curiosity': return 'story';
      default: return 'poem';
    }
  }

  function cancel() {
    if (intervalId) { clearInterval(intervalId); intervalId = null; }
    active = false;
    Visualizer.stopTraining();
  }

  function isActive() { return active; }
  function getProgress() { return progress; }

  return { start, cancel, isActive, getProgress };
})();
