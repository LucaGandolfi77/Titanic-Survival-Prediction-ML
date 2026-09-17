export function createTrainingSimulator(model) {
  let running = false;
  let epoch = 0;
  let loss = 1.0;
  let history = [];
  let weightsSnapshot = null;

  function start() {
    weightsSnapshot = deepCloneWeights(model);
    running = true;
    epoch = 0;
    loss = 1.0;
    history = [];
  }

  function step(learningRate = 0.01) {
    if (!running) return null;
    epoch++;

    loss = Math.max(0.01, loss * (1 - learningRate * 0.1) + (Math.random() - 0.5) * 0.02);
    history.push({ epoch, loss });

    if (weightsSnapshot) {
      for (let L = 0; L < model.weights.length; L++) {
        if (!model.weights[L]) continue;
        for (let to = 0; to < model.weights[L].length; to++) {
          for (let from = 0; from < model.weights[L][to].length; from++) {
            const gradient = (Math.random() - 0.5) * 0.1 * (1.0 / (1 + epoch * 0.05));
            model.weights[L][to][from] -= gradient * learningRate;
            model.weights[L][to][from] = Number(model.weights[L][to][from].toFixed(4));
          }
        }
        if (model.biases[L]) {
          for (let t = 0; t < model.biases[L].length; t++) {
            const gradient = (Math.random() - 0.5) * 0.05;
            model.biases[L][t] -= gradient * learningRate;
            model.biases[L][t] = Number(model.biases[L][t].toFixed(4));
          }
        }
      }
    }

    return { epoch, loss };
  }

  function stop() {
    running = false;
  }

  function isRunning() {
    return running;
  }

  function getHistory() {
    return history;
  }

  function getEpoch() {
    return epoch;
  }

  function getLoss() {
    return loss;
  }

  function reset() {
    if (weightsSnapshot) {
      model.weights = deepCloneWeights(weightsSnapshot.model);
      model.biases = deepCloneBiases(weightsSnapshot.model);
    }
    running = false;
    epoch = 0;
    loss = 1.0;
    history = [];
  }

  function deepCloneWeights(m) {
    return {
      model: {
        weights: m.weights.map((W) => W.map((row) => [...row])),
        biases: m.biases.map((b) => [...b]),
      },
    };
  }

  function deepCloneBiases(m) {
    return m.biases.map((b) => [...b]);
  }

  return { start, step, stop, isRunning, getHistory, getEpoch, getLoss, reset };
}
