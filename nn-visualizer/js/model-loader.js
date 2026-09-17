export function createSampleModel() {
  return {
    name: 'Sample MLP',
    layers: [
      { type: 'input', size: 3 },
      { type: 'dense', size: 4, activation: 'relu' },
      { type: 'output', size: 2, activation: 'softmax' },
    ],
    weights: [],
    biases: [],
  };
}

export function initModelWeights(model) {
  const rand = (a = -1, b = 1) => a + Math.random() * (b - a);
  for (let L = 1; L < model.layers.length; L++) {
    const to = model.layers[L].size;
    const from = model.layers[L - 1].size;
    const W = Array.from({ length: to }, () =>
      Array.from({ length: from }, () => Number(rand().toFixed(3)))
    );
    const B = Array.from({ length: to }, () => Number(rand(-0.5, 0.5).toFixed(3)));
    model.weights.push(W);
    model.biases.push(B);
  }
  return model;
}

export async function loadModelFromJSON(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const model = await response.json();
    if (!model.layers || !Array.isArray(model.layers)) {
      throw new Error('Invalid model format: missing layers array');
    }
    return model;
  } catch (error) {
    console.error('Failed to load model:', error);
    return null;
  }
}

export function buildNodeDetailHtml(model, layer, idx) {
  let html = `<strong>Node ${layer}:${idx}</strong>`;
  if (layer > 0) {
    const weights = model.weights[layer - 1]?.[idx];
    const bias = model.biases[layer - 1]?.[idx];
    if (weights) {
      const sum = weights.reduce((a, b) => a + b, 0);
      const activation = 1 / (1 + Math.exp(-sum));
      html += `<div class="detail-section">Bias: ${bias ?? 'N/A'}</div>`;
      html += `<div class="detail-section">Activation: <span class="${activation > 0.5 ? 'detail-weight-positive' : 'detail-weight-negative'}">${activation.toFixed(4)}</span></div>`;
      html += `<div class="detail-section">Incoming weights: <ul>` +
        weights.map((w, i) => `<li>${i} → <span class="${w > 0 ? 'detail-weight-positive' : 'detail-weight-negative'}">${w}</span></li>`).join('') + `</ul></div>`;
    }
  } else {
    html += `<div>Input node (no weights)</div>`;
  }
  if (layer < model.layers.length - 1) {
    const Wnext = model.weights[layer];
    if (Wnext) {
      const out = [];
      for (let to = 0; to < Wnext.length; to++) {
        out.push(`${layer}:${idx} → ${to} = ${Wnext[to][idx]}`);
      }
      html += `<div class="detail-section">Outgoing: <ul>` +
        out.map((l) => `<li>${l}</li>`).join('') + `</ul></div>`;
    }
  }
  return html;
}
