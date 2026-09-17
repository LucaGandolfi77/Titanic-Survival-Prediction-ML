export function createInitialState(model) {
  return {
    model,
    scale: 1,
    tx: 0,
    ty: 0,
    dragging: false,
    last: [0, 0],
    selected: null,
  };
}

export function computeNodePositions(model, width = 1200, height = 800, margin = 80) {
  const layers = model.layers.length;
  if (layers === 0) return [];
  const layerX = (i) => margin + (i * (width - 2 * margin)) / Math.max(layers - 1, 1);
  const positions = [];
  for (let i = 0; i < layers; i++) {
    const size = model.layers[i].size;
    positions[i] = [];
    const colYStart = (height - size * 80) / 2 + 40;
    for (let n = 0; n < size; n++) {
      positions[i].push({ x: layerX(i), y: colYStart + n * 80 });
    }
  }
  return positions;
}

export function clampScale(s) {
  return Math.max(0.2, Math.min(4, s));
}
