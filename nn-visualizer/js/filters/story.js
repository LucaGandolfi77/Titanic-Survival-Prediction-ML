let active = false;
let svgRoot = null;
let appState = null;
let overlays = [];
let highlights = [];

const LAYER_STORIES = [
  'Input Layer — riceve i dati grezzi dal mondo reale. Ogni nodo rappresenta una feature.',
  'Hidden Layers — trasformano progressivamente i dati in rappresentazioni astratte. Più strati = apprendimento più profondo.',
  'Output Layer — produce la risposta finale della rete. I valori indicano la predizione o la classificazione.',
];

export function createStoryFilter(svg, state) {
  svgRoot = svg;
  appState = state;
  return {
    toggle() {
      active = !active;
      if (active) {
        build();
      } else {
        destroy();
      }
    },
    isActive() {
      return active;
    },
    destroy() {
      destroy();
    },
  };
}

function build() {
  destroy();
  if (!svgRoot || !appState || !appState.model) return;

  const layers = appState.model.layers.length;
  const width = 1200;
  const height = 800;
  const positions = [];

  for (let i = 0; i < layers; i++) {
    const size = appState.model.layers[i].size;
    positions[i] = [];
    const x = (i / Math.max(layers - 1, 1)) * width;
    for (let n = 0; n < size; n++) {
      const y = ((n / Math.max(size, 1)) - 0.5) * (height * 0.7);
      positions[i].push({ x, y });
    }
  }

  const viewer = svgRoot.parentElement;
  if (!viewer) return;

  for (let i = 0; i < layers; i++) {
    const story = LAYER_STORIES[i] || LAYER_STORIES[LAYER_STORIES.length - 1];
    const p = positions[i][0];
    if (!p) continue;

    const overlay = document.createElement('div');
    overlay.className = 'story-layer-label';
    overlay.style.cssText = `
      position: absolute;
      left: ${(p.x / width) * 100}%;
      top: ${(p.y / height) * 100}%;
      transform: translate(-50%, -120%);
      background: rgba(15,23,36,0.92);
      color: #e6eef8;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      max-width: 280px;
      text-align: center;
      pointer-events: none;
      border: 1px solid rgba(96,165,250,0.3);
      animation: storyFadeIn 0.4s ease ${i * 0.15}s both;
      z-index: 10;
      line-height: 1.4;
    `;
    overlay.textContent = story;
    viewer.appendChild(overlay);
    overlays.push(overlay);
  }

  for (let i = 0; i < layers; i++) {
    for (let n = 0; n < positions[i].length; n++) {
      const p = positions[i][n];
      const highlight = document.createElement('div');
      highlight.className = 'story-highlight';
      highlight.style.cssText = `
        position: absolute;
        left: ${(p.x / width) * 100}%;
        top: ${(p.y / height) * 100}%;
        width: 40px;
        height: 40px;
        transform: translate(-50%, -50%);
        border: 2px solid rgba(96,165,250,0.6);
        border-radius: 50%;
        animation: storyPulse 2s ease-in-out infinite;
        animation-delay: ${(i + n) * 0.1}s;
        pointer-events: none;
        z-index: 9;
      `;
      viewer.appendChild(highlight);
      highlights.push(highlight);
    }
  }
}

function destroy() {
  overlays.forEach((el) => {
    if (el.parentElement) el.parentElement.removeChild(el);
  });
  highlights.forEach((el) => {
    if (el.parentElement) el.parentElement.removeChild(el);
  });
  overlays = [];
  highlights = [];
}
