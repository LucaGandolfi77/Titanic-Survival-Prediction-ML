const SVG_NS = 'http://www.w3.org/2000/svg';
import { computeNodePositions } from './state.js';

function createSvgElement(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

export function createSvgRoot(width = 1200, height = 800) {
  const svg = createSvgElement('svg', {
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: 'xMidYMid meet',
  });
  svg.style.width = '100%';
  svg.style.height = '100%';
  svg.style.display = 'block';
  return svg;
}

export function renderModel(svgRoot, state, onNodeClick, cosmic = false) {
  const model = state.model;
  const width = 1200;
  const height = 800;

  while (svgRoot.firstChild) {
    svgRoot.removeChild(svgRoot.firstChild);
  }

  const positions = computeNodePositions(model);
  if (positions.length === 0) return positions;

  if (cosmic) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="cosmic-glow" x="-100%" y="-100%" width="300%" height="300%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <radialGradient id="nodeGrad0" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#4a90d9" stop-opacity="0.9"/>
        <stop offset="100%" stop-color="#0a1628" stop-opacity="0.3"/>
      </radialGradient>
      <radialGradient id="nodeGrad1" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#e05577" stop-opacity="0.8"/>
        <stop offset="100%" stop-color="#0a1628" stop-opacity="0.3"/>
      </radialGradient>
      <linearGradient id="linkPos" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#4a90d9"/>
        <stop offset="100%" stop-color="#60a5fa"/>
      </linearGradient>
      <linearGradient id="linkNeg" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#e05577"/>
        <stop offset="100%" stop-color="#fb7185"/>
      </linearGradient>
    `;
    svgRoot.appendChild(defs);
  }

  const g = createSvgElement('g', {
    transform: `translate(${state.tx},${state.ty}) scale(${state.scale})`,
  });
  g.setAttribute('id', 'transform-group');
  svgRoot.appendChild(g);

  const layers = model.layers.length;

  // Draw links
  for (let L = 1; L < layers; L++) {
    const W = model.weights[L - 1];
    if (!W) continue;
    for (let to = 0; to < W.length; to++) {
      for (let from = 0; from < W[to].length; from++) {
        const w = W[to][from];
        const p1 = positions[L - 1][from];
        const p2 = positions[L][to];
        if (!p1 || !p2) continue;
        const line = createSvgElement('line', {
          x1: p1.x, y1: p1.y,
          x2: p2.x, y2: p2.y,
          'stroke-width': cosmic
            ? Math.min(8, Math.max(1, Math.abs(w) * 8)).toFixed(2)
            : Math.min(6, Math.max(0.6, Math.abs(w) * 6)).toFixed(2),
          stroke: cosmic
            ? (w > 0 ? 'url(#linkPos)' : 'url(#linkNeg)')
            : (w > 0 ? '#60a5fa' : '#fb7185'),
          opacity: cosmic ? '0.7' : '0.6',
          filter: cosmic ? 'url(#glow)' : '',
        });
        g.appendChild(line);
      }
    }
  }

  // Draw nodes
  for (let i = 0; i < layers; i++) {
    for (let n = 0; n < positions[i].length; n++) {
      const p = positions[i][n];
      const group = createSvgElement('g', {
        transform: `translate(${p.x},${p.y})`,
      });

      const circle = createSvgElement('circle', {
        r: cosmic ? '26' : '22',
        fill: cosmic
          ? (i === 0 ? 'url(#nodeGrad0)' : 'url(#nodeGrad1)')
          : (i === 0 ? '#111827' : '#0b1220'),
        stroke: cosmic ? 'transparent' : '#20344d',
        'stroke-width': cosmic ? '0' : '2',
        filter: cosmic ? 'url(#cosmic-glow)' : '',
      });
      circle.style.cursor = 'pointer';
      if (cosmic) {
        circle.style.filter = 'drop-shadow(0 0 8px rgba(96,165,250,0.4))';
      }
      group.appendChild(circle);

      const label = createSvgElement('text', {
        y: '6',
        'text-anchor': 'middle',
        'font-size': cosmic ? '13' : '11',
        fill: cosmic ? '#fff' : '#9aa3c7',
        'font-weight': cosmic ? '700' : '400',
      });
      label.textContent = `${i}:${n}`;
      group.appendChild(label);

      group.addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (onNodeClick) onNodeClick(i, n);
      });

      g.appendChild(group);
    }
  }

  return positions;
}

export function applyTransform(svgRoot, state) {
  const g = svgRoot.querySelector('#transform-group');
  if (!g) return;
  g.setAttribute('transform', `translate(${state.tx},${state.ty}) scale(${state.scale})`);
}

export function focusNode(svgRoot, state, x, y) {
  const vbW = 1200;
  const vbH = 800;
  state.tx = (vbW / 2 - x) * state.scale;
  state.ty = (vbH / 2 - y) * state.scale;
  applyTransform(svgRoot, state);
}
