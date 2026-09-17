let active = false;
let overlay = null;
let svgRoot = null;
let appState = null;
let viewer = null;
let ripples = [];
let startTime = 0;
let notesPlayed = 0;
let rafId = null;

const NOTES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25];
const NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C5'];

export function createMusicMinigame(svg, state, audio) {
  svgRoot = svg;
  appState = state;
  return {
    toggle() {
      active = !active;
      if (active) {
        start(audio);
      } else {
        stop();
      }
    },
    isActive() {
      return active;
    },
    destroy() {
      stop();
    },
  };
}

function start(audio) {
  if (!svgRoot) return;
  viewer = svgRoot.parentElement;
  if (!viewer) return;

  notesPlayed = 0;
  startTime = Date.now();

  overlay = document.createElement('div');
  overlay.className = 'music-minigame-overlay';
  overlay.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 20;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px;
  `;

  const header = document.createElement('div');
  header.style.cssText = `
    background: rgba(15,23,36,0.92);
    color: #fbbf24;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 16px;
    font-weight: 700;
    pointer-events: auto;
    border: 1px solid rgba(251,191,36,0.4);
    animation: musicPop 0.3s ease;
  `;
  header.textContent = '🎵 Music Neurons — Click nodes to play!';
  overlay.appendChild(header);

  const scoreBar = document.createElement('div');
  scoreBar.id = 'music-score';
  scoreBar.style.cssText = `
    background: rgba(15,23,36,0.85);
    color: #60a5fa;
    padding: 4px 14px;
    border-radius: 12px;
    font-size: 13px;
    margin-top: 6px;
    pointer-events: auto;
    border: 1px solid rgba(96,165,250,0.3);
  `;
  scoreBar.textContent = '♫ 0 notes | ⏱ 60s';
  overlay.appendChild(scoreBar);

  viewer.appendChild(overlay);

  if (svgRoot) {
    svgRoot.style.cursor = 'pointer';
    svgRoot._musicClickHandler = (e) => {
      if (!active) return;
      const target = e.target.closest ? e.target.closest('g') : null;
      if (!target) return;
      playNodeSound(target, audio);
    };
    svgRoot.addEventListener('click', svgRoot._musicClickHandler);
  }

  updateTimer();
}

function playNodeSound(group, audio) {
  const circles = group.querySelectorAll('circle');
  let layer = -1;
  let idx = -1;
  group.querySelectorAll('text').forEach((text) => {
    const parts = text.textContent.split(':');
    if (parts.length === 2) {
      layer = parseInt(parts[0], 10);
      idx = parseInt(parts[1], 10);
    }
  });

  const noteIndex = ((layer * 3 + idx) % NOTES.length + NOTES.length) % NOTES.length;
  const freq = NOTES[noteIndex];
  const name = NOTE_NAMES[noteIndex];

  notesPlayed++;

  if (audio && audio.playNote) {
    try { audio.playNote(layer, idx, 0.4); } catch (e) { /* audio not ready */ }
  }

  const scoreEl = document.getElementById('music-score');
  if (scoreEl) {
    scoreEl.textContent = `♫ ${notesPlayed} notes | ⏱ ${Math.max(0, 60 - Math.floor((Date.now() - startTime) / 1000))}s`;
  }

  const ripple = document.createElement('div');
  ripple.style.cssText = `
    position: absolute;
    pointer-events: none;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: 2px solid #fbbf24;
    transform: translate(-50%, -50%);
    animation: musicRipple 0.6s ease-out forwards;
    z-index: 21;
  `;

  const svgRect = svgRoot.getBoundingClientRect();
  const viewerRect = viewer.getBoundingClientRect();
  const svgBox = svgRoot.getBoundingClientRect();

  let cx = 0, cy = 0;
  if (group && group.style && group.style.transform) {
    const transform = group.style.transform;
    const match = transform.match(/translate\(([-\d.]+)px,([-\d.]+)px\)/);
    if (match) {
      const vb = 1200;
      const vh = 800;
      cx = (svgRect.left - viewerRect.left) + (parseFloat(match[1]) / vb) * svgRect.width + 15;
      cy = (svgRect.top - viewerRect.top) + (parseFloat(match[2]) / vh) * svgRect.height + 15;
    }
  }

  if (cx === 0 && cy === 0) {
    cx = svgRect.left - viewerRect.left + svgRect.width / 2;
    cy = svgRect.top - viewerRect.top + svgRect.height / 2;
  }

  ripple.style.left = cx + 'px';
  ripple.style.top = cy + 'px';
  viewer.appendChild(ripple);
  ripples.push(ripple);

  if (notesPlayed % 10 === 0) {
    const scoreEl = document.getElementById('music-score');
    if (scoreEl) {
      scoreEl.textContent = `🎉 ${notesPlayed} notes! Amazing!`;
      setTimeout(() => {
        if (scoreEl && active) {
          scoreEl.textContent = `♫ ${notesPlayed} notes | ⏱ ${Math.max(0, 60 - Math.floor((Date.now() - startTime) / 1000))}s`;
        }
      }, 1500);
    }
  }

  if (Date.now() - startTime > 60000) {
    endGame();
  }
}

function updateTimer() {
  if (!active) return;
  const remaining = Math.max(0, 60 - Math.floor((Date.now() - startTime) / 1000));
  const scoreEl = document.getElementById('music-score');
  if (scoreEl) {
    scoreEl.textContent = `♫ ${notesPlayed} notes | ⏱ ${remaining}s`;
  }
  if (remaining <= 0) {
    endGame();
    return;
  }
  rafId = setTimeout(() => updateTimer(), 1000);
}

function endGame() {
  if (!active) return;
  const scoreEl = document.getElementById('music-score');
  if (scoreEl) {
    scoreEl.textContent = `🎵 ${notesPlayed} notes played!`;
  }
  stop();
}

function stop() {
  if (rafId) {
    clearTimeout(rafId);
    rafId = null;
  }
  if (overlay && overlay.parentElement) {
    overlay.parentElement.removeChild(overlay);
  }
  overlay = null;
  if (svgRoot) {
    svgRoot.style.cursor = '';
    if (svgRoot._musicClickHandler) {
      svgRoot.removeEventListener('click', svgRoot._musicClickHandler);
      svgRoot._musicClickHandler = null;
    }
  }
  ripples.forEach((el) => {
    if (el.parentElement) el.parentElement.removeChild(el);
  });
  ripples = [];
}
