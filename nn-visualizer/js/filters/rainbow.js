let active = false;
let angle = 0;
let rafId = null;
let svgRoot = null;
let appState = null;

export function createRainbowFilter(svg, state) {
  svgRoot = svg;
  appState = state;
  return {
    toggle() {
      active = !active;
      svgRoot.classList.toggle('rainbow-mode', active);
      if (active) {
        animate();
      } else {
        stop();
        svgRoot.style.filter = '';
      }
    },
    isActive() {
      return active;
    },
    destroy() {
      stop();
      if (svgRoot) {
        svgRoot.classList.remove('rainbow-mode');
        svgRoot.style.filter = '';
      }
    },
  };
}

function animate() {
  if (!active || !svgRoot) return;
  angle = (angle + 1.2) % 360;
  svgRoot.style.filter = `hue-rotate(${angle}deg) saturate(1.5)`;
  rafId = requestAnimationFrame(animate);
}

function stop() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}
