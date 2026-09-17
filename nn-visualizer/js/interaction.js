import { clampScale } from './state.js';

export function setupPanZoom(svgRoot, state, applyTransformFn, onMoveCallback) {
  svgRoot.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = -e.deltaY * 0.001;
    const oldScale = state.scale;
    state.scale = clampScale(state.scale * (1 + delta));
    const rect = svgRoot.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    state.tx -= cx / state.scale - cx / oldScale;
    state.ty -= cy / state.scale - cy / oldScale;
    applyTransformFn();
  }, { passive: false });

  svgRoot.addEventListener('mousedown', (e) => {
    state.dragging = true;
    state.last = [e.clientX, e.clientY];
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.dragging) return;
    const dx = e.clientX - state.last[0];
    const dy = e.clientY - state.last[1];
    state.last = [e.clientX, e.clientY];
    state.tx += dx;
    state.ty += dy;
    applyTransformFn();
    if (onMoveCallback) onMoveCallback();
  });

  window.addEventListener('mouseup', () => {
    state.dragging = false;
  });

  svgRoot.addEventListener('click', () => {});
}

export function setupTouchGestures(svgRoot, state, applyTransformFn) {
  let lastTouchDist = 0;
  let lastTouchCenter = { x: 0, y: 0 };

  svgRoot.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      state.dragging = true;
      state.last = [e.touches[0].clientX, e.touches[0].clientY];
    } else if (e.touches.length === 2) {
      state.dragging = false;
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      lastTouchDist = Math.sqrt(dx * dx + dy * dy);
      lastTouchCenter = {
        x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
        y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
      };
    }
  }, { passive: true });

  svgRoot.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && state.dragging) {
      const dx = e.touches[0].clientX - state.last[0];
      const dy = e.touches[0].clientY - state.last[1];
      state.last = [e.touches[0].clientX, e.touches[0].clientY];
      state.tx += dx;
      state.ty += dy;
      applyTransformFn();
    } else if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const scaleDelta = dist / (lastTouchDist || 1);
      state.scale = clampScale(state.scale * scaleDelta);
      lastTouchDist = dist;
      const center = {
        x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
        y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
      };
      state.tx += center.x - lastTouchCenter.x;
      state.ty += center.y - lastTouchCenter.y;
      lastTouchCenter = center;
      applyTransformFn();
    }
  }, { passive: false });

  svgRoot.addEventListener('touchend', () => {
    state.dragging = false;
    lastTouchDist = 0;
  });
}
