export function $(sel) { return document.querySelector(sel); }
export function $$(sel) { return document.querySelectorAll(sel); }

export function rand(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function formatNumber(n) {
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return String(n);
}

export function generateName() {
  const first = ['Bruno', 'Elena', 'Marcus', 'Aria', 'Lucan', 'Sera', 'Draven', 'Nora', 'Pax', 'Lyra', 'Gaius', 'Freya', 'Odin', 'Zara', 'Kael', 'Mira', 'Theron', 'Isolde', 'Rex', 'Nova'];
  return first[rand(0, first.length - 1)];
}

export function hexToRGBA(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
