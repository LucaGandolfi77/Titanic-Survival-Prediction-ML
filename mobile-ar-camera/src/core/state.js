export function createAppState() {
  const state = {
    selectedFilter: null,
    facingMode: 'user',
    zoom: 1,
    torchEnabled: false,
    stream: null,
    lastBlob: null,
    lastObjectURL: null,
    previewing: false,
    webglSupported: true,
    renderer: null,
    scene: null,
    camera: null,
    mesh: null,
    material: null,
    videoTexture: null,
    arState: createARState(),
    detectionBusy: false,
    arUnavailable: false,
    gesture: { mode: null, startX: 0, startDistance: 0, startZoom: 1 },
    loopHandle: 0,
    lastFrameAt: performance.now(),
    mirrorMode: true,
  };

  const listeners = new Map();

  function subscribe(key, callback) {
    if (!listeners.has(key)) listeners.set(key, new Set());
    listeners.get(key).add(callback);
    return () => listeners.get(key)?.delete(callback);
  }

  function notify(key, value) {
    const subs = listeners.get(key);
    if (subs) subs.forEach((cb) => cb(value));
  }

  function set(key, value) {
    const old = state[key];
    state[key] = value;
    notify(key, { old, value });
  }

  function get(key) {
    return state[key];
  }

  function getAll() {
    return { ...state };
  }

  return { state, set, get, getAll, subscribe };
}

export function createARState() {
  return {
    particles: {
      fire: new ParticleSystem(),
      sparkles: new ParticleSystem(),
      jet: new ParticleSystem(),
    },
    wandTrail: [],
    ghostTrails: [],
    previousHipY: null,
    hue: 0,
  };
}

class ParticleSystem {
  constructor() {
    this.particles = [];
  }

  spawn(x, y, config = {}) {
    this.particles.push({
      x,
      y,
      vx: config.vx ?? (Math.random() - 0.5) * 2,
      vy: config.vy ?? (Math.random() - 0.5) * 2,
      life: config.life ?? 0.8,
      age: 0,
      size: config.size ?? 8,
      color: config.color ?? 'rgba(255,160,60,0.8)',
    });
  }

  update(dt) {
    this.particles = this.particles.filter((p) => {
      p.age += dt;
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.05;
      return p.age < p.life;
    });
  }

  draw(ctx) {
    for (const p of this.particles) {
      const alpha = 1 - p.age / p.life;
      drawGlowCircle(ctx, p.x, p.y, p.size * alpha, withAlpha(p.color, alpha));
    }
  }
}

function withAlpha(color, alpha) {
  if (color.startsWith('rgba(')) {
    const parts = color.slice(5, -1).split(',').map((part) => part.trim());
    return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  if (color.startsWith('rgb(')) {
    const parts = color.slice(4, -1).split(',').map((part) => part.trim());
    return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  return color;
}

function drawGlowCircle(ctx, x, y, r, color) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, color);
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}
