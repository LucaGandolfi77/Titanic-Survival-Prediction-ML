import { mapState } from './map.js';
import { hexToPixelCenter } from '../utils/hex-math.js';
import { rand } from '../utils/helpers.js';
import { haptic } from './ui/notifications.js';

export const particles = [];

const MAX_PARTICLES = 200;

export function spawnParticle(x, y, type) {
  if (particles.length > MAX_PARTICLES) return;
  const count = type === 'gather' ? 3 : type === 'combat' ? 5 : type === 'build' ? 4 : 2;
  for (let i = 0; i < count; i++) {
    particles.push({
      x,
      y,
      vx: (Math.random() - 0.5) * 3,
      vy: (Math.random() - 0.5) * 3 - 1,
      life: 1.0,
      decay: 0.015 + Math.random() * 0.02,
      size: 2 + Math.random() * 3,
      type,
      color: getParticleColor(type),
    });
  }
}

function getParticleColor(type) {
  switch (type) {
    case 'gather': return '#4a8022';
    case 'combat': return '#9b2226';
    case 'build': return '#d4a017';
    case 'ageup': return '#8B6914';
    case 'death': return '#6a6a6a';
    default: return '#faf7f0';
  }
}

export function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.05;
    p.life -= p.decay;
    if (p.life <= 0) {
      particles.splice(i, 1);
    }
  }
}

export function renderParticles(ctx, camera) {
  for (const p of particles) {
    const alpha = Math.max(0, p.life);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size * camera.zoom, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}
