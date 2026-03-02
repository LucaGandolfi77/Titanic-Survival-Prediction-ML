/* ===== Canvas 2D rink renderer ===== */
import { lerp } from './utils.js';

export class RinkRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = canvas.width;
    this.height = canvas.height;
    this.teamColor = '#7dd3fc';
    this.sparkle = false;
    this.sparkleParticles = [];
  }

  setTeamColor(color) {
    this.teamColor = color;
  }

  clear() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    // Ice gradient
    const grad = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w * 0.6);
    grad.addColorStop(0, '#e8f4fd');
    grad.addColorStop(1, '#c8dff0');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Rink outline (rounded rectangle)
    const pad = 12;
    const rad = 40;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(pad + rad, pad);
    ctx.lineTo(w - pad - rad, pad);
    ctx.arcTo(w - pad, pad, w - pad, pad + rad, rad);
    ctx.lineTo(w - pad, h - pad - rad);
    ctx.arcTo(w - pad, h - pad, w - pad - rad, h - pad, rad);
    ctx.lineTo(pad + rad, h - pad);
    ctx.arcTo(pad, h - pad, pad, h - pad - rad, rad);
    ctx.lineTo(pad, pad + rad);
    ctx.arcTo(pad, pad, pad + rad, pad, rad);
    ctx.closePath();
    ctx.stroke();

    // Center circle
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 50, 0, Math.PI * 2);
    ctx.stroke();

    // Center line
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, pad);
    ctx.lineTo(w / 2, h - pad);
    ctx.stroke();

    // Blue lines
    ctx.strokeStyle = 'rgba(100,150,220,0.25)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w * 0.33, pad);
    ctx.lineTo(w * 0.33, h - pad);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(w * 0.67, pad);
    ctx.lineTo(w * 0.67, h - pad);
    ctx.stroke();
  }

  drawSkater(skater, index) {
    const ctx = this.ctx;
    const x = skater.renderX;
    const y = skater.renderY;
    const radius = 15;

    ctx.save();

    if (skater.state === 'wobbling') {
      // Pulsing red ring
      const pulse = 1 + 0.15 * Math.sin(Date.now() * 0.01);
      const scaledR = radius * pulse;

      // Red glow
      ctx.shadowColor = '#f87171';
      ctx.shadowBlur = 15;

      ctx.fillStyle = 'rgba(248,113,113,0.3)';
      ctx.beginPath();
      ctx.arc(x, y, scaledR + 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.fillStyle = this.teamColor;
      ctx.beginPath();
      ctx.arc(x, y, scaledR, 0, Math.PI * 2);
      ctx.fill();

      // ⚠ icon
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px Inter';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('⚠', x, y - scaledR - 8);
    } else if (skater.state === 'fallen') {
      // X mark, gray
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = '#64748b';
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#f87171';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x - 7, y - 7);
      ctx.lineTo(x + 7, y + 7);
      ctx.moveTo(x + 7, y - 7);
      ctx.lineTo(x - 7, y + 7);
      ctx.stroke();
    } else {
      // Normal or formation
      const inFormation = skater.state === 'formation';

      if (inFormation) {
        // Gold ring outline + glow
        ctx.shadowColor = 'rgba(245,158,11,0.5)';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Formation target dot (semi-transparent)
      if (inFormation && skater.targetX !== undefined) {
        ctx.fillStyle = 'rgba(245,158,11,0.2)';
        ctx.beginPath();
        ctx.arc(skater.targetX, skater.targetY, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = this.teamColor;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Border
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Number (always)
    if (skater.state !== 'fallen') {
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px "Share Tech Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(index + 1, x, y + 1);
    }

    ctx.restore();
  }

  drawFormationTarget(x, y) {
    const ctx = this.ctx;
    ctx.fillStyle = 'rgba(245,158,11,0.15)';
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(245,158,11,0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  drawSparkles() {
    if (!this.sparkle) return;
    const ctx = this.ctx;
    const now = Date.now();

    // Add new sparkles
    if (Math.random() < 0.3) {
      this.sparkleParticles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        size: 1 + Math.random() * 3,
        life: 1,
        speed: 0.01 + Math.random() * 0.02
      });
    }

    // Update & draw
    for (let i = this.sparkleParticles.length - 1; i >= 0; i--) {
      const p = this.sparkleParticles[i];
      p.life -= p.speed;
      if (p.life <= 0) {
        this.sparkleParticles.splice(i, 1);
        continue;
      }
      ctx.save();
      ctx.globalAlpha = p.life * 0.8;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
      ctx.fill();

      // Star shape
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 0.5;
      const s = p.size * 3 * p.life;
      ctx.beginPath();
      ctx.moveTo(p.x - s, p.y);
      ctx.lineTo(p.x + s, p.y);
      ctx.moveTo(p.x, p.y - s);
      ctx.lineTo(p.x, p.y + s);
      ctx.stroke();
      ctx.restore();
    }
  }

  drawScorePopup(x, y, text, alpha) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = '#f59e0b';
    ctx.font = 'bold 14px "Orbitron", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  render(skaters, scorePopups) {
    this.clear();

    // Draw formation target dots
    for (const sk of skaters) {
      if (sk.state === 'formation' && sk.targetX !== undefined) {
        this.drawFormationTarget(sk.targetX, sk.targetY);
      }
    }

    // Draw sparkles (finale)
    this.drawSparkles();

    // Draw skaters
    skaters.forEach((sk, i) => this.drawSkater(sk, i));

    // Draw score popups
    if (scorePopups) {
      for (const pop of scorePopups) {
        this.drawScorePopup(pop.x, pop.y, pop.text, pop.alpha);
      }
    }
  }
}
