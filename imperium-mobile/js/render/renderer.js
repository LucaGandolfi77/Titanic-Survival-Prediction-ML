import { hexToPixelCenter } from '../utils/hex-math.js';
import { selectedCitizen as selCitizen } from '../citizen.js';

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.resize();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = window.innerWidth * dpr;
    this.canvas.height = window.innerHeight * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = window.innerWidth;
    this.height = window.innerHeight;
  }

  render(camera, citizensList, unitsList, buildingsList, particleList, weatherState) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0d0a04';
    ctx.fillRect(0, 0, w, h);

    const map = mapState.tiles;
    if (!map) return;

    const camLeft = camera.x - w / camera.zoom / 2 - 40;
    const camRight = camera.x + w / camera.zoom / 2 + 40;
    const camTop = camera.y - h / camera.zoom / 2 - 40;
    const camBottom = camera.y + h / camera.zoom / 2 + 40;

    for (let r = 0; r < Math.min(30, map.length); r++) {
      if (!map[r]) continue;
      for (let q = 0; q < Math.min(40, map[r].length); q++) {
        const tile = map[r][q];
        const { x: cx, y: cy } = hexToPixelCenter(q, r);
        if (cx < camLeft || cx > camRight || cy < camTop || cy > camBottom) continue;
        this.drawTile(ctx, tile, cx, cy, camera.zoom);
      }
    }

    for (const b of buildingsList) {
      const { x: cx, y: cy } = hexToPixelCenter(b.q, b.r);
      this.drawBuilding(ctx, b, cx, cy, camera.zoom);
    }

    for (const u of unitsList) {
      if (!u.alive) continue;
      this.drawUnit(ctx, u, camera.zoom);
    }

    for (const c of citizensList) {
      this.drawCitizen(ctx, c, camera.zoom);
    }

    if (particleList && particleList.length > 0) {
      for (const p of particleList) {
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

    if (weatherState && weatherState.getOverlayColor && weatherState.getOverlayColor() !== 'rgba(200,200,200,0)') {
      ctx.fillStyle = weatherState.getOverlayColor();
      ctx.fillRect(0, 0, w, h);
    }
  }

  drawTile(ctx, tile, cx, cy, zoom) {
    const size = 22 * zoom;
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = Math.PI / 180 * (60 * i - 30);
      const x = cx + size * Math.cos(angle);
      const y = cy + size * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    if (!tile.explored) {
      ctx.fillStyle = '#0d0a04';
      ctx.fill();
      ctx.restore();
      return;
    }
    ctx.fillStyle = tile.color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.12)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
    if (tile.terrain !== 'water' && tile.terrain !== 'sand') {
      ctx.font = `${Math.max(8, Math.round(10 * zoom))}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      const emoji = tile.terrain === 'forest' ? '🌲' : tile.terrain === 'hill' ? '⛰️' : tile.terrain === 'plain' ? '🌿' : tile.terrain === 'swamp' ? '💧' : '🟩';
      ctx.fillText(emoji, cx, cy);
    }
    if (tile.building && tile.explored) {
      ctx.font = `${Math.round(16 * zoom)}px serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(tile.building.emoji || '🏗️', cx, cy - size * 0.6);
    }
    if (tile.resourceAmount > 0 && tile.explored) {
      ctx.font = `${Math.max(6, Math.round(7 * zoom))}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.fillText('+' + tile.resourceAmount, cx, cy + size * 0.5);
    }
    ctx.restore();
  }

  drawBuilding(ctx, b, cx, cy, zoom) {
    ctx.save();
    ctx.font = `${Math.round(16 * zoom)}px serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(b.emoji, cx, cy);
    if (!b.constructed && b.progress < 1) {
      const barW = 20 * zoom;
      const barH = 3 * zoom;
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(cx - barW / 2, cy + 12 * zoom, barW, barH);
      ctx.fillStyle = '#d4a017';
      ctx.fillRect(cx - barW / 2, cy + 12 * zoom, barW * (b.progress || 0), barH);
    }
    ctx.restore();
  }

  drawUnit(ctx, u, zoom) {
    ctx.save();
    const radius = 8 * zoom;
    ctx.beginPath();
    ctx.arc(u.x, u.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = u.color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.font = `${Math.max(7, Math.round(9 * zoom))}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff';
    ctx.fillText(u.name.charAt(0), u.x, u.y);
    const hpBarW = 14 * zoom;
    const hpBarH = 2 * zoom;
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(u.x - hpBarW / 2, u.y + radius + 3 * zoom, hpBarW, hpBarH);
    ctx.fillStyle = u.hp / u.maxHp > 0.5 ? '#357a38' : '#9b2226';
    ctx.fillRect(u.x - hpBarW / 2, u.y + radius + 3 * zoom, hpBarW * Math.max(0, u.hp / u.maxHp), hpBarH);
    ctx.restore();
  }

  drawCitizen(ctx, c, zoom) {
    ctx.save();
    const radius = 7 * zoom;
    ctx.beginPath();
    ctx.arc(c.x, c.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = c.color || '#8B6914';
    ctx.fill();
    ctx.strokeStyle = c === selCitizen ? '#faf7f0' : 'rgba(0,0,0,0.6)';
    ctx.lineWidth = c === selCitizen ? 2.5 : 1.5;
    ctx.stroke();
    ctx.font = `${Math.max(7, Math.round(9 * zoom))}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'middle';
    ctx.fillText(c.name.charAt(0), c.x, c.y);
    if (c.task === 'gather') {
      ctx.strokeStyle = 'rgba(53, 122, 56, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(c.x, c.y, radius + 3 * zoom, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }
}
