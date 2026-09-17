export class Camera {
  constructor() {
    this.x = 0;
    this.y = 0;
    this.zoom = 1;
    this.targetX = 0;
    this.targetY = 0;
    this.targetZoom = 1;
    this.smoothing = 0.12;
  }

  pan(dx, dy) {
    this.targetX += dx;
    this.targetY += dy;
  }

  setTarget(x, y) {
    this.targetX = x;
    this.targetY = y;
  }

  setZoom(z) {
    this.targetZoom = Math.max(0.5, Math.min(3, z));
  }

  zoomAt(delta, centerX, centerY) {
    const oldZoom = this.zoom;
    this.setZoom(this.zoom + delta);
    const ratio = this.zoom / oldZoom;
    this.targetX = centerX - (centerX - this.targetX) * ratio;
    this.targetY = centerY - (centerY - this.targetY) * ratio;
  }

  update() {
    this.x = lerp(this.x, this.targetX, this.smoothing);
    this.y = lerp(this.y, this.targetY, this.smoothing);
    this.zoom = lerp(this.zoom, this.targetZoom, this.smoothing);
  }

  worldToScreen(wx, wy) {
    return {
      x: (wx - this.x) * this.zoom + canvas.width / 2,
      y: (wy - this.y) * this.zoom + canvas.height / 2,
    };
  }

  screenToWorld(sx, sy) {
    return {
      x: (sx - canvas.width / 2) / this.zoom + this.x,
      y: (sy - canvas.height / 2) / this.zoom + this.y,
    };
  }
}

let canvas = null;
export function setCameraCanvas(c) { canvas = c; }
