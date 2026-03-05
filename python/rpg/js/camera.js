class Camera {
  constructor(viewW = CANVAS_W, viewH = CANVAS_H, worldW = MAP_W * TILE_SIZE, worldH = MAP_H * TILE_SIZE){
    this.viewW = viewW; this.viewH = viewH;
    this.worldW = worldW; this.worldH = worldH;
    this.x = 0; this.y = 0; // top-left world pixel
    this.lerp = 0.08;
  }

  // follow world coordinate (center on target) using lerp; dt optional (not required for fixed lerp)
  follow(worldX, worldY, dt = 1){
    const targetX = Math.max(0, Math.min(this.worldW - this.viewW, worldX - this.viewW / 2));
    const targetY = Math.max(0, Math.min(this.worldH - this.viewH, worldY - this.viewH / 2));
    const t = this.lerp * dt;
    this.x += (targetX - this.x) * t;
    this.y += (targetY - this.y) * t;
    this._clamp();
  }

  setInstant(worldX, worldY){
    this.x = Math.max(0, Math.min(this.worldW - this.viewW, worldX - this.viewW / 2));
    this.y = Math.max(0, Math.min(this.worldH - this.viewH, worldY - this.viewH / 2));
  }

  _clamp(){
    this.x = Math.max(0, Math.min(this.x, Math.max(0, this.worldW - this.viewW)));
    this.y = Math.max(0, Math.min(this.y, Math.max(0, this.worldH - this.viewH)));
  }

  worldToScreen(wx, wy){ return { x: Math.round(wx - this.x), y: Math.round(wy - this.y) } }
  screenToWorld(sx, sy){ return { x: sx + this.x, y: sy + this.y } }

  // apply camera transform to ctx (translate world so top-left becomes 0,0)
  apply(ctx){ ctx.save(); ctx.translate(-this.x, -this.y); }
  // restore or reset transform
  reset(ctx){ ctx.restore(); }
}

export default Camera;
