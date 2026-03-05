class Renderer {
  constructor(ctx, tileMap, camera){ this.ctx = ctx; this.tileMap = tileMap; this.camera = camera; }

  // state is the Game instance
  render(state){
    const ctx = this.ctx; ctx.clearRect(0,0, CANVAS_W, CANVAS_H);
    const time = (state.time.hour || 12) + ((state.time.minute||0)/60);
    this.drawSky(time);
    // apply camera transform
    this.camera.apply(ctx);
    // draw tiles (ground + objects)
    if (this.tileMap && this.tileMap.draw) this.tileMap.draw(ctx, this.camera, time);
    // ambient occlusion / ground shadows around heavy objects (buildings, trees, rocks, mines)
    if (this.tileMap){
      const s = TILE_SIZE || 16;
      const camX = this.camera.x||0, camY = this.camera.y||0;
      const viewW = this.camera.viewW||CANVAS_W, viewH = this.camera.viewH||CANVAS_H;
      const startTx = Math.max(0, Math.floor(camX / s) - 1);
      const endTx = Math.min(this.tileMap.width-1, Math.ceil((camX + viewW) / s) + 1);
      const startTy = Math.max(0, Math.floor(camY / s) - 1);
      const endTy = Math.min(this.tileMap.height-1, Math.ceil((camY + viewH) / s) + 1);
      const aoIds = new Set([6,7,9,10]);
      ctx.save();
      for (let ty = startTy; ty <= endTy; ty++){
        for (let tx = startTx; tx <= endTx; tx++){
          const id = this.tileMap.getTile(tx,ty);
          if (aoIds.has(id)){
            const cx = tx * s + s/2, cy = ty * s + s*0.7;
            const rg = ctx.createRadialGradient(cx, cy, 2, cx, cy, s*0.9);
            rg.addColorStop(0, 'rgba(0,0,0,0.18)'); rg.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(cx, cy, s*0.9, 0, Math.PI*2); ctx.fill();
          }
        }
      }
      ctx.restore();
    }
    // draw entities sorted by world Y
    const entities = [];
    if (state.npcs) entities.push(...state.npcs);
    if (state.player) entities.push(state.player);
    entities.sort((a,b)=> (a.y || 0) - (b.y || 0));
    for (const e of entities){
      // draw shadow under entity if it supports shadow drawing
      if (e.drawShadow){ e.drawShadow(ctx, this.camera); }
      if (e.draw) e.draw(ctx, this.camera, state.time.day);
    }
    // particles
    if (state.particles && state.particles.draw) state.particles.draw(ctx, this.camera);
    // reset camera transform
    this.camera.reset(ctx);
    // UI layer (HUD)
    if (state.ui && state.ui.drawHUD) state.ui.drawHUD();
  }

  drawSky(time){
    const ctx = this.ctx; const h = CANVAS_H; const w = CANVAS_W;
    // choose colors by time of day
    let top = '#7ab8d4', bot = '#c5eaf8';
    if (time < 8){ // pre-dawn
      top = '#d4805a'; bot = '#e8a87c';
    } else if (time < 10){ top = '#e8a87c'; bot = '#a8daf0'; }
    else if (time >= 18 && time < 20){ top = '#e8a87c'; bot = '#d4805a'; }
    else if (time >= 20){ top = '#1a1428'; bot = '#0e0c18'; }
    const g = ctx.createLinearGradient(0,0,0,h); g.addColorStop(0, top); g.addColorStop(1, bot);
    ctx.fillStyle = g; ctx.fillRect(0,0,w,h);
    // simple stars at night
    if (time >= 20 || time < 6){
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      for (let i=0;i<30;i++){ const x = (i*53) % w + (Math.sin(i)*5); const y = (i*37) % (h/2) + 8; ctx.fillRect(x, y, 1,1); }
    }
  }
}

export default Renderer;
