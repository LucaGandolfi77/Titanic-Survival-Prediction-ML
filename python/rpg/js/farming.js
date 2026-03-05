// Farming system with soil grid and a fishing minigame
export default class FarmSystem {
  constructor(particles = null){
    this.particles = particles;
    this.w = window.MAP_W || 40; this.h = window.MAP_H || 30; this.tileSize = window.TILE_SIZE || 16;
    // soilState[y][x]
    this.state = Array.from({length:this.h}, ()=>Array.from({length:this.w}, ()=>({ tilled:false, watered:false, crop:null })));
    this.groundItems = [];
    this._cropDefs = window.CROPS || { turnip:{days:3}, carrot:{days:4}, pumpkin:{days:5}, tomato:{days:4} };

    // Fishing state
    this.fishState = 'IDLE';
    this._fish = {
      timer:0, wait:0, activeTime:0, catchY:0, fishY:0, speed:1.0, result:null
    };
  }

  inBounds(tx,ty){ return tx >= 0 && ty >= 0 && tx < this.w && ty < this.h; }

  till(tx,ty){ if (!this.inBounds(tx,ty)) return false; const s = this.state[ty][tx]; s.tilled = true; s.watered = false; return true; }
  water(tx,ty){ if (!this.inBounds(tx,ty)) return false; const s = this.state[ty][tx]; if (!s.tilled) return false; s.watered = true; return true; }
  plant(tx,ty,type){ if (!this.inBounds(tx,ty)) return false; const s = this.state[ty][tx]; if (!s.tilled || s.crop) return false; s.crop = { type, day_planted:0, days_watered:0, stage:0 }; try{ if (this.particles){ const px = tx * this.tileSize + this.tileSize/2; const py = ty * this.tileSize + this.tileSize/2; this.particles.emit('sparkle', px, py); } }catch(e){} return true; }

  harvest(tx,ty){ if (!this.inBounds(tx,ty)) return null; const s = this.state[ty][tx]; if (!s.crop) return null; if ((s.crop.stage || 0) < 3) return null; const item = s.crop.type; s.crop = null; s.tilled = false; s.watered = false; // emit harvest particles
    try{ if (this.particles){ const px = tx * this.tileSize + this.tileSize/2; const py = ty * this.tileSize + this.tileSize/2; this.particles.emit('harvest', px, py); } }catch(e){}
    return { id: item, qty: 1 };
  }

  advanceDay(){
    for (let y=0;y<this.h;y++){
      for (let x=0;x<this.w;x++){
        const s = this.state[y][x];
        if (s.crop){
          if (s.watered){ s.crop.days_watered = (s.crop.days_watered||0) + 1; const def = this._cropDefs[s.crop.type] || { days:4 }; const prog = Math.min(def.days, s.crop.days_watered); s.crop.stage = Math.floor((prog / def.days) * 3); }
          else { s.crop.unwatered = (s.crop.unwatered||0) + 1; }
        }
        s.watered = false;
      }
    }
  }

  drawCrops(ctx, camera, time){
    const s = this.tileSize;
    for (let y=0;y<this.h;y++){
      for (let x=0;x<this.w;x++){
        const cell = this.state[y][x]; if (!cell.crop) continue;
        const wx = x * s, wy = y * s; const pos = camera.worldToScreen(wx + s/2, wy + s/2);
        const cx = pos.x, cy = pos.y; const st = cell.crop.stage || 0;
        ctx.save();
        if (st === 0){ ctx.fillStyle = '#6b8f5e'; ctx.fillRect(cx-1, cy-6, 2, 4); }
        else if (st === 1){ ctx.fillStyle = '#6b8f5e'; ctx.fillRect(cx-2, cy-8, 4, 6); }
        else if (st === 2){ ctx.fillStyle = '#6b8f5e'; ctx.fillRect(cx-3, cy-10, 6, 8); }
        else { // full
          const color = (window.CROPS && window.CROPS[cell.crop.type] && window.CROPS[cell.crop.type].colors) ? window.CROPS[cell.crop.type].colors[2] : '#d45f35';
          const bob = Math.sin(time + x*0.3) * 2;
          ctx.fillStyle = color; ctx.beginPath(); ctx.ellipse(cx, cy + bob - 4, 6, 6, 0, 0, Math.PI*2); ctx.fill();
        }
        ctx.restore();
      }
    }
  }

  /* Fishing minigame API */
  startFishing(wx, wy){
    if (this.fishState !== 'IDLE') return false;
    this.fishState = 'CASTING'; this._fish.timer = 0; this._fish.wait = 0.5; this._fish.activeTime = 0; this._fish.result = null;
    return true;
  }

  updateFishing(dt, spacePressed){
    const f = this._fish;
    if (this.fishState === 'IDLE') return null;
    if (this.fishState === 'CASTING'){
      f.timer += dt; if (f.timer >= f.wait){ this.fishState = 'WAITING'; f.timer = 0; f.wait = 1 + Math.random()*2; }
      return null;
    }
    if (this.fishState === 'WAITING'){
      f.timer += dt; if (f.timer >= f.wait){ // enter active
        this.fishState = 'ACTIVE'; f.timer = 0; f.activeTime = 0; f.speed = 0.8; f.catchY = 40 + Math.random()*(200-40); f.fishY = 100; }
      return null;
    }
    if (this.fishState === 'ACTIVE'){
      f.activeTime += dt; // speed ramps
      f.speed = 0.8 + Math.min(1.7, (f.activeTime / 8) * 1.7);
      f.fishY = 100 + Math.sin(f.activeTime * f.speed*2) * 80;
      // check input
      if (spacePressed){ const diff = Math.abs(f.fishY - f.catchY); const success = diff <= 20; this.fishState = 'RESULT'; f.result = success ? this._sampleFish() : { fail:true }; return f.result; }
      if (f.activeTime >= 8){ this.fishState = 'RESULT'; f.result = { fail:true }; return f.result; }
      return null;
    }
    if (this.fishState === 'RESULT'){
      const r = f.result; this.fishState = 'IDLE'; return r;
    }
    return null;
  }

  drawFishing(ctx){
    if (this.fishState === 'IDLE') return;
    const W = ctx.canvas.width, H = ctx.canvas.height; const barH = 200, barW = 28; const x = W - 80, y = (H - barH)/2;
    ctx.save(); ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(x-8,y-8, barW+16, barH+16);
    ctx.fillStyle = '#444'; ctx.fillRect(x,y,barW,barH);
    if (this.fishState === 'ACTIVE' || this.fishState === 'RESULT'){
      // catch zone
      ctx.fillStyle = '#2f7a40'; ctx.fillRect(x, y + this._fish.catchY - 20, barW, 40);
      // fish icon
      const fy = y + this._fish.fishY; ctx.fillStyle = '#ffd890'; ctx.beginPath(); ctx.arc(x + barW/2, fy, 8,0,Math.PI*2); ctx.fill();
    }
    ctx.restore();
  }

  _sampleFish(){
    const r = Math.random()*100; if (r < 80) return { fish:'Common Fish', rarity:'common' }; if (r < 95) return { fish:'Uncommon Fish', rarity:'uncommon' }; return { fish:'Golden Carp', rarity:'legendary' };
  }
}

