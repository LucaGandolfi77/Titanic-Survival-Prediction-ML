// FILE: Player.js
class Player {
  constructor(particles=null, x=MAP_W*TILE_SIZE/2, y=MAP_H*TILE_SIZE/2){
    this.particles = particles;
    
    this.x = x; this.y = y; // world coords (px)
    this.speed = 24; // px per second
    this.energy = 100; this.maxEnergy = 100;
    this.leaves = 100;
    this.inventory = Array.from({length:8}, ()=>({item:null, qty:0}));
    this.selectedSlot = 0;
    this.tools = ['hand','hoe','watering','axe','rod','pickaxe'];
    this.toolIndex = 0;
    this.footTimer = 0;
    this.facing = 'down';
  }

  cycleTool(delta){ this.toolIndex = (this.toolIndex + (delta>0?1:-1) + this.tools.length) % this.tools.length; }
  get currentTool(){ return this.tools[this.toolIndex]; }

  // input: {left,right,up,down,toolNext,toolPrev}
  update(dt, input={}, bounds=null, audio=null){
    let dx = 0, dy = 0;
    if (input.left) dx -= 1; if (input.right) dx += 1;
    if (input.up) dy -= 1; if (input.down) dy += 1;
    if (dx !== 0 || dy !== 0){
      const len = Math.hypot(dx,dy) || 1; dx/=len; dy/=len;
      this.x += dx * this.speed * dt; this.y += dy * this.speed * dt;
      this.facing = Math.abs(dx) > Math.abs(dy) ? (dx>0?'right':'left') : (dy>0?'down':'up');
      this.footTimer += dt; if (this.footTimer >= 0.4){ if (audio) audio.play('footstep'); this.footTimer = 0; }
    }
    if (bounds){ this.x = Math.max(bounds[0], Math.min(bounds[2], this.x)); this.y = Math.max(bounds[1], Math.min(bounds[3], this.y)); }
    if (input.toolNext) this.cycleTool(1); if (input.toolPrev) this.cycleTool(-1);
  }

  useTool(tx, ty, farmSys, tileMap, audio=null){
    const tool = this.currentTool;
    const costMap = {hoe:5, watering:3, axe:8, rod:2, pickaxe:6, hand:1};
    const cost = costMap[tool] || 0; if (this.energy < cost){ if (audio) audio.play('error'); return false; }
    const wx = Math.floor(tx), wy = Math.floor(ty);
    if (tool === 'hoe'){ farmSys.till(wx,wy); this.energy -= cost; if (audio) audio.play('pop'); return true; }
    if (tool === 'watering'){ farmSys.water(wx,wy); this.energy -= cost; if (audio) audio.play('splash'); return true; }
    if (tool === 'hand'){ const res = farmSys.harvest(wx,wy); if (res){ this._addToInventory(res.item, res.qty); if (audio) audio.play('harvest'); return true; } if (audio) audio.play('error'); return false; }
    if (tool === 'axe'){ const id = tileMap.map[wy] && tileMap.map[wy][wx]; if (id===7){ tileMap.map[wy][wx]=0; this._addToInventory('wood',3); this.energy-=cost; try{ if (this.particles){ const px = wx * (window.TILE_SIZE||16) + (window.TILE_SIZE||16)/2; const py = wy * (window.TILE_SIZE||16) + (window.TILE_SIZE||16)/2; this.particles.emit('chopLeaf', px, py); } }catch(e){} if (audio) audio.play('pop'); return true; } if (audio) audio.play('error'); return false; }
    if (tool === 'rod'){ const id = tileMap.map[wy] && tileMap.map[wy][wx]; if (id===3){ farmSys.startFishing && farmSys.startFishing(wx,wy); this.energy -= cost; if (audio) audio.play('splash'); return true; } if (audio) audio.play('error'); return false; }
    if (tool === 'pickaxe'){ const id = tileMap.map[wy] && tileMap.map[wy][wx]; if (id===9){ // break rock
      tileMap.map[wy][wx] = 0; this._addToInventory('stone',2); this.energy -= cost;
      try{ if (this.particles){ const px = wx * (window.TILE_SIZE||16) + (window.TILE_SIZE||16)/2; const py = wy * (window.TILE_SIZE||16) + (window.TILE_SIZE||16)/2; this.particles.emit('sparkle', px, py); } }catch(e){}
      if (audio) audio.play('pop'); return true; } if (audio) audio.play('error'); return false; }
    return false;
  }

  _addToInventory(item, qty=1){
    for (const slot of this.inventory){ if (slot.item === item){ slot.qty += qty; return; } }
    for (const slot of this.inventory){ if (slot.item === null){ slot.item = item; slot.qty = qty; return; } }
  }

  draw(ctx, camera){
    const scr = camera.worldToScreen(this.x, this.y);
    const x = scr.x, y = scr.y; const sx = x-8, sy = y-16;
    // shadow
    if (ctx.save){ ctx.save(); ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.beginPath(); ctx.ellipse(x, y+6, 8, 4, 0, 0, Math.PI*2); ctx.fill(); ctx.restore(); }
    ctx.fillStyle='#b84c2a'; ctx.fillRect(sx+5, sy+12, 6, 4); // boots
    ctx.fillStyle='#d4a35a'; ctx.fillRect(sx+3, sy+4, 10, 8); // body
    ctx.fillStyle='#a06b3e'; ctx.fillRect(sx+1, sy+5, 2, 5); ctx.fillRect(sx+13, sy+5, 2, 5); // arms
    ctx.fillStyle='#e8b870'; ctx.fillRect(sx+4, sy-4, 8, 7); // head
    ctx.fillStyle='#5c3d1e'; ctx.fillRect(sx+3, sy-8, 10, 4); ctx.fillStyle='#7a4e2d'; ctx.fillRect(sx+4, sy-13, 8,5); // hat
    ctx.fillStyle='#2d1b0e'; ctx.fillRect(sx+6, sy-2, 2, 2); // eyes
    ctx.fillStyle='#ffd890'; ctx.font='9px monospace'; ctx.fillText(this.currentTool, sx-18, sy-10);
  }
}

export default Player;
