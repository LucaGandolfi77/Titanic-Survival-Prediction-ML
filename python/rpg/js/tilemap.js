class TileMap {
  constructor(width = MAP_W, height = MAP_H){
    this.width = width; this.height = height;
    // tile id map [row][col]
    this.map = new Array(this.height);
    for (let y=0;y<this.height;y++){
      this.map[y] = new Array(this.width);
      for (let x=0;x<this.width;x++) this.map[y][x] = this._initialTile(x,y);
    }
  }

  _initialTile(tx, ty){
    // tile ids (match player expectations: TREE=7, WATER=3, ROCK=9)
    const GRASS = 0, SOIL = 1, TILLED = 2, WATER = 3, PATH = 4, FLOWER = 5,
          BUILDING = 6, TREE = 7, MOUNTAIN = 8, ROCK = 9, MINE = 10, SAND = 11, BRIDGE = 12;
    // Forest
    if (ty <= 7){
      // denser forest near top
      const r = Math.random(); if (r < 0.18) return TREE; if (r < 0.28) return FLOWER; return GRASS;
    }
    // Farm
    if (ty <= 17){
      // place a building at (18,11) if within bounds
        if (tx === 16 && ty === 12 && tx < this.width && ty < this.height) return BUILDING;
      // farm soil fields in a band
      if (ty >= 10 && ty <= 15){ if (Math.random() < 0.76) return SOIL; }
      // paths
      if (ty === 9 || ty === 16) return PATH;
      return GRASS;
    }
    // Town
    if (ty <= 21){
      if (Math.random() < 0.18) return BUILDING;
      if (Math.random() < 0.14) return FLOWER;
      if (Math.random() < 0.10) return TREE;
      return PATH;
    }
    // River
    if (ty <= 25){
      // bridge at center
      const bridgeX = Math.floor(this.width/2);
      // make the bridge slightly wider across 23
        if (tx >= bridgeX-1 && tx <= bridgeX+1 && (ty === 23 || ty === 24)) return BRIDGE;
      return (Math.random() < 0.82) ? WATER : SAND;
    }
    // Mine area
    if (ty <= 29){
        const mineEntranceX = Math.min(20, this.width-1);
        if (tx === 22 && ty === 27) return MINE;
          return (Math.random() < 0.78) ? ROCK : GRASS;
    }
    // default
    return GRASS;
  }

  getTile(tx, ty){ if (tx < 0 || ty < 0 || tx >= this.width || ty >= this.height) return -1; return this.map[ty][tx]; }
  setTile(tx, ty, id){ if (tx < 0 || ty < 0 || tx >= this.width || ty >= this.height) return; this.map[ty][tx] = id; }

  isWalkable(tx, ty){
    const id = this.getTile(tx,ty); if (id === -1) return false;
    // non-walkable types: WATER(3), TREE(7), BUILDING(6), MINE(10), ROCK(9)
    const nonWalk = new Set([3,7,6,10,9]); return !nonWalk.has(id);
  }

  // draws a single tile at pixel position (px,py). 'time' is seconds float for animations
  drawTile(ctx, id, px, py, time=0, tx=0, ty=0){
    const s = TILE_SIZE || 16;
    // helper lerp
    const lerp = (a,b,t)=> a + (b-a)*t;
    switch(id){
      case 0: // GRASS
        ctx.fillStyle = (PALETTE && PALETTE.ground) ? PALETTE.ground : '#5a7a50'; ctx.fillRect(px,py,s,s);
        ctx.fillStyle = (PALETTE && PALETTE.green) ? PALETTE.green : '#6b8f5e';
        const sway = Math.sin(time*1.2 + (tx||0)*0.4 + (ty||0)*0.25) * 1.8;
        ctx.fillRect(Math.round(px + s/2 + sway) - 1, py + s - 4, 2, 4);
        break;
      case 1: // SOIL
        ctx.fillStyle = '#5c3d1e'; ctx.fillRect(px,py,s,s);
        ctx.fillStyle = '#7a4e2d'; for (let i=0;i<6;i++){ const rx = px + (i*7 % s); const ry = py + ((i*13)%s); ctx.fillRect(rx+1, ry+1, 1,1); }
        break;
      case 2: // TILLED
        ctx.fillStyle = '#3d2510'; ctx.fillRect(px,py,s,s);
        ctx.fillStyle = '#5c3d1e'; for (let y=py+2;y<py+s;y+=4) ctx.fillRect(px+2,y,s-4,1);
        break;
      case 3: // WATER
        // shimmer between two blues
        const t = (Math.sin(time*1.8 + (tx||0)*0.35) + 1)/2;
        const c1 = {r:0x48, g:0x80, b:0xa6}; const c2 = {r:0x62, g:0xa0, b:0xc8};
        const cr = Math.round(lerp(c1.r,c2.r,t)), cg = Math.round(lerp(c1.g,c2.g,t)), cb = Math.round(lerp(c1.b,c2.b,t));
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(px,py,s,s);
        break;
      case 4: // PATH
        ctx.fillStyle = '#7a7060'; ctx.fillRect(px,py,s,s);
        ctx.fillStyle = '#8a8070'; ctx.fillRect(px+2,py+2,s-4,s-4);
        ctx.strokeStyle = '#606050'; ctx.lineWidth = 1; ctx.strokeRect(px+0.5,py+0.5,s-1,s-1);
        break;
      case 5: // FLOWER
        ctx.fillStyle = '#5a7a50'; ctx.fillRect(px,py,s,s);
        const color = PALETTE && PALETTE.accent ? PALETTE.accent[(tx+ty) % PALETTE.accent.length] : '#ffd890';
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(px + s/2, py + s/2, 2.5,0,Math.PI*2); ctx.fill();
        break;
      case 6: // BUILDING
        ctx.fillStyle = '#7a5a3a'; ctx.fillRect(px+2,py+6,s-4,s-6);
        ctx.fillStyle = '#5c3d1e'; ctx.fillRect(px+2,py+2,s-4,4);
        ctx.fillStyle = '#ffd890'; ctx.fillRect(px + s/2 - 4, py + s/2 - 2, 6,6);
        break;
      case 7: // TREE
        ctx.fillStyle = '#5c3d1e'; ctx.fillRect(px + s/2 - 4, py + s - 8, 8, 8);
        ctx.fillStyle = '#4e8045'; ctx.beginPath(); ctx.arc(px + s/2, py + s/2 - 2, 10,0,Math.PI*2); ctx.fill();
        break;
      case 8: // MOUNTAIN
        ctx.fillStyle = '#5a5060'; ctx.fillRect(px,py,s,s);
        ctx.fillStyle = '#6a6070'; ctx.beginPath(); ctx.moveTo(px,py+s); ctx.lineTo(px + s/2, py+4); ctx.lineTo(px+s,py+s); ctx.fill();
        break;
      case 9: // ROCK (pickaxe target)
        ctx.fillStyle = '#6a6168'; ctx.fillRect(px,py,s,s);
        ctx.fillStyle = '#4f4850'; ctx.beginPath(); ctx.moveTo(px+2,py+s-2); ctx.lineTo(px + s/2, py+6); ctx.lineTo(px+s-2,py+s-2); ctx.fill();
        break;
      case 12: // BRIDGE
        ctx.fillStyle = '#8b5e3c'; ctx.fillRect(px,py+6,s,10);
        ctx.fillStyle = '#5c3d1e'; for (let i=0;i<4;i++) ctx.fillRect(px + 4 + i*10, py+6, 2,10);
        break;
      case 10: // MINE
        ctx.fillStyle = '#5a5060'; ctx.fillRect(px,py,s,s);
        ctx.fillStyle = '#2d1b0e'; ctx.beginPath(); ctx.arc(px + s/2, py + s/2 + 4, s/3, Math.PI, 0); ctx.fill();
        break;
      case 11: // SAND
        ctx.fillStyle = '#c9b58e'; ctx.fillRect(px,py,s,s); break;
      default:
        ctx.fillStyle = '#5a7a50'; ctx.fillRect(px,py,s,s); break;
    }
  }
}

export default TileMap;

// Draw full map helper: renders visible tiles using camera to cull
TileMap.prototype.draw = function(ctx, camera, time=0){
  const s = TILE_SIZE || 16;
  const camX = camera.x || 0, camY = camera.y || 0;
  const viewW = camera.viewW || CANVAS_W, viewH = camera.viewH || CANVAS_H;
  const startTx = Math.max(0, Math.floor(camX / s) - 1);
  const endTx = Math.min(this.width-1, Math.ceil((camX + viewW) / s) + 1);
  const startTy = Math.max(0, Math.floor(camY / s) - 1);
  const endTy = Math.min(this.height-1, Math.ceil((camY + viewH) / s) + 1);
  for (let ty = startTy; ty <= endTy; ty++){
    for (let tx = startTx; tx <= endTx; tx++){
      const id = this.getTile(tx, ty);
      const px = tx * s; const py = ty * s;
      this.drawTile(ctx, id, px, py, time, tx, ty);
    }
  }
};
