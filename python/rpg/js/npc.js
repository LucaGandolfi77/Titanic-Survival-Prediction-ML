// Clean ES module for NPCs
export class NPC {
  constructor({name, role, x=0, y=0, homeX=null, homeY=null, body='#d4a35a', accent='#ffd890', shopAccess=false, dialogues=null, specialIndex=2}){
    this.name = name; this.role = role; this.x = x; this.y = y;
    this.homeX = (homeX !== null) ? homeX : x; this.homeY = (homeY !== null) ? homeY : y;
    this.body = body; this.accent = accent; this.shopAccess = !!shopAccess;
    this.wanderRadius = 3 * (typeof TILE_SIZE !== 'undefined' ? TILE_SIZE : 16);
    this.wanderTimer = Math.random()*3 + 3;
    this.target = {x:this.x, y:this.y}; this.speed = 0.6 * (typeof TILE_SIZE !== 'undefined' ? TILE_SIZE : 16);
    this.bob = Math.random()*Math.PI*2; this._dialogues = dialogues || ['Hello.']; this._specialIndex = specialIndex;
  }
  update(dt){ this.bob += dt * 0.8; this.wanderTimer -= dt; if (this.wanderTimer <= 0){ this.wanderTimer = 3 + Math.random()*3; const angle = Math.random()*Math.PI*2; const r = Math.random()*this.wanderRadius; this.target.x = this.homeX + Math.cos(angle)*r; this.target.y = this.homeY + Math.sin(angle)*r; } const dx = this.target.x - this.x; const dy = this.target.y - this.y; const dist = Math.hypot(dx,dy); if (dist > 1){ const nx = dx/dist, ny = dy/dist; const step = Math.min(this.speed * dt, dist); this.x += nx*step; this.y += ny*step; } }
  isSpecialDay(day){ return (day % 4) === this._specialIndex; }
  getDialogueForDay(day){ const idx = day % (this._dialogues.length||1); return this._dialogues[idx] || { text: 'Hello' }; }
  draw(ctx, camera, day=1){ const pos = camera.worldToScreen(this.x, this.y); const cx = pos.x, cy = pos.y; const sx = cx - 8, sy = cy - 16; ctx.save(); ctx.fillStyle = 'rgba(0,0,0,0.18)'; ctx.beginPath(); ctx.ellipse(cx, cy+6, 7, 3.5, 0,0,Math.PI*2); ctx.fill(); ctx.restore(); ctx.fillStyle = this.body; ctx.fillRect(sx+3, sy+4, 10, 8); ctx.fillStyle = '#e8b870'; ctx.fillRect(sx+4, sy-4, 8, 7); ctx.fillStyle = '#2d1b0e'; ctx.fillRect(sx+6, sy-2, 2,2); ctx.fillStyle = this.accent; ctx.fillRect(sx+4, sy+2, 8,3); ctx.fillStyle = '#ffd890'; ctx.font='10px monospace'; ctx.fillText(this.name, cx - ctx.measureText(this.name).width/2, sy+26); if (this.isSpecialDay(day)){ ctx.fillStyle='#ffd890'; ctx.font='14px monospace'; ctx.fillText('!', cx-3, sy-18 + Math.sin(this.bob)*2); } }
}

export const NPC_CONFIGS = [
  { name:'Mira', role:'Baker', x:22*16, y:20*16, homeX:22*16, homeY:20*16, body:'#f2c97a', accent:'#ffd890', dialogues:[ 'Fresh bread today!', 'I saw fireflies near the forest.', 'Here, take these seeds!', 'Have you tried fishing at dawn?' ], specialIndex:2 },
  { name:'Old Tom', role:'Farmer', x:18*16, y:13*16, homeX:18*16, homeY:13*16, body:'#8b5e3c', accent:'#a06b3e', dialogues:[ 'Water your crops every day.', 'My old notes on farming.', 'Turnips grow fastest.', 'Strange lights near the mine.' ], specialIndex:1 },
  { name:'Lily', role:'Herbalist', x:24*16, y:19*16, homeX:24*16, homeY:19*16, body:'#4e8045', accent:'#62975a', dialogues:[ 'Forest mushrooms are potent after rain.', 'An energy potion, on the house!', 'Bring me five herbs and I\'ll brew something.', 'The fairy ring...' ], specialIndex:1, shopAccess:true }
];
