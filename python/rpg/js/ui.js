export default class UIManager {
  constructor(ctx, canvas, player, time){
    this.ctx = ctx; this.canvas = canvas; this.player = player; this.time = time;
    this._toastQueue = [];
    this._lastToastId = 0;
  }

  drawHUD(){
    if (!this.ctx) return;
    this.drawTopLeft();
    this.drawInventoryBar();
  }

  drawTopLeft(){
    const ctx = this.ctx; const CANVAS_W = (window.CANVAS_W || (this.canvas && this.canvas.width) || 640);
    const CANVAS_H = (window.CANVAS_H || (this.canvas && this.canvas.height) || 480);
    const x = 8, y = 8, w = 140, h = 52;
    ctx.save();
    ctx.fillStyle = (window.PALETTE && window.PALETTE.ui && window.PALETTE.ui.bg) || '#2d1b0e';
    ctx.fillRect(x,y,w,h);
    ctx.strokeStyle = (window.PALETTE && window.PALETTE.ui && window.PALETTE.ui.border) || '#a06b3e';
    ctx.lineWidth = 2; ctx.strokeRect(x+1,y+1,w-2,h-2);
    // Day / Time
    ctx.fillStyle = (window.PALETTE && window.PALETTE.ui && window.PALETTE.ui.text) || '#ffd890';
    ctx.font = '13px monospace';
    const day = this.time?.day || 1; const hour = this.time?.hour || 8; const minute = this.time?.minute || 0;
    const ampm = hour >= 12 ? 'PM' : 'AM'; const hh = ((hour+11)%12)+1; const mm = String(minute).padStart(2,'0');
    ctx.fillText(`Day ${day}  |  ${hh}:${mm} ${ampm}`, x+8, y+20);
    // Energy bar
    const energy = this.player?.energy ?? 100;
    ctx.fillStyle = '#3b2a18'; ctx.fillRect(x+8, y+28, 120, 10);
    const fillW = Math.max(0, Math.min(120, Math.round((energy/this.player?.maxEnergy || 1) * 120)));
    ctx.fillStyle = energy < 25 ? '#b84c2a' : '#d4a35a'; ctx.fillRect(x+8, y+28, fillW, 10);
    ctx.restore();
  }

  drawMinimap(tileMap){
    if (!this.ctx || !tileMap) return;
    const ctx = this.ctx; const sizeW = 96, sizeH = 72; const pad = 8;
    const CANVAS_W = (window.CANVAS_W || (this.canvas && this.canvas.width) || 640);
    const x = CANVAS_W - sizeW - pad, y = pad;
    ctx.save(); ctx.fillStyle = 'rgba(0,0,0,0.2)'; ctx.fillRect(x,y,sizeW,sizeH);
    const tw = tileMap.width, th = tileMap.height; const pxW = Math.max(1, Math.floor(sizeW / tw)); const pxH = Math.max(1, Math.floor(sizeH / th));
    for (let ty=0; ty<th; ty++){
      for (let tx=0; tx<tw; tx++){
        const id = tileMap.getTile(tx,ty);
        let c = '#5a7a50';
        if (id === 3) c = '#62a0c8'; else if (id === 7) c = '#4e8045'; else if (id === 6) c = '#7a5a3a'; else if (id === 9) c = '#6a6168';
        ctx.fillStyle = c; ctx.fillRect(x + tx*pxW, y + ty*pxH, pxW, pxH);
      }
    }
    // player dot
    if (this.player){ const ptx = Math.floor(this.player.x / (tileMap.tileSize || 16)); const pty = Math.floor(this.player.y / (tileMap.tileSize || 16)); ctx.fillStyle = '#ffd890'; ctx.fillRect(x + ptx*pxW -1, y + pty*pxH -1, 3,3); }
    ctx.restore();
  }

  drawInventoryBar(){
    if (!this.ctx || !this.player) return;
    const ctx = this.ctx; const CANVAS_W = (window.CANVAS_W || (this.canvas && this.canvas.width) || 640);
    const slots = this.player.inventory || [];
    const slotW = 36, slotH = 36, gap = 8; const totalW = slots.length * slotW + (slots.length-1)*gap;
    const x = Math.round((CANVAS_W - totalW)/2), y = (window.CANVAS_H || (this.canvas && this.canvas.height) || 480) - slotH - 12;
    ctx.save();
    for (let i=0;i<slots.length;i++){
      const sx = x + i*(slotW+gap); ctx.fillStyle = '#2d1b0e'; ctx.fillRect(sx, y, slotW, slotH);
      if (i === this.player.selectedSlot) { ctx.strokeStyle = '#ffd890'; ctx.lineWidth = 2; ctx.strokeRect(sx+1,y+1,slotW-2,slotH-2); }
      const it = slots[i]; if (it && it.item){ ctx.fillStyle = '#ffd890'; ctx.font='11px monospace'; ctx.fillText(it.item, sx+6, y+20); ctx.fillText(String(it.qty), sx+6, y+32); }
    }
    // Leaves count at bottom-right
    ctx.fillStyle = '#ffd890'; ctx.font='14px monospace'; ctx.fillText(`♣ ${this.player.leaves || 0}`, CANVAS_W - 80, y + slotH/2 + 6);
    ctx.restore();
  }

  showToast(text){
    const id = ++this._lastToastId; const d = document.createElement('div'); d.className = 'toast'; d.textContent = text;
    Object.assign(d.style, { position:'fixed', right:'16px', bottom: `${16 + (this._toastQueue.length*44)}px`, background:'#2d1b0e', color:'#ffd890', padding:'8px 12px', border:'1px solid #a06b3e', fontFamily:'monospace', zIndex:9999, borderRadius:'6px' });
    document.body.appendChild(d); this._toastQueue.push(d);
    setTimeout(()=>{ d.style.transition='opacity 300ms'; d.style.opacity='0'; setTimeout(()=>{ d.remove(); this._toastQueue.shift(); }, 350); }, 2500);
  }

  drawPause(){
    if (!this.ctx) return; const ctx = this.ctx; const W = (window.CANVAS_W || (this.canvas && this.canvas.width) || 640); const H = (window.CANVAS_H || (this.canvas && this.canvas.height) || 480);
    ctx.save(); ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(0,0,W,H); ctx.fillStyle = '#ffd890'; ctx.font='28px monospace'; ctx.fillText('PAUSED', W/2 - 60, H/2);
    ctx.restore();
  }

  setupMobileControls(){
    if (typeof navigator !== 'undefined' && ('maxTouchPoints' in navigator ? navigator.maxTouchPoints > 0 : false)){
      const container = document.getElementById('mobile-controls'); if (!container) return;
      container.innerHTML = '';
      const btn = document.createElement('button'); btn.textContent='A'; btn.style.padding='12px'; btn.style.fontSize='18px'; btn.style.borderRadius='50%'; btn.style.background='#5c3d1e'; btn.style.color='#ffd890';
      btn.addEventListener('pointerdown', ()=>{ window.dispatchEvent(new KeyboardEvent('keydown',{key:' '})); }); container.appendChild(btn);
    }
  }
}
