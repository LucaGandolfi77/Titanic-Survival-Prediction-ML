// Global ShopSystem used by game.js (non-module script style)
class ShopSystem {
  constructor(domElement, player){
    this.dom = domElement || null; this.player = player || null;
    this.opened = false; this.onClose = null;
    this.buyList = [
      { id:'turnip_seed', name:'Turnip Seeds', cost:5, icon:'#7da668' },
      { id:'carrot_seed', name:'Carrot Seeds', cost:8, icon:'#e87048' },
      { id:'pumpkin_seed', name:'Pumpkin Seeds', cost:12, icon:'#d45f35' },
      { id:'tomato_seed', name:'Tomato Seeds', cost:8, icon:'#b84c2a' },
      { id:'energy_potion', name:'Energy Potion', cost:25, icon:'#ffd890' },
      { id:'herb_bundle', name:'Herb Bundle', cost:15, icon:'#72b8d4' },
    ];
    this.sellPrices = { turnip:8, carrot:14, pumpkin:22, tomato:12, Bass:12, Trout:15, Catfish:30, 'Golden Carp':200, Herb:6, Mushroom:8, Wood:3, Stone:2 };
    this._hoveredIndex = -1;
    this._domOverlay = null;
  }

  open(onClose=null){ this.opened = true; this.onClose = onClose; }
  open(onClose=null){ this.opened = true; this.onClose = onClose; if (typeof document !== 'undefined'){ this._createDOMOverlay(); } }
  close(){ this.opened = false; if (this.onClose) this.onClose(); this._removeDOMOverlay(); }

  // create a DOM overlay with buttons if a container isn't supplied
  _createDOMOverlay(){
    if (this._domOverlay) return this._domOverlay;
    const wrap = document.getElementById('game-wrapper') || document.body;
    const overlay = document.createElement('div'); overlay.id = 'shop-overlay';
    overlay.style.position = 'absolute'; overlay.style.left = '50%'; overlay.style.top = '50%'; overlay.style.transform = 'translate(-50%, -50%)';
    overlay.style.width = '520px'; overlay.style.height = '400px'; overlay.style.background = 'rgba(20,12,8,0.95)'; overlay.style.border = '2px solid #a06b3e'; overlay.style.padding = '10px'; overlay.style.zIndex = 50;
    overlay.style.color = '#ffd890'; overlay.style.fontFamily = 'monospace';
    const header = document.createElement('div'); header.style.display='flex'; header.style.justifyContent='space-between'; header.style.alignItems='center'; header.style.marginBottom='8px';
    const title = document.createElement('div'); title.textContent = "🌿 LILY'S EMPORIUM"; title.style.fontSize = '18px'; header.appendChild(title);
    const leaves = document.createElement('div'); leaves.style.fontSize='14px'; leaves.style.color='#ffd890'; leaves.textContent = `Leaves: ${this.player ? this.player.leaves : 0} ♣`;
    header.appendChild(leaves);
    overlay.appendChild(header);
    this._leavesElem = leaves;
    const grid = document.createElement('div'); grid.style.display = 'grid'; grid.style.gridTemplateColumns = '1fr 1fr'; grid.style.gap = '8px';
    for (const it of this.buyList){
      const card = document.createElement('div'); card.style.background = '#2d1b0e'; card.style.padding = '6px'; card.style.display = 'flex'; card.style.alignItems = 'center';
      const icon = document.createElement('div'); icon.style.width='36px'; icon.style.height='36px'; icon.style.background = it.icon || '#555'; icon.style.marginRight='8px'; card.appendChild(icon);
      const info = document.createElement('div'); info.style.flex='1'; info.innerHTML = `<div style="font-size:13px">${it.name}</div><div style="font-size:12px;color:#ffd890">${it.cost} ♣</div>`;
      card.appendChild(info);
      const btn = document.createElement('button'); btn.textContent = 'Buy'; btn.style.marginLeft='8px'; btn.style.background='#5c3d1e'; btn.style.color='#ffd890'; btn.style.border='none'; btn.style.padding='6px 8px'; btn.style.cursor='pointer';
      btn.addEventListener('mouseover', ()=>{ btn.style.background='#735034'; }); btn.addEventListener('mouseout', ()=>{ btn.style.background='#5c3d1e'; });
      btn.addEventListener('click', ()=>{ const ok = this.buy(it.id, this.player); if (ok){ if (window.UIManagerInstance && window.UIManagerInstance.showToast) window.UIManagerInstance.showToast(`Bought ${it.name}`); this._updateLeavesDisplay(); } else if (window.UIManagerInstance) window.UIManagerInstance.showToast('Not enough Leaves'); });
      card.appendChild(btn);
      grid.appendChild(card);
    }
    overlay.appendChild(grid);
    const closeBtn = document.createElement('button'); closeBtn.textContent='Close'; closeBtn.style.position='absolute'; closeBtn.style.right='8px'; closeBtn.style.bottom='8px'; closeBtn.style.background='#333'; closeBtn.style.color='#ffd890'; closeBtn.addEventListener('click', ()=>{ this.close(); this._removeDOMOverlay(); });
    overlay.appendChild(closeBtn);
    wrap.appendChild(overlay); this._domOverlay = overlay; this._updateLeavesDisplay(); return overlay;
  }

  _removeDOMOverlay(){ if (this._domOverlay && this._domOverlay.parentNode){ this._domOverlay.parentNode.removeChild(this._domOverlay); this._domOverlay = null; } }

  buy(itemId){
    const it = this.buyList.find(i=>i.id===itemId); if (!it || !this.player) return false;
    if (this.player.leaves >= it.cost){ this.player.leaves -= it.cost; this.player._addToInventory(itemId, 1); return true; }
    return false;
  }

  sell(itemName, qty=1){
    if (!this.player) return false; const price = this.sellPrices[itemName]; if (!price) return false;
    let remaining = qty;
    for (const slot of this.player.inventory){ if (slot.item === itemName){ const take = Math.min(slot.qty, remaining); slot.qty -= take; remaining -= take; if (slot.qty <= 0) slot.item = null; if (remaining<=0) break; } }
    if (remaining>0) return false;
    this.player.leaves += price * qty; return true;
  }

  _updateLeavesDisplay(){ if (this._leavesElem && this.player){ this._leavesElem.textContent = `Leaves: ${this.player.leaves} ♣`; } }

  render(ctx){ if (!this.opened) return;
    const w = 500, h = 380; const x = (CANVAS_W - w)/2, y = (CANVAS_H - h)/2;
    ctx.fillStyle = PALETTE.ui.bg; ctx.fillRect(x,y,w,h);
    ctx.strokeStyle = PALETTE.ui.border; ctx.lineWidth = 2; ctx.strokeRect(x+1,y+1,w-2,h-2);
    ctx.fillStyle = '#ffd890'; ctx.font='20px monospace'; ctx.fillText("🌿 LILY'S EMPORIUM", x+20, y+34);
    // buy grid
    const bx = x+20, by = y+60; const cols = 2, cellW = 220, cellH = 64;
    for (let i=0;i<this.buyList.length;i++){
      const col = i%cols, row=Math.floor(i/cols);
      const ix = bx + col*(cellW+10), iy = by + row*(cellH+10);
      const it = this.buyList[i]; ctx.fillStyle='#2d1b0e'; ctx.fillRect(ix,iy,cellW,cellH); ctx.strokeStyle='#5c3d1e'; ctx.strokeRect(ix+1,iy+1,cellW-2,cellH-2);
      // hover highlight
      if (this._hoveredIndex === i){ ctx.fillStyle = 'rgba(255,216,144,0.08)'; ctx.fillRect(ix+2, iy+2, cellW-4, cellH-4); ctx.strokeStyle='#ffd890'; ctx.lineWidth=1; ctx.strokeRect(ix+2, iy+2, cellW-4, cellH-4); }
      ctx.fillStyle = it.icon || '#666'; ctx.fillRect(ix+8, iy+8, 48, 48);
      ctx.fillStyle = '#ffd890'; ctx.font='12px monospace'; ctx.fillText(it.name, ix+64, iy+28); ctx.fillText(it.cost + ' ♣', ix+64, iy+46);
    }
    // sell prices
    ctx.fillStyle = '#ffd890'; ctx.font='12px monospace'; ctx.fillText('Sell Prices', x + w - 140, y+60);
    let sy = y+80; for (const [k,v] of Object.entries(this.sellPrices)){ ctx.fillStyle='#ffd890'; ctx.fillText(`${k}: ${v} ♣`, x + w - 140, sy); sy += 18; if (sy > y+h-30) break; }
    ctx.fillStyle = '#999'; ctx.font='12px monospace'; ctx.fillText('[ESC] Close', x + w - 110, y + h - 24);
  }

  // Handle mouse click in canvas coordinates (sx, sy)
  // returns true if shop consumed the click
  handleClick(sx, sy, ui=null){
    if (!this.opened) return false;
    const w = 500, h = 380; const x = (CANVAS_W - w)/2, y = (CANVAS_H - h)/2;
    const bx = x+20, by = y+60; const cols = 2, cellW = 220, cellH = 64;
    for (let i=0;i<this.buyList.length;i++){
      const col = i%cols, row = Math.floor(i/cols);
      const ix = bx + col*(cellW+10), iy = by + row*(cellH+10);
      if (sx >= ix && sx <= ix+cellW && sy >= iy && sy <= iy+cellH){
        const it = this.buyList[i]; const ok = this.buy(it.id, this.player);
        if (ui && ui.showToast) ui.showToast(ok ? `Bought ${it.name}` : `Not enough Leaves`);
        return true;
      }
    }
    // click outside items but inside panel -> ignore
    if (sx >= x && sx <= x+w && sy >= y && sy <= y+h) return true;
    return false;
  }

  // mouse move to update hovered item for visual highlight
  handleMouseMove(sx, sy){
    if (!this.opened) { this._hoveredIndex = -1; return; }
    const w = 500, h = 380; const x = (CANVAS_W - w)/2, y = (CANVAS_H - h)/2;
    const bx = x+20, by = y+60; const cols = 2, cellW = 220, cellH = 64;
    let found = -1;
    for (let i=0;i<this.buyList.length;i++){
      const col = i%cols, row = Math.floor(i/cols);
      const ix = bx + col*(cellW+10), iy = by + row*(cellH+10);
      if (sx >= ix && sx <= ix+cellW && sy >= iy && sy <= iy+cellH){ found = i; break; }
    }
    this._hoveredIndex = found;
    // change cursor
    try{ const canvas = document.querySelector('#gameCanvas'); if (canvas) canvas.style.cursor = (found>=0 ? 'pointer' : 'default'); }catch(e){}
  }
}

export default ShopSystem;
