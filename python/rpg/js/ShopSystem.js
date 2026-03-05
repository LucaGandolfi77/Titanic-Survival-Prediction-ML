// FILE: ShopSystem.js
class ShopSystem {
  constructor(){
    // Lily's shop inventory for buying
    this.buyList = [
      { id:'turnip_seed', name:'Turnip Seeds', cost:5, iconColor:'#7da668' },
      { id:'carrot_seed', name:'Carrot Seeds', cost:8, iconColor:'#e87048' },
      { id:'pumpkin_seed', name:'Pumpkin Seeds', cost:12, iconColor:'#d45f35' },
      { id:'tomato_seed', name:'Tomato Seeds', cost:8, iconColor:'#b84c2a' },
      { id:'energy_potion', name:'Energy Potion', cost:25, iconColor:'#ffd890' },
      { id:'herb_bundle', name:'Herb Bundle', cost:15, iconColor:'#72b8d4' },
    ];
    // sell prices
    this.sellPrices = {
      turnip:8, carrot:14, pumpkin:22, tomato:12, Bass:12, Trout:15, Catfish:30, 'Golden Carp':200,
      Herb:6, Mushroom:8, Wood:3, Stone:2
    };
    this.opened = false;
    this.onClose = null;
  }

  open(onClose=null){ this.opened = true; this.onClose = onClose; }
  close(){ this.opened = false; if (this.onClose) this.onClose(); }

  buy(itemId, player){
    const it = this.buyList.find(i=>i.id===itemId); if (!it) return false;
    if (player.leaves >= it.cost){ player.leaves -= it.cost; player._addToInventory(itemId.replace('_seed','_seed'), 1); return true; }
    return false;
  }

  sell(itemName, qty, player){
    const price = this.sellPrices[itemName]; if (!price) return false;
    // remove qty from inventory
    let remaining = qty;
    for (const slot of player.inventory){ if (slot.item === itemName){ const take = Math.min(slot.qty, remaining); slot.qty -= take; remaining -= take; if (slot.qty <= 0) slot.item = null; if (remaining<=0) break; } }
    if (remaining>0) return false; // not enough
    player.leaves += price * qty; return true;
  }

  draw(ctx){ if (!this.opened) return;
    const w = 500, h = 380; const x = (CANVAS_W - w)/2, y = (CANVAS_H - h)/2;
    ctx.fillStyle = PALETTE.ui.bg; ctx.fillRect(x,y,w,h);
    ctx.strokeStyle = PALETTE.ui.border; ctx.lineWidth = 2; ctx.strokeRect(x+1,y+1,w-2,h-2);
    // title
    ctx.fillStyle = '#ffd890'; ctx.font='20px monospace'; ctx.fillText("🌿 LILY'S EMPORIUM", x+20, y+34);
    // draw buy items in grid
    const bx = x+20, by = y+60; const cols = 2; const cellW = 220, cellH = 64;
    for (let i=0;i<this.buyList.length;i++){
      const col = i % cols; const row = Math.floor(i/cols);
      const ix = bx + col*(cellW+10); const iy = by + row*(cellH+10);
      const it = this.buyList[i];
      ctx.fillStyle = '#2d1b0e'; ctx.fillRect(ix, iy, cellW, cellH);
      ctx.strokeStyle = '#5c3d1e'; ctx.strokeRect(ix+1, iy+1, cellW-2, cellH-2);
      // icon
      ctx.fillStyle = it.iconColor; ctx.fillRect(ix+8, iy+8, 48, 48);
      ctx.fillStyle = '#ffd890'; ctx.font='12px monospace'; ctx.fillText(it.name, ix+64, iy+28);
      ctx.fillText(it.cost + ' ♣', ix+64, iy+46);
    }
    // sell column (player sells items) -- show prices
    ctx.fillStyle = '#ffd890'; ctx.font='12px monospace'; ctx.fillText('Sell Prices', x + w - 140, y+60);
    let sy = y+80; for (const [k,v] of Object.entries(this.sellPrices)){ ctx.fillStyle='#ffd890'; ctx.fillText(`${k}: ${v} ♣`, x + w - 140, sy); sy += 18; if (sy > y+h-30) break; }
    // close hint
    ctx.fillStyle = '#999'; ctx.font='12px monospace'; ctx.fillText('[ESC] Close', x + w - 110, y + h - 24);
  }
}

export default ShopSystem;
