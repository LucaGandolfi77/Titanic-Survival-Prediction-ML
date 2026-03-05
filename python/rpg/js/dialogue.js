export default class DialogueSystem {
  constructor(domElement = null){
    this.domRoot = domElement || document.body;
    this.active = false;
    this.npc = null;
    this.day = 1;
    this.fullText = '';
    this.charIndex = 0;
    this.speed = 40; // ms per char
    this._raf = null;
    this._lastTick = 0;
    this._onComplete = null;
    this.player = null;
    this.ui = null;
    this._createDOM();
  }

  _createDOM(){
    if (this._container) return;
    const container = document.createElement('div'); container.id = 'dialogue-box';
    Object.assign(container.style, { position:'fixed', left:'50%', bottom:'8%', transform:'translateX(-50%)', width:'640px', maxWidth:'90%', background:'rgba(20,12,8,0.95)', border:'2px solid #a06b3e', color:'#ffd890', fontFamily:'monospace', padding:'10px', zIndex:9999, display:'none' });
    // layout: portrait + content
    const row = document.createElement('div'); row.style.display='flex'; row.style.gap='12px';
    const portraitWrap = document.createElement('div'); portraitWrap.style.width='64px'; portraitWrap.style.flex='0 0 64px';
    const portrait = document.createElement('canvas'); portrait.width = 64; portrait.height = 64; portrait.style.background='transparent'; portraitWrap.appendChild(portrait);
    this._portrait = portrait;
    const content = document.createElement('div'); content.style.flex='1 1 auto';
    const nameEl = document.createElement('div'); nameEl.style.fontSize='16px'; nameEl.style.fontWeight='bold'; nameEl.style.marginBottom='6px';
    this._nameEl = nameEl;
    const textEl = document.createElement('div'); textEl.style.minHeight='48px'; textEl.style.whiteSpace='pre-wrap'; textEl.style.fontSize='14px';
    this._textEl = textEl;
    const hint = document.createElement('div'); hint.style.fontSize='12px'; hint.style.opacity='0.8'; hint.style.marginTop='6px'; hint.textContent = '[SPACE] Continue • [ESC] Close';
    content.appendChild(nameEl); content.appendChild(textEl); content.appendChild(hint);
    row.appendChild(portraitWrap); row.appendChild(content);
    container.appendChild(row);
    this._container = container; this._textEl = textEl; this.domRoot.appendChild(container);

    // key handlers
    this._onKey = (e) => { if (!this.active) return; if (e.key === ' ') { e.preventDefault(); this.handleSpace(); } else if (e.key === 'Escape'){ e.preventDefault(); this.handleEsc(); } };
    window.addEventListener('keydown', this._onKey);
  }

  open(npc, day = 1, { player = null, ui = null, onComplete = null } = {}){
    this.npc = npc; this.day = day; this.player = player; this.ui = ui; this._onComplete = onComplete;
    const dlg = (npc && (typeof npc.getDialogueForDay === 'function' ? npc.getDialogueForDay(day) : (npc.dialogueForDay ? npc.dialogueForDay(day) : null))) || (npc && npc.dialogues && npc.dialogues[day % npc.dialogues.length]) || { text: (npc && npc.name) ? `Hello, I'm ${npc.name}.` : 'Hello.' };
    this.fullText = typeof dlg === 'string' ? dlg : (dlg.text || String(dlg));
    this._pendingGift = dlg && dlg.gift ? dlg.gift : null;
    this.charIndex = 0; this.active = true; this._container.style.display = 'block'; this._nameEl.textContent = npc?.name ? `${npc.name} • ${npc.role || ''}` : '...';
    this._renderPortrait(); this._updateText(); this._lastTick = performance.now(); this._startLoop();
  }

  _renderPortrait(){
    try{
      const c = this._portrait; const ctx = c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
      // simple circle head + body using npc colors
      const body = this.npc?.body || '#d4a35a'; const accent = this.npc?.accent || '#ffd890';
      ctx.fillStyle = body; ctx.beginPath(); ctx.arc(32,26,14,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = accent; ctx.fillRect(18,40,28,14);
      ctx.fillStyle = '#2d1b0e'; ctx.fillRect(26,22,4,4); ctx.fillRect(34,22,4,4);
    }catch(e){}
  }

  _updateText(){ this._textEl.textContent = this.fullText.slice(0, this.charIndex); }

  _startLoop(){ if (this._raf) return; const step = (t)=>{ if (!this.active){ this._raf = null; return; } const elapsed = t - this._lastTick; if (elapsed >= this.speed){ const advance = Math.floor(elapsed / this.speed); this.charIndex = Math.min(this.fullText.length, this.charIndex + advance); this._lastTick = t; this._updateText(); if (this.charIndex >= this.fullText.length){ /* finished typing */ } } this._raf = requestAnimationFrame(step); }; this._raf = requestAnimationFrame(step); }

  handleSpace(){
    if (!this.active) return;
    if (this.charIndex < this.fullText.length){ this.charIndex = this.fullText.length; this._updateText(); return; }
    // fully shown -> close and give gift if any
    if (this._pendingGift && this.player){ if (this._pendingGift.item){ this.player._addToInventory(this._pendingGift.item, this._pendingGift.qty || 1); } if (this._pendingGift.leaves){ this.player.leaves = (this.player.leaves || 0) + this._pendingGift.leaves; } if (this.ui && this.ui.showToast) this.ui.showToast('Received a gift!'); }
    this.close(); if (this._onComplete) try{ this._onComplete(); }catch(e){}
  }

  handleEsc(){ this.close(); }

  close(){ this.active = false; if (this._container) this._container.style.display = 'none'; if (this._raf){ cancelAnimationFrame(this._raf); this._raf = null; } }

  // optional canvas fallback draw (called by game loop)
  draw(ctx){ if (!this.active) return; try{
    const W = ctx.canvas.width, H = ctx.canvas.height; const w = Math.min(640, W-40), h = 120; const x = (W - w)/2, y = H - h - 20;
    ctx.save(); ctx.fillStyle = 'rgba(20,12,8,0.95)'; ctx.fillRect(x,y,w,h); ctx.strokeStyle='#a06b3e'; ctx.lineWidth=2; ctx.strokeRect(x+1,y+1,w-2,h-2);
    ctx.fillStyle='#ffd890'; ctx.font='14px monospace'; ctx.fillText(this.npc?.name || '...', x+80, y+24);
    ctx.fillStyle='#ffd890'; ctx.font='12px monospace'; ctx.fillText(this.fullText.slice(0,this.charIndex), x+80, y+52);
    // portrait
    ctx.fillStyle = this.npc?.body || '#d4a35a'; ctx.beginPath(); ctx.arc(x+40, y+40, 20,0,Math.PI*2); ctx.fill(); ctx.restore();
  }catch(e){}}
}
