export default class Journal {
  constructor(domElement = null, questSystem = null){
    this.log = []; this.discovered = new Set(); this.questSystem = questSystem;
    this._visible = false; this._container = null; this._mountPoint = domElement || document.body;
    this._createOverlay();
  }

  _createOverlay(){
    if (this._container) return;
    const wrap = document.createElement('div'); wrap.id = 'journal-overlay';
    Object.assign(wrap.style, { position:'fixed', left:'50%', top:'50%', transform:'translate(-50%,-50%)', width:'640px', height:'420px', background:'#2d1b0e', border:'2px solid #a06b3e', color:'#ffd890', fontFamily:'monospace', zIndex:1000, display:'none', padding:'10px', boxSizing:'border-box' });
    // header with tabs
    const header = document.createElement('div'); header.style.display='flex'; header.style.gap='8px'; header.style.marginBottom='8px';
    const tabs = ['FARM LOG','QUESTS','BESTIARY']; this._tabElems = {};
    tabs.forEach(t => { const b = document.createElement('button'); b.textContent = t; b.style.background='transparent'; b.style.color='#ffd890'; b.style.border='1px solid rgba(255,255,255,0.04)'; b.style.padding='6px 8px'; b.addEventListener('click', ()=>{ this._setTab(t); }); header.appendChild(b); this._tabElems[t]=b; });
    wrap.appendChild(header);
    // content area
    const content = document.createElement('div'); content.id='journal-content'; Object.assign(content.style, { flex: '1 1 auto', height:'340px', overflow:'auto', background:'#3d2510', padding:'8px' }); wrap.appendChild(content);
    // close button
    const close = document.createElement('button'); close.textContent='Close'; close.style.position='absolute'; close.style.right='10px'; close.style.bottom='10px'; close.addEventListener('click', ()=>this.toggle()); wrap.appendChild(close);
    this._container = wrap; this._content = content; this._mountPoint.appendChild(wrap);
    this._currentTab = 'FARM LOG'; this._setTab(this._currentTab);
  }

  _setTab(name){ this._currentTab = name; for (const k in this._tabElems) this._tabElems[k].style.opacity = (k===name)?'1':'0.7'; this.render(); }

  toggle(){ this._visible = !this._visible; if (this._container) this._container.style.display = this._visible ? 'block' : 'none'; if (this._visible) this.render(); }

  addLog(text){ const entry = {text, ts: Date.now()}; this.log.unshift(entry); if (this.log.length>200) this.log.length=200; this.render(); }

  discover(itemId){ this.discovered.add(itemId); this.render(); }

  render(){ if (!this._container) return; const content = this._content; content.innerHTML = '';
    if (this._currentTab === 'FARM LOG'){
      const list = document.createElement('div'); this.log.slice(0,50).forEach(e => { const d = document.createElement('div'); d.style.padding='6px'; d.style.borderBottom='1px solid rgba(255,255,255,0.03)'; d.textContent = `${new Date(e.ts).toLocaleTimeString()} — ${e.text}`; list.appendChild(d); }); content.appendChild(list);
    } else if (this._currentTab === 'QUESTS'){
      const qwrap = document.createElement('div'); qwrap.style.display='grid'; qwrap.style.gridTemplateColumns='1fr 1fr'; qwrap.style.gap='8px';
      const sampleQuests = this.questSystem && this.questSystem.quests ? this.questSystem.quests : [ {id:'crops5', desc:'Plant your first 5 crops', reward:{leaves:50}, done:false}, {id:'allNpcs', desc:'Talk to all villagers', reward:{leaves:30}, done:false} ];
      sampleQuests.forEach(q=>{ const card = document.createElement('div'); card.style.padding='8px'; card.style.background='#2d1b0e'; card.style.border='1px solid rgba(255,255,255,0.03)'; const title = document.createElement('div'); title.textContent = q.desc; title.style.fontWeight='bold'; card.appendChild(title); const reward = document.createElement('div'); reward.textContent = `Reward: ${q.reward?.leaves ?? 0} ♣`; reward.style.marginTop='6px'; card.appendChild(reward); qwrap.appendChild(card); }); content.appendChild(qwrap);
    } else if (this._currentTab === 'BESTIARY'){
      const grid = document.createElement('div'); grid.style.display='grid'; grid.style.gridTemplateColumns='repeat(6,1fr)'; grid.style.gap='6px'; const items = ['turnip','carrot','pumpkin','tomato','Bass','Trout','Catfish','Golden Carp'];
      items.forEach(id=>{ const cell = document.createElement('div'); cell.style.height='64px'; cell.style.background='#2d1b0e'; cell.style.display='flex'; cell.style.alignItems='center'; cell.style.justifyContent='center'; if (!this.discovered.has(id)){ cell.style.filter='grayscale(100%)'; cell.textContent='???'; } else { cell.textContent = id; } grid.appendChild(cell); }); content.appendChild(grid);
    }
  }
}

