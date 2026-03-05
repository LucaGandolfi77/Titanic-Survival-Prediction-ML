export default class ParticleSystem {
  constructor(){
    this.pool = [];
    this._max = 800;
  }

  emit(type, x, y, opts = {}){
    if (!type) return;
    const now = performance.now();
    const push = (p)=>{ if (this.pool.length < this._max) this.pool.push(p); };
    const palette = (window.PALETTE && window.PALETTE.accent) || ['#ffd890','#d45f35','#e87048'];

    if (type === 'harvest'){
      const colors = palette;
      for (let i=0;i<12;i++){
        const ang = (Math.PI*2) * Math.random(); const sp = 40 + Math.random()*80;
        push({ x, y, vx: Math.cos(ang)*sp, vy: -Math.abs(Math.sin(ang))*sp - 20, age:0, ttl:0.8 + Math.random()*0.4, size:2+Math.random()*3, color: colors[i%colors.length], created:now });
      }
    } else if (type === 'leafFall'){
      const colors = ['#d45f35','#e87048','#d4a35a'];
      for (let i=0;i<8;i++){ push({ x: x + Math.random()*40 - 20, y: y + Math.random()*8, vx: 10 + Math.random()*20, vy: 20 + Math.random()*30, age:0, ttl:2 + Math.random()*2, size:2+Math.random()*3, color: colors[i%colors.length], created:now }); }
    } else if (type === 'ripple'){
      push({ x, y, r:4, vr: 60, age:0, ttl:0.8, size:0, color: (opts.color||'#5a9ac0'), ring:true, created:now });
    } else if (type === 'firefly'){
      for (let i=0;i<6;i++) push({ x: x + Math.random()*40-20, y: y + Math.random()*40-20, vx: (Math.random()*2-1)*10, vy:(Math.random()*2-1)*10, age:0, ttl:4+Math.random()*6, size:2.5+Math.random()*1.5, color:'#d4e870', blink:true, created:now });
    } else if (type === 'smoke'){
      for (let i=0;i<6;i++) push({ x: x + Math.random()*8-4, y: y + Math.random()*6, vx: (Math.random()*2-1)*6, vy: -10 - Math.random()*20, age:0, ttl:1.2 + Math.random()*0.8, size:6+Math.random()*6, color:'#7a7878', created:now });
    } else if (type === 'sparkle'){
      for (let i=0;i<8;i++) push({ x: x + Math.random()*10-5, y: y + Math.random()*10-5, vx: (Math.random()*2-1)*80, vy: (Math.random()*2-1)*80, age:0, ttl:0.6 + Math.random()*0.4, size:1+Math.random()*2, color: (Math.random()>0.6? '#ffffff' : '#ffd890'), created:now });
    } else if (type === 'chopLeaf'){
      const colors = ['#62975a','#4e8045'];
      for (let i=0;i<6;i++){ const ang = Math.PI*1.2 + (i/5-0.5); push({ x: x + Math.random()*6-3, y: y, vx: Math.cos(ang)*(40+Math.random()*40), vy: -Math.abs(Math.sin(ang))*(40+Math.random()*40), age:0, ttl:0.8 + Math.random()*0.6, size:2+Math.random()*3, color: colors[i%colors.length], created:now }); }
    }
  }

  update(dt){
    if (!dt) return;
    const g = 80; // gravity
    for (let i = this.pool.length-1; i>=0; i--){ const p = this.pool[i]; p.age += dt; if (p.ttl && p.age >= p.ttl){ this.pool.splice(i,1); continue; }
      if (p.r !== undefined){ p.r += (p.vr||40)*dt; }
      if (p.vx !== undefined) p.x += p.vx * dt; if (p.vy !== undefined) p.y += p.vy * dt;
      if (!p.ring && !p.blink){ p.vy += g * dt; }
      // fade alpha via age/ttl
      if (p.ttl) p.alpha = Math.max(0, 1 - (p.age / p.ttl));
    }
  }

  draw(ctx, camera){
    if (!ctx) return;
    ctx.save();
    for (let i=0;i<this.pool.length;i++){
      const p = this.pool[i]; const alpha = (p.alpha === undefined) ? 1 : p.alpha; ctx.globalAlpha = alpha;
      if (p.ring){ ctx.strokeStyle = p.color || '#72b8d4'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, p.r || 0, 0, Math.PI*2); ctx.stroke(); }
      else if (p.blink){ ctx.fillStyle = p.color || '#d4e870'; const s = p.size || 3; ctx.beginPath(); ctx.arc(p.x, p.y, s,0,Math.PI*2); ctx.fill(); }
      else { ctx.fillStyle = p.color || '#ffd890'; const s = p.size || 3; ctx.fillRect(Math.round(p.x - s/2), Math.round(p.y - s/2), s, s); }
    }
    ctx.restore();
  }
}

