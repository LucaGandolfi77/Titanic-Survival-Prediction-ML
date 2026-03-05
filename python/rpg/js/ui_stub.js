export default class UIManager {
  constructor(ctx, canvas, player, time){ this.ctx = ctx; this.canvas = canvas; this.player = player; this.time = time; }
  drawHUD(){ /* stubbed HUD drawing handled by Renderer/UI in future */ }
  showToast(text){ const d = document.createElement('div'); d.className='toast'; d.textContent = text; document.body.appendChild(d); setTimeout(()=>d.remove(), 2500); }
}
