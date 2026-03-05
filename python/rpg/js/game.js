import TileMap from './tilemap.js';
import Camera from './camera.js';
import Renderer from './renderer.js';
import ParticleSystem from './particles.js';
import AudioManager from './audio.js';
import Player from './player.js';
import { NPC, NPC_CONFIGS } from './npc.js';
import FarmSystem from './farming.js';
import DialogueSystem from './dialogue.js';
import ShopSystem from './shop.js';
import QuestSystem from './quest.js';
import Journal from './journal.js';
import UIManager from './ui.js';

// Provide sane global defaults (legacy code expects these globals).
if (typeof window !== 'undefined'){
  window.MAP_W = window.MAP_W || 40;
  window.MAP_H = window.MAP_H || 30;
  window.TILE_SIZE = window.TILE_SIZE || 16;
  window.CANVAS_W = window.CANVAS_W || 640;
  window.CANVAS_H = window.CANVAS_H || 480;
  window.PALETTE = window.PALETTE || { ui:{ bg:'#2d1b0e', border:'#a06b3e', text:'#ffd890' }, ground:'#5a7a50', green:'#6b8f5e', accent:['#ffd890','#d45f35','#e87048'] };
  window.GAME_STATES = window.GAME_STATES || { PLAYING:0, PAUSED:1 };
}

class Game {
  constructor(canvas) {
    // Instantiate all systems, pass dependencies
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.state    = GAME_STATES.PLAYING;
    this.time     = { hour:8, minute:0, day:1 };
    this.tileMap  = new TileMap();
    this.camera   = new Camera();
    this.renderer = new Renderer(this.ctx, this.tileMap, this.camera);
    this.particles= new ParticleSystem();
    this.audio    = new AudioManager();
    this.player   = new Player(this.particles);
    this.npcs     = NPC_CONFIGS.map(c => new NPC(c));
    this.farmSys  = new FarmSystem(this.particles);
    this.dlgSys   = new DialogueSystem();
    this.shopSys  = new ShopSystem(null, this.player);
    this.quests   = new QuestSystem();
    this.journal  = new Journal();
    this.ui       = new UIManager(this.ctx, canvas, this.player, this.time);
    // expose UI instance for DOM overlays
    try{ window.UIManagerInstance = this.ui; }catch(e){}
    this.keys     = {};
    this._lastTS  = 0;
    this._bindInput();
    // mouse -> forward to shop when open
    canvas.addEventListener('mousedown', (e)=>{
      const rect = canvas.getBoundingClientRect(); const sx = e.clientX - rect.left; const sy = e.clientY - rect.top;
      if (this.shopSys && this.shopSys.opened){ const handled = this.shopSys.handleClick(sx, sy, this.ui); if (handled) e.preventDefault(); }
    });
    canvas.addEventListener('mousemove', (e)=>{
      const rect = canvas.getBoundingClientRect(); const sx = e.clientX - rect.left; const sy = e.clientY - rect.top;
      if (this.shopSys){ this.shopSys.handleMouseMove(sx, sy); }
    });
  }

  loop(ts){
    if (!this._lastTS) this._lastTS = ts;
    let dt = Math.min(0.1, (ts - this._lastTS)/1000);
    this._lastTS = ts;
    if (this.state !== GAME_STATES.PAUSED){ this.update(dt); }
    // render frame
    this.renderer.render(this);
    // overlays
    if (this.dlgSys && this.dlgSys.active) this.dlgSys.draw && this.dlgSys.draw(this.ctx);
    if (this.shopSys && this.shopSys.opened) this.shopSys.render(this.ctx);
    // request next frame
    requestAnimationFrame(t => this.loop(t));
  }

  update(dt){
    // Advance in-game time: 1 real second = 1 game minute
    this.time.minute += Math.floor(dt * 1);
    if (this.time.minute >= 60){ this.time.hour += Math.floor(this.time.minute/60); this.time.minute = this.time.minute % 60; }
    // simple day rollover
    if (this.time.hour >= 24){ this.time.hour = this.time.hour % 24; this.advanceDay(); }

    // build input object
    const input = {
      left: this.keys['ArrowLeft'] || this.keys['a'],
      right: this.keys['ArrowRight'] || this.keys['d'],
      up: this.keys['ArrowUp'] || this.keys['w'],
      down: this.keys['ArrowDown'] || this.keys['s'],
      toolNext: this.keys['e'] || this.keys['E'],
      toolPrev: this.keys['q'] || this.keys['Q']
    };

    const bounds = [0,0, this.tileMap.width * TILE_SIZE, this.tileMap.height * TILE_SIZE];
    this.player.update(dt, input, bounds, this.audio);
    for (const n of this.npcs) n.update(dt, this.time.day);
    this.particles && this.particles.update && this.particles.update(dt);
    this.farmSys && this.farmSys.update && this.farmSys.update(dt);
    this.camera.follow(this.player.x, this.player.y, dt);
  }
  advanceDay() {
    this.time.day++;
    this.time.hour = 8; this.time.minute = 0;
    this.player.energy = 100;
    this.farmSys.advanceDay();
    this.journal.addLog(`Day ${this.time.day} begins.`);
    this.audio.play('dayJingle');
    this.save();
  }
  save() {
    try{
      const farmState = this.farmSys && (this.farmSys.state || null);
      localStorage.setItem('cozyfarm_save', JSON.stringify({
        day: this.time.day,
        leaves: this.player.leaves,
        energy: this.player.energy,
        inventory: this.player.inventory,
        farmState: farmState,
        questProgress: this.quests && this.quests.progress,
        log: this.journal && this.journal.log && this.journal.log.slice(0,50),
        bestiary: this.journal && this.journal.discovered ? [...this.journal.discovered] : [],
      }));
    }catch(e){ console.warn('Save failed', e); }
  }
  load() { /* reverse of save, called on init if save exists */ }
  _bindInput() {
    // map keys
    window.addEventListener('keydown', (e) => {
      const k = e.key; this.keys[k] = true;
      // Open/close shop with B when near Lily
      if (k === 'b' || k === 'B'){
        if (this.shopSys && this.shopSys.opened){ this.shopSys.close(); e.preventDefault(); return; }
        // find Lily or any NPC with shopAccess
        const shopNpc = this.npcs.find(n => (n.name === 'Lily' || n.shopAccess));
        if (shopNpc){ const dx = shopNpc.x - this.player.x, dy = shopNpc.y - this.player.y; if (Math.hypot(dx,dy) <= TILE_SIZE * 2){ this.shopSys && this.shopSys.open(); } else { this.ui && this.ui.showToast && this.ui.showToast("Lily isn't nearby."); } }
        else { this.ui && this.ui.showToast && this.ui.showToast("No shop found."); }
        e.preventDefault(); return;
      }

      // Close overlays / pause
      if (k === 'Escape'){
        if (this.shopSys && this.shopSys.opened){ this.shopSys.close(); e.preventDefault(); return; }
        if (this.dlgSys && this.dlgSys.active){
          if (typeof this.dlgSys.handleEsc === 'function') { this.dlgSys.handleEsc(); }
          else if (typeof this.dlgSys.close === 'function') { this.dlgSys.close(); }
          else { this.dlgSys.active = false; }
          e.preventDefault(); return;
        }
        this.state = (this.state === GAME_STATES.PAUSED) ? GAME_STATES.PLAYING : GAME_STATES.PAUSED;
        e.preventDefault(); return;
      }

      // Dialogue advance or player action / talk
      if (k === ' '){
        if (this.dlgSys && this.dlgSys.active){
          if (typeof this.dlgSys.handleSpace === 'function') { this.dlgSys.handleSpace(); }
          else if (typeof this.dlgSys.skipOrAdvance === 'function') { this.dlgSys.skipOrAdvance(); }
          else if (typeof this.dlgSys.handleKey === 'function') { this.dlgSys.handleKey(' '); }
          e.preventDefault(); return;
        }
        // try talk to nearby NPC first (within ~1.5 tiles)
        if (this.state === GAME_STATES.PLAYING){
          const interactRange = (window.TILE_SIZE || 16) * 1.5;
          const nearby = this.npcs.find(n => Math.hypot(n.x - this.player.x, n.y - this.player.y) <= interactRange);
          if (nearby && this.dlgSys){ this.dlgSys.open(nearby, this.time.day, { player: this.player, ui: this.ui }); e.preventDefault(); return; }
          // otherwise use current tool on player's facing tile
          const tx = Math.floor((this.player.x + 8)/TILE_SIZE), ty = Math.floor((this.player.y + 8)/TILE_SIZE);
          this.player.useTool(tx, ty, this.farmSys, this.tileMap, this.audio);
          e.preventDefault(); return;
        }
      }

      if (k === 'q' || k === 'Q'){ this.player.cycleTool(-1); e.preventDefault(); return; }
      if (k === 'e' || k === 'E'){ this.player.cycleTool(1); e.preventDefault(); return; }
      if (k === 'j' || k === 'J'){ this.journal && this.journal.toggle && this.journal.toggle(); e.preventDefault(); return; }
      if (k === 'Tab'){ this.player.selectedSlot = (this.player.selectedSlot + 1) % this.player.inventory.length; e.preventDefault(); return; }
    });

    window.addEventListener('keyup', (e) => { this.keys[e.key] = false; });
  }
}

window.addEventListener('load', () => {
  const canvas = document.getElementById('gameCanvas');
  const game = new Game(canvas);
  game.load();
  requestAnimationFrame(ts => game.loop(ts));
});
