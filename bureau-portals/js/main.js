import * as THREE from 'three';
import { SceneManager } from './scene.js';
import { WorldGenerator } from './world.js';
import { PortalManager, Portal } from './portals.js';
import { Player } from './player.js';
import { NPCManager } from './npcs.js';
import { ItemManager } from './items.js';
import { PuzzleManager } from './puzzle.js';
import { ParticleSystem } from './particles.js';
import { AudioManager } from './audio.js';
import { HUDManager } from './hud.js';
import { UIManager } from './ui.js';
import { Controls } from './controls.js';
import { NonEuclideanEffects } from './noneuclidean.js';
import { MathUtils } from './utils.js';

export class GameController {
  constructor() {
    this.state = 'menu'; // menu, playing, paused, gameover, win
    this.lastFrameTime = Date.now();
    this.deltaTime = 0;
    
    // Initialize core systems
    this.scene = new SceneManager();
    this.world = new WorldGenerator(this.scene.scene);
    this.portals = new PortalManager(this.scene.scene);
    this.particles = new ParticleSystem(this.scene.scene);
    this.audio = new AudioManager();
    this.effects = new NonEuclideanEffects();
    this.puzzle = new PuzzleManager();
    
    // Initialize game entities
    this.player = null;
    this.npcs = null;
    this.items = null;
    this.controls = null;
    this.hud = null;
    this.ui = new UIManager(this);
    
    // Room connectivity mapping (12 rooms, bidirectional portal connections)
    this.roomConnectivity = this.buildRoomConnectivity();
    
    // Portal rendering state
    this.maxPortalDepth = 2;
    this.portalRenderQueue = [];
    
    // Game stats
    this.debugMode = false;
  }

  buildRoomConnectivity() {
    return {
      0: { portals: [
        { dest: 1, type: 'NORMAL', pos: [6, 1.5, 5], rot: 0 },
        { dest: 3, type: 'UPSIDE_DOWN', pos: [-6, 1.5, 5], rot: 0 }
      ]},
      1: { portals: [
        { dest: 0, type: 'NORMAL', pos: [4, 1.5, -6], rot: 180 },
        { dest: 5, type: 'LOOP', pos: [-4, 1.5, 6], rot: 90 }
      ]},
      2: { portals: [
        { dest: 6, type: 'SIDEWAYS_L', pos: [4, 1.5, 4], rot: 0 }
      ]},
      3: { portals: [
        { dest: 0, type: 'UPSIDE_DOWN', pos: [4, 1.5, -8], rot: 0 },
        { dest: 2, type: 'NORMAL', pos: [-2, 1.5, 8], rot: 0 },
        { dest: 4, type: 'SIDEWAYS_R', pos: [2, 1.5, -8], rot: 0 }
      ]},
      4: { portals: [
        { dest: 3, type: 'SIDEWAYS_L', pos: [3, 1.5, -3], rot: 0 },
        { dest: 5, type: 'NORMAL', pos: [-3, 1.5, -3], rot: 0 }
      ]},
      5: { portals: [
        { dest: 1, type: 'LOOP', pos: [4, 1.5, -5], rot: 270 },
        { dest: 4, type: 'NORMAL', pos: [4, 1.5, 5], rot: 0 },
        { dest: 6, type: 'MIRROR', pos: [-4, 1.5, 0], rot: 0 }
      ]},
      6: { portals: [
        { dest: 2, type: 'SIDEWAYS_R', pos: [-4, 1.5, -4], rot: 0 },
        { dest: 5, type: 'MIRROR', pos: [4, 1.5, 0], rot: 180 },
        { dest: 7, type: 'NORMAL', pos: [6, 1.5, 0], rot: 0 }
      ]},
      7: { portals: [
        { dest: 6, type: 'NORMAL', pos: [-3, 1.5, 0], rot: 180 },
        { dest: 8, type: 'SCALED', pos: [3, 1.5, -3], rot: 0 }
      ]},
      8: { portals: [
        { dest: 7, type: 'SCALED', pos: [-5, 1.5, 3], rot: 180 },
        { dest: 9, type: 'NORMAL', pos: [5, 1.5, -3], rot: 0 }
      ]},
      9: { portals: [
        { dest: 8, type: 'NORMAL', pos: [-4, 1.5, 5], rot: 180 },
        { dest: 10, type: 'NORMAL', pos: [4, 1.5, 5], rot: 0 }
      ]},
      10: { portals: [
        { dest: 9, type: 'NORMAL', pos: [0, 0.8, -4], rot: 180 },
        { dest: 11, type: 'TIME_LAG', pos: [0, 0.8, 4], rot: 0 }
      ]},
      11: { portals: [
        { dest: 10, type: 'TIME_LAG', pos: [8, 2, -6], rot: 180 },
        { dest: 0, type: 'VOID', pos: [-8, 2, 6], rot: 0 }
      ]}
    };
  }

  initializeGame() {
    // Clear previous game state
    this.scene.scene.clear();
    this.puzzle = new PuzzleManager();
    
    // Build world with all rooms
    this.world.rooms.forEach((room, idx) => {
      this.scene.scene.add(room.group);
    });
    
    // Create player
    this.player = new Player();
    this.player.position.set(0, 1.8, 0);
    this.player.currentRoom = this.world.rooms[0];
    
    // Create NPCs and items
    this.npcs = new NPCManager(this.scene.scene, this.world);
    this.items = new ItemManager(this.scene.scene, this.world);
    
    // Create controls
    this.controls = new Controls(this.player);
    
    // Create HUD
    this.hud = new HUDManager(this.puzzle, this.player);
    
    // Link portals with room connectivity
    this.linkAllPortals();
    
    // Set initial player metrics
    this.player.sanity = 80;
    this.player.flashlightBattery = 100;
  }

  linkAllPortals() {
    Object.entries(this.roomConnectivity).forEach(([roomIdxStr, roomData]) => {
      const roomIdx = parseInt(roomIdxStr);
      roomData.portals.forEach((portalData, portalIdx) => {
        const portal = new Portal(
          new THREE.Vector3(...portalData.pos),
          new THREE.Euler(0, portalData.rot * Math.PI / 180, 0),
          this.getPortalTypeId(portalData.type),
          portalData.dest,
          this.world.rooms[portalData.dest]
        );
        // Build visual components (buildFrame/buildSurface already attach to portal.group)
        try {
          portal.frame = portal.buildFrame();
        } catch (err) {
          portal.frame = null;
        }
        try {
          portal.surface = portal.buildSurface();
        } catch (err) {
          portal.surface = null;
        }

        if (portal.frame) portal.group.add(portal.frame);
        if (portal.surface) portal.group.add(portal.surface);

        if (this.world.rooms[roomIdx] && this.world.rooms[roomIdx].group) {
          this.world.rooms[roomIdx].group.add(portal.group);
        }

        this.portals.portals.push(portal);
      });
    });
  }

  getPortalTypeId(typeName) {
    const types = {
      'NORMAL': 0, 'UPSIDE_DOWN': 1, 'SIDEWAYS_L': 2, 'SIDEWAYS_R': 3,
      'FORWARD_DOWN': 4, 'ROTATED_45': 5, 'LOOP': 6, 'MIRROR': 7,
      'SCALED': 8, 'TIME_LAG': 9, 'VOID': 10, 'CORRECT': 11
    };
    return types[typeName] || 0;
  }

  startGame() {
    this.state = 'playing';
    this.initializeGame();
    this.audio.resume(); // Resume Web Audio on interaction
    this.animate();
  }

  pauseGame() {
    if (this.state === 'playing') {
      this.state = 'paused';
      this.ui.showPause();
    }
  }

  resumeGame() {
    this.state = 'playing';
    this.lastFrameTime = Date.now();
  }

  restartGame() {
    this.state = 'playing';
    this.scene.scene.clear();
    this.initializeGame();
    this.animate();
  }

  endGame(reason) {
    this.state = 'gameover';
    this.ui.showGameOver(reason);
  }

  winGame() {
    this.state = 'win';
    const stats = {
      sanity: this.player.sanity,
      rooms: this.puzzle.completedObjectives.length,
      items: this.puzzle.inventory.length
    };
    this.ui.showWin(stats);
  }

  update(delta) {
    if (this.state !== 'playing') return;

    // Update player
    const moveDir = new THREE.Vector3();
    if (this.controls.keys['w']) moveDir.z -= 1;
    if (this.controls.keys['s']) moveDir.z += 1;
    if (this.controls.keys['a']) moveDir.x -= 1;
    if (this.controls.keys['d']) moveDir.x += 1;
    if (this.controls.keys['shift']) this.player.sprint = true;
    else this.player.sprint = false;
    
    if (this.controls.keys[' ']) this.player.tryJump();
    if (this.controls.keys['c']) this.player.setCrouch(true);
    else this.player.setCrouch(false);
    
    this.player.moveInDirection(moveDir, delta);
    this.player.update(delta);
    
    // Check portal crossing
    this.portals.portals.forEach(portal => {
      if (portal.checkPlayerCrossing(this.player.lastPos, this.player.position)) {
        this.player.teleportThroughPortal(portal);
        this.audio.playSwoosh();
      }
    });

    // Update current room based on position
    let closestRoom = this.world.rooms[0];
    let minDist = Infinity;
    this.world.rooms.forEach(room => {
      const roomCenter = room.group.position;
      const dist = this.player.position.distanceTo(roomCenter);
      if (dist < minDist) {
        minDist = dist;
        closestRoom = room;
      }
    });
    this.player.currentRoom = closestRoom;

    // Check void proximity for audio/visual effects
    this.portals.portals.forEach(portal => {
      if (portal.type === 10) { // VOID
        const distToVoid = this.player.position.distanceTo(portal.position);
        if (distToVoid < 5) {
          const intensity = 1 - (distToVoid / 5);
          this.audio.playVoidRumble(intensity);
          this.player.sanity -= intensity * 0.5;
        }
      }
    });

    // Update NPCs
    this.npcs.update(delta, this.player);

    // Update items
    this.items.update(delta);
    this.items.items.forEach(item => {
      if (this.player.position.distanceTo(item.position) < 2.0) {
        if (this.controls.tryInteract()) {
          this.player.collectItem(item);
          this.puzzle.addItem(item);
          this.audio.playPickup();
          this.particles.emitBurst(item.position, 8, 'spark');
          this.items.removeItem(item);
        }
      }
    });

    // Update particles
    this.particles.update(delta);

    // Update audio
    if (this.player.onGround && this.player.velocity.length() > 0.1) {
      this.audio.playFootstep();
    }
    this.audio.setAmbience('office');

    // Update effects
    this.effects.updateSanity(this.player.sanity);
    this.effects.applyDistortions();

    // Update HUD
    this.hud.updateSanity(this.player.sanity);
    this.hud.updateBattery(this.player.flashlightBattery);
    this.hud.updateInventory(this.puzzle.inventory);
    this.hud.updateObjectives(this.puzzle);
    this.hud.updateMap(this.player.currentRoom);

    // Check win/loss conditions
    if (this.player.sanity <= -100) {
      this.endGame('ASSIMILATED INTO BUREAUCRACY\nYour consciousness merges with the filing systems.');
      return;
    }

    if (this.npcs.auditor && this.npcs.auditor.isAttacking && this.player.position.distanceTo(this.npcs.auditor.position) < 1.5) {
      this.endGame('AUDITED\nYour employment and soul have been permanently revoked.');
      return;
    }

    // Check exit condition
    if (this.player.currentRoom.id === 11) { // EXIT_HALL
      const exitDeskPos = new THREE.Vector3(8, 1.5, 0);
      if (this.player.position.distanceTo(exitDeskPos) < 2.0) {
        if (this.controls.tryInteract()) {
          if (this.puzzle.canExitGame()) {
            this.winGame();
          } else {
            this.ui.showNotification('MISSING REQUIRED ITEMS', 'warning');
          }
        }
      }
    }

    // Check secret ending (8 sticky notes)
    if (this.puzzle.hasSecretEnding() && this.player.currentRoom.id === 0) {
      // Reveal secret portal to supply closet (Portal #0 becomes CORRECT)
      if (this.portals.portals[0].type !== 11) {
        this.portals.portals[0].type = 11;
        this.ui.showNotification('PORTAL 0 RECALIBRATED - SEEK GREG\'S CLOSET', 'success');
      }
    }
  }

  renderPortals() {
    const visiblePortals = [];
    
    // Find portals within view
    this.portals.portals.forEach(portal => {
      const distToCamera = this.scene.camera.position.distanceTo(portal.position);
      const frustumTest = this.scene.camera.frustum.containsPoint(portal.position);
      if (distToCamera < 50 && frustumTest) {
        visiblePortals.push({ portal, depth: 0 });
      }
    });

    // Render each portal with stencil technique (max 2 levels of recursion)
    visiblePortals.forEach(({ portal }) => {
      this.renderPortalStencil(portal, 1);
    });
  }

  renderPortalStencil(portal, depth) {
    if (depth > this.maxPortalDepth) return;

    // PASS 1: Write stencil ID (no color output)
    const gl = this.scene.renderer.getContext();
    gl.colorMask(false, false, false, false);
    gl.stencilFunc(gl.ALWAYS, portal.stencilID, 0xFF);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
    
    this.scene.renderer.render(portal.frameScene, this.scene.camera);
    
    // PASS 2: Render destination through portal with stencil test
    gl.colorMask(true, true, true, true);
    gl.stencilFunc(gl.EQUAL, portal.stencilID, 0xFF);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    
    const virtualCamera = portal.getDestinationCamera(this.scene.camera);
    this.scene.renderer.render(this.scene.scene, virtualCamera);
    
    // PASS 3: Depth repair (render portal frame again to fix depth)
    gl.colorMask(false, false, false, false);
    gl.depthFunc(gl.ALWAYS);
    this.scene.renderer.render(portal.surfaceScene, this.scene.camera);
    gl.depthFunc(gl.LESS);
    
    gl.stencilFunc(gl.ALWAYS, 0, 0);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const now = Date.now();
    this.deltaTime = (now - this.lastFrameTime) / 1000;
    this.lastFrameTime = now;

    // Clamp delta to prevent large jumps
    if (this.deltaTime > 0.05) this.deltaTime = 0.05;

    if (this.state === 'playing') {
      this.update(this.deltaTime);
      this.renderPortals();
    }

    this.scene.renderer.render(this.scene.scene, this.scene.camera);
  }
}

// Global game instance
window.game = null;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  window.game = new GameController();
  
  // Resume audio context on any interaction
  ['click', 'keydown', 'touchstart'].forEach(event => {
    document.addEventListener(event, () => {
      if (window.game && window.game.audio) {
        window.game.audio.resume();
      }
    }, { once: true });
  });
});