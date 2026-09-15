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
import { bus } from './event-bus.js';
import { createPostProcessing } from './post-processing.js';
import { PortalRenderer } from './portal-renderer.js';
import { SaveSystem } from './save-system.js';
import { ShadowSystem } from './shadow-system.js';

export class GameController {
  constructor() {
    this.state = 'menu';
    this.lastFrameTime = Date.now();
    this.deltaTime = 0;
    this.exitTriggered = false;
    this._exitDeskPos = new THREE.Vector3(8, 1.5, 0);
    this._moveDir = new THREE.Vector3();
    this._lastAmbienceTime = 0;

    // Initialize core systems
    this.scene = new SceneManager();
    this.world = new WorldGenerator(this.scene.scene);
    this.portals = new PortalManager(this.scene.scene);
    this.particles = new ParticleSystem(this.scene.scene);
    this.audio = new AudioManager();
    this.effects = new NonEuclideanEffects(this.scene.renderer, this.scene.scene);
    this.puzzle = new PuzzleManager();

    // Initialize game entities
    this.player = null;
    this.npcs = null;
    this.items = null;
    this.controls = null;
    this.hud = null;
    this.ui = new UIManager(this);
    this.portalRenderer = null;
    this.saveSystem = new SaveSystem();
    this.shadowSystem = new ShadowSystem();
    this.noteMarkers = [];
    this.phoneRinging = false;
    this.phoneTimer = 20;
    this.phoneRingDuration = 0;
    this.phonePosition = new THREE.Vector3(0, 0, -3);
    this.coffeeActive = false;
    this.coffeeRound = 1;
    this.coffeeMeterValue = 50;
    this.coffeeMeterDir = 1;
    this.coffeeMeterSpeed = 1.5;
    this.coffeeRoundTimer = 0;
    this.coffeeCooldown = 0;
    this.coffeePosition = new THREE.Vector3(-2, 0, 2);
    this.formOpen = false;
    this.formFields = { name: true, dept: true, stamp: false, signature: false, date: true };
    this.paperScoutActive = false;
    this.paperScoutMesh = null;
    this.paperScoutTimer = 0;
    this.newspaperOpen = false;
    this.faxActive = false;
    this.faxDigits = [];
    this.faxInput = [];
    this.faxTimer = 10;
    this.faxPosition = new THREE.Vector3(3, 0, 5);

    // Photocopier Multiverse
    this.photocopierActive = false;
    this.photocopierCopies = [];
    this.photocopierTimer = 30;
    this.photocopierPosition = new THREE.Vector3(4, 0, -2);
    this.photocopierCurrentCopy = 0;
    this._photocopierKeyHeld = [false, false, false];

    // Elevator of Wrong Floors
    this.elevatorActive = false;
    this.elevatorCooldown = 0;
    this.elevatorPosition = new THREE.Vector3(0, 0, 8);

    // Water Cooler Gossip
    this.gossipActive = false;
    this.gossipMessages = [];
    this.gossipTimer = 0;
    this.gossipCooldown = 0;
    this.gossipPosition = new THREE.Vector3(-3, 0, 5);

    // Employee of the Month
    this.employeeActive = false;
    this.employeeVoted = false;
    this.employeePosition = new THREE.Vector3(2, 0, 2);
    this.employeeCooldown = 0;

    // Time Clock Punch
    this.timeClockActive = false;
    this.timeClockTimer = 0;
    this.timeClockMisses = 0;
    this.timeClockPromptTimer = 0;
    this.timeClockPosition = new THREE.Vector3(0, 0, -5);
    this.timeClockPunching = false;

    // Lunch Hour at the Void
    this.lunchActive = false;
    this.lunchTimer = 0;
    this.lunchEaten = false;
    this.lunchPosition = new THREE.Vector3(0, 0, 0);

    // Quarterly Performance Review
    this.reviewActive = false;
    this.reviewQuestions = [];
    this.reviewCurrentQ = 0;
    this.reviewScore = 0;

    // Office Fire Drill
    this.fireDrillActive = false;
    this.fireDrillTimer = 0;
    this.fireDrillDoorsSealed = 0;
    this.fireDrillOriginalPortals = [];

    // Telecommuting Dimension
    this.telecommuteActive = false;
    this.telecommuteInDimension = false;
    this.telecommuteTimer = 0;
    this.telecommuteMonitor = null;
    this._throwKeyHeld = false;
    this._newspaperKeyHeld = false;
    this._faxDigitHeld = Array(10).fill(false);
    this._timeClockKeyHeld = false;
    this._reviewKeyHeld = [false, false, false, false, false];
    this._photocopierKeyHeld = [false, false, false];

    // Room connectivity mapping (12 rooms, bidirectional portal connections)
    this.roomConnectivity = this.buildRoomConnectivity();

    // Portal rendering state
    this.portalRenderQueue = [];

    // Game stats
    this.debugMode = false;
    this.currentRoomTheme = null;
  }

  buildRoomConnectivity() {
    return {
      0: {
        portals: [
          { dest: 1, type: 'NORMAL', pos: [6, 1.5, 5], rot: 0 },
          { dest: 3, type: 'UPSIDE_DOWN', pos: [-6, 1.5, 5], rot: 0 }
        ]
      },
      1: {
        portals: [
          { dest: 0, type: 'NORMAL', pos: [4, 1.5, -6], rot: 180 },
          { dest: 5, type: 'LOOP', pos: [-4, 1.5, 6], rot: 90 }
        ]
      },
      2: { portals: [{ dest: 6, type: 'SIDEWAYS_L', pos: [4, 1.5, 4], rot: 0 }] },
      3: {
        portals: [
          { dest: 0, type: 'UPSIDE_DOWN', pos: [4, 1.5, -8], rot: 0 },
          { dest: 2, type: 'NORMAL', pos: [-2, 1.5, 8], rot: 0 },
          { dest: 4, type: 'SIDEWAYS_R', pos: [2, 1.5, -8], rot: 0 }
        ]
      },
      4: {
        portals: [
          { dest: 3, type: 'SIDEWAYS_L', pos: [3, 1.5, -3], rot: 0 },
          { dest: 5, type: 'NORMAL', pos: [-3, 1.5, -3], rot: 0 }
        ]
      },
      5: {
        portals: [
          { dest: 1, type: 'LOOP', pos: [4, 1.5, -5], rot: 270 },
          { dest: 4, type: 'NORMAL', pos: [4, 1.5, 5], rot: 0 },
          { dest: 6, type: 'MIRROR', pos: [-4, 1.5, 0], rot: 0 }
        ]
      },
      6: {
        portals: [
          { dest: 2, type: 'SIDEWAYS_R', pos: [-4, 1.5, -4], rot: 0 },
          { dest: 5, type: 'MIRROR', pos: [4, 1.5, 0], rot: 180 },
          { dest: 7, type: 'NORMAL', pos: [6, 1.5, 0], rot: 0 }
        ]
      },
      7: {
        portals: [
          { dest: 6, type: 'NORMAL', pos: [-3, 1.5, 0], rot: 180 },
          { dest: 8, type: 'SCALED', pos: [3, 1.5, -3], rot: 0 }
        ]
      },
      8: {
        portals: [
          { dest: 7, type: 'SCALED', pos: [-5, 1.5, 3], rot: 180 },
          { dest: 9, type: 'NORMAL', pos: [5, 1.5, -3], rot: 0 }
        ]
      },
      9: {
        portals: [
          { dest: 8, type: 'NORMAL', pos: [-4, 1.5, 5], rot: 180 },
          { dest: 10, type: 'NORMAL', pos: [4, 1.5, 5], rot: 0 }
        ]
      },
      10: {
        portals: [
          { dest: 9, type: 'NORMAL', pos: [0, 0.8, -4], rot: 180 },
          { dest: 11, type: 'TIME_LAG', pos: [0, 0.8, 4], rot: 0 }
        ]
      },
      11: {
        portals: [
          { dest: 10, type: 'TIME_LAG', pos: [8, 2, -6], rot: 180 },
          { dest: 0, type: 'VOID', pos: [-8, 2, 6], rot: 0 }
        ]
      }
    };
  }

  initializeGame() {
    this.shadowSystem.reset();
    this.noteMarkers.forEach((m) => this.scene.scene.remove(m));
    this.noteMarkers = [];
    this.phoneRinging = false;
    this.phoneTimer = 20;
    this.phoneRingDuration = 0;
    this.coffeeActive = false;
    this.coffeeRound = 1;
    this.coffeeMeterValue = 50;
    this.coffeeMeterDir = 1;
    this.coffeeMeterSpeed = 1.5;
    this.coffeeRoundTimer = 0;
    this.coffeeCooldown = 0;
    this.formOpen = false;
    this.formFields = { name: true, dept: true, stamp: false, signature: false, date: true };
    this.paperScoutActive = false;
    this.paperScoutMesh = null;
    this.paperScoutTimer = 0;
    this.newspaperOpen = false;
    this.faxActive = false;
    this.faxDigits = [];
    this.faxInput = [];
    this.faxTimer = 10;
    // Reset all Phase 11+ features
    this.photocopierActive = false;
    this.photocopierCopies = [];
    this.photocopierTimer = 30;
    this.photocopierCurrentCopy = 0;
    this._photocopierKeyHeld = [false, false, false];
    this.elevatorActive = false;
    this.elevatorCooldown = 0;
    this.gossipActive = false;
    this.gossipMessages = [];
    this.gossipTimer = 0;
    this.gossipCooldown = 0;
    this.employeeActive = false;
    this.employeeVoted = false;
    this.employeeCooldown = 0;
    this.timeClockActive = false;
    this.timeClockTimer = 0;
    this.timeClockMisses = 0;
    this.timeClockPromptTimer = 0;
    this.timeClockPunching = false;
    this.lunchActive = false;
    this.lunchTimer = 0;
    this.lunchEaten = false;
    this.reviewActive = false;
    this.reviewQuestions = [];
    this.reviewCurrentQ = 0;
    this.reviewScore = 0;
    this.fireDrillActive = false;
    this.fireDrillTimer = 0;
    this.fireDrillDoorsSealed = 0;
    this.fireDrillOriginalPortals = [];
    this.telecommuteActive = false;
    this.telecommuteInDimension = false;
    this.telecommuteTimer = 0;
    this.telecommuteMonitor = null;
    // Clear previous game state
    this.scene.scene.clear();
    // Recreate lighting removed by clearing the scene
    if (typeof this.scene.setupLighting === 'function') this.scene.setupLighting();
    this.puzzle = new PuzzleManager();

    // Build world with all rooms
    this.world.rooms.forEach((room, idx) => {
      this.scene.scene.add(room.group);
    });

    // Create player
    this.player = new Player(this.scene.camera, this.world);
    this.player.position.set(0, 1.8, 0);
    this.player.currentRoom = this.world.rooms[0];

    // Link player to puzzle system
    this.puzzle.player = this.player;

    // Create NPCs and items
    this.npcs = new NPCManager(this.scene.scene, this.world);
    this.items = new ItemManager(this.scene.scene, this.world);

    // Create controls
    this.controls = new Controls(this.player, bus);

    // Create HUD
    this.hud = new HUDManager(this.player, this.puzzle, this);

    // Link portals with room connectivity
    this.linkAllPortals();
    this.portalRenderer = new PortalRenderer(this.scene, this.portals);
    this.shadowSystem.createGhostMesh(this.scene.scene);

    // Setup EventBus handlers
    this.setupHandlers();

    // Set initial player metrics
    this.player.sanity = 80;
    this.player.flashlightBattery = 100;
  }

  linkAllPortals() {
    const portalKeys = [];

    Object.entries(this.roomConnectivity).forEach(([roomIdxStr, roomData]) => {
      const roomIdx = parseInt(roomIdxStr);
      roomData.portals.forEach((portalData, portalIdx) => {
        const portal = new Portal(
          new THREE.Vector3(...portalData.pos),
          new THREE.Euler(0, (portalData.rot * Math.PI) / 180, 0),
          this.getPortalTypeId(portalData.type),
          portalData.dest,
          this.world.rooms[portalData.dest]
        );

        if (this.world.rooms[roomIdx] && this.world.rooms[roomIdx].group) {
          this.world.rooms[roomIdx].group.add(portal.group);
        }

        this.portals.portals.push(portal);
        portalKeys.push([roomIdx, portalData.dest]);
      });
    });

    portalKeys.forEach(([src, dest], idx) => {
      const reverseIdx = portalKeys.findIndex(([s, d]) => s === dest && d === src);
      if (reverseIdx !== -1 && reverseIdx !== idx) {
        this.portals.linkPortals(idx, reverseIdx);
      }
    });
  }

  getPortalTypeId(typeName) {
    const types = {
      NORMAL: 0,
      UPSIDE_DOWN: 1,
      SIDEWAYS_L: 2,
      SIDEWAYS_R: 3,
      FORWARD_DOWN: 4,
      ROTATED_45: 5,
      LOOP: 6,
      MIRROR: 7,
      SCALED: 8,
      TIME_LAG: 9,
      VOID: 10,
      CORRECT: 11
    };
    return types[typeName] || 0;
  }

  getRoomTheme(roomId) {
    const themes = {
      0: {
        fogColor: 0x0a0a14,
        fogDensity: 0.08,
        ambient: 0.6,
        dirColor: 0xffffff,
        dirIntensity: 1.2,
        pointColor: 0xffffff
      },
      1: {
        fogColor: 0x1a1608,
        fogDensity: 0.14,
        ambient: 0.35,
        dirColor: 0xddcc99,
        dirIntensity: 0.5,
        pointColor: 0xfffacd
      },
      2: {
        fogColor: 0x141008,
        fogDensity: 0.1,
        ambient: 0.5,
        dirColor: 0xffaa44,
        dirIntensity: 0.7,
        pointColor: 0xffddaa
      },
      3: {
        fogColor: 0x141400,
        fogDensity: 0.18,
        ambient: 0.25,
        dirColor: 0xffff88,
        dirIntensity: 0.6,
        pointColor: 0xffffaa
      },
      4: {
        fogColor: 0x14100a,
        fogDensity: 0.06,
        ambient: 0.7,
        dirColor: 0xffd4a8,
        dirIntensity: 1.0,
        pointColor: 0xffe8c8
      },
      5: {
        fogColor: 0x000000,
        fogDensity: 0.22,
        ambient: 0.1,
        dirColor: 0x330000,
        dirIntensity: 0.15,
        pointColor: 0x220000
      },
      6: {
        fogColor: 0x140808,
        fogDensity: 0.08,
        ambient: 0.5,
        dirColor: 0xff6644,
        dirIntensity: 1.0,
        pointColor: 0xff8866
      },
      7: {
        fogColor: 0x0a0a0a,
        fogDensity: 0.08,
        ambient: 0.5,
        dirColor: 0xddeeff,
        dirIntensity: 1.0,
        pointColor: 0xeeffff
      },
      8: {
        fogColor: 0x0c0c12,
        fogDensity: 0.06,
        ambient: 0.6,
        dirColor: 0xffffff,
        dirIntensity: 1.2,
        pointColor: 0xffffff
      },
      9: {
        fogColor: 0x041408,
        fogDensity: 0.12,
        ambient: 0.3,
        dirColor: 0x44ff88,
        dirIntensity: 0.6,
        pointColor: 0x33ff66
      },
      10: {
        fogColor: 0x0a0a14,
        fogDensity: 0.14,
        ambient: 0.4,
        dirColor: 0xaaccff,
        dirIntensity: 0.8,
        pointColor: 0xbbddff
      },
      11: {
        fogColor: 0x080814,
        fogDensity: 0.03,
        ambient: 0.85,
        dirColor: 0xffffff,
        dirIntensity: 1.5,
        pointColor: 0xffffff
      }
    };
    return themes[roomId] || themes[0];
  }

  applyRoomTheme(roomId) {
    if (this.currentRoomTheme === roomId) return;
    this.currentRoomTheme = roomId;
    const theme = this.getRoomTheme(roomId);
    const scene = this.scene.scene;

    scene.fog = new THREE.FogExp2(theme.fogColor, theme.fogDensity);
    this.scene.directionalLight.color.setHex(theme.dirColor);
    this.scene.directionalLight.intensity = theme.dirIntensity;
    this.scene.pointLights.forEach((light) => {
      light.color.setHex(theme.pointColor);
      light.intensity = Math.max(0.15, theme.ambient * 0.7);
    });
  }

  startGame() {
    this.state = 'playing';
    this.initializeGame();
    this.scene.postProcessing = createPostProcessing(
      this.scene.renderer,
      this.scene.scene,
      this.scene.camera,
      () => this.portalRenderer.renderVisiblePortals()
    );
    this.audio.resume();
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
    // Cleanup previous game state
    if (this.player) this.player.destroy();
    if (this.controls) this.controls.destroy();
    this.scene.scene.clear();
    this.initializeGame();
    this.animate();
  }

  setupHandlers() {
    bus.on('audio:play', (data) => {
      if (data.type === 'jump') this.audio.playJump();
      else if (data.type === 'dialog') this.audio.playDialog();
      else this.audio.playAmbience(data.type || 'office');
    });

    bus.on('audio:swoosh', () => this.audio.playSwoosh());
    bus.on('audio:pickup', () => this.audio.playPickup());
    bus.on('audio:footstep', () => this.audio.playFootstep());

    bus.on('game:over', (data) => this.endGame(data.reason));
    bus.on('game:win', () => this.winGame());
    bus.on('input:pause', () => this.pauseGame());

    bus.on('input:interact', () => {
      if (this.phoneRinging && this.player.position.distanceTo(this.phonePosition) < 2.0) {
        this.answerPhone();
        return;
      }
      if (
        !this.coffeeActive &&
        this.coffeeCooldown <= 0 &&
        this.player.position.distanceTo(this.coffeePosition) < 1.5
      ) {
        this.startCoffeeGame();
        return;
      }
      if (!this.faxActive && this.player.position.distanceTo(this.faxPosition) < 2.0) {
        this.startFax();
        return;
      }
      if (
        !this.photocopierActive &&
        this.player.position.distanceTo(this.photocopierPosition) < 2.0
      ) {
        this.startPhotocopier();
        return;
      }
      if (this.player.position.distanceTo(this.elevatorPosition) < 2.0) {
        this.activateElevator();
        return;
      }
      if (
        this.player.position.distanceTo(this.gossipPosition) < 2.0 &&
        this.player.position.distanceTo(this.coffeePosition) > 1.5
      ) {
        this.startGossip();
        return;
      }
      if (this.player.position.distanceTo(this.employeePosition) < 2.0) {
        this.activateEmployeeOfMonth();
        return;
      }
      if (this.player.position.distanceTo(this.timeClockPosition) < 2.0) {
        this.activateTimeClock();
        return;
      }
      if (this.player.position.distanceTo(this.lunchPosition) < 2.0 && !this.lunchActive) {
        this.startLunchHour();
        return;
      }
      if (
        this.player.position.distanceTo(new THREE.Vector3(0, 1.5, 0)) < 2.0 &&
        !this.reviewActive
      ) {
        this.startReview();
        return;
      }
      if (
        this.telecommuteMonitor &&
        this.player.position.distanceTo(this.telecommuteMonitor.position) < 2.0
      ) {
        this.destroyTelecommuteMonitor();
        return;
      }
      const item = this.items.checkInteraction(this.player.position);
      if (item && !item.collected) {
        this.player.collectItem(item);
        this.audio.playSpatialPickup(item.position);
        this.particles.emitBurst(item.position, 8, 'spark');
        this.items.collectItem(item);
        this.puzzle.addItemToInventory(item);
        if (item.type === 'stamp') this.formFields.stamp = true;
        if (item.type === 'note') this.spawnNoteMarker(item.position);
        if (item.type === 'apple' && this.lunchActive) {
          this.eatApple();
        }
      }
    });

    bus.on('player:sanity-changed', (data) => {
      this.hud.updateSanity(data.value);
    });

    bus.on('inventory:add', (data) => {
      this.puzzle.addItemToInventory(data.item);
    });

    bus.on('ui:notify', (data) => {
      this.ui.showNotification(data.message, data.type);
    });

    bus.on('npc:dialog', (data) => {
      this.ui.showNPCDialog(data.position, data.text);
      this.audio.playSpatialDialog(data.position);
    });

    bus.on('npc:interacted', (data) => {
      this.formFields.signature = true;
      this.ui.updateExitForm(this.formFields);
    });

    bus.on('ui:toggle-inventory', () => {
      this.ui.toggleInventoryView();
    });
  }

  answerPhone() {
    this.phoneRinging = false;
    this.phoneTimer = 45 + Math.random() * 30;
    this.player.modifySanity(5);
    this.ui.showNotification('CLERK NOTIFIED', 'success');
    this.audio.playClick();
  }

  startCoffeeGame() {
    this.coffeeActive = true;
    this.coffeeRound = 1;
    this.coffeeMeterValue = 50;
    this.coffeeMeterDir = 1;
    this.coffeeMeterSpeed = 1.5;
    this.coffeeRoundTimer = 2;
    this.ui.showCoffeeMeter();
    this.audio.playCoffeePour();
  }

  stopCoffeeMeter() {
    const value = this.coffeeMeterValue;
    let result, sanity;
    if (value >= 40 && value <= 60) {
      result = 'PERFECT';
      sanity = 20;
    } else if (value >= 25 && value <= 75) {
      result = 'GOOD';
      sanity = 15;
    } else {
      result = 'OK';
      sanity = 5;
    }
    this.player.modifySanity(sanity);
    this.ui.showNotification(`COFFEE ${result} — SANITY +${sanity}`, 'success');
    if (this.coffeeRound >= 3) {
      this.endCoffeeGame();
    } else {
      this.coffeeRound++;
      this.coffeeMeterValue = 50;
      this.coffeeMeterDir = 1;
      this.coffeeMeterSpeed = 1.5 + (this.coffeeRound - 1) * 0.5;
      this.coffeeRoundTimer = 2;
      this.audio.playCoffeePour();
    }
  }

  endCoffeeGame() {
    this.coffeeActive = false;
    this.coffeeCooldown = 60;
    this.ui.hideCoffeeMeter();
    this.ui.showNotification('COFFEE SESSION ENDED', 'info');
  }

  openExitForm() {
    this.formOpen = true;
    this.formFields.stamp = this.puzzle.inventory.some((i) => i.type === 'stamp');
    const dateEl = document.getElementById('form-date');
    if (dateEl) dateEl.textContent = new Date().toISOString().slice(0, 10);
    this.ui.showExitForm(this.formFields);
    this.ui.showNotification('FORM 27-Γ OPENED', 'info');
  }

  closeExitForm() {
    this.formOpen = false;
    this.ui.hideExitForm();
  }

  submitExitForm() {
    if (
      this.formFields.name &&
      this.formFields.dept &&
      this.formFields.stamp &&
      this.formFields.signature &&
      this.formFields.date
    ) {
      this.formOpen = false;
      this.ui.hideExitForm();
      this.winGame();
    } else {
      this.ui.showNotification('FORM INCOMPLETE — FILL ALL FIELDS', 'warning');
    }
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

  startFax() {
    this.faxActive = true;
    this.faxDigits = Array.from({ length: 10 }, () => Math.floor(Math.random() * 10));
    this.faxInput = Array(10).fill(null);
    this.faxTimer = 15;
    this.ui.showFaxKeypad();
    this.ui.showNotification('FAX MACHINE — ENTER 10 DIGITS', 'info');
  }

  dialFaxDigit(digit) {
    if (!this.faxActive) return;
    const pos = this.faxInput.indexOf(null);
    if (pos !== -1) {
      this.faxInput[pos] = digit;
      this.audio.playClick();
      this.ui.updateFaxDisplay(this.faxDigits, this.faxInput);
      if (this.faxInput.every((d) => d !== null)) {
        this.checkFaxResult();
      }
    }
  }

  checkFaxResult() {
    const correct = this.faxInput.every((d, i) => d === this.faxDigits[i]);
    this.faxActive = false;
    this.ui.hideFaxKeypad();
    if (correct) {
      this.player.modifySanity(5);
      this.ui.showNotification('FAX SENT — EMPLOYEE OF MONTH?', 'success');
    } else {
      this.failFax('WRONG NUMBER');
    }
  }

  failFax(reason) {
    this.faxActive = false;
    this.ui.hideFaxKeypad();
    this.player.modifySanity(-5);
    this.ui.showNotification(`FAX EXPLODED — ${reason}`, 'warning');
  }

  throwPaperScout() {
    if (this.paperScoutActive) return;
    this.player.modifySanity(-2);
    const geo = new THREE.TriangleGeometry(0.2, 0.15);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9
    });
    this.paperScoutMesh = new THREE.Mesh(geo, mat);
    this.paperScoutMesh.position.copy(this.scene.camera.position);
    this.scene.scene.add(this.paperScoutMesh);
    this.paperScoutActive = true;
    this.paperScoutTimer = 3;
    this.ui.showNotification('PAPER AIRPLANE THROWN', 'info');
  }

  collectPaperScout() {
    if (!this.paperScoutMesh) return;
    const pos = new THREE.Vector3();
    this.paperScoutMesh.getWorldPosition(pos);

    this.scene.scene.remove(this.paperScoutMesh);
    this.paperScoutMesh = null;
    this.paperScoutActive = false;
    this.paperScoutTimer = 0;

    let nearest = null;
    let minDist = Infinity;
    this.portals.portals.forEach((portal) => {
      const d = pos.distanceTo(portal.position);
      if (d < minDist) {
        minDist = d;
        nearest = portal;
      }
    });

    if (nearest && minDist < 20) {
      const dest = nearest.destinationRoom;
      const typeName = [
        'Normal',
        'Upside-Down',
        'Sideways L',
        'Sideways R',
        'Forward-Down',
        'Rotated',
        'Loop',
        'Mirror',
        'Scaled',
        'Time-Lag',
        'VOID',
        'Correct'
      ][nearest.type];
      this.ui.showNotification(`SCOUTED: Room ${dest?.id ?? '?'} via ${typeName} portal`, 'info');
    } else {
      this.ui.showNotification('SCOUTED: No portal nearby', 'info');
    }
  }

  toggleNewspaper() {
    if (this.newspaperOpen) {
      this.hideNewspaper();
    } else {
      this.showNewspaper();
    }
  }

  showNewspaper() {
    this.newspaperOpen = true;
    const el = document.getElementById('newspaper');
    if (el) el.classList.remove('hidden');
    this.updateNewspaper();
  }

  hideNewspaper() {
    this.newspaperOpen = false;
    const el = document.getElementById('newspaper');
    if (el) el.classList.add('hidden');
  }

  updateNewspaper() {
    const titleEl = document.getElementById('news-title');
    const dateEl = document.getElementById('news-date');
    const roomsEl = document.getElementById('news-rooms');
    const itemsEl = document.getElementById('news-items');
    const sanityEl = document.getElementById('news-sanity');
    const statusEl = document.getElementById('news-status');

    if (titleEl) titleEl.textContent = 'BUREAU GAZETTE';
    if (dateEl) dateEl.textContent = new Date().toLocaleDateString();
    if (roomsEl)
      roomsEl.textContent = `Rooms Explored: ${this.puzzle.completedObjectives.length} / 12`;
    if (itemsEl) itemsEl.textContent = `Items Collected: ${this.puzzle.inventory.length} / 8`;
    if (sanityEl) sanityEl.textContent = `Sanity: ${Math.round(this.player.sanity)} / 100`;

    const sanity = this.player.sanity;
    let status = 'ACCEPTABLE';
    if (sanity < -50) status = 'AUDITOR ACTIVE';
    else if (sanity < 30) status = 'CRITICAL';
    else if (sanity < 50) status = 'WARNING';
    else if (sanity < 70) status = 'MILD HALLUCINATIONS';
    if (statusEl) {
      statusEl.textContent = status;
      statusEl.style.color =
        sanity < -50
          ? 'var(--portal-red)'
          : sanity < 30
            ? 'var(--sanity-bad)'
            : sanity < 50
              ? 'var(--sanity-warn)'
              : 'var(--sanity-good)';
    }
  }

  startPhotocopier() {
    if (this.photocopierActive) return;
    this.photocopierActive = true;
    this.photocopierTimer = 30;
    this.photocopierCurrentCopy = 0;
    this._photocopierKeyHeld = [false, false, false];
    this.photocopierCopies = [];
    const colors = [0xff4444, 0x44ff44, 0x4444ff];
    for (let i = 0; i < 3; i++) {
      const copy = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 1.8, 0.5),
        new THREE.MeshBasicMaterial({
          color: colors[i],
          transparent: true,
          opacity: 0.35,
          wireframe: true
        })
      );
      copy.position.copy(this.player.position);
      this.scene.scene.add(copy);
      this.photocopierCopies.push(copy);
    }
    this.ui.showPhotocopierOverlay();
    this.audio.playClick();
  }

  switchPhotocopierCopy(idx) {
    if (!this.photocopierActive || idx >= this.photocopierCopies.length) return;
    this.photocopierCurrentCopy = idx;
    this.ui.updatePhotocopierDisplay(idx);
    this.audio.playClick();
  }

  endPhotocopier() {
    this.photocopierCopies.forEach((c) => this.scene.scene.remove(c));
    this.photocopierCopies = [];
    this.photocopierActive = false;
    this.ui.hidePhotocopierOverlay();
    this.player.modifySanity(10);
    this.ui.showNotification('MULTIVERSE COPIES MERGED — SANITY +10', 'success');
  }

  activateElevator() {
    if (this.elevatorActive || this.elevatorCooldown > 0) return;
    this.elevatorActive = true;
    const rooms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    const wrongFloor = rooms[Math.floor(Math.random() * rooms.length)];
    this.ui.showNotification(`ELEVATOR — WRONG FLOOR: ROOM ${wrongFloor}`, 'warning');
    this.player.teleportToRoom(wrongFloor);
    const gravityAngles = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];
    this.player.gravityAngle = gravityAngles[Math.floor(Math.random() * gravityAngles.length)];
    this.elevatorCooldown = 20;
    this.elevatorActive = false;
    this.audio.playSwoosh();
  }

  startGossip() {
    if (this.gossipActive || this.gossipCooldown > 0) return;
    this.gossipActive = true;
    this.gossipTimer = 15;
    const rumors = [
      'Heard EMPLOYEE #4471-B was seen in the VOID Office... strange choice.',
      'EMPLOYEE #4471-B runs fast. SUSPICIOUS.',
      'Someone saw EMPLOYEE #4471-B carrying a FORKLIFT at 3am.',
      'Rumor: EMPLOYEE #4471-B is actually a PORTAL in disguise.',
      'EMPLOYEE #4471-B has been filing INCORRECTLY. Investigate.',
      'Whispers say EMPLOYEE #4471-B knows about the AUDITOR.',
      'EMPLOYEE #4471-B was seen SPEAKING to the VOID.',
      'CONFIDENTIAL: EMPLOYEE #4471-B drinks coffee black.',
      'EMPLOYEE #4471-B skipped 3 fire drills. FLAGGED.',
      'Word is EMPLOYEE #4471-B has a twin in the SHADOW dimension.'
    ];
    this.gossipMessages.push(rumors[Math.floor(Math.random() * rumors.length)]);
    if (this.gossipMessages.length > 5) this.gossipMessages.shift();
    this.ui.showGossipBoard(this.gossipMessages);
  }

  activateEmployeeOfMonth() {
    if (this.employeeActive || this.employeeCooldown > 0) return;
    this.employeeActive = true;
    const win = Math.random() > 0.5;
    if (win) {
      this.player.modifySanity(10);
      this.puzzle.addItemToInventory({
        type: 'secret',
        name: 'Golden Clipboard',
        position: this.employeePosition.clone()
      });
      this.ui.showNotification('EMPLOYEE OF THE MONTH! +10 SANITY', 'success');
    } else {
      this.player.modifySanity(-10);
      this.ui.showNotification('DENIED — Coworkers mock your ambition.', 'warning');
    }
    this.employeeActive = false;
    this.employeeCooldown = 120;
    this.audio.playClick();
  }

  activateTimeClock() {
    if (this.timeClockPunching) return;
    this.timeClockPunching = true;
    this.timeClockTimer = 2;
    this.ui.showTimeClockOverlay(2, this.timeClockMisses);
  }

  punchTimeClock() {
    if (!this.timeClockPunching) return;
    this.timeClockPunching = false;
    this.player.modifySanity(5);
    this.ui.hideTimeClockOverlay();
    this.ui.showNotification('CLOCK PUNCHED — SANITY +5', 'success');
    this.audio.playClick();
  }

  startLunchHour() {
    if (this.lunchActive) return;
    this.lunchActive = true;
    this.lunchTimer = 30;
    this.lunchEaten = false;
    this.player.teleportToRoom(5);
    this.ui.showLunchOverlay(30, false);
    this.ui.showNotification('LUNCH HOUR — FIND THE APPLE!', 'warning');
  }

  eatApple() {
    if (!this.lunchActive || this.lunchEaten) return;
    this.lunchEaten = true;
    this.ui.showNotification('APPLE EATEN — SURVIVED LUNCH', 'success');
    this.player.modifySanity(10);
  }

  endLunchHour() {
    this.lunchActive = false;
    this.ui.hideLunchOverlay();
    if (!this.lunchEaten) {
      this.player.modifySanity(-20);
      this.ui.showNotification('STARVED AT LUNCH — SANITY -20', 'warning');
    }
  }

  startReview() {
    if (this.reviewActive) return;
    this.reviewActive = true;
    this.reviewCurrentQ = 0;
    this.reviewScore = 0;
    this.reviewQuestions = [
      {
        q: 'What is the capital of the Bureau?',
        options: ['Administration', '42', 'Portal Division', 'Qardly'],
        correct: 1
      },
      {
        q: 'How many legs does a desk have?',
        options: ['2', '4', '7', 'Depends on dimension'],
        correct: 1
      },
      {
        q: 'What is the answer to life?',
        options: ['42', 'Nothing', 'Paperwork', 'Greg'],
        correct: 0
      },
      { q: 'Is the fax working?', options: ['Yes', 'No', 'Sometimes', '42'], correct: 1 },
      { q: 'Form 27-Γ or Form 27-B/6?', options: ['27-Γ', '27-B/6', 'Both', 'Neither'], correct: 2 }
    ];
    this.ui.showReviewQuestion(this.reviewQuestions[0], 0);
  }

  answerReview(idx) {
    const q = this.reviewQuestions[this.reviewCurrentQ];
    if (!q) return;
    if (idx === q.correct) {
      this.reviewScore++;
      this.player.modifySanity(3);
    } else if (idx === 1 && q.options[1] === '42') {
      this.reviewScore += 2;
      this.player.modifySanity(10);
      this.ui.showNotification('SECRET ANSWER — 42 WAS CORRECT', 'success');
    } else {
      this.player.modifySanity(-5);
    }
    this.reviewCurrentQ++;
    if (this.reviewCurrentQ >= this.reviewQuestions.length) {
      this.endReview();
    } else {
      this.ui.showReviewQuestion(this.reviewQuestions[this.reviewCurrentQ], this.reviewScore);
    }
  }

  endReview() {
    this.reviewActive = false;
    this.ui.hideReviewOverlay();
    const bonus = this.reviewScore * 2;
    this.player.modifySanity(bonus);
    this.ui.showNotification(`REVIEW COMPLETE — SCORE: ${this.reviewScore}`, 'success');
  }

  startFireDrill() {
    if (this.fireDrillActive) return;
    this.fireDrillActive = true;
    this.fireDrillTimer = 60;
    this.fireDrillDoorsSealed = 0;
    this.fireDrillOriginalPortals = this.portals.portals.map((p) => p.type);
    this.portals.portals.forEach((p) => {
      if (p.type !== 10) p.type = 10;
    });
    this.ui.showFireDrillOverlay(60, 0);
    this.ui.showNotification('FIRE DRILL — EVACUATE IN 60s!', 'danger');
  }

  endFireDrill() {
    this.fireDrillActive = false;
    this.ui.hideFireDrillOverlay();
    if (this.fireDrillDoorsSealed >= 8) {
      this.player.modifySanity(15);
      this.ui.showNotification('FIRE DRILL PASSED — SANITY +15', 'success');
    } else {
      this.player.modifySanity(-10);
      this.ui.showNotification('FIRE DRILL FAILED — SANITY -10', 'warning');
    }
    this.portals.portals.forEach((p, i) => {
      p.type = this.fireDrillOriginalPortals[i] || 0;
    });
  }

  activateTelecommute() {
    if (this.telecommuteActive || this.telecommuteInDimension) return;
    this.telecommuteActive = true;
    this.telecommuteInDimension = true;
    this.telecommuteTimer = 0;
    this.ui.showTelecommuteOverlay();
    this.ui.showNotification('TELECOMMUTING — HOME OFFICE DIMENSION', 'info');
    const monitorGeo = new THREE.BoxGeometry(1.5, 1, 0.1);
    const monitorMat = new THREE.MeshBasicMaterial({ color: 0x333333 });
    this.telecommuteMonitor = new THREE.Mesh(monitorGeo, monitorMat);
    this.telecommuteMonitor.position.copy(this.player.position).add(new THREE.Vector3(0, 1.5, 2));
    this.scene.scene.add(this.telecommuteMonitor);
  }

  destroyTelecommuteMonitor() {
    if (!this.telecommuteMonitor) return;
    this.scene.scene.remove(this.telecommuteMonitor);
    this.telecommuteMonitor = null;
    this.telecommuteInDimension = false;
    this.telecommuteActive = false;
    this.ui.hideTelecommuteOverlay();
    this.player.modifySanity(5);
    this.ui.showNotification('MONITOR DESTROYED — RETURNING HOME', 'success');
  }

  update(delta) {
    if (this.state !== 'playing') return;

    // Store position BEFORE movement for portal crossing detection
    this.player._prevPosition.copy(this.player.position);

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
    this.player.update(delta, this.world, this.portals.portals);

    // Update shadow system (portal echo)
    if (this.player && this.shadowSystem) {
      const pos = this.player.position;
      const cam = this.player.camera;
      const ghost = this.shadowSystem.update(
        pos.x,
        pos.y,
        pos.z,
        this.player.yaw,
        this.player.pitch,
        this.player.sanity,
        performance.now() / 1000
      );
      this.shadowSystem.updateGhostMesh(ghost);
    }

    // Check portal crossing
    this.portals.portals.forEach((portal) => {
      if (portal.checkPlayerCrossing(this.player._prevPosition, this.player.position)) {
        this.player.teleportThroughPortal(portal);
        this.audio.playSpatialSwoosh(portal.position);
      }
    });

    // Update spatial audio
    if (this.player) {
      const camPos = this.player.camera.position;
      this.audio.setListenerPosition(camPos.x, camPos.y, camPos.z);
      this.audio.updateSpatials();
    }

    // Update current room based on position
    let closestRoom = this.world.rooms[0];
    let minDist = Infinity;
    this.world.rooms.forEach((room) => {
      const roomCenter = room.group.position;
      const dist = this.player.position.distanceTo(roomCenter);
      if (dist < minDist) {
        minDist = dist;
        closestRoom = room;
      }
    });
    this.player.currentRoom = closestRoom;
    this.applyRoomTheme(closestRoom.id);

    // Pulsing fog — breathes with sanity
    if (this.scene.scene.fog && this.player) {
      const theme = this.getRoomTheme(closestRoom.id);
      const time = performance.now() / 1000;
      const sanity = this.player.sanity;
      const intensity = 1 - Math.max(0, Math.min(1, (sanity + 100) / 200));
      const pulseSpeed = 0.5 + intensity * 2.5;
      const pulseAmp = 0.02 + intensity * 0.06;
      this.scene.scene.fog.density = theme.fogDensity + Math.sin(time * pulseSpeed) * pulseAmp;
    }

    // Dynamic lighting — reacts to sanity
    if (this.scene.pointLights && this.player) {
      const sanity = this.player.sanity;
      const t = performance.now() / 1000;
      const critical = sanity < 20;
      const moderate = sanity < 50 && !critical;
      this.scene.pointLights.forEach((light, idx) => {
        if (critical) {
          light.visible = Math.sin(t * 3 + idx * 1.7) > -0.2;
          light.intensity = 0.2 + Math.sin(t * 6 + idx) * 0.15;
        } else if (moderate) {
          light.intensity = 0.4 + Math.sin(t * 2 + idx) * 0.1;
          light.visible = Math.random() > 0.03;
        } else {
          light.intensity = 0.8;
          light.visible = true;
        }
      });
    }

    // Phone ringing mechanic
    if (this.phoneRinging) {
      this.phoneRingDuration -= this.deltaTime;
      const prevFloor = Math.floor(this.phoneRingDuration + this.deltaTime);
      const currFloor = Math.floor(this.phoneRingDuration);
      if (this.phoneRingDuration <= 0) {
        this.player.modifySanity(-10);
        this.ui.showNotification('MISSED CALL — WRITTEN UP', 'warning');
        this.phoneRinging = false;
        this.phoneTimer = 45 + Math.random() * 30;
      } else if (prevFloor !== currFloor && this.phoneRingDuration < 4.5) {
        this.ui.showNotification('TELEPHONE RINGING', 'warning');
        this.audio.playPhoneRing();
      }
    } else {
      this.phoneTimer -= this.deltaTime;
      if (this.phoneTimer <= 0) {
        this.phoneRinging = true;
        this.phoneRingDuration = 5;
        this.ui.showNotification('TELEPHONE RINGING — GO ANSWER', 'warning');
        this.audio.playPhoneRing();
      }
    }

    // Coffee mini-game
    if (this.coffeeActive) {
      this.coffeeMeterValue += this.coffeeMeterDir * this.coffeeMeterSpeed * this.deltaTime;
      if (this.coffeeMeterValue >= 100) {
        this.coffeeMeterValue = 100;
        this.coffeeMeterDir = -1;
      }
      if (this.coffeeMeterValue <= 0) {
        this.coffeeMeterValue = 0;
        this.coffeeMeterDir = 1;
      }
      this.coffeeRoundTimer -= this.deltaTime;
      if (this.coffeeRoundTimer <= 0) {
        this.stopCoffeeMeter();
      }
      this.ui.updateCoffeeMeter(this.coffeeRound, this.coffeeMeterValue);
    } else {
      this.coffeeCooldown -= this.deltaTime;
    }

    // Paper airplane throw (T key)
    if (this.controls.keys['t'] && !this._throwKeyHeld && !this.paperScoutActive) {
      this._throwKeyHeld = true;
      this.throwPaperScout();
    }
    if (!this.controls.keys['t']) {
      this._throwKeyHeld = false;
    }

    // Paper airplane scout
    if (this.paperScoutActive && this.paperScoutMesh) {
      const forward = new THREE.Vector3();
      this.scene.camera.getWorldDirection(forward);
      this.paperScoutMesh.position.add(forward.multiplyScalar(12 * this.deltaTime));
      this.paperScoutTimer -= this.deltaTime;
      if (this.paperScoutTimer <= 0) {
        this.collectPaperScout();
      }
    }

    // Newspaper toggle (N key, single press)
    if (this.controls.keys['n'] && !this._newspaperKeyHeld) {
      this._newspaperKeyHeld = true;
      this.toggleNewspaper();
    }
    if (!this.controls.keys['n']) {
      this._newspaperKeyHeld = false;
    }

    // Fax machine
    if (this.faxActive) {
      this.faxTimer -= this.deltaTime;
      if (this.faxTimer <= 0) {
        this.failFax('TIME UP');
      }
      this.ui.updateFaxDisplay(this.faxDigits, this.faxInput);
      for (let d = 0; d <= 9; d++) {
        if (this.controls.keys[String(d)] && !this._faxDigitHeld[d]) {
          this._faxDigitHeld[d] = true;
          this.dialFaxDigit(d);
        }
        if (!this.controls.keys[String(d)]) {
          this._faxDigitHeld[d] = false;
        }
      }
    }

    // Photocopier Multiverse
    if (this.photocopierActive) {
      this.photocopierTimer -= this.deltaTime;
      this.ui.updatePhotocopierTimer(this.photocopierTimer);
      for (let k = 1; k <= 3; k++) {
        if (this.controls.keys[String(k)] && !this._photocopierKeyHeld[k - 1]) {
          this._photocopierKeyHeld[k - 1] = true;
          this.switchPhotocopierCopy(k - 1);
        }
        if (!this.controls.keys[String(k)]) {
          this._photocopierKeyHeld[k - 1] = false;
        }
      }
      if (this.photocopierCopies[this.photocopierCurrentCopy] && this.player) {
        this.player.position.copy(this.photocopierCopies[this.photocopierCurrentCopy].position);
      }
      if (this.photocopierTimer <= 0) {
        this.endPhotocopier();
      }
    }

    // Elevator cooldown
    if (this.elevatorCooldown > 0) this.elevatorCooldown -= this.deltaTime;

    // Water Cooler Gossip
    if (this.gossipActive) {
      this.gossipTimer -= this.deltaTime;
      if (this.gossipTimer <= 0) {
        this.gossipActive = false;
        this.gossipCooldown = 30;
        this.ui.hideGossipBoard();
      }
    } else {
      this.gossipCooldown -= this.deltaTime;
      if (this.gossipCooldown <= 0 && this.player) {
        const speed = this.player.velocity.length();
        if (speed > 2) {
          this.startGossip();
        }
      }
    }

    // Employee of the Month cooldown
    if (this.employeeCooldown > 0) this.employeeCooldown -= this.deltaTime;

    // Time Clock Punch
    if (this.timeClockPunching) {
      this.timeClockTimer -= this.deltaTime;
      this.ui.updateTimeClockTimer(this.timeClockTimer);
      if (this.controls.keys['e'] && !this._timeClockKeyHeld) {
        this._timeClockKeyHeld = true;
        this.punchTimeClock();
      }
    } else {
      this._timeClockKeyHeld = false;
      this.timeClockPromptTimer -= this.deltaTime;
      if (this.timeClockPromptTimer <= 0) {
        this.timeClockPromptTimer = 300 + Math.random() * 120;
        if (this.timeClockMisses < 3) {
          this.activateTimeClock();
        }
      }
    }

    // Lunch Hour at Void
    if (this.lunchActive) {
      this.lunchTimer -= this.deltaTime;
      this.ui.updateLunchTimer(this.lunchTimer, this.lunchEaten);
      if (this.lunchTimer <= 0) {
        this.endLunchHour();
      }
    }

    // Quarterly Review - key answers
    if (this.reviewActive) {
      for (let k = 1; k <= 4; k++) {
        if (this.controls.keys[String(k)] && !this._reviewKeyHeld[k]) {
          this._reviewKeyHeld[k] = true;
          this.answerReview(k - 1);
        }
        if (!this.controls.keys[String(k)]) {
          this._reviewKeyHeld[k] = false;
        }
      }
    } else {
      this._reviewKeyHeld = [false, false, false, false, false];
    }

    // Office Fire Drill
    if (this.fireDrillActive) {
      this.fireDrillTimer -= this.deltaTime;
      this.fireDrillDoorsSealed = 0;
      this.portals.portals.forEach((p) => {
        if (p.type === 10) this.fireDrillDoorsSealed++;
      });
      this.ui.updateFireDrill(this.fireDrillTimer, this.fireDrillDoorsSealed);
      if (
        this.player.currentRoom.id === 11 &&
        this.player.position.distanceTo(this._exitDeskPos) < 2.0
      ) {
        this.endFireDrill();
      }
      if (this.fireDrillTimer <= 0) {
        this.endFireDrill();
      }
    }

    // Telecommuting Dimension
    if (this.telecommuteActive && this.telecommuteInDimension) {
      this.telecommuteTimer += this.deltaTime;
      this.effects.applyDistortions();
    }

    // Check void proximity for audio/visual effects
    this.portals.portals.forEach((portal) => {
      if (portal.type === 10) {
        // VOID
        const distToVoid = this.player.position.distanceTo(portal.position);
        if (distToVoid < 5) {
          const intensity = 1 - distToVoid / 5;
          this.audio.playVoidRumble(intensity);
          this.player.modifySanity(-intensity * 0.5);
        }
      }
    });

    // Update NPCs
    this.npcs.update(delta, this.player);

    // Update items
    this.items.update(delta, this.player.position);
    this.items.items.forEach((item) => {
      if (this.player.position.distanceTo(item.position) < 2.0) {
        this.player.collectItem(item);
        this.audio.playSpatialPickup(item.position);
        this.particles.emitBurst(item.position, 8, 'spark');
        this.items.collectItem(item);
        this.puzzle.addItemToInventory(item);
        if (item.type === 'note') this.spawnNoteMarker(item.position);
      }
    });

    // Update particles
    this.particles.update(delta);

    // Update audio
    if (this.player.onGround && this.player.velocity.lengthSq() > 0.01) {
      this.audio.playFootstep();
    }
    const now = performance.now();
    if (now - this._lastAmbienceTime > 2) {
      this.audio.playAmbience('office');
      this._lastAmbienceTime = now;
    }

    // Update effects
    this.effects.updateSanity(this.player.sanity);
    this.effects.applyDistortions();

    // Update bloom with sanity level
    if (this.scene.postProcessing && this.player) {
      const sanity = this.player.sanity;
      const intensity = sanity < 0 ? Math.min(1.0, Math.abs(sanity) / 100) : 0.3;
      this.scene.postProcessing.bloomPass.strength = intensity;
      this.scene.postProcessing.bokehPass.uniforms.focus.value = 15.0;
    }

    // Update HUD
    this.hud.updateBattery(this.player.flashlightBattery);
    this.hud.updateInventory(this.puzzle.inventory);
    this.hud.updateObjectives(this.puzzle);
    this.hud.updateMap(this.player.currentRoom);

    // Check win/loss conditions
    if (this.player.sanity <= -100) {
      this.endGame(
        'ASSIMILATED INTO BUREAUCRACY\nYour consciousness merges with the filing systems.'
      );
      return;
    }

    if (
      this.npcs.auditor &&
      this.npcs.auditor.isAttacking &&
      this.player.position.distanceTo(this.npcs.auditor.position) < 1.5
    ) {
      this.endGame('AUDITED\nYour employment and soul have been permanently revoked.');
      return;
    }

    // Check exit condition
    if (this.player.currentRoom.id === 11) {
      const exitDeskPos = this._exitDeskPos;
      if (this.player.position.distanceTo(exitDeskPos) < 2.0) {
        if (this.controls.keys['e'] && !this.exitTriggered) {
          this.exitTriggered = true;
          if (this.formOpen) {
            this.submitExitForm();
          } else {
            this.openExitForm();
          }
        }
      } else {
        this.exitTriggered = false;
        if (this.formOpen) this.closeExitForm();
      }
    }

    // Check secret ending (8 sticky notes)
    if (this.puzzle.hasSecretEnding() && this.player.currentRoom.id === 0) {
      // Reveal secret portal to supply closet (Portal #0 becomes CORRECT)
      if (this.portals.portals[0].type !== 11) {
        this.portals.portals[0].type = 11;
        this.ui.showNotification("PORTAL 0 RECALIBRATED - SEEK GREG'S CLOSET", 'success');
      }
    }
  }

  saveGame() {
    if (!this.player) return false;
    const data = this.saveSystem.serialize(this);
    const ok = this.saveSystem.save(data);
    if (ok) {
      this.ui.showNotification('SAVED', 'success');
    } else {
      this.ui.showNotification('SAVE FAILED', 'warning');
    }
    return ok;
  }

  loadGame() {
    const data = this.saveSystem.load();
    if (!data) {
      this.ui.showNotification('NO SAVE FOUND', 'warning');
      return false;
    }
    const ok = this.saveSystem.deserialize(this, data);
    if (ok) {
      this.hud.updateSanity(this.player.sanity);
      this.hud.updateBattery(this.player.flashlightBattery);
      this.hud.updateInventory(this.puzzle.inventory);
      this.hud.updateObjectives(this.puzzle);
      this.ui.showNotification('LOADED', 'success');
    }
    return ok;
  }

  spawnNoteMarker(worldPosition) {
    const marker = new THREE.Mesh(
      new THREE.CircleGeometry(0.15, 16),
      new THREE.MeshBasicMaterial({
        color: 0xffff00,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide
      })
    );
    marker.position.set(worldPosition.x, 0.02, worldPosition.z);
    marker.rotation.x = -Math.PI / 2;
    this.scene.scene.add(marker);
    this.noteMarkers.push(marker);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const now = performance.now();
    this.deltaTime = (now - this.lastFrameTime) / 1000;
    this.lastFrameTime = now;

    // Clamp delta to prevent large jumps
    if (this.deltaTime > 0.05) this.deltaTime = 0.05;

    if (this.state === 'playing') {
      this.update(this.deltaTime);
    }

    if (this.scene.postProcessing) {
      this.scene.postProcessing.composer.render();
    } else {
      this.scene.renderer.render(this.scene.scene, this.scene.camera);
    }
  }
}

// Global game instance
window.game = null;

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  window.game = new GameController();

  // Resume audio context on any interaction
  ['click', 'keydown', 'touchstart'].forEach((event) => {
    document.addEventListener(
      event,
      () => {
        if (window.game && window.game.audio) {
          window.game.audio.resume();
        }
      },
      { once: true }
    );
  });
});
