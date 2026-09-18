import * as THREE from 'three';
import { initScene, getScene, getCamera, getRenderer } from './scene.js';
import { Track } from './track.js';
import { Player } from './player.js';
import { InputController } from './controls.js';
import { ObstacleManager } from './obstacles.js';
import { CollectibleManager } from './collectibles.js';
import { CollisionSystem } from './collision.js';
import { EffectsManager } from './effects.js';
import { AudioManager } from './audio.js';
import { HUD } from './hud.js';
import { UIManager } from './ui.js';
import { CharacterManager } from './characters.js';

const GameState = { MENU: 0, PLAYING: 1, PAUSED: 2, GAME_OVER: 3 };

class Game {
  constructor() {
    this.state = GameState.MENU;
    this.score = 0;
    this.coins = 0;
    this.distance = 0;
    this.gameSpeed = 12;
    this.baseSpeed = 12;
    this.maxSpeed = 35;
    this.speedIncrement = 0.3;
    this.lastSpeedScore = 0;
    this.combo = 0;
    this.lastCoinTime = 0;
    this.comboTimeout = 800;
    this.difficulty = 1;
    this.lastFrameTime = 0;
    this.clock = new THREE.Clock();
    this.inputState = { left: false, right: false, jump: false, slide: false, pause: false };
    this._leftLatch = false;
    this._rightLatch = false;
    this._upLatch = false;
    this._downLatch = false;
    this._pauseLatch = false;
    this._wakeLock = null;
  }

  async init() {
    const canvas = document.getElementById('game-canvas');
    const { renderer, scene, camera } = initScene(canvas);

    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;

    this.track = new Track(scene);
    this.obstacles = new ObstacleManager(scene);
    this.collectibles = new CollectibleManager(scene);
    this.collision = new CollisionSystem();
    this.effects = new EffectsManager(scene);
    this.audio = new AudioManager();
    this.hud = new HUD();
    this.ui = new UIManager();
    this.characters = new CharacterManager();
    this.input = new InputController();

    this.player = new Player(scene, this.characters.getSelected());

    this._setupUI();
    this._setupWakeLock();
    this.ui.showScreen('menu');
    this.characters.renderSelection(document.getElementById('character-select'));

    window.addEventListener('beforeunload', () => this._cleanup());
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.state === GameState.PLAYING) {
        this.pause();
      }
    });

    this._loop();
  }

  _setupUI() {
    const $ = (id) => document.getElementById(id);

    $('btn-play').addEventListener('click', () => this.start());
    $('btn-leaderboard').addEventListener('click', () => {
      this.ui.renderLeaderboard();
      this.ui.showToast('Leaderboard');
    });

    $('btn-resume').addEventListener('click', () => this.resume());
    $('btn-restart-pause').addEventListener('click', () => { this.resume(); this.start(); });
    $('btn-quit').addEventListener('click', () => { this.resume(); this.backToMenu(); });

    $('btn-retry').addEventListener('click', () => this.start());
    $('btn-share').addEventListener('click', () => this.ui.shareScore(this.score, this.distance));
    $('btn-menu').addEventListener('click', () => this.backToMenu());
  }

  _setupWakeLock() {
    if ('wakeLock' in navigator) {
      navigator.wakeLock.request('screen').then(wl => {
        this._wakeLock = wl;
      }).catch(() => {});
    }
  }

  _releaseWakeLock() {
    if (this._wakeLock) {
      this._wakeLock.release().catch(() => {});
      this._wakeLock = null;
    }
  }

  start() {
    this.audio.init();
    this.audio.resume();

    this.state = GameState.PLAYING;
    this.score = 0;
    this.coins = 0;
    this.distance = 0;
    this.gameSpeed = this.baseSpeed;
    this.difficulty = 1;
    this.lastSpeedScore = 0;
    this.combo = 0;
    this.lastCoinTime = 0;

    const selectedChar = this.characters.getSelected();
    this.player = new Player(this.scene, selectedChar);
    this.track = new Track(this.scene);
    this.obstacles.reset();
    this.collectibles.reset();
    this.effects.reset();

    this.hud.hideAllScreens();
    this.hud.show();
    this.ui.hideAllScreens();
    this.clock.start();
    this.audio.startEngine();
    this._setupWakeLock();
  }

  pause() {
    if (this.state !== GameState.PLAYING) return;
    this.state = GameState.PAUSED;
    this.ui.showScreen('pause');
    this.clock.stop();
  }

  resume() {
    if (this.state !== GameState.PAUSED) return;
    this.state = GameState.PLAYING;
    this.ui.hideAllScreens();
    this.hud.show();
    this.clock.start();
  }

  gameOver() {
    if (this.state === GameState.GAME_OVER) return;
    this.state = GameState.GAME_OVER;
    this.audio.playGameOver();
    this.audio.stopEngine();
    this.hud.hide();
    this.effects.triggerScreenShake(0.5, 0.5);

    const isNew = this.ui.isHighScore(this.score);
    this.ui.saveHighScore(this.score, this.coins, this.distance);

    const newUnlocks = this.characters.checkUnlocks(this.score);
    if (newUnlocks.length > 0) {
      this.ui.showToast(`Unlocked: ${newUnlocks.map(c => c.name).join(', ')}!`);
    }

    this.ui.showGameOver(this.score, this.coins, this.distance, isNew);
    this._releaseWakeLock();

    if (navigator.vibrate) {
      navigator.vibrate([50, 30, 80]);
    }
  }

  backToMenu() {
    this.state = GameState.MENU;
    this.hud.hideAllScreens();
    this.ui.showScreen('menu');
    this.audio.stopEngine();
    this.characters.renderSelection(document.getElementById('character-select'));
    this._releaseWakeLock();
  }

  _processInput() {
    const raw = this.input.getInput();

    const latch = (val, latchKey) => {
      if (val && !this.inputState[latchKey]) {
        this.inputState[latchKey] = true;
        return true;
      }
      if (!val) this.inputState[latchKey] = false;
      return false;
    };

    if (latch(raw.left, '_leftLatch')) this.player.switchLane(-1);
    if (latch(raw.right, '_rightLatch')) this.player.switchLane(1);
    if (latch(raw.jump, '_upLatch')) {
      if (this.player.jump()) this.audio.playJump();
    }
    if (latch(raw.slide, '_downLatch')) {
      if (this.player.slide()) this.audio.playSlide();
    }

    if (this.input.consumePause()) {
      if (this.state === GameState.PLAYING) this.pause();
      else if (this.state === GameState.PAUSED) this.resume();
    }
  }

  _updateGameplay(delta) {
    const speedMult = this.characters.applyAbility(this.player, this.collectibles);

    if (this.collectibles.hasPowerup('jetpack')) {
      this.gameSpeed = this.maxSpeed * 1.2;
      this.player.worldY = Math.max(this.player.worldY, 6);
      if (this.player.worldY < 6) {
        this.player.velocityY = 10;
        this.player.isGrounded = false;
      }
    } else {
      this.gameSpeed = Math.min(this.maxSpeed, this.baseSpeed + this.speedIncrement * Math.floor(this.score / 1000));
    }

    this.gameSpeed *= speedMult;
    this.distance += this.gameSpeed * delta;
    this.score += this.gameSpeed * delta * (this.collectibles.hasPowerup('multiplier') ? 2 : 1);

    if (this.score - this.lastSpeedScore >= 1000) {
      this.difficulty += 0.1;
      this.lastSpeedScore = this.score;
    }

    this.player.update(delta, this.gameSpeed);
    this.track.update(this.player.worldZ);
    this.obstacles.update(this.player.worldZ, this.difficulty);
    this.collectibles.update(this.player.worldZ, this.player.worldX, this.player.mesh.position, delta);
    this.effects.update(delta);
    this.audio.updateEngine(this.gameSpeed);

    this._checkCollisions();

    if (this.collectibles.hasPowerup('shield') && this.player.isDead) {
      this.player.isDead = false;
      this.player.worldY = 0.9;
      this.player.velocityY = 0;
      this.player.isGrounded = true;
      this.player.mesh.rotation.set(0, 0, 0);
      delete this.collectibles.activePowerups['shield'];
    }

    if (this.combo > 0 && performance.now() - this.lastCoinTime > this.comboTimeout) {
      this.combo = 0;
    }
  }

  _checkCollisions() {
    const hitObstacle = this.collision.checkPlayerObstacle(this.player, this.obstacles.getObstacles());
    if (hitObstacle) {
      if (!this.player.isInvincible && !this.collectibles.hasPowerup('shield')) {
        this.player.die();
        this.effects.spawnHitEffect(this.player.mesh.position);
        this.effects.triggerScreenShake(0.8, 0.4);
        this.hud.flashDamage();
        this.audio.playHit();
        if (navigator.vibrate) navigator.vibrate([50, 30, 80]);
        setTimeout(() => this.gameOver(), 800);
        return;
      }
    }

    const hitCoins = this.collision.checkPlayerCoins(this.player, this.collectibles);
    for (const coin of hitCoins) {
      const value = this.collectibles.collectCoin(coin);
      if (value > 0) {
        this.coins += value;
        this.effects.spawnCoinBurst(coin.position);
        this.audio.playCoinCollect();

        this.combo++;
        this.lastCoinTime = performance.now();
        if (this.combo >= 5) {
          this.score += this.combo * 10;
          this.hud.showCombo(this.combo);
        }
        if (navigator.vibrate) navigator.vibrate(15);
      }
    }

    const hitPowerups = this.collision.checkPlayerPowerups(this.player, this.collectibles);
    for (const pu of hitPowerups) {
      const type = this.collectibles.collectPowerup(pu);
      if (type) {
        this.effects.spawnPowerupEffect(pu.position, 0x00e5ff);
        this.audio.playPowerup();

        if (type === 'shield') {
          this.player.setInvincible(0.1);
        }
        if (type === 'jetpack') {
          this.player.velocityY = 12;
          this.player.isGrounded = false;
        }

        this.ui.showToast(type.toUpperCase() + ' activated!');
        if (navigator.vibrate) navigator.vibrate([10, 20, 10]);
      }
    }
  }

  _updateCamera(delta) {
    const targetX = this.player.worldX * 0.3;
    const targetY = 10 + (this.player.worldY - 0.9) * 0.3;
    const targetZ = this.player.worldZ + 16;

    this.camera.position.x = THREE.MathUtils.lerp(this.camera.position.x, targetX, 3 * delta);
    this.camera.position.y = THREE.MathUtils.lerp(this.camera.position.y, targetY, 3 * delta);
    this.camera.position.z = THREE.MathUtils.lerp(this.camera.position.z, targetZ, 5 * delta);

    const lookZ = this.player.worldZ - 5;
    this.camera.lookAt(targetX * 0.5, 2, lookZ);

    const shake = this.effects.getShakeOffset();
    this.camera.position.x += shake.x;
    this.camera.position.y += shake.y;
  }

  _updateHUD() {
    this.hud.updateScore(this.score);
    this.hud.updateCoins(this.coins);
    this.hud.updateDistance(this.distance);
    this.hud.setMultiplier(this.collectibles.hasPowerup('multiplier') ? 2 : 1);
    this.hud.updatePowerups(this.collectibles.activePowerups);
  }

  _loop() {
    requestAnimationFrame(() => this._loop());

    const delta = Math.min(this.clock.getDelta(), 0.05);

    if (this.state === GameState.PLAYING) {
      this._processInput();
      this._updateGameplay(delta);
      this._updateCamera(delta);
      this._updateHUD();
    } else if (this.state === GameState.MENU) {
      const time = performance.now() * 0.0003;
      this.camera.position.x = Math.sin(time) * 8;
      this.camera.position.y = 10 + Math.sin(time * 0.5) * 2;
      this.camera.position.z = Math.cos(time) * 12;
      this.camera.lookAt(0, 2, 0);
    }

    if (this.state === GameState.PLAYING || this.state === GameState.MENU) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  _cleanup() {
    this.audio.dispose();
    this._releaseWakeLock();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const game = new Game();
  window.__subwayRunner = game;
  game.init();

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js')
      .then(() => console.log('SW registered'))
      .catch(err => console.warn('SW registration failed', err));
  }
});
