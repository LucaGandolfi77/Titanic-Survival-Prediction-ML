/* ── js/main.js ── Game loop, state management, system wiring ── */

import { initScene, initLights }   from './scene.js';
const THREE = globalThis.THREE;
import { buildWorld }              from './world.js';
import { Drone }                   from './drone.js';
import { InputController }         from './controls.js';
import { EnemyManager }            from './enemies.js';
import { WeaponSystem }            from './weapons.js';
import { CollisionSystem }         from './collision.js';
import { EffectsManager }          from './effects.js';
import { AudioManager }            from './audio.js';
import { HUD }                     from './hud.js';
import { UIManager }               from './ui.js';

/* ══════════════════════════════════════════════════════════
   Game state enum
   ══════════════════════════════════════════════════════════ */
const State = {
  MENU:          'menu',
  PLAYING:       'playing',
  PAUSED:        'paused',
  WAVE_COMPLETE: 'wave_complete',
  GAME_OVER:     'game_over',
};

/* ══════════════════════════════════════════════════════════
   Main Game class
   ══════════════════════════════════════════════════════════ */
class Game {
  constructor() {
    this.state = State.MENU;
    this.score = 0;
    this.wave  = 0;
    this.clock = new THREE.Clock(false);

    // Systems (initialized in init())
    this.renderer  = null;
    this.scene     = null;
    this.camera    = null;
    this.drone     = null;
    this.controls  = null;
    this.enemies   = null;
    this.weapons   = null;
    this.collision = null;
    this.effects   = null;
    this.audio     = null;
    this.hud       = null;
    this.ui        = null;
    this.buildings = [];

    // Wave transition timer
    this._waveDelay = 0;
    this._muzzleFlashCD = 0;
    this._smokeCD = 0;
  }

  /* ── Initialize everything ── */
  init() {
    // Three.js scene
    const { renderer, scene, camera } = initScene();
    this.renderer = renderer;
    this.scene    = scene;
    this.camera   = camera;
    initLights(scene);

    // World
    const world    = buildWorld(scene);
    this.buildings = world.buildings;

    // Systems
    this.drone     = new Drone(scene, camera);
    this.controls  = new InputController(renderer.domElement);
    this.enemies   = new EnemyManager(scene);
    this.weapons   = new WeaponSystem(scene);
    this.collision = new CollisionSystem(this.buildings);
    this.effects   = new EffectsManager(scene);
    this.audio     = new AudioManager();
    this.hud       = new HUD();
    this.ui        = new UIManager();

    // Wire UI callbacks
    this.ui.onStart    = () => this.startGame();
    this.ui.onResume   = () => this.resume();
    this.ui.onRestart  = () => this.startGame();
    this.ui.onNextWave = () => this.nextWave();
    this.ui.onMainMenu = () => this.goToMenu();

    // Show menu
    this.ui.showMenu();
    this.hud.hide();

    // Start render loop (always runs for background animation)
    this._loop();
  }

  /* ── Start / restart game ── */
  startGame() {
    this.audio.init();
    this.audio.resume();

    this.score = 0;
    this.wave  = 0;

    this.drone.reset();
    this.weapons.reset();
    this.enemies.clearAll();
    this.effects.clear();

    this.state = State.PLAYING;
    this.ui.showGame();
    this.hud.show();
    this.clock.start();

    // Request pointer lock for desktop
    this.controls.requestPointerLock();

    // Start wave 1
    this.nextWave();
  }

  /* ── Next wave ── */
  nextWave() {
    this.wave++;
    this.enemies.spawnWave(this.wave, this.drone.position);
    this.hud.showWaveBanner(this.wave);
    this.state = State.PLAYING;
    this.ui.showGame();
    this.hud.show();
    this.audio.resume();
  }

  /* ── Pause / resume ── */
  pause() {
    if (this.state !== State.PLAYING) return;
    this.state = State.PAUSED;
    this.clock.stop();
    this.ui.showPause();
    document.exitPointerLock?.();
  }

  resume() {
    if (this.state !== State.PAUSED) return;
    this.state = State.PLAYING;
    this.clock.start();
    this.ui.hidePause();
    this.controls.requestPointerLock();
  }

  /* ── Game over ── */
  gameOver() {
    this.state = State.GAME_OVER;
    this.clock.stop();
    this.hud.hide();

    // Explosion on drone position
    this.effects.spawnExplosion(this.drone.position.clone(), 2);
    this.audio.playExplosion();

    document.exitPointerLock?.();
    this.ui.showGameOver(this.score, this.wave);
  }

  /* ── Return to menu ── */
  goToMenu() {
    this.state = State.MENU;
    this.clock.stop();
    this.enemies.clearAll();
    this.effects.clear();
    this.weapons.reset();
    this.hud.hide();
    this.ui.showMenu();
    document.exitPointerLock?.();

    // Reset camera for menu backdrop
    this.camera.position.set(0, 50, 100);
    this.camera.lookAt(0, 10, 0);
  }

  /* ═══════════════════════════════════════════════════════
     GAME LOOP
     ═══════════════════════════════════════════════════════ */
  _loop() {
    requestAnimationFrame(() => this._loop());

    const delta = Math.min(this.clock.getDelta(), 0.05); // cap at 50ms

    if (this.state === State.MENU) {
      // Gentle camera orbit for menu backdrop
      const t = performance.now() * 0.0001;
      this.camera.position.x = Math.sin(t) * 120;
      this.camera.position.z = Math.cos(t) * 120;
      this.camera.position.y = 60;
      this.camera.lookAt(0, 10, 0);
      this.renderer.render(this.scene, this.camera);
      return;
    }

    if (this.state === State.PAUSED || this.state === State.WAVE_COMPLETE || this.state === State.GAME_OVER) {
      // Still render (for effects fade-out)
      this.effects.update(delta || 0.016);
      this.renderer.render(this.scene, this.camera);
      return;
    }

    if (this.state !== State.PLAYING || delta === 0) {
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // ── 1. Input ──
    const input = this.controls.getInput();

    // Pause check
    if (input.pause) {
      this.pause();
      return;
    }

    // ── 2. Update drone ──
    this.drone.update(delta, input);

    // ── 3. Fire weapons ──
    if (input.firePrimary) {
      const origin = this.drone.position.clone();
      const dir    = this.drone.getForward();
      if (this.weapons.firePrimary(origin, dir)) {
        this.audio.playGunshot();
        // Muzzle flash (throttled)
        this._muzzleFlashCD -= delta;
        if (this._muzzleFlashCD <= 0) {
          this.effects.spawnMuzzleFlash(
            origin.clone().addScaledVector(dir, 1.5),
            dir
          );
          this._muzzleFlashCD = 0.08;
        }
      }
    }

    if (input.fireMissile) {
      const origin = this.drone.position.clone();
      const dir    = this.drone.getForward();
      if (this.weapons.fireMissile(origin, dir, this.enemies.allEnemies)) {
        this.audio.playMissileLaunch();
      }
    }

    // Boost sparks
    if (input.boost && this.drone.throttle > 0.3) {
      this._smokeCD -= delta;
      if (this._smokeCD <= 0) {
        this.effects.spawnSparks(
          this.drone.position.clone(),
          this.drone.getForward()
        );
        this._smokeCD = 0.05;
      }
    }

    // ── 4. Update systems ──
    this.weapons.update(delta);
    this.enemies.update(delta, this.drone.position);
    this.effects.update(delta);
    this.audio.updateEngine(this.drone.throttle);

    // ── 5. Collision detection ──
    this._handleCollisions();

    // ── 6. Check wave end ──
    if (this.enemies.aliveCount === 0 && this.wave > 0) {
      this._onWaveCleared();
    }

    // ── 7. Check game over ──
    if (!this.drone.isAlive) {
      this.gameOver();
      return;
    }

    // ── 8. Update HUD ──
    this.hud.update(this.drone, this.weapons, this.enemies, this.score, this.wave);

    // ── 9. Render ──
    this.renderer.render(this.scene, this.camera);
  }

  /* ── Collision handling ── */
  _handleCollisions() {
    // Drone vs buildings
    const buildingHit = this.collision.checkDroneVsBuildings(this.drone.position, 1.5);
    if (buildingHit) {
      this.drone.pushOut(buildingHit.normal, buildingHit.depth);
      this.drone.takeDamage(5 * (buildingHit.depth > 0.5 ? 2 : 1));
      this.hud.flashDamage();
      this.audio.playDamage();
    }

    // Drone vs ground
    const groundHit = this.collision.checkDroneVsGround(this.drone.position);
    if (groundHit) {
      if (this.drone.velocity.y < -15) {
        this.drone.takeDamage(20);
        this.hud.flashDamage();
        this.audio.playDamage();
      }
    }

    // Player projectiles vs enemies
    const allPlayerProj = this.weapons.getAllProjectiles();
    const hits = this.collision.checkProjectileVsEnemies(allPlayerProj, this.enemies.allEnemies);
    for (const hit of hits) {
      // Damage enemy
      hit.enemy.takeDamage(hit.projectile.damage);
      this.audio.playHit();

      // Remove projectile
      this.weapons.removeProjectile(hit.projectile);

      // Spawn hit effect
      this.effects.spawnExplosion(hit.projectile.mesh.position.clone(), 0.3);

      // If enemy died
      if (!hit.enemy.alive) {
        this.score += hit.enemy.scoreValue;
        this.effects.spawnExplosion(hit.enemy.position.clone(), 1.5);
        this.audio.playExplosion();
        this.effects.spawnSmoke(hit.enemy.position.clone());
        // Defer removal to avoid array mutation during iteration
        setTimeout(() => this.enemies.removeEnemy(hit.enemy), 50);
      }
    }

    // Enemy projectiles vs drone
    const enemyProj = this.enemies.getEnemyProjectiles();
    const droneHits = this.collision.checkEnemyProjectileVsDrone(
      enemyProj, this.drone.position, 1.5
    );
    for (const proj of droneHits) {
      this.drone.takeDamage(proj.damage);
      this.hud.flashDamage();
      this.audio.playDamage();
      this.effects.spawnExplosion(proj.mesh.position.clone(), 0.4);
      // Release projectile back to pool
      this.enemies.projectilePool.release(proj);
    }
  }

  /* ── Wave cleared ── */
  _onWaveCleared() {
    const bonus = this.wave * 100;
    this.score += bonus;
    this.state = State.WAVE_COMPLETE;
    this.clock.stop();
    this.audio.playWaveComplete();
    document.exitPointerLock?.();
    this.ui.showWaveComplete(this.wave, bonus);
  }
}

/* ══════════════════════════════════════════════════════════
   Bootstrap
   ══════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  const game = new Game();
  game.init();
});
