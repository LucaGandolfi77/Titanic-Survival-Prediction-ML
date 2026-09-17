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
import { PostProcessingManager }   from './postprocessing.js';
import { PowerUpManager, POWERUP_TYPES } from './powerups.js';

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
    this.postProcess = null;
    this.powerups  = null;

    // Wave transition timer
    this._waveDelay = 0;
    this._muzzleFlashCD = 0;
    this._smokeCD = 0;

    // Day/Night cycle
    this._worldTime = 0;
    this._cycleLength = 120;

    // Power-up state
    this._activePowerUps = {};
    this._powerUpExpiry = {};
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
    this.postProcess = new PostProcessingManager(renderer, scene, camera);
    this.powerups  = new PowerUpManager(scene);

    // Resize handler (post-processing aware)
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      if (this.postProcess) this.postProcess.resize(window.innerWidth, window.innerHeight);
    });

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

  /* ── Apply active power-up effects to drone ── */
  _applyPowerUpEffects() {
    const now = performance.now();
    const drone = this.drone;
    const active = {};

    for (const type in this._activePowerUps) {
      if (now < this._powerUpExpiry[type]) {
        active[type] = this._activePowerUps[type];
      }
    }
    this._activePowerUps = active;

    // Reset drone base values
    drone.maxSpeed = 60;
    drone.maxThrust = 25;
    drone._shieldMesh = null;

    // Apply effects
    for (const type in active) {
      switch (type) {
        case 'overdrive':
          drone.maxSpeed = 90;
          drone.maxThrust = 40;
          break;
        case 'shield':
          if (!drone._shieldMesh) {
            const geo = new THREE.IcosahedronGeometry(3, 1);
            const mat = new THREE.MeshBasicMaterial({
              color: 0x4488ff,
              transparent: true,
              opacity: 0.2,
              wireframe: true,
            });
            drone._shieldMesh = new THREE.Mesh(geo, mat);
            drone.scene.add(drone._shieldMesh);
          }
          break;
        case 'ghost':
          drone.mesh.traverse(c => {
            if (c.isMesh && c.material) {
              if (!c.material._origOpacity) c.material._origOpacity = c.material.opacity;
              c.material.transparent = true;
              c.material.opacity = 0.15;
            }
          });
          break;
        case 'nanoRegen':
          drone.health = Math.min(drone.maxHealth, drone.health + 2 * (1/60));
          break;
      }
    }

    // Restore ghost materials when expired
    if (!active.ghost && drone._shieldMesh) {
      // shield stays if still active
    }
    // Clean up ghost transparency
    if (!active.ghost) {
      drone.mesh.traverse(c => {
        if (c.isMesh && c.material && c.material._origOpacity !== undefined) {
          c.material.opacity = c.material._origOpacity;
          c.material.transparent = c.material.opacity < 1;
          delete c.material._origOpacity;
        }
      });
    }
  }

  /* ── Reset power-up and game state ── */
  _resetState() {
    this._activePowerUps = {};
    this._powerUpExpiry = {};
    this._worldTime = 0;
    if (this.drone && this.drone._shieldMesh) {
      this.drone.scene.remove(this.drone._shieldMesh);
      this.drone._shieldMesh.geometry.dispose();
      this.drone._shieldMesh.material.dispose();
      this.drone._shieldMesh = null;
    }
    this.drone.mesh.traverse(c => {
      if (c.isMesh && c.material && c.material._origOpacity !== undefined) {
        c.material.opacity = c.material._origOpacity;
        c.material.transparent = c.material.opacity < 1;
        delete c.material._origOpacity;
      }
    });
  }

  /* ── Start / restart game ── */
  startGame() {
    this._resetState();
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
    if (navigator.vibrate) navigator.vibrate([50, 30, 80]);

    // Clean up shield
    if (this.drone && this.drone._shieldMesh) {
      this.drone.scene.remove(this.drone._shieldMesh);
      this.drone._shieldMesh.geometry.dispose();
      this.drone._shieldMesh.material.dispose();
      this.drone._shieldMesh = null;
    }

    // Explosion on drone position
    this.effects.spawnExplosion(this.drone.position.clone(), 2);
    this.audio.playExplosion();

    document.exitPointerLock?.();
    this.ui.showGameOver(this.score, this.wave);
  }

  /* ── Return to menu ── */
  goToMenu() {
    this._resetState();
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
      const t = performance.now() * 0.0001;
      this.camera.position.x = Math.sin(t) * 120;
      this.camera.position.z = Math.cos(t) * 120;
      this.camera.position.y = 60;
      this.camera.lookAt(0, 10, 0);
      this.postProcess.render();
      return;
    }

    if (this.state === State.PAUSED || this.state === State.WAVE_COMPLETE || this.state === State.GAME_OVER) {
      this.effects.update(delta || 0.016);
      this.postProcess.render();
      return;
    }

    if (this.state !== State.PLAYING || delta === 0) {
      this.postProcess.render();
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
    this._updateDayNight(delta);
    this._applyPowerUpEffects();
    this.drone.update(delta, input);
    this._checkPowerUpCollection();

    // ── 3. Fire weapons ──
    if (input.firePrimary) {
      const origin = this.drone.position.clone();
      const dir    = this.drone.getForward();
      if (this.weapons.firePrimary(origin, dir)) {
        this.audio.playGunshot();
        this.postProcess.boostBloom(0.3);
        setTimeout(() => this.postProcess.resetBloom(), 100);
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

    if (input.firePlasma) {
      const origin = this.drone.position.clone();
      const dir    = this.drone.getForward();
      if (this.weapons.firePlasma(origin, dir)) {
        this.audio.playMissileLaunch();
        this.postProcess.boostBloom(1.2);
        setTimeout(() => this.postProcess.resetBloom(), 300);
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
    this.postProcess.render(delta);
  }

  /* ── Day/Night cycle update ── */
  _updateDayNight(delta) {
    this._worldTime = (this._worldTime + delta) % this._cycleLength;
    const t = this._worldTime / this._cycleLength;

    let bgColor, fogColor, hemiIntensity, sunIntensity, sunColor;

    if (t < 0.33) {
      bgColor = new THREE.Color(0x1a1a2e);
      fogColor = new THREE.Color(0x1a1a2e);
      hemiIntensity = 0.4;
      sunIntensity = 1.5;
      sunColor = new THREE.Color(0xffffff);
    } else if (t < 0.5) {
      const f = (t - 0.33) / 0.17;
      bgColor = new THREE.Color().lerpColors(new THREE.Color(0x1a1a2e), new THREE.Color(0x2a1520), f);
      fogColor = new THREE.Color().lerpColors(new THREE.Color(0x1a1a2e), new THREE.Color(0x2a1520), f);
      hemiIntensity = 0.4 - f * 0.2;
      sunIntensity = 1.5 - f * 0.8;
      sunColor = new THREE.Color().lerpColors(new THREE.Color(0xffffff), new THREE.Color(0xff6633), f);
    } else if (t < 0.83) {
      bgColor = new THREE.Color(0x080818);
      fogColor = new THREE.Color(0x080818);
      hemiIntensity = 0.15;
      sunIntensity = 0.2;
      sunColor = new THREE.Color(0x4455aa);
    } else {
      const f = (t - 0.83) / 0.17;
      bgColor = new THREE.Color().lerpColors(new THREE.Color(0x080818), new THREE.Color(0x1a1a2e), f);
      fogColor = new THREE.Color().lerpColors(new THREE.Color(0x080818), new THREE.Color(0x1a1a2e), f);
      hemiIntensity = 0.15 + f * 0.25;
      sunIntensity = 0.2 + f * 1.3;
      sunColor = new THREE.Color().lerpColors(new THREE.Color(0x4455aa), new THREE.Color(0xffffff), f);
    }

    this.scene.background = bgColor;
    this.scene.fog = new THREE.FogExp2(fogColor.getHex(), 0.002 + (t > 0.5 ? 0.002 : 0));

    this.scene.traverse(child => {
      if (child.isHemisphereLight) child.intensity = hemiIntensity;
      if (child.isDirectionalLight && child.castShadow) {
        child.intensity = sunIntensity;
        child.color = sunColor;
      }
    });

    if (this.postProcess && t > 0.5 && t < 0.83) {
      this.postProcess.boostBloom(0.4);
    } else if (this.postProcess) {
      this.postProcess.resetBloom();
    }
  }

  /* ── Collect power-ups ── */
  _checkPowerUpCollection() {
    for (const pu of this.powerups.active) {
      const dist = pu.position.distanceTo(this.drone.position);
      if (dist < 4) {
        const type = this.powerups.collect(pu);
        if (type) {
          this._activePowerUps[type] = true;
          this._powerUpExpiry[type] = performance.now() + (1000 * 60 * POWERUP_TYPES[type].duration);
          this.audio.playGunshot();
          if (navigator.vibrate) navigator.vibrate([10, 20, 10]);
        }
      }
    }
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
      if (navigator.vibrate) navigator.vibrate(20);
    }

    // Drone vs ground
    const groundHit = this.collision.checkDroneVsGround(this.drone.position);
    if (groundHit) {
      if (this.drone.velocity.y < -15) {
          this.drone.takeDamage(20);
          this.hud.flashDamage();
          this.audio.playDamage();
          if (navigator.vibrate) navigator.vibrate(50);
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
        const explosionScale = hit.enemy.type === 'boss' ? 3 : 1.5;
        this.effects.spawnExplosion(hit.enemy.position.clone(), explosionScale);
        this.audio.playExplosion();
        this.effects.spawnSmoke(hit.enemy.position.clone());
        if (navigator.vibrate) navigator.vibrate(30);
        if (!hit.enemy._removing) {
          hit.enemy._removing = true;
          setTimeout(() => {
            hit.enemy._removing = false;
            this.enemies.removeEnemy(hit.enemy);
          }, 50);
        }
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
      if (navigator.vibrate) navigator.vibrate(15);
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
    if (navigator.vibrate) navigator.vibrate([20, 20, 40, 20, 60]);
    document.exitPointerLock?.();
    this.ui.showWaveComplete(this.wave, bonus);
  }
}

/* ══════════════════════════════════════════════════════════
    Bootstrap
    ══════════════════════════════════════════════════════════ */
window.addEventListener('beforeunload', () => {
  if (window.__droneGame && window.__droneGame.audio) {
    window.__droneGame.audio.dispose();
  }
});

window.addEventListener('DOMContentLoaded', () => {
  const game = new Game();
  window.__droneGame = game;
  game.init();
});
