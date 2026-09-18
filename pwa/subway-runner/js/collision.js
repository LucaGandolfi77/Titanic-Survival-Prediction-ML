import * as THREE from 'three';

export class CollisionSystem {
  constructor() {
    this._playerBox = new THREE.Box3();
    this._objBox = new THREE.Box3();
  }

  checkPlayerObstacle(player, obstacles) {
    if (player.isDead || player.isInvincible) return null;
    const pBox = player.getHitbox();

    for (const obs of obstacles) {
      this._objBox.setFromObject(obs);
      if (pBox.intersectsBox(this._objBox)) {
        return obs;
      }
    }
    return null;
  }

  checkPlayerCoins(player, collectibleManager) {
    if (player.isDead) return [];
    const pBox = player.getHitbox();
    const collected = [];

    for (const coin of collectibleManager.coins) {
      if (coin.userData.collected) continue;
      this._objBox.setFromObject(coin);
      if (pBox.intersectsBox(this._objBox)) {
        collected.push(coin);
      }
    }
    return collected;
  }

  checkPlayerPowerups(player, collectibleManager) {
    if (player.isDead) return [];
    const pBox = player.getHitbox();
    const collected = [];

    for (const pu of collectibleManager.powerups) {
      if (pu.userData.collected) continue;
      this._objBox.setFromObject(pu);
      if (pBox.intersectsBox(this._objBox)) {
        collected.push(pu);
      }
    }
    return collected;
  }
}
