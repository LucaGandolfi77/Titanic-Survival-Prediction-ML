import * as THREE from 'three';
import { MathUtils, createBox, createSphere, createCylinder } from './utils.js';
import { LANE_POSITIONS, LANE_WIDTH } from './track.js';

const GRAVITY = -35;
const JUMP_FORCE = 14;
const SLIDE_DURATION = 0.6;
const LANE_SWITCH_SPEED = 12;
const PLAYER_Y_BASE = 0.9;
const PLAYER_HEIGHT = 1.8;

export class Player {
  constructor(scene, characterDef) {
    this.scene = scene;
    this.characterDef = characterDef;

    this.currentLane = 0;
    this.targetLane = 0;
    this.worldX = 0;
    this.worldY = PLAYER_Y_BASE;
    this.worldZ = 0;

    this.velocityY = 0;
    this.isGrounded = true;
    this.isJumping = false;
    this.isSliding = false;
    this.slideTimer = 0;
    this.isDead = false;
    this.isInvincible = false;
    this.invincibleTimer = 0;

    this.runCycle = 0;
    this.mesh = this.buildMesh();
    scene.add(this.mesh);

    this.hitbox = new THREE.Box3();
  }

  buildMesh() {
    const group = new THREE.Group();
    const bodyColor = this.characterDef?.color || 0x2196f3;
    const skinColor = 0xfdd9b5;

    const torso = createBox(0.6, 0.7, 0.4, bodyColor);
    torso.position.y = 0.95;
    torso.castShadow = true;
    group.add(torso);
    this.torso = torso;

    const head = createSphere(0.25, skinColor);
    head.position.y = 1.55;
    head.castShadow = true;
    group.add(head);

    const eyeGeo = new THREE.SphereGeometry(0.04, 6, 6);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0x2c3e50 });
    const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
    leftEye.position.set(-0.08, 1.58, 0.2);
    group.add(leftEye);
    const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
    rightEye.position.set(0.08, 1.58, 0.2);
    group.add(rightEye);

    const hatColor = this.characterDef?.hatColor || 0xff9800;
    const hatBrim = createCylinder(0.28, 0.28, 0.04, hatColor, 8);
    hatBrim.position.y = 1.76;
    group.add(hatBrim);
    const hatTop = createCylinder(0.18, 0.2, 0.15, hatColor, 8);
    hatTop.position.y = 1.84;
    group.add(hatTop);

    const legGeo = new THREE.BoxGeometry(0.2, 0.5, 0.25);
    const legMat = new THREE.MeshStandardMaterial({ color: 0x1a237e });
    this.leftLeg = new THREE.Mesh(legGeo, legMat);
    this.leftLeg.position.set(-0.15, 0.35, 0);
    this.leftLeg.castShadow = true;
    group.add(this.leftLeg);

    this.rightLeg = new THREE.Mesh(legGeo, legMat);
    this.rightLeg.position.set(0.15, 0.35, 0);
    this.rightLeg.castShadow = true;
    group.add(this.rightLeg);

    const armGeo = new THREE.BoxGeometry(0.15, 0.5, 0.2);
    const armMat = new THREE.MeshStandardMaterial({ color: bodyColor });
    this.leftArm = new THREE.Mesh(armGeo, armMat);
    this.leftArm.position.set(-0.45, 0.9, 0);
    this.leftArm.castShadow = true;
    group.add(this.leftArm);

    this.rightArm = new THREE.Mesh(armGeo, armMat);
    this.rightArm.position.set(0.45, 0.9, 0);
    this.rightArm.castShadow = true;
    group.add(this.rightArm);

    this.mesh = group;
    return group;
  }

  switchLane(direction) {
    const newLane = MathUtils.clamp(this.targetLane + direction, -1, 1);
    if (newLane !== this.targetLane) {
      this.targetLane = newLane;
    }
  }

  jump() {
    if (this.isGrounded && !this.isDead) {
      this.velocityY = JUMP_FORCE;
      this.isGrounded = false;
      this.isJumping = true;
      this.isSliding = false;
      this.slideTimer = 0;
      return true;
    }
    return false;
  }

  slide() {
    if (this.isGrounded && !this.isDead) {
      this.isSliding = true;
      this.slideTimer = SLIDE_DURATION;
      this.isJumping = false;
      return true;
    }
    return false;
  }

  die() {
    if (this.isDead) return;
    this.isDead = true;
  }

  setInvincible(duration) {
    this.isInvincible = true;
    this.invincibleTimer = duration;
  }

  getHitbox() {
    const hw = 0.35;
    const hh = this.isSliding ? 0.5 : PLAYER_HEIGHT * 0.5;
    const hd = 0.25;
    this.hitbox.set(
      new THREE.Vector3(this.worldX - hw, this.worldY - PLAYER_Y_BASE + 0.05, this.worldZ - hd),
      new THREE.Vector3(this.worldX + hw, this.worldY - PLAYER_Y_BASE + 0.05 + hh * 2, this.worldZ + hd)
    );
    return this.hitbox;
  }

  update(delta, gameSpeed) {
    if (this.isDead) {
      this.worldY -= 8 * delta;
      this.mesh.position.y = this.worldY;
      this.mesh.rotation.x += 3 * delta;
      return;
    }

    this.worldZ -= gameSpeed * delta;

    const targetX = LANE_POSITIONS[this.targetLane + 1];
    this.worldX = MathUtils.smoothDamp(this.worldX, targetX, 0, 0.08, 100, delta).value;
    if (Math.abs(this.worldX - targetX) < 0.05) {
      this.worldX = targetX;
      this.currentLane = this.targetLane;
    }

    if (this.isInvincible) {
      this.invincibleTimer -= delta;
      if (this.invincibleTimer <= 0) {
        this.isInvincible = false;
      }
      this.mesh.visible = Math.floor(this.invincibleTimer * 10) % 2 === 0;
    } else {
      this.mesh.visible = true;
    }

    if (!this.isGrounded) {
      this.velocityY += GRAVITY * delta;
      this.worldY += this.velocityY * delta;
      if (this.worldY <= PLAYER_Y_BASE) {
        this.worldY = PLAYER_Y_BASE;
        this.velocityY = 0;
        this.isGrounded = true;
        this.isJumping = false;
      }
    }

    if (this.isSliding) {
      this.slideTimer -= delta;
      if (this.slideTimer <= 0) {
        this.isSliding = false;
      }
    }

    this.runCycle += delta * gameSpeed * 0.5;

    this.mesh.position.set(this.worldX, this.worldY, this.worldZ);

    if (this.isSliding) {
      this.mesh.scale.y = 0.5;
      this.mesh.position.y = this.worldY - 0.3;
    } else {
      this.mesh.scale.y = 1;
    }

    const legSwing = Math.sin(this.runCycle) * 0.5;
    if (this.leftLeg && this.rightLeg) {
      this.leftLeg.rotation.x = legSwing;
      this.rightLeg.rotation.x = -legSwing;
    }
    if (this.leftArm && this.rightArm) {
      this.leftArm.rotation.x = -legSwing;
      this.rightArm.rotation.x = legSwing;
    }
  }

  reset() {
    this.currentLane = 0;
    this.targetLane = 0;
    this.worldX = 0;
    this.worldY = PLAYER_Y_BASE;
    this.worldZ = 0;
    this.velocityY = 0;
    this.isGrounded = true;
    this.isJumping = false;
    this.isSliding = false;
    this.slideTimer = 0;
    this.isDead = false;
    this.isInvincible = false;
    this.invincibleTimer = 0;
    this.runCycle = 0;
    this.mesh.position.set(0, PLAYER_Y_BASE, 0);
    this.mesh.rotation.set(0, 0, 0);
    this.mesh.scale.set(1, 1, 1);
    this.mesh.visible = true;
  }
}

export { PLAYER_Y_BASE, PLAYER_HEIGHT };
