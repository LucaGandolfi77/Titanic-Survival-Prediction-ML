import * as THREE from 'three';
import { MathUtils, rotateVectorAroundAxis } from './utils.js';

export class Player {
  constructor(camera, world) {
    this.camera = camera;
    this.world = world;
    
    // Position & Physics
    this.position = new THREE.Vector3(0, 1.8, 0);
    this.velocity = new THREE.Vector3();
    this.acceleration = new THREE.Vector3();
    this.onGround = false;
    
    // Gravity (variable)
    this.gravityVector = new THREE.Vector3(0, -1, 0);
    this.targetGravityVector = new THREE.Vector3(0, -1, 0);
    this.gravityMagnitude = 9.8;
    
    // Controls
    this.keys = {};
    this.mouseDelta = { x: 0, y: 0 };
    this.yaw = 0;
    this.pitch = 0;
    
    // Movement params
    this.height = 1.8;
    this.radius = 0.3;
    this.walkSpeed = 5;
    this.runSpeed = 9;
    this.jumpForce = 7;
    this.mouseSensitivity = 1 / 500;
    
    // Sanity & State
    this.sanity = 100;
    this.flashlightBattery = 100;
    this.isCrouching = false;
    this.maxHealth = 100;
    this.health = 100;
    
    // Flashlight
    this.hasFlashlight = true;
    this.flashlightRange = 15;
    this.flashlightIntensity = 1.0;
    this.batteryDrainRate = 0.5; // %/sec when on
    
    // Animation state
    this.headBob = 0;
    this.headBobAmount = 0.1;
    this.headBobSpeed = 6;
    
    // Collision channels
    this.currentRoom = null;
    
    this.setupControls();
  }

  setupControls() {
    document.addEventListener('keydown', (e) => {
      this.keys[e.key.toLowerCase()] = true;
    });
    document.addEventListener('keyup', (e) => {
      this.keys[e.key.toLowerCase()] = false;
    });

    document.addEventListener('mousemove', (e) => {
      if (document.pointerLockElement === document.documentElement) {
        this.mouseDelta.x += e.movementX;
        this.mouseDelta.y += e.movementY;
      }
    });

    // Pointer lock
    document.addEventListener('click', () => {
      document.documentElement.requestPointerLock();
    });

    document.addEventListener('pointerlockchange', () => {
      // Handle lock change
    });
  }

  update(dt, world, portals) {
    // Lerp gravity smoothly
    this.gravityVector.lerp(this.targetGravityVector, dt * 2);

    // Update camera orientation
    this.updateLook(dt);

    // Handle movement input
    this.handleMovement(dt, world);

    // Apply gravity
    this.applyGravity(dt);

    // Collision & ground detection
    this.resolveCollisions(world);

    // Update camera position
    this.camera.position.copy(this.position);
    this.camera.position.add(new THREE.Vector3(0, this.height / 2, 0));

    // Head bob animation
    this.updateHeadBob(dt);

    // Flashlight battery
    if (this.hasFlashlight && this.flashlightBattery > 0) {
      this.flashlightBattery = Math.max(0, this.flashlightBattery - this.batteryDrainRate * dt);
    }

    // Check portal crossings
    if (portals) {
      this.checkPortalCrossing(portals);
    }
  }

  updateLook(dt) {
    // Mouse look
    this.yaw += this.mouseDelta.x * this.mouseSensitivity;
    this.pitch += this.mouseDelta.y * this.mouseSensitivity;
    
    // Clamp pitch
    this.pitch = MathUtils.clamp(this.pitch, -Math.PI / 2, Math.PI / 2);
    
    // Reset mouse delta
    this.mouseDelta.x = 0;
    this.mouseDelta.y = 0;

    // Apply rotation to camera
    this.camera.rotation.order = 'YXZ';
    this.camera.rotation.y = this.yaw;
    this.camera.rotation.x = this.pitch;
  }

  handleMovement(dt, world) {
    // Get forward/right vectors relative to gravity
    const up = this.gravityVector.clone().multiplyScalar(-1).normalize();
    const forward = new THREE.Vector3(0, 0, -1);
    const right = new THREE.Vector3(1, 0, 0);

    // Rotate based on yaw
    forward.applyAxisAngle(up, this.yaw);
    right.applyAxisAngle(up, this.yaw);

    // Also rotate relative to pitch
    const pitchAxis = right.clone();
    forward.applyAxisAngle(pitchAxis, this.pitch);

    // But movement should be on the "ground" plane, not up/down
    forward.sub(up.clone().multiplyScalar(forward.dot(up)));
    forward.normalize();
    right.sub(up.clone().multiplyScalar(right.dot(up)));
    right.normalize();

    // Input
    const input = new THREE.Vector3();
    if (this.keys['w']) input.add(forward);
    if (this.keys['s']) input.sub(forward);
    if (this.keys['a']) input.sub(right);
    if (this.keys['d']) input.add(right);

    if (input.lengthSq() > 0) {
      input.normalize();
    }

    // Speed
    const targetSpeed = this.keys['shift'] ? this.runSpeed : this.walkSpeed;
    const targetVel = input.clone().multiplyScalar(targetSpeed);

    // Move along ground
    this.velocity.add(input.clone().multiplyScalar(targetSpeed * dt));
    this.velocity.multiplyScalar(0.95); // friction

    // Apply velocity
    this.position.add(this.velocity.clone().multiplyScalar(dt));

    // Jump
    if (this.keys[' '] && this.onGround && !this.isCrouching) {
      const jumpDir = up.clone().multiplyScalar(this.jumpForce);
      this.velocity.add(jumpDir);
      this.onGround = false;
      if (window.game && window.game.audio) {
        window.game.audio.playJump();
      }
    }

    // Crouch (for maintenance shaft)
    if (this.keys['c']) {
      this.isCrouching = true;
      this.height = 0.9;
    } else {
      this.isCrouching = false;
      this.height = 1.8;
    }
  }

  applyGravity(dt) {
    const gravityForce = this.gravityVector.clone().multiplyScalar(this.gravityMagnitude);
    this.acceleration.add(gravityForce.multiplyScalar(dt));
    this.velocity.add(this.acceleration.clone().multiplyScalar(dt));
    this.acceleration.set(0, 0, 0);
  }

  resolveCollisions(world) {
    // Simple AABB collision against room bounds
    const up = this.gravityVector.clone().multiplyScalar(-1);
    
    // Get current room
    let currentRoom = null;
    for (let room of world.rooms) {
      const localPos = this.position.clone();
      room.group.worldToLocal(localPos);
      
      if (MathUtils.checkPointInAABB(
        localPos,
        room.bounds.min,
        room.bounds.max
      )) {
        currentRoom = room;
        break;
      }
    }

    if (currentRoom) {
      this.currentRoom = currentRoom;
      
      // Clamp within room bounds
      const halfW = currentRoom.width / 2 - this.radius;
      const halfD = currentRoom.depth / 2 - this.radius;
      
      this.position.x = MathUtils.clamp(
        this.position.x,
        currentRoom.group.position.x - halfW,
        currentRoom.group.position.x + halfW
      );
      this.position.z = MathUtils.clamp(
        this.position.z,
        currentRoom.group.position.z - halfD,
        currentRoom.group.position.z + halfD
      );
    }

    // Ground detection (based on gravity direction)
    const groundCheckDist = 0.2;
    const groundCheckPos = this.position.clone().add(
      this.gravityVector.clone().multiplyScalar(groundCheckDist)
    );
    
    const raycaster = new THREE.Raycaster(this.position, this.gravityVector, 0, groundCheckDist * 2);
    
    if (currentRoom) {
      const intersects = raycaster.intersectObjects(currentRoom.group.children, true);
      this.onGround = intersects.length > 0;
      
      if (this.onGround && this.velocity.dot(this.gravityVector) > 0) {
        this.velocity.sub(
          this.gravityVector.clone().multiplyScalar(this.velocity.dot(this.gravityVector))
        );
      }
    }
  }

  updateHeadBob(dt) {
    if (this.velocity.lengthSq() > 0.1 && this.onGround) {
      this.headBob += dt * this.headBobSpeed * (this.keys['shift'] ? 1.5 : 1);
      const bob = Math.sin(this.headBob) * this.headBobAmount;
      this.camera.position.y += bob;
    }
  }

  checkPortalCrossing(portals) {
    const prevPos = this.position.clone().sub(this.velocity);
    
    for (let portal of portals) {
      if (portal.checkPlayerCrossing(prevPos, this.position)) {
        this.teleportThroughPortal(portal);
        break;
      }
    }
  }

  teleportThroughPortal(portal) {
    // Position
    const localPos = this.position.clone();
    portal.group.worldToLocal(localPos);
    const transformedPos = portal.applyPortalTransform(localPos);
    
    if (portal.destinationPortal) {
      portal.destinationPortal.group.localToWorld(transformedPos);
      this.position.copy(transformedPos);
      
      // Velocity transform
      const localVel = this.velocity.clone();
      const transformedVel = portal.applyPortalTransform(localVel);
      this.velocity.copy(transformedVel);
    }

    // Gravity change
    this.targetGravityVector.copy(portal.getGravityAfterTransit());
    
    // Emit event
    if (window.game) {
      window.game.audioManager.playSwoosh();
      window.game.ui.showNotification('PORTAL TRANSIT', 'success');
    }

    // Sanity loss for certain portals
    if (portal.type === 6) { // LOOP
      this.modifySanity(-5);
    } else if (portal.type === 10) { // VOID
      this.gameOver('You entered the void. Game Over.');
    }
  }

  modifySanity(amount) {
    this.sanity = MathUtils.clamp(this.sanity + amount, -100, 100);
    if (window.game && window.game.hud) {
      window.game.hud.updateSanity(this.sanity);
    }
  }

  gameOver(reason) {
    if (window.game) {
      window.game.gameOver(reason);
    }
  }

  rechargeFlashlight() {
    this.flashlightBattery = 100;
  }

  takeDamage(amount) {
    this.health = Math.max(0, this.health - amount);
    if (this.health <= 0) {
      this.gameOver('You have been eliminated.');
    }
  }

  collectItem(item) {
    if (window.game) {
      window.game.inventory.addItem(item);
    }
  }
}