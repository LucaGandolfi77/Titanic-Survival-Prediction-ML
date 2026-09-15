import * as THREE from 'three';
import { MathUtils, rotateVectorAroundAxis } from './utils.js';
import { bus } from './event-bus.js';

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

    // Cached objects
    this._raycaster = new THREE.Raycaster();
    this._prevPosition = new THREE.Vector3();
    this._up = new THREE.Vector3();
    this._forward = new THREE.Vector3(0, 0, -1);
    this._right = new THREE.Vector3(1, 0, 0);
    this._pitchAxis = new THREE.Vector3();
    this._input = new THREE.Vector3();
    this._inputScaled = new THREE.Vector3();
    this._velocityDt = new THREE.Vector3();
    this._jumpDir = new THREE.Vector3();
    this._gravityForce = new THREE.Vector3();
    this._accelDt = new THREE.Vector3();
    this._groundCheckPos = new THREE.Vector3();
    this._localPos = new THREE.Vector3();
    this._heightOffset = new THREE.Vector3(0, this.height / 2, 0);

    // Event handler references for cleanup
    this._onKeyDown = null;
    this._onKeyUp = null;
    this._onMouseMove = null;
    this._onClick = null;

    this.setupControls();
  }

  setupControls() {
    this._onKeyDown = (e) => {
      this.keys[e.key.toLowerCase()] = true;
    };
    this._onKeyUp = (e) => {
      this.keys[e.key.toLowerCase()] = false;
    };
    this._onMouseMove = (e) => {
      if (document.pointerLockElement === document.body) {
        this.mouseDelta.x += e.movementX;
        this.mouseDelta.y += e.movementY;
      }
    };
    this._onClick = () => {
      document.body.requestPointerLock();
    };

    document.addEventListener('keydown', this._onKeyDown);
    document.addEventListener('keyup', this._onKeyUp);
    document.addEventListener('mousemove', this._onMouseMove);
    document.addEventListener('click', this._onClick);
  }

  destroy() {
    if (this._onKeyDown) document.removeEventListener('keydown', this._onKeyDown);
    if (this._onKeyUp) document.removeEventListener('keyup', this._onKeyUp);
    if (this._onMouseMove) document.removeEventListener('mousemove', this._onMouseMove);
    if (this._onClick) document.removeEventListener('click', this._onClick);
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
    this.camera.position.add(this._heightOffset);

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
    // Get up vector
    this._up.copy(this.gravityVector).multiplyScalar(-1).normalize();

    // Get forward/right vectors relative to gravity
    this._forward.set(0, 0, -1);
    this._right.set(1, 0, 0);

    // Rotate based on yaw
    this._forward.applyAxisAngle(this._up, this.yaw);
    this._right.applyAxisAngle(this._up, this.yaw);

    // Also rotate relative to pitch
    this._pitchAxis.copy(this._right);
    this._forward.applyAxisAngle(this._pitchAxis, this.pitch);

    // But movement should be on the "ground" plane, not up/down
    this._forward.sub(this._up.clone().multiplyScalar(this._forward.dot(this._up))).normalize();
    this._right.sub(this._up.clone().multiplyScalar(this._right.dot(this._up))).normalize();

    // Input
    this._input.set(0, 0, 0);
    if (this.keys['w']) this._input.add(this._forward);
    if (this.keys['s']) this._input.sub(this._forward);
    if (this.keys['a']) this._input.sub(this._right);
    if (this.keys['d']) this._input.add(this._right);

    if (this._input.lengthSq() > 0) {
      this._input.normalize();
    }

    // Speed
    const targetSpeed = this.keys['shift'] ? this.runSpeed : this.walkSpeed;
    this._inputScaled.copy(this._input).multiplyScalar(targetSpeed);

    // Move along ground
    this.velocity.add(this._inputScaled.multiplyScalar(dt));
    this.velocity.multiplyScalar(0.95); // friction

    // Apply velocity
    this.position.add(this._velocityDt.copy(this.velocity).multiplyScalar(dt));

    // Jump
    if (this.keys[' '] && this.onGround && !this.isCrouching) {
      this._jumpDir.copy(this._up).multiplyScalar(this.jumpForce);
      this.velocity.add(this._jumpDir);
      this.onGround = false;
      bus.emit('audio:play', { type: 'jump' });
    }

    // Crouch (for maintenance shaft)
    if (this.keys['c']) {
      this.isCrouching = true;
      this.height = 0.9;
      this._heightOffset.y = 0.45;
    } else {
      this.isCrouching = false;
      this.height = 1.8;
      this._heightOffset.y = 0.9;
    }
  }

  applyGravity(dt) {
    const gravityForce = this._gravityForce
      .copy(this.gravityVector)
      .multiplyScalar(this.gravityMagnitude);
    this.acceleration.add(gravityForce.multiplyScalar(dt));
    this.velocity.add(this._accelDt.copy(this.acceleration).multiplyScalar(dt));
    this.acceleration.set(0, 0, 0);
  }

  resolveCollisions(world) {
    // Simple AABB collision against room bounds
    const up = this.gravityVector.clone().multiplyScalar(-1);

    // Get current room
    let currentRoom = null;
    for (const room of world.rooms) {
      const localPos = this.position.clone();
      room.group.worldToLocal(localPos);

      if (MathUtils.checkPointInAABB(localPos, room.bounds.min, room.bounds.max)) {
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
    const groundCheckPos = this.position
      .clone()
      .add(this.gravityVector.clone().multiplyScalar(groundCheckDist));

    const raycaster = this._raycaster.set(
      this.position,
      this.gravityVector,
      0,
      groundCheckDist * 2
    );

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
    // Handled in main.js update loop
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

    bus.emit('audio:swoosh');
    bus.emit('ui:notify', { message: 'PORTAL TRANSIT', type: 'success' });

    // Sanity loss for certain portals
    if (portal.type === 6) {
      // LOOP
      this.modifySanity(-5);
    } else if (portal.type === 10) {
      // VOID
      bus.emit('game:over', { reason: 'You entered the void. Game Over.' });
    }
  }

  modifySanity(amount) {
    this.sanity = MathUtils.clamp(this.sanity + amount, -100, 100);
    bus.emit('player:sanity-changed', { value: this.sanity });
  }

  teleportToRoom(roomIndex) {
    const room = this.world.rooms[roomIndex];
    if (room && room.group) {
      this.position.set(room.group.position.x, 1.8, room.group.position.z);
      this.currentRoom = room;
    }
  }

  gameOver(reason) {
    bus.emit('game:over', { reason });
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
    bus.emit('inventory:add', { item });
  }
}
