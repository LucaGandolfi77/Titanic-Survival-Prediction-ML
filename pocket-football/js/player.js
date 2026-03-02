import { Vector2, clamp, lerp } from './utils.js';

export class Player {
  constructor(team, role, startPos) {
    this.pos = new Vector2(startPos.x, startPos.y);
    this.vel = new Vector2(0, 0);
    this.facing = team === 0 ? 0 : Math.PI; // 0 = right, PI = left
    this.team = team; // 0 = home, 1 = away
    this.role = role; // 0=GK, 1=DEF, 2=ATT
    this.number = Math.floor(Math.random() * 99) + 1;
    
    // Stats
    this.maxSpeed = 160;
    this.dribbleSpeed = 120;
    this.acceleration = 400;
    this.friction = 0.85;
    this.radius = 14;

    // State
    this.hasBall = false;
    this.isControlled = false;
    this.stamina = 100;
    this.isTackling = false;
    this.tackleTimer = 0;
    
    // AI target
    this.targetPos = new Vector2();
  }

  update(dt, inputVec) {
    // Tackle cooldown
    if (this.isTackling) {
      this.tackleTimer -= dt;
      if (this.tackleTimer <= 0) {
        this.isTackling = false;
        this.maxSpeed = 160; // Reset speed penalty
      }
    }

    // Stamina logic
    const speed = this.vel.mag();
    if (speed > this.maxSpeed * 0.8) {
      this.stamina = Math.max(0, this.stamina - 5 * dt);
    } else {
      this.stamina = Math.min(100, this.stamina + 3 * dt);
    }

    // Movement Physics
    let acc = new Vector2(0, 0);
    
    // Apply input (AI or Human)
    if (inputVec && inputVec.magSq() > 0.01) {
      acc.copy(inputVec).normalize().multiplyScalar(this.acceleration);
      
      // Update facing direction smoothly
      const targetAngle = Math.atan2(inputVec.y, inputVec.x);
      // Simple lerp angle logic (handling wrapping)
      let diff = targetAngle - this.facing;
      while (diff < -Math.PI) diff += Math.PI * 2;
      while (diff > Math.PI) diff -= Math.PI * 2;
      this.facing += diff * 0.15;
    }

    // Apply acceleration
    this.vel.add(acc.multiplyScalar(dt));

    // Cap speed
    const currentMax = this.hasBall ? this.dribbleSpeed : this.maxSpeed;
    const staminaFactor = this.stamina < 30 ? 0.8 : 1.0;
    const effectiveMax = currentMax * staminaFactor;

    if (this.vel.mag() > effectiveMax) {
      this.vel.normalize().multiplyScalar(effectiveMax);
    }

    // Friction
    this.vel.multiplyScalar(this.friction); // Simple damping

    // Update position
    this.pos.add(new Vector2(this.vel.x * dt, this.vel.y * dt));

    // Field boundary constraint (simple box)
    // Field is 0-800 x 0-500. Goal areas allowed.
    // Simplifying to strict bounds for now, can refine for goals later
    this.pos.x = clamp(this.pos.x, 15, 785);
    this.pos.y = clamp(this.pos.y, 15, 485);
  }

  startTackle() {
      if (this.isTackling) return;
      this.isTackling = true;
      this.tackleTimer = 0.5; // 500ms lunge
      // Lunging burst
      const lunge = new Vector2(Math.cos(this.facing), Math.sin(this.facing)).multiplyScalar(150);
      this.vel.add(lunge);
  }
}