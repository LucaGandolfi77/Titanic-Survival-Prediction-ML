import { Vector2, clamp } from './utils.js';

export class Ball {
  constructor(x, y) {
    this.pos = new Vector2(x, y);
    this.vel = new Vector2(0, 0);
    this.owner = null; // Reference to Player object
    this.isAirborne = false;
    this.height = 0; // Visual Z height
    this.spin = 0;
    
    this.radius = 8;
    this.friction = 0.97; // Rolling friction per frame
    this.restitution = 0.6; // Bounce factor
  }

  update(dt) {
    if (this.owner) {
      // Snapped to player
      const offset = new Vector2(Math.cos(this.owner.facing), Math.sin(this.owner.facing)).multiplyScalar(16);
      this.pos.copy(this.owner.pos).add(offset);
      this.vel.copy(this.owner.vel);
      this.height = 0;
    } else {
      // Free physics
      // Apply friction
      if (!this.isAirborne) {
        this.vel.multiplyScalar(1 - 0.03 * 60 * dt); // Approx matches standard friction
      } else {
          // Air drag less
          this.vel.multiplyScalar(0.99);
      }

      // Update position
      this.pos.add(new Vector2(this.vel.x * dt, this.vel.y * dt));

      // Wall Bouncing
      if (this.pos.y < this.radius) {
        this.pos.y = this.radius;
        this.vel.y *= -this.restitution;
      }
      if (this.pos.y > 500 - this.radius) {
        this.pos.y = 500 - this.radius;
        this.vel.y *= -this.restitution;
      }

      // Side walls (Goals are open, but let's bounce off corners)
      // Left Goal Range: Y 210-290. x < 0 is goal.
      // Right Goal Range: Y 210-290. x > 800 is goal.
      
      // Left Wall
      if (this.pos.x < this.radius) {
         if (this.pos.y < 210 || this.pos.y > 290) {
             this.pos.x = this.radius;
             this.vel.x *= -this.restitution;
         }
      }

      // Right Wall
      if (this.pos.x > 800 - this.radius) {
          if (this.pos.y < 210 || this.pos.y > 290) {
              this.pos.x = 800 - this.radius;
              this.vel.x *= -this.restitution;
          }
      }
    }
  }

  shoot(direction, power) {
      this.owner = null;
      // Power 0.0 to 1.0 -> Speed 300 to 800
      const speed = 300 + (power * 500);
      this.vel.copy(direction).normalize().multiplyScalar(speed);
      
      // Add some loft
      this.isAirborne = true;
      // Simple fake height logic for now? 
      // For now just 2D physics
  }

  pass(targetPos) {
      this.owner = null;
      const dir = targetPos.clone().sub(this.pos).normalize();
      this.vel.copy(dir).multiplyScalar(450); // Snappy pass
  }
}