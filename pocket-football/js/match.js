import { Player } from './player.js';
import { Ball } from './ball.js';
import { AI } from './ai.js';
import { audio } from './audio.js';
import { Vector2 } from './utils.js';

export const MatchState = {
  KICKOFF: 0,
  PLAYING: 1,
  GOAL_SCORED: 2,
  HALFTIME: 3,
  FULLTIME: 4,
  PAUSED: 5
};

export class Match {
  constructor(difficulty, durationMinutes, onGoal, onHalftime, onFulltime) {
    this.difficulty = difficulty;
    this.duration = durationMinutes * 60; // Seconds
    this.timer = this.duration / 2; // Start with 1st half length? No, countdown.
    this.currentTime = 0; // Running up
    this.half = 1;
    this.state = MatchState.KICKOFF;
    
    this.scores = [0, 0];
    this.players = [];
    this.ball = new Ball(400, 250);
    this.ai = new AI(difficulty);
    
    // Callbacks
    this.onGoal = onGoal;
    this.onHalftime = onHalftime;
    this.onFulltime = onFulltime;
    
    this.initPlayers();
    this.resetPositions(true); // Kickoff setup
  }

  initPlayers() {
    // 5 vs 5
    // Team 0 (Home - Blue)
    // Roles: 0:GK, 1:DEF, 2:DEF, 3:ATT, 4:ATT
    // Default positions (Kickoff)
    const homePos = [
      {r:0, x:40, y:250}, {r:1, x:200, y:150}, {r:1, x:200, y:350}, {r:2, x:320, y:200}, {r:2, x:320, y:300}
    ];
    // Team 1 (Away - Red - AI)
    const awayPos = [
      {r:0, x:760, y:250}, {r:1, x:600, y:150}, {r:1, x:600, y:350}, {r:2, x:480, y:200}, {r:2, x:480, y:300}
    ];

    this.players = [];
    homePos.forEach((p, i) => this.players.push(new Player(0, p.r, new Vector2(p.x, p.y))));
    awayPos.forEach((p, i) => this.players.push(new Player(1, p.r, new Vector2(p.x, p.y))));
  }

  resetPositions(isKickoff) {
    // Determine who kicks off? (Alternate)
    // For now, Team 0 starts Half 1, Team 1 starts Half 2
    // Or just always center
    
    this.ball.pos.set(400, 250);
    this.ball.vel.set(0, 0);
    this.ball.owner = null;
    
    // Reset players to own halves
    this.players.forEach(p => {
       p.vel.set(0, 0);
       p.isTackling = false;
       // Simple reset to initial slots
       if(p.team === 0) {
           p.pos.x = p.role === 0 ? 40 : 300; // Simplified line
           if (p.role === 0) p.pos.x = 40;
           else if (p.role === 1) p.pos.x = 200;
           else p.pos.x = 350;
           p.facing = 0;
       } else {
           if (p.role === 0) p.pos.x = 760;
           else if (p.role === 1) p.pos.x = 600;
           else p.pos.x = 450;
           p.facing = Math.PI;
       }
       // Spread Y based on index roughly to avoid stacking
       // Just using random jitter for now to prevent total overlap
       p.pos.y = 250 + (Math.random() - 0.5) * 200;
    });
    
    this.state = MatchState.KICKOFF;
  }

  update(dt, input) {
    if (this.state === MatchState.PAUSED || this.state === MatchState.GOAL_SCORED) return;
    
    // Timer
    if (this.state === MatchState.PLAYING) {
      this.currentTime += dt;
      // Check half time
      const halfDuration = this.duration / 2;
      
      if (this.half === 1 && this.currentTime >= halfDuration) {
         this.halfEnd();
         return;
      } else if (this.half === 2 && this.currentTime >= this.duration) {
         this.matchEnd();
         return;
      }
    }

    // Update Entities
    this.ball.update(dt);
    
    // Update Players
    this.players.forEach(p => {
       // Get Input for Controlled Player
       let moveVec = null;
       
       if (p === input.controlledPlayer) {
           moveVec = input.joyVec;
           // Actions
           if (input.buttons.shoot && !input.isCharging) {
               // Released shoot?
           }
           if (input.buttons.tackle) p.startTackle();
           if (input.buttons.pass && p.hasBall) {
               // Find teammate
               // Logic handled in Main usually? Or here.
               // Let's defer "Event" handling to Main Loop's input processing
           }
       } else if (p.team === 1) {
           // AI
           // Validated externally or integrated?
           // AI update separate loop usually
       }

       // We will pass AI intent from outside in Main.js
       // Here we just accept Physics update with "current velocity"
       // Actually, we need to apply forces here.
       
       // So: Update takes applied forces.
       // AI decisions set velocity/target.
       p.update(dt, p === input.controlledPlayer ? input.joyVec : p.aiInputVec);
    });

    // Collisions
    this.checkCollisions();
    
    // Goal Check
    this.checkGoal();
  }
  
  checkCollisions() {
     // Player-Ball interaction (Possession)
     if (!this.ball.owner && !this.ball.isAirborne) {
         let closest = null;
         let minDist = 16; // Capture radius
         
         this.players.forEach(p => {
             const dist = p.pos.distanceTo(this.ball.pos);
             if (dist < minDist) {
                 minDist = dist;
                 closest = p;
             }
         });
         
         if (closest) {
             this.ball.owner = closest;
             closest.hasBall = true;
             // Slow down player slightly? handled in Player class
         }
     }
     
     // Player-Player (Tackle / Bump)
     // Simple repulsion
     for (let i = 0; i < this.players.length; i++) {
         for (let j = i + 1; j < this.players.length; j++) {
             const p1 = this.players[i];
             const p2 = this.players[j];
             const distSq = p1.pos.distanceToSquared(p2.pos);
             const radSum = 28; // 14+14
             
             if (distSq < radSum * radSum) {
                 // Push apart
                 const dist = Math.sqrt(distSq);
                 const overlap = radSum - dist;
                 const normal = p1.pos.clone().sub(p2.pos).normalize();
                 
                 p1.pos.add(normal.clone().multiplyScalar(overlap * 0.5));
                 p2.pos.sub(normal.clone().multiplyScalar(overlap * 0.5));
             }
         }
     }
  }

  checkGoal() {
      // Goal dimensions: Y 210-290
      
      // Home Goal (Left) -> Away Team Scores
      if (this.ball.pos.x < 0 && this.ball.pos.y > 210 && this.ball.pos.y < 290) {
          this.score(1);
      }
      // Away Goal (Right) -> Home Team Scores
      else if (this.ball.pos.x > 800 && this.ball.pos.y > 210 && this.ball.pos.y < 290) {
          this.score(0);
      }
  }

  score(teamIndex) {
      if (this.state === MatchState.GOAL_SCORED) return;
      
      this.scores[teamIndex]++;
      this.state = MatchState.GOAL_SCORED;
      audio.playCheer(); // Synthesized cheer
      
      this.onGoal(teamIndex);
      
      // Reset after delay
      setTimeout(() => {
          if (this.state !== MatchState.FULLTIME) {
            this.resetPositions();
            this.state = MatchState.KICKOFF; // Wait for user to kick off again? 
            // Design says "Kickoff button". Correct.
          }
      }, 2500);
  }
  
  halfEnd() {
      this.state = MatchState.HALFTIME;
      this.onHalftime();
  }
  
  startSecondHalf() {
      this.half = 2;
      this.resetPositions();
      this.state = MatchState.KICKOFF; // Or straight to play?
      // Usually kickoff
  }
  
  matchEnd() {
      this.state = MatchState.FULLTIME;
      this.onFulltime();
      audio.playWhistle();
  }
}