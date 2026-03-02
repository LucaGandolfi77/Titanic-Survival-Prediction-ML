import { Vector2, randomRange, checkCircleCollision } from './utils.js';

export class AI {
  constructor(difficulty) {
    this.difficulty = difficulty; // 'easy', 'medium', 'hard'
    this.updateInterval = this.getUpdateInterval();
    this.timer = 0;
    
    // Config based on difficulty
    this.reactionTime = difficulty === 'hard' ? 0.05 : (difficulty === 'medium' ? 0.2 : 0.4);
    this.aggressiveness = difficulty === 'hard' ? 0.85 : (difficulty === 'medium' ? 0.65 : 0.4);
    this.accuracy = difficulty === 'hard' ? 0.95 : (difficulty === 'medium' ? 0.8 : 0.6);
    
    // Store intents
    this.intents = new Map(); // PlayerID -> { targetPos, action }
  }

  getUpdateInterval() {
    return this.reactionTime;
  }

  update(dt, matchState) {
    this.timer -= dt;
    if (this.timer <= 0) {
      this.timer = this.updateInterval + Math.random() * 0.1; // Jitter
      this.decide(matchState);
    }
    
    return this.intents;
  }

  decide(matchState) {
    // matchState = { ball, players (all), myTeamIndex }
    const { ball, players, myTeamIndex } = matchState;
    const teamPlayers = players.filter(p => p.team === myTeamIndex);
    const oppPlayers = players.filter(p => p.team !== myTeamIndex);

    // AI Logic Loop per player
    teamPlayers.forEach(p => {
      let action = 'idle';
      let targetV = new Vector2(0, 0);
      let pressing = false; // Is sprinting/pressing?

      // 1. HAS BALL?
      if (p.hasBall) {
        // Goal distance
        const goalCenter = new Vector2(myTeamIndex === 0 ? 800 : 0, 250); // Attacking goal
        const distToGoal = p.pos.distanceTo(goalCenter);

        // SHOOT?
        if (distToGoal < 300) {
            // Check clear line of sight roughly
            action = 'shoot';
            targetV = goalCenter; // Aim at goal
        } 
        // PASS?
        else {
            // Find best teammate forward
            let bestPass = null;
            let bestScore = -Infinity;
            
            teamPlayers.forEach(mate => {
               if (mate === p) return;
               // Must be forward
               const isForward = (myTeamIndex === 0) ? (mate.pos.x > p.pos.x) : (mate.pos.x < p.pos.x);
               if (isForward) {
                   const score = mate.pos.distanceTo(goalCenter) * -1; // Closer to goal is better
                   if (score > bestScore) {
                       bestScore = score;
                       bestPass = mate;
                   }
               }
            });

            if (bestPass && Math.random() < this.accuracy) {
                action = 'pass';
                targetV = bestPass.pos;
            } else {
                action = 'dribble';
                targetV = goalCenter;
            }
        }
      } 
      // 2. DEFENDING? (Opponent has ball)
      else if (ball.owner && ball.owner.team !== myTeamIndex) {
         const distToBall = p.pos.distanceTo(ball.pos);
         
         // Chase if close
         if (distToBall < 150) {
             action = 'move';
             targetV = ball.pos;
             pressing = true;
             
             // Tackle?
             if (distToBall < 20 && Math.random() < this.aggressiveness) {
                 action = 'tackle';
             }
         } else {
             // Fall back to formation
             targetV = this.getFormationPos(p.role, myTeamIndex, ball.pos);
             action = 'move';
         }
      } 
      // 3. LOOSE BALL?
      else {
          const distToBall = p.pos.distanceTo(ball.pos);
          if (distToBall < 200) { // Someone go get it
              // Basic check: am I the closest? (Skipping for performance/simplicity)
              action = 'move';
              targetV = ball.pos;
              pressing = true;
          } else {
              targetV = this.getFormationPos(p.role, myTeamIndex, ball.pos);
              action = 'move';
          }
      }

      this.intents.set(p, { action, target: targetV, press: pressing });
    });
  }

  getFormationPos(role, teamIndex, ballPos) {
      // Simple 1-2-1 formation (GK-DEF-DEF-ATT-ATT) mapped to 5
      // Role 0: GK, 1,2: DEF, 3,4: ATT
      const isHome = teamIndex === 0; // Attacks Right (800, 250)
      
      let baseX = 0;
      let baseY = 250;

      // Formation Anchors (Normalized 0-1)
      if (role === 0) { // GK
          baseX = isHome ? 0.05 : 0.95;
          baseY = 0.5;
      } else if (role === 1) { // DEF Top
          baseX = isHome ? 0.25 : 0.75;
          baseY = 0.3;
      } else if (role === 2) { // DEF Bottom
          baseX = isHome ? 0.25 : 0.75;
          baseY = 0.7;
      } else if (role === 3) { // ATT Top
          baseX = isHome ? 0.45 : 0.55;
          baseY = 0.4;
      } else if (role === 4) { // ATT Bottom
          baseX = isHome ? 0.45 : 0.55;
          baseY = 0.6;
      }

      // Dynamic Shifting
      // If attacking (ball x > center), shift formation forward
      // If defending (ball x < center), shift back
      
      const fieldW = 800;
      const fieldH = 500;
      
      // Calculate Ball X Factor (0 to 1)
      const ballFactor = ballPos.x / fieldW;
      
      // Shift logic
      let shiftX = (ballFactor - 0.5) * 0.2; // +/- 20% shift
      if (!isHome) shiftX *= -1; // Invert for away team logic if needed

      // Apply
      let finalX = (baseX + shiftX) * fieldW;
      let finalY = baseY * fieldH;

      return new Vector2(finalX, finalY);
  }
}