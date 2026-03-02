import { Renderer } from './renderer.js';
import { Controls } from './controls.js';
import { MatchState } from './match.js'; // We'll implement Match logic inline or minimal class here to avoid circular dep hell if not careful, but imported Match is better
import { Match } from './match.js'; // Re-using the class we defined
import { UI } from './ui.js';
import { audio } from './audio.js';
import { Vector2, checkCircleCollision, clamp } from './utils.js';

// Global Game State
const GAME = {
  renderer: null,
  controls: null,
  match: null,
  lastTime: 0,
  settings: {
    difficulty: 'medium',
    duration: 3,
    teamName: 'HOME',
    teamColor: '#3b82f6',
    sound: true
  },
  animationFrameId: null
};

// Main Loop
function loop(timestamp) {
  if (!GAME.lastTime) GAME.lastTime = timestamp;
  const dt = Math.min((timestamp - GAME.lastTime) / 1000, 0.05); // Cap at 50ms
  GAME.lastTime = timestamp;

  update(dt);
  render();

  GAME.animationFrameId = requestAnimationFrame(loop);
}

function update(dt) {
  if (!GAME.match) return;

  const m = GAME.match;
  const input = GAME.controls;

  // Handle Pause Input
  // (Simple tap on top center handled by UI overlays or invisible button?)
  // For now, let's assume UI handles pause via button in HUD if added.
  // Actually, UI click on HUD Top pauses?
  // Let's implement a pause zone logic here or event listener.
  
  if (m.state === MatchState.PLAYING) {
      // 1. Determine Controlled Player
      // Auto-switch logic
      let bestPlayer = null;
      let minDist = Infinity;
      
      // Filter My Team (Index 0)
      const myPlayers = m.players.filter(p => p.team === 0);
      
      // If manually switching, we skip auto logic for a bit? 
      // Simplified: Just auto-switch to closest to ball always unless button held?
      // Spec says manual switch button cycles.
      
      if (input.buttons.switch && !input.wasSwitchPressed) {
          // Cycle to next player
          input.wasSwitchPressed = true;
          // Find current index
          let idx = myPlayers.indexOf(GAME.controlledPlayer);
          idx = (idx + 1) % myPlayers.length;
          GAME.controlledPlayer = myPlayers[idx];
      } else if (!input.buttons.switch) {
          input.wasSwitchPressed = false;
          
          // Auto switch
          if (document.getElementById('btn-autoswitch').classList.contains('active')) {
             myPlayers.forEach(p => {
                 const d = p.pos.distanceTo(m.ball.pos);
                 if (d < minDist) {
                     minDist = d;
                     bestPlayer = p;
                 }
             });
             // Only switch if significantly closer to avoid jitter
             if (bestPlayer && bestPlayer !== GAME.controlledPlayer) {
                 const currentDist = GAME.controlledPlayer.pos.distanceTo(m.ball.pos);
                 if (minDist < currentDist - 40) { // Hysteresis
                     GAME.controlledPlayer = bestPlayer;
                 }
             }
             // Initial set
             if (!GAME.controlledPlayer) GAME.controlledPlayer = bestPlayer;
          }
      }

      // 2. Process Input for Controlled Player
      GAME.controlledPlayer.isControlled = true;
      
      // Vector from joystick
      const joy = input.getOutput();
      
      // Apply to player velocity (handled in match.update -> player.update)
      // We pass the Input State to Match
      
      // Actions
      if (input.buttons.pass) {
          if (GAME.controlledPlayer.hasBall && !input.wasPassPressed) {
             // Pass Logic
             // Find teammate in direction of joystick, or nearest forward
             let target = null;
             let bestScore = -Infinity;
             const joyHeading = joy.mag() > 0.1 ? joy.angle() : GAME.controlledPlayer.facing;
             
             myPlayers.forEach(p => {
                 if (p === GAME.controlledPlayer) return;
                 const toP = p.pos.clone().sub(GAME.controlledPlayer.pos);
                 const dist = toP.mag();
                 const angle = toP.angle();
                 
                 // Score: Alignment with stick + Distance valid
                 const angleDiff = Math.abs(angle - joyHeading);
                 if (angleDiff < 1.0) { // Within ~60 deg cone
                     const score = 1000 - dist;
                     if (score > bestScore) {
                         bestScore = score;
                         target = p;
                     }
                 }
             });
             
             if (target) {
                 m.ball.pass(target.pos);
                 audio.playPass();
             } else {
                 // Pass into space
                 const shootDir = new Vector2(Math.cos(joyHeading), Math.sin(joyHeading));
                 m.ball.vel = shootDir.multiplyScalar(400);
                 m.ball.owner = null;
                 audio.playKick(0.5);
             }
             input.wasPassPressed = true;
          }
      } else {
          input.wasPassPressed = false;
      }
      
      if (input.buttons.shoot) {
          // Charging handled in controls.js visual
      } else {
           // If was charging, release shoot
           if (input.isCharging) {
               // This state management is tricky across modules.
               // Relying on controls.js to manage 'isCharging' state effectively.
               // Actually controls.js sets isCharging=false on touchend.
               // We need to capture the 'just released' moment.
           }
      }
      
      // Hacky Shoot Release Check:
      // If we tracked start time in specific property in controls, check it
      if (GAME.lastShootState && !input.buttons.shoot && GAME.shootChargeTime > 0) {
          // Release!
          if (GAME.controlledPlayer.hasBall) {
              const power = Math.min((performance.now() - GAME.shootChargeTime) / 800, 1.0);
              // Direction: Joystick or Goal
              let dir = joy.mag() > 0.1 ? joy.clone() : new Vector2(1, 0); // Default right
              // Auto-aim to goal if no stick
              if (joy.mag() < 0.1) {
                  dir = new Vector2(800 - GAME.controlledPlayer.pos.x, 250 - GAME.controlledPlayer.pos.y).normalize();
              }
              
              m.ball.shoot(dir, power);
              audio.playKick(power);
              GAME.shootChargeTime = 0;
          }
      }
      
      if (input.buttons.shoot) {
          if (!GAME.lastShootState) GAME.shootChargeTime = performance.now();
      } else {
          GAME.shootChargeTime = 0;
      }
      GAME.lastShootState = input.buttons.shoot;

      // Tackle
      if (input.buttons.tackle && !input.wasTacklePressed) {
          GAME.controlledPlayer.startTackle();
          audio.playTackle();
          input.wasTacklePressed = true;
      } else if (!input.buttons.tackle) {
          input.wasTacklePressed = false;
      }

      // AI Update
      m.ai.update(dt, { ball: m.ball, players: m.players, myTeamIndex: 0 }); 
      // Note: myTeamIndex 0 is human. AI usually controls 1. 
      // Wait, AI class logic needs to know which team it is controlling.
      // We should instantiate AI for Team 1.
      // Let's fix AI usage in Match.js
  }
  
  // Pass controlled player into match update for physics focus
  m.update(dt, { 
      joyVec: input.getOutput(),
      buttons: input.buttons, 
      controlledPlayer: GAME.controlledPlayer,
      isCharging: input.isCharging
  });
  
  // Update UI
  UI.updateTime(m.duration - m.currentTime, m.half);
  UI.updateScore(m.scores[0], m.scores[1]);
}

function render() {
  if (!GAME.renderer || !GAME.match) return;
  GAME.renderer.render({
      players: GAME.match.players,
      ball: GAME.match.ball,
      matchTime: GAME.match.currentTime,
      scores: GAME.match.scores,
      isGoal: GAME.match.state === MatchState.GOAL_SCORED,
      controlledPlayer: GAME.controlledPlayer
  });
}

// Init
window.onload = () => {
  const canvas = document.getElementById('game-canvas');
  GAME.renderer = new Renderer(canvas);
  GAME.controls = new Controls();
  
  UI.setupListeners({
      onSelectDifficulty: (diff) => GAME.settings.difficulty = diff,
      onSelectColor: (col) => GAME.settings.teamColor = col,
      onStartMatch: (name) => {
          GAME.settings.teamName = name;
          document.getElementById('home-name').textContent = name;
          audio.init(); // User interaction required
          startMatch();
      },
      onResume: () => {
          UI.showHUD();
          UI.showScreen(null); // Hide all screens
          if (GAME.match) GAME.match.state = MatchState.PLAYING;
      },
      onRestart: () => {
          startMatch();
      },
      onQuit: () => {
          GAME.match = null;
          UI.showScreen('menu');
      },
      onNextHalf: () => {
          UI.showHUD();
          UI.showScreen(null);
          GAME.match.state = MatchState.PLAYING;
          GAME.match.half = 2; // Simple logic
          GAME.match.resetPositions();
          audio.playWhistle();
      },
      onSetDuration: (min) => GAME.settings.duration = min
  });

  // Start Loop
  requestAnimationFrame(loop);
};

function startMatch() {
  GAME.match = new Match(
      GAME.settings.difficulty,
      GAME.settings.duration,
      (team) => UI.showEvent('GOAL!'), // onGoal
      () => { // onHalftime
          UI.showScreen('halftime');
      },
      () => { // onFulltime
          const home = GAME.match.scores[0];
          const away = GAME.match.scores[1];
          const res = home > away ? "YOU WIN!" : (home < away ? "YOU LOSE" : "DRAW");
          document.getElementById('fulltime-result').textContent = res;
          UI.showScreen('fulltime');
          
          // Save Record
          const recs = JSON.parse(localStorage.getItem('pf_records') || '[]');
          recs.unshift({
              date: new Date().toLocaleDateString(),
              result: res,
              score: `${home}-${away}`
          });
          localStorage.setItem('pf_records', JSON.stringify(recs.slice(0, 10)));
      }
  );
  
  // Set initial controlled player
  GAME.controlledPlayer = GAME.match.players[3]; // An attacker
  
  UI.showHUD();
  UI.showScreen(null);
  
  // Kickoff Event
  UI.showEvent("KICK OFF!", 1000);
  audio.playWhistle();
  GAME.match.state = MatchState.PLAYING;
}