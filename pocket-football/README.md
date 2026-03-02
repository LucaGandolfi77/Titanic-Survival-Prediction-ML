# POCKET FOOTBALL ⚽

A fully playable 5-a-side football (soccer) game optimized for mobile devices with responsive touch controls, synthesized audio, and AI opponents.

## Features

- **Mobile First:** Virtual joystick and action buttons (Pass, Shoot, Tackle, Switch).
- **Responsive Canvas:** Adapts to any screen size (Portrait/Landscape) maintaining aspect ratio.
- **AI Opponent:** 3 Difficulty levels with varying speed, aggression, and accuracy.
- **Physics Engine:** Custom 2D physics for ball rolling, bouncing, and player collisions.
- **Game Modes:** Configurable match duration (1, 3, 5 mins).
- **Audio:** Web Audio API implementation (Zero assets required). Effects for kicks, whistles, and crowds.
- **Progress:** LocalStorage saved match history.

## How to Play

### Controls
- **Joystick (Left):** Move your active player.
- **PASS (Green):** Pass to nearest teammate in direction of movement.
- **SHOOT (Amber):** Hold to charge power, release to shoot.
- **TACKLE (Red):** Attempt to steal the ball.
- **SWITCH (Purple):** Cycle control to next player (Auto-switch enabled by default).

### Rules
- 5 players per team (1 GK, 2 Defenders, 2 Attackers).
- Match is split into two halves.
- Score more goals than the AI to win!

## Development

### Structure
- `index.html`: Main entry and UI structure.
- `css/`: Styles for UI, HUD, and Controls.
- `js/`: Modular ES6 JavaScript.
  - `main.js`: Game loop and initialization.
  - `renderer.js`: Canvas 2D drawing logic.
  - `match.js`: Rules, scoring, and time management.
  - `ai.js`: Decision trees for opponent behavior.
  - `player.js`: Physics and state for entities.
  - `ball.js`: Ball physics.
  - `audio.js`: Synthesized sound effects.

### Running Locally
Since this project uses ES6 modules, you must serve it via a local web server to avoid CORS errors.

```bash
python3 -m http.server 8080
```
Then navigate to `http://localhost:8080` on your device or simulator.

## Technical Details

- **No Libraries:** Pure Vanilla JS, HTML5 Canvas.
- **Rendering:** 60 FPS requestAnimationFrame loop.
- **State Management:** Simple state machine (Kickoff -> Playing -> Goal -> HalfTime -> End).
- **AI:** Normalized formation vectors with dynamic shifting based on ball position.

Created by GitHub Copilot.