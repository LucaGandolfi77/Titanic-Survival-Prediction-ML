# Gravity Coffee Simulator

A surreal first-person space bar simulator where gravity shifts constantly, and you must serve coffee to alien customers without spilling.

## Features

* Custom 3D rigid body physics (boxes, spheres, cylinders) with spatial hash broadphase
* Fluid simulation via custom WebGL ShaderMaterial for coffee tilting (depth shading, crema swirl, foam bubbles)
* 11 different Gravity states affecting play (Normal, Zero-G, Reversed, Diagonal, etc)
* First-person grab and tilt mechanics
* Procedural Web Audio API sound effects (no external audio files required)
* PostFX bloom/glow overlay for neon ambiance
* Atmospheric dust particles + exponential fog
* Emissive alien customers with colored glow halos
* High score leaderboard (localStorage)
* Multiple cup types (Mug, Espresso, Pitcher) with different fill rates and scores
* EventBus architecture for decoupled communication

## Tech Stack

| Technology | Version | Usage |
|------------|---------|-------|
| JavaScript | ES6+ | Game logic, physics, rendering |
| Three.js | r158 | 3D rendering, WebGL |
| Web Audio API | Browser-native | Sound synthesis (alarm, shift, collision, happy, angry, pour) |
| WebGL Shaders | GLSL | Coffee liquid shader (vertex + fragment), bloom + SSAO post-processing |
| CSS3 | Latest | UI styling, animations
| localStorage | Browser-native | High scores, tutorial state |
| EventBus | Custom | Module communication (observer pattern) |

## Installation & Running

Everything is contained in this folder, and built via vanilla JS + Three.js (`r158`).

To run:
1. Open this folder in your terminal.
2. Run a local web server to bypass CORS issues for module imports:
   `python3 -m http.server 8000`
3. Open `http://localhost:8000` in your browser window.

## Gameplay

- **Mouse Look**: Look around the bar.
- **Left Click**: Grab a coffee pot or cup. Click again to drop/throw.
- **Right Click (Hold) + Scroll Wheel**: Tilt your held object to pour coffee from the pot to the cup.
- **Listen for alarms**: The HUD will flash and an alarm will sound 3 seconds before gravity shifts. Brace your cups!

*Never spill the coffee.*

## Cups

| Type | Size | Fill Speed | Score Multiplier |
|------|------|-----------|-----------------|
| Espresso | Small | Fast (0.15/s) | 1.5x |
| Mug | Medium | Normal (0.1/s) | 1.0x |
| Pitcher | Large | Slow (0.06/s) | 0.7x |

## Bugfixes Applied

| # | Fix | Impact |
|---|-----|--------|
| 1 | Animation uses `clock.getElapsedTime()` instead of `Date.now()` | Frame-rate independent |
| 2 | Physics drag uses exponential decay instead of linear | Correct physics |
| 3 | Raycaster uses single quaternion (no double rotation) | Correct object detection |
| 4 | Counter physics plane aligned with visual position | Counter exists in physics space |
| 5 | Sugar cube dimensions match collision bounding | Correct collision |
| 6 | Pour threshold consistent (PI/3) across modules | Consistent spilling |
| 7 | Dead shader code removed (vWorldPosition) | Clean shader |
| 8 | Customers spawn at floor level | No underground spawn |
| 9 | Gravity arrow handles all 3D vectors correctly | Correct UI |
| 10 | Audio node cleanup prevents memory leak | No audio context issues |
| 11 | Particle system uses object pool | No memory leaks |
| 12 | Cup respawn uses game clock (not setTimeout) | No ghost cups |
| 13 | Dead code removed from scene.js | Clean code |
| 14 | PostFX canvas styled properly | Visual polish |

## Architecture

```
gravity-coffee/
├── index.html
├── README.md
├── css/
│   ├── variables.css      # CSS custom properties, colors, fonts
│   ├── reset.css          # Normalize + canvas rules
│   ├── ui.css             # Screen layouts, menus, buttons
│   ├── hud.css            # HUD panels, orders, pour meter
│   └── animations.css     # Keyframe animations (flash, shake, float)
├── js/
│   ├── main.js            # GameEngine class, game loop, state management
│   ├── physics.js         # PhysicsWorld + PhysicsBody (spatial hash collision)
│   ├── scene.js           # SceneManager (renderer, lights, postFX, dust)
│   ├── gravity.js         # GravityDirector (11 states, transitions, chaos)
│   ├── bar.js             # SpaceBar (room, counter, shelves, neon sign)
│   ├── objects.js         # ItemManager (cups, pots, sugar with types)
│   ├── customers.js       # CustomerSystem (aliens, orders, delivery)
│   ├── controls.js        # Controls (pointer lock, grab, tilt)
│   ├── hud.js             # HUD (score, orders, gravity, pour meter)
│   ├── audio.js           # AudioSystem (Web Audio API synthesis)
│   ├── coffee.js          # Coffee shader (GLSL vertex + fragment)
│   ├── ui.js              # UI manager (screen transitions)
│   ├── event-bus.js       # EventBus (observer pattern for decoupling)
│   └── utils.js           # MathUtils, domUtils, CONFIG
└── sounds/                # Sound files (legacy - now synthesized)
```

### Backend Modules

- **GameEngine**: Central controller, game loop, FSM, high scores
- **PhysicsWorld**: Custom rigid body physics with spatial hash broadphase
- **SceneManager**: Three.js renderer, lighting, atmosphere, PostFX bloom
- **GravityDirector**: 11 gravity states with smooth transitions and chaos mode
- **ItemManager**: Object lifecycle, coffee cup types, liquid simulation
- **CustomerSystem**: Alien customers with orders and delivery checking
- **AudioSystem**: Procedural sound synthesis via Web Audio API

### Frontend Modules

- **UI**: Screen management (menu, howto, pause, gameover, settings)
- **Controls**: Pointer lock, raycasting, grab/tilt mechanics
- **HUD**: DOM-based HUD (score, stats, orders, gravity, pour meter)
- **EventBus**: Decoupled communication between all modules
- **CONFIG**: Centralized constants (no magic numbers)

## Known Issues

- Mobile touch controls implemented for grab, tilt, look (#mobile-controls)
- No server-side persistence (high scores are client-only)
- Spatial hash grid resets each frame (acceptable for <50 bodies)
- PostFX uses WebGL bloom + SSAO shaders

## Future Improvements (TODO)

### High Priority
- [x] Add touch event handlers for mobile controls (#mobile-controls buttons)
- [x] Implement InstancedMesh for repeated geometries (stools, sugar cubes)
- [x] Add loading screen for Three.js CDN resources
- [x] Add frustum culling for off-screen objects

### Graphics
- [x] True WebGL bloom post-processing shader
- [x] SSAO approximation for ambient occlusion
- [x] Procedural wall textures (wood grain, concrete)
- [x] Animated menu background (rotating 3D cup)
- [x] More alien variety (Jellyfish, Robot, Star)

### Gameplay
- [x] Power-ups: Gravity Anchor, Super Grip, Time Warp
- [x] Multiplayer spectator mode (WASD camera + player follow)
- [x] Level editor with shareable codes
- [x] Stress mode (10 customers, 5s gravity shifts)
- [x] Coffee journal tracking achievements (localStorage)
- [x] Speedrun timer with splits
- [x] Seasonal skins (Halloween, Christmas)
- [x] Ambient procedural music that shifts with gravity

### Architecture
- [ ] Formal FSM with state entry/exit handlers per state
- [ ] InputManager with action mapping
- [ ] AssetManager with geometry/material caching
- [ ] Add unit tests for physics collision resolution
- [ ] Implement proper save/load game state

### Performance
- [ ] Persistent spatial hash (don't rebuild each frame)
- [ ] Add worker thread for physics computation
- [ ] Implement LOD for distant objects
- [ ] Add frame rate monitor and adaptive quality

## API Reference

### EventBus Events

| Event | Data | Description |
|-------|------|-------------|
| `audio:ready` | AudioSystem | Audio system initialized |
| `engine:ready` | GameEngine | Engine initialized |
| `state:enter` | {from, to} | State changed |
| `state:exit` | {from, to} | State changed |
| `particle:spill` | Vector3 | Spill particle position |
| `score:add` | number | Score points added |
| `score:spill` | - | Spill occurred |
| `audio:playAlarm` | - | Play shift warning |
| `audio:playShift` | - | Play gravity shift |
| `audio:playCollision` | force | Play collision sound |
| `audio:playHappy` | - | Play happy sound |
| `audio:playAngry` | - | Play angry sound |
| `audio:setPouring` | boolean | Start/stop pouring |

## License

Gravity Coffee Simulator - First-person coffee physics
