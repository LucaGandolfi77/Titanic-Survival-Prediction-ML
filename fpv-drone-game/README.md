# DRONE STRIKE — FPV Combat Game

A fully playable first-person drone combat game built with **Three.js** and vanilla JavaScript.
Pilot an FPV racing drone, destroy waves of enemies, and survive as long as you can.

## How to Run

```bash
# Option A – Node.js
npx serve .

# Option B – Python
python3 -m http.server 8080
```

Then open **http://localhost:8080** (or the port shown).

## Controls

| Action | Keyboard / Mouse | Touch |
|---|---|---|
| Throttle Up / Down | W / S or ↑ / ↓ | Left Joystick Y |
| Yaw Left / Right | A / D or ← / → | Left Joystick X |
| Pitch Up / Down | Mouse Y (pointer-lock) | Right Joystick Y |
| Roll Left / Right | Mouse X (pointer-lock) | Right Joystick X |
| Fire Cannon | Left Click | 🔥 Button |
| Fire Missile | Right Click | 🚀 Button |
| Boost | Space | — |
| Pause | Escape | — |

## Features

- Realistic arcade flight physics with FPV camera
- 3 enemy types: Ground Turret, Patrol Drone, Heavy Gunship
- Wave-based survival with scaling difficulty
- Rapid-fire cannon + homing missiles
- Procedural terrain, buildings, trees
- Particle explosions, smoke trails, muzzle flash
- Synthesized audio (Web Audio API, no files needed)
- Full HUD: health, ammo, altitude, speed, compass, crosshair
- Touch-friendly virtual joysticks for mobile play
- High score board with localStorage persistence

## Tech Stack

- Three.js r158 (CDN)
- Vanilla JS ES Modules
- Web Audio API
- CSS3 for HUD and UI
- No external dependencies

## File Structure

```
fpv-drone-game/
├── index.html
├── README.md
├── css/
│   ├── variables.css
│   ├── reset.css
│   ├── hud.css
│   ├── controls.css
│   └── ui.css
└── js/
    ├── main.js
    ├── scene.js
    ├── world.js
    ├── drone.js
    ├── enemies.js
    ├── weapons.js
    ├── controls.js
    ├── hud.js
    ├── effects.js
    ├── audio.js
    ├── collision.js
    ├── ui.js
    └── utils.js
```

## Browser Support

Chrome 90+, Firefox 88+, Safari 15+, Edge 90+
