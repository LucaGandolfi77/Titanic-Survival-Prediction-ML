# Gravity Coffee Simulator

A surreal first-person space bar simulator where gravity shifts constantly, and you must serve coffee to alien customers without spilling.

## Features
* Custom 3D rigid body physics (boxes, spheres, cylinders)
* Fluid simulation via custom WebGL ShaderMaterial for coffee tilting
* 11 different Gravity states affecting play (Normal, Zero-G, Reversed, Diagonal, etc)
* First-person grab and tilt mechanics 
* Procedural Web Audio API sound effects (no external audio files required)

## Installation & Running

Everything is contained in this folder, and built via vanilla Vanilla JS + Three.js (`r158`).

To run:
1. Open this folder in your terminal.
2. Run a local web server to bypass CORS issues for module imports:
   \`python3 -m http.server 8000\`
3. Open \`http://localhost:8000\` in your browser window.

## Gameplay
- **Mouse Look**: Look around the bar.
- **Left Click**: Grab a coffee pot or cup. Click again to drop/throw.
- **Right Click (Hold) + Scroll Wheel**: Tilt your held object to pour coffee from the pot to the cup.
- **Listen for alarms**: The HUD will flash and an alarm will sound 3 seconds before gravity shifts. Brace your cups! 

*Never spill the coffee.*