# Hypercube Delivery Service

A surreal 3D puzzle-driving game where the player drives a delivery van through the faces of a 4D hypercube (tesseract) projected into 3D space. 

Built entirely with vanilla JavaScript, CSS, and Three.js (no other frameworks).

## Features
- **4D Navigation:** 8 completely distinct 3D cells connected via portals that flip the world 90 degrees around you.
- **Inception-style Transitions:** Seamless procedural world-flipping that reorients gravity and your perspective.
- **Physics-based Van:** Simple arcade driving model with drifting.
- **Procedural Audio:** Engine sounds, portal whooshes, and music synthed through Web Audio API. 
- **Time Pressure:** Deliver packages before they expire. Strategy and map-reading are essential.

## How to Play
- **W, A, S, D** or **Arrow Keys** to drive.
- **Spacebar** to brake.
- **H** to honk the horn.
- Follow the glowing portals. Watch your 4D minimap to navigate the 8 connected topological rooms.

## Installation
No build step required. Just serve the directory with any local static web server to avoid CORS issues when loading modules:
\`\`\`bash
python3 -m http.server 8000
# or
npx serve .
\`\`\`
Then visit `http://localhost:8000`.