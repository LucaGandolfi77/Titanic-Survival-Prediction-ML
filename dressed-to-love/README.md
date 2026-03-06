
════════════════════════════════════════════════════════════════════
PROJECT STRUCTURE — GENERATE ALL FILES
════════════════════════════════════════════════════════════════════

dressed-to-love/
├── index.html                  ← entry point only (no game logic)
├── README.md
│
├── css/
│   ├── variables.css           ← all CSS custom properties + palette
│   ├── reset.css
│   ├── layout.css              ← panels, grid, sidebar
│   ├── wardrobe.css            ← outfit cards, color swatches
│   ├── ui.css                  ← modals, toasts, HUD, badges
│   ├── relationships.css       ← couple cards, status badges
│   └── animations.css          ← all @keyframes
│
├── js/
│   ├── main.js                 ← init Three.js, game loop, state machine
│   ├── state.js                ← global reactive state (all game data)
│   ├── characters.js           ← character definitions, personalities
│   ├── outfits.js              ← outfit catalog, style tags, color palettes
│   ├── wardrobe.js             ← wardrobe UI, drag-drop, outfit assignment
│   ├── relationships.js        ← pairing logic, compatibility engine
│   ├── events.js               ← random events, problems, solutions
│   ├── romance.js              ← flirt / dating / engagement / wedding
│   ├── betrayal.js             ← affair system, discovery mechanics
│   ├── divorce.js              ← divorce proceedings, asset split UI
│   ├── social.js               ← friendship, rivalry, hate system
│   ├── scene3d.js              ← Three.js scene: characters, stage, camera
│   ├── character3d.js          ← 3D character builder (body + outfit layers)
│   ├── ui.js                   ← all screens, modals, panels, toasts
│   ├── audio.js                ← Web Audio API: music + SFX
│   └── utils.js                ← helpers: random, lerp, UUID, save/load
│
└── assets/
    └── (no external assets needed — all shapes/colors via Three.js + CSS)
