# ⚔️ Imperium Mobile

> *Command your citizens. Conquer the ages.*

Age of Empires-style real-time strategy PWA optimized for iPhone. Gather resources, move unique citizens across a hex-tile map, build structures, train units, age up through 4 historical eras, and conquer against AI factions.

---

## 📋 Stack

| Technology | Role |
|------------|------|
| Canvas 2D | Map, units, buildings, effects rendering |
| HTML/CSS DOM | UI screens, HUD, touch controls overlay |
| Vanilla ES Modules | Architecture (no build step) |
| Service Worker | Offline caching (Stale-While-Revalidate) |
| Web Audio API | Procedural SFX + adaptive music (planned) |
| IndexedDB | Save games, citizen data, chronicle |
| localStorage | Quick settings, campaign progress, high scores |
| Google Fonts | Chakra Petch (display) + Inter (UI) |

---

## 📂 Project Structure

```
imperium-mobile/
├── index.html              # Entry point: canvas + all UI screens + inline event wiring
├── manifest.json           # PWA manifest (iPhone-optimized)
├── sw.js                   # Service Worker (Stale-While-Revalidate precache)
├── apple-touch-icon        # Inline SVG in index.html
├── css/
│   ├── variables.css       # Design tokens (colors, spacing, fonts)
│   ├── reset.css           # CSS base reset
│   ├── ui.css              # Screens, menus, buttons, transitions
│   ├── hud.css             # HUD overlay (resources, age, population)
│   ├── controls.css        # Joystick, action buttons, touch controls
│   └── animations.css      # Transitions, pulse, shake, shimmer effects
├── js/
│   ├── main.js             # GameController: init, game loop, touch input, render loop
│   ├── map.js              # Hex-tile map state, A* pathfinding, exploration
│   ├── citizen.js          # Citizen entities (traits, movement, tasks, morale, gather)
│   ├── building.js         # Building placement, construction, production
│   ├── resource.js         # 4 resources (food/wood/gold/stone), income, population
│   ├── manpower.js         # Manpower pool: regen, costs, overdraft, capacity
│   ├── unit.js             # Military unit management (spawn, move, auto-attack)
│   ├── combat.js           # Combat resolution, AI attack triggers
│   ├── ai.js               # AI opponents (build, train, attack, mercenaries, diplomacy-aware)
│   ├── diplomacy.js        # Relation state machine, diplomatic actions, events, reputation
│   ├── mercenary.js        # Gold-based unit hiring system
│   ├── fog.js              # Fog of war (revealed, visible, explored)
│   ├── time-ripple.js      # Time Ripple: accelerate/slow/freeze zones
│   ├── weather.js          # Dynamic weather system with gameplay effects
│   ├── dna-mutation.js     # Citizen DNA mutation system
│   ├── hero.js             # Hero units: hire, level, XP, abilities
│   ├── territory.js        # Territory claiming around buildings
│   ├── campaign.js         # Campaign missions: objectives, progression, win/loss
│   ├── assets.js           # Asset preloader
│   ├── data/
│   │   ├── factions.js     # 6 civilizations with relations & reputation
│   │   ├── buildings.js    # 14 building types with costs, manpower, production
│   │   ├── units.js        # 13+ units with manpowerCost, stats, faction unique
│   │   ├── tech-tree.js    # 4 ages + 10 technologies
│   │   └── map-templates.js# Procedural hex-map generator with terrain types
│   ├── render/
│   │   ├── camera.js       # Camera: pan, zoom, smooth interpolation
│   │   ├── renderer.js     # Canvas 2D renderer (tiles, buildings, units, particles, weather overlay)
│   │   └── particles.js    # Particle effects (gather, combat, build, age-up)
│   ├── ui/
│   │   ├── screens.js      # Screen manager (show/hide navigation)
│   │   ├── hud.js          # HUD controller (resources, age, population, manpower)
│   │   └── notifications.js# Toast system, haptic feedback, toggle helper
│   └── utils/
│       ├── helpers.js      # rand, clamp, lerp, format, generateName
│       ├── hex-math.js     # Hex grid math (distance, neighbors, pixel conversion)
│       └── storage.js      # IndexedDB wrapper (saves, citizens, chronicle)
└── README.md
```

---

## 🎮 Gameplay

### Core Loop
```
Gather → Build → Train → Age Up → Conquer → Repeat
```

### Controls (iPhone Touch)
| Gesture | Action |
|---------|--------|
| Tap citizen | Select/deselect |
| Drag citizen to target | Move/assign task |
| Tap empty land (with citizen selected) | Move to location |
| Single-finger drag (empty) | Pan camera (joystick) |
| Pinch | Zoom in/out |

### Resources
- 🌾 **Food** — Gathered from plain/swamp terrain, farms
- 🪵 **Wood** — Gathered from forest terrain, lumber camps
- 🪙 **Gold** — Gathered from ruins terrain, mines, markets
- 🪨 **Stone** — Gathered from hill terrain, quarries

### Ages
| Age | Unlock |
|-----|--------|
| Dark Age | Town Center, Houses, Farms, Lumber Camps, Walls |
| Feudal Age | Barracks, Archery Range, Market, Militia, Archers |
| Castle Age | Stable, Workshop, Tower, Knights, Trebuchets |
| Imperial Age | Dock, All technologies, Powerful unique units |

### Factions
| Faction | Bonus | Unique Unit |
|---------|-------|-------------|
| 🏛️ Roma | +15% wall strength | Legionary |
| ⚔️ Vikings | +20% naval speed | Berserker |
| 🐎 Mongols | +25% cavalry speed | Horde Rider |
| 𓂀 Egyptians | +20% gold generation | Pharaoh Guard |
| 🌿 Celts | +20% wood gathering | Woad Warrior |
| ⛩️ Japan | +15% attack speed | Samurai |

---

## 🚀 Running the Game

### Local Development
```bash
cd imperium-mobile
python3 -m http.server 8080
# Open http://localhost:8080
```

> ⚠️ **Important**: PWA features (Service Worker, installability) require HTTPS or localhost.

### Deploy
```bash
# Vercel
npx vercel

# Netlify
npx netlify deploy --prod

# GitHub Pages
gh pages deploy --dir .
```

---

## 📱 PWA Features

- **Installable** — Add to Home Screen on iPhone Safari
- **Offline** — Fully playable without network after first load
- **Standalone mode** — Runs as a native app (no browser UI)
- **Haptic feedback** — Taptic engine on combat, building, aging
- **Safe area** — Respects iPhone notch and home indicator
- **Portrait optimized** — Designed for iPhone screen proportions

---

## 🔧 Browser Requirements

| Feature | Minimum |
|---------|---------|
| Safari iOS | 15+ |
| Chrome Android | 90+ |
| WebGL | Not required (Canvas 2D only) |
| AudioContext | Required for SFX |
| Touchscreen | Required for mobile |

---

## 🗺️ Feature Roadmap

### Phase 1 — Foundation (✅ Complete)
- PWA scaffolding, manifest, service worker
- Hex map generation with procedural terrain
- Citizen selection, movement, auto-gather
- Building placement (Town Center, House)
- Resource system with HUD
- Touch input (tap, drag, pinch, joystick)
- Camera pan/zoom
- Multiple UI screens (menu, settings, how-to-play, campaign, skirmish)
- Loading screen with skeleton UI

### Phase 2 — Gameplay (✅ Complete)
- All 14 buildings with construction progress & production
- All 13 units + 6 faction unique units
- Combat system with auto-attack, HP bars, damage calculation
- AI opponents (build, train, attack, mercenary hiring)
- Aging up with resource verification and unlocks
- Technology tree research (10 technologies across 4 ages)
- Fog of war (unexplored, explored, visible states)
- Particle effects (gather, build, combat, age-up)
- Mini-map with fog and AI base indicator

### Phase 3 — Strategy Depth (✅ Complete)
- **Manpower system** — Finite army cap, regenerates with population/buildings, overdraft penalties, mercenary alternative
- **Diplomacy system** — 5 relation tiers (Allied/Friendly/Neutral/Tense/At War), 5 diplomatic actions (Peace/Ally/Trade/NAP/Declare War), random events, reputation tracking, AI-aware behavior
- **Mercenary system** — Gold-based unit hiring (3× cost), available when manpower insufficient
- Build panel & Train panel with age-gated UI
- Diplomacy panel with relation indicators and action buttons
- Manpower bar in HUD with overdraft warning

### Phase 4 — Innovation (✅ Complete)
- **Time Ripple** — Accelerate/slow/freeze zones via `[RIP]` button; 30s cooldown, 50 gold cost
- **Dynamic Weather** — Clear/rain/storm/fog/snow; affects combat visibility & gather rates
- **DNA Mutation** — 5% chance per new citizen for positive or negative traits
- **Hero Units** — Hire for 200 gold; level 1-10; XP from combat; abilities at levels 3/5/8/10
- **Territory Claiming** — Auto-claim around buildings (radius 5); +10% bonus per 100 tiles owned

### Phase 5 — Multiplayer & Launch (In Progress)
- Campaign missions (8 missions, objectives, win/loss, localStorage progression)
- 2-Player mode (hot-switch, faction selection, active player indicator)
- Polish and launch preparation

---

## 📜 License

MIT — Feel free to use, modify, and distribute.
