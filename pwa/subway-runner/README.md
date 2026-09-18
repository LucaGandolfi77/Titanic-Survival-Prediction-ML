# Subway Runner

**An endless 3D runner PWA — dash through train tracks, dodge obstacles, collect coins, and beat your high score.**

Built with vanilla JavaScript and Three.js. Zero build tools, zero npm dependencies. Installable as a Progressive Web App on any device.

![Stack](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![Stack](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![Stack](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Stack](https://img.shields.io/badge/Three.js-000000?style=flat&logo=three.js&logoColor=white)
![Stack](https://img.shields.io/badge/PWA-5A0FC8?style=flat&logo=pwa&logoColor=white)

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/LucaGandolfi77/Titanic-Survival-Prediction-ML.git
cd Titanic-Survival-Prediction-ML/pwa/subway-runner

# Serve locally (any static server works)
python3 -m http.server 8080
# or
npx serve .

# Open in browser
open http://localhost:8080
```

No build step. No install. Just serve and play.

---

## Features

- **3D Endless Runner** — Three.js powered with dynamic camera, shadows, fog, and procedural track generation
- **3-Lane System** — Swipe or use arrow keys to switch lanes, jump over barriers, slide under obstacles
- **6 Playable Characters** — Each with unique abilities, unlockable via score milestones
- **4 Power-ups** — Magnet, Multiplier (x2), Shield, Jetpack
- **Combo System** — Chain coin collections for bonus points
- **Progressive Difficulty** — Speed increases every 1,000 points
- **Full PWA** — Installable, offline-ready, service worker cached
- **Haptic Feedback** — Vibration API for coin collection, hits, and game over
- **Screen Wake Lock** — Keeps screen on during gameplay
- **Web Share API** — Share your high score on social media
- **Synthesized Audio** — All sounds generated via Web Audio API (zero audio files)
- **Touch Controls** — Virtual buttons on mobile, keyboard on desktop
- **Local Leaderboard** — Top 10 scores persisted in localStorage

---

## Architecture

```
subway-runner/
├── index.html              App shell + import maps
├── manifest.webmanifest    PWA manifest
├── sw.js                   Service worker (cache-first)
├── css/
│   ├── variables.css       Design tokens
│   ├── reset.css           CSS reset
│   ├── hud.css             In-game HUD
│   ├── ui.css              Menu/pause/gameover screens
│   └── controls.css        Touch controls + responsive
├── js/
│   ├── main.js             Game orchestrator + state machine
│   ├── scene.js            Three.js renderer, camera, lights
│   ├── track.js            Infinite procedural track generation
│   ├── player.js           Player character + physics
│   ├── controls.js         Keyboard + touch swipe input
│   ├── obstacles.js        Obstacle system (barriers, trains, signs)
│   ├── collectibles.js     Coins + power-ups
│   ├── collision.js        AABB collision detection
│   ├── effects.js          Particle effects + screen shake
│   ├── audio.js            Web Audio API synthesized sounds
│   ├── hud.js              DOM HUD overlay updates
│   ├── ui.js               Screen management + leaderboard
│   ├── characters.js       Character definitions + unlock system
│   └── utils.js            ObjectPool, MathUtils, mesh helpers
└── README.md               This file
```

### Game States

```
MENU → PLAYING → GAME_OVER → MENU
              ↕
           PAUSED
```

### Rendering Pipeline

```
Three.js Scene → PerspectiveCamera → WebGLRenderer (ACES Filmic, PCF Soft Shadows)
```

### Input System

| Input | Desktop | Mobile |
|-------|---------|--------|
| Move Left | ← / A | Swipe Left / Left Button |
| Move Right | → / D | Swipe Right / Right Button |
| Jump | ↑ / W / Space | Swipe Up / Up Button |
| Slide | ↓ / S | Swipe Down / Down Button |
| Pause | Esc / P | Esc / P |

---

## Characters

| Character | Ability | Unlock Score |
|-----------|---------|-------------|
| Jake | — | 0 |
| Tricky | +2s Magnet | 10,000 |
| Fresh | 2x Coins for 5s | 50,000 |
| Spike | Speed +10% | 100,000 |
| Yolanda | Auto Shield /30s | 250,000 |
| King | 3x Score | 500,000 |

---

## Contributing

### Branching Strategy

- `main` — production-ready
- `develop` — integration branch
- `feature/*` — new features
- `fix/*` — bug fixes
- `refactor/*` — code improvements

### Code Style

- ES Modules with named exports
- No frameworks, no build tools
- CSS custom properties for theming
- Consistent file naming: `camelCase.js`
- Each module exports one primary class or function

### Commit Convention

```
feat: add jetpack power-up
fix: prevent double-jump at lane boundary
refactor: extract ObjectPool from obstacles
docs: update README with character table
```

---

## Browser Support

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| Three.js WebGL | ✅ | ✅ | ✅ | ✅ |
| Service Worker | ✅ | ✅ | ✅ | ✅ |
| Web Share API | ✅ | ❌ | ✅ | ✅ |
| Wake Lock API | ✅ | ❌ | ✅ | ✅ |
| Vibration API | ✅ | ✅ | ❌ | ✅ |

---

## Performance

- **Object pooling** for obstacles and collectibles (zero GC spikes)
- **Delta time** capped at 50ms to prevent physics explosion on tab-away
- **Pixel ratio** capped at 2x to avoid GPU overload on high-DPI displays
- **Fog culling** for visual distance masking
- **Shadow map** 2048x2048 with bias tuning
- **Service Worker** pre-caches all assets for instant offline load

---

## Future Roadmap

### Phase 1: Scalability

- [ ] Customizable character skins (colors, patterns, accessories)
- [ ] "Stunt" mode — specific missions (jump 5 trains in a row, collect 100 coins without dying)
- [ ] Daily challenges with rewards
- [ ] Detailed statistics (distance, coins, time played, best streaks)
- [ ] Achievement system with unlockable titles

### Phase 2: Intelligence

- [ ] **Transformers.js** offline voice commands ("jump!", "slide!") as alternative input
- [ ] **WebNN** adaptive track generation based on player skill level
- [ ] Machine learning difficulty prediction — tracks get harder when you're doing well, easier when struggling
- [ ] Procedural music generation using Web Audio API oscillators

### Phase 3: Ecosystem

- [ ] **Web Share Target API** — receive challenges from friends via shared links
- [ ] **Background Sync** — sync offline scores when connection returns
- [ ] **Push Notifications** — daily challenges and special events
- [ ] **Multiplayer** — real-time racing on parallel tracks via WebRTC
- [ ] **Cloud Leaderboard** — Firebase/Supabase for global rankings
- [ ] **Track Editor** — JSON-based modding system for community-created tracks
- [ ] **Seasonal Events** — holiday-themed tracks and limited-time characters

---

## License

MIT
