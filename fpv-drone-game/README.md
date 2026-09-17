# 🛸 DRONE STRIKE — FPV Combat Simulator

> A fully playable first-person drone combat game built with Three.js and vanilla JavaScript. Pilot an FPV racing drone, destroy waves of enemies, and survive as long as you can.

![Lighthouse](https://img.shields.io/badge/Lighthouse-PWA--Ready-A3E635?logo=lighthouse&style=flat-square)
![Three.js](https://img.shields.io/badge/Three.js-r158-00ff41?logo=three.js&style=flat-square)
![PWA](https://img.shields.io/badge/PWA-Installable-06b6d4?style=flat-square)
![Web Audio](https://img.shields.io/badge/Audio-Web%20API-ff6600?style=flat-square)
![ES Modules](https://img.shields.io/badge/JS-ES%20Modules-blue?style=flat-square)

---

## 🚀 Quick Start

```bash
# Option A — Node.js (recommended)
npx serve .

# Option B — Python
python3 -m http.server 8080

# Open http://localhost:8080 (or shown port)
```

> ⚠️ Must be served via HTTP/HTTPS (not `file://`) for ES Modules and Web Audio.

---

## 🎮 Controls

| Action | Keyboard / Mouse | Touch |
|---|---|---|
| Throttle Up / Down | W / S or ↑ / ↓ | Left Joystick Y |
| Yaw Left / Right | A / D or ← / → | Left Joystick X |
| Pitch Up / Down | Mouse Y (pointer-lock) | Right Joystick Y |
| Roll Left / Right | Mouse X (pointer-lock) | Right Joystick X |
| Fire Cannon | Left Click / Space | 🔥 Button |
| Fire Missile | Right Click / E | 🚀 Button |
| Boost | Shift | — |
| Pause | Escape | — |

---

## ✨ Features

- **Real Arcade Flight Physics** — gravity, drag, lift, roll
- **3 Enemy Types + Boss** — Ground Turret, Patrol Drone, Heavy Gunship, and multi-phase Boss (Wave 10+) with distinct AI
- **Wave Survival** — progressive difficulty with 5+ procedurally scaled waves
- **Weapons** — rapid-fire cannon (10/s), homing missiles (8 with recharge), **Plasma Beam** (Q key)
- **Power-Ups** — Shield, Overdrive, Ghost, Nano-Regen — collect in the world
- **Day/Night Cycle** — dynamic lighting, fog, and bloom intensity every 120s
- **Post-Processing** — Bloom, Vignette, Tone Mapping via EffectComposer
- **Procedural World** — terrain, buildings, trees generated on the fly
- **Particle Effects** — explosions, smoke trails, drone trail, muzzle flash with physics
- **Synthesized Audio** — Web Audio API, zero external audio files
- **Full HUD** — health bar, ammo, altitude, speed, compass, crosshair, damage vignette
- **Touch Controls** — dual virtual joysticks + fire buttons for mobile/tablet
- **High Score Board** — persistent via `localStorage`
- **PWA** — installable, service worker cached, offline-capable core
- **Haptic Feedback** — vibration on hits, damage, explosions, wave completion

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Three.js r158** (CDN + cached) | 3D rendering, camera, scene graph |
| **Vanilla JavaScript ES Modules** | All game logic — no frameworks |
| **Web Audio API** | Real-time procedural sound synthesis |
| **CSS3 + CSS Variables** | HUD, UI overlays, animations |
| **Service Worker + Cache API** | Offline support, CDN caching |
| **Web App Manifest** | PWA installability |
| **Web Speech API** | Voice commands *(Phase 2 — planned)* |

---

## 📱 System Requirements

- **Browser**: Chrome 90+, Firefox 88+, Safari 15+, Edge 90+
- **OS**: Android 10+, iOS 14+, Windows 10+, macOS 11+
- **Hardware**: WebGL 2.0 capable GPU, touch screen for mobile
- **Network**: Internet on first load (Three.js CDN). Subsequent loads work offline via Service Worker cache.

---

## 📁 Project Structure

```
fpv-drone-game/
├── index.html           # Entry point — canvas, HUD, UI screens, SW registration
├── manifest.json        # PWA manifest (theme, icons, display mode)
├── sw.js                # Service Worker — caching strategies
├── README.md
├── css/
│   ├── variables.css    # Design tokens (colors, fonts, spacing)
│   ├── reset.css        # Normalize + base styles
│   ├── ui.css           # Menu screens, game over, controls info
│   ├── hud.css          # HUD elements — health, ammo, compass, meters
│   └── controls.css     # Virtual joysticks, fire buttons
└── js/
    ├── main.js          # Game loop, state machine, system orchestration
    ├── drone.js         # Flight physics, FPV camera, mesh
    ├── world.js         # Terrain, buildings, trees, grid
    ├── scene.js         # Renderer init, lights, camera, resize handler
    ├── collision.js     # Sphere-AABB, sphere-sphere, ground checks
    ├──     enemies.js       # Enemy AI (3 types + Boss), wave system, projectile pool
    ├── weapons.js       # Cannon (bullet pool) + missiles (homing)
    ├── effects.js       # Particles, explosions, muzzle flash, sparks
    ├── audio.js         # Web Audio synthesis — all sounds procedurally
    ├── controls.js      # Keyboard, mouse (pointer-lock), touch joysticks
    ├── hud.js           # HUD DOM updates (per-frame)
    ├── ui.js            # Screen management, high scores (localStorage)
    └── utils.js         # ObjectPool (O(1)), MathUtils, helpers
```

---

## 🤝 Contributing

### Branching Strategy

```
main            → Production stable
develop         → Integration branch
feature/*       → New features (e.g., feature/voice-controls)
fix/*           → Bug fixes (e.g., fix/joystick-memory-leak)
perf/*          → Performance optimizations
```

### Commit Convention (Conventional Commits)

```
feat:     Add homing missile system
fix:      Resolve touch listener memory leak
perf:     Optimize ObjectPool from O(n) to O(1)
refactor: Restructure collision detection module
docs:     Update README with PWA instructions
chore:    Update Three.js dependency
```

### Code Style

- **Indentation**: 2 spaces
- **Variables**: `camelCase`; constants: `UPPER_SNAKE_CASE`
- **Modules**: Named ES module exports
- **Comments**: JSDoc for public APIs; section dividers (`/* ── ... ── */`) for internals
- **No redundant comments** — code should be self-documenting

---

## 🌐 Browser Support

| Browser | Minimum Version | Status |
|---|---|---|
| Chrome | 90+ | ✅ Full Support |
| Firefox | 88+ | ✅ Full Support |
| Safari | 15+ | ✅ Full Support |
| Edge | 90+ | ✅ Full Support |

---

## 📋 Piano di Implementazione (TODO Live)

> Stato: **In Corso** — Fase A (Grafica Shock) quasi completa, Fase B in corso

### 🔬 FASE A: Grafica Shock (Priorità Massima)

| ID | Task | Stato | Stima | Note |
|----|------|-------|-------|------|
| A1 | **Post-Processing Pipeline** — Bloom + Vignette + Tone Mapping via EffectComposer | ✅ Completato | — | `js/postprocessing.js` |
| A2 | **Day/Night Cycle** — Luce dinamica, cambio colori cielo/nebbia ogni 120s | ✅ Completato | — | Integrato in main.js |
| A3 | **Particelle Avanzate** — Trail effect, volumetric light cone (GPU smoke in corso) | ⏳ In Corso | 5 giorni | `js/drone.js` |

### ⚔️ FASE B: Gameplay Profondo

| ID | Task | Stato | Stima | Note |
|----|------|-------|-------|------|
| B1 | **Plasma Gun** — Raggio continuo cangiante (tasto Q) | ✅ Completato | — | `js/weapons.js` |
| B2 | **Power-Up System** — Shield + Overdrive + Ghost + Nano-Regen | ✅ Completato | — | `js/powerups.js` |
| B3 | **Boss Fight** — Boss speciale Wave 10+ con fasi | ✅ Completato | — | `js/enemies.js` |
| B4 | **Wingman AI** — Drone compagno con 3 modalità | ⏳ Pending | 6 giorni | `js/wingman.js` |

### 🌍 FASE C: Espansione Mondi

| ID | Task | Stato | Stima | Note |
|----|------|-------|-------|------|
| C1 | **Volcano Base** — Lava, fumo tossico, terreno deformabile | ⏳ Pending | 1 settimana | Nuovo terrain shader |
| C2 | **Submarine** — Camera subacquea, physics acqua | ⏳ Pending | 1 settimana | Shader acqua + bolle |
| C3 | **Space Station** — Zero-G, asteroidi, luce diretta | ⏳ Pending | 1 settimana | Physics modificata |
| C4 | **Mushroom Forest** — Bioluminescenza, spore lente | ⏳ Pending | 1 settimana | Particelle + shader |
| C5 | **Clockwork Factory** — Steampunk, ingranaggi, vapore | ⏳ Pending | 1 settimana | Animazioni meccaniche |
| C6 | **The Void** — Infinito, nemici dal buio, shader psichedelico | ⏳ Pending | 1 settimana | Shader custom |

### 🎮 FASE D: Gamification & Progression

| ID | Task | Stato | Stima | Note |
|----|------|-------|-------|------|
| D1 | **Pilot Level System** (1-50) con XP | ⏳ Pending | 3 giorni | IndexedDB |
| D2 | **Challenge System** (giornaliero/settimanale) | ⏳ Pending | 4 giorni | Timer + storage |
| D3 | **Drone Skins** ( rarità comune → mitica) | ⏳ Pending | 5 giorni | Custom shader + LOD |
| D4 | **Generative Music Engine** | ⏳ Pending | 3 giorni | Web Audio API layers |

### 🧪 FASE E: Sperimentazione

| ID | Task | Stato | Stima | Note |
|----|------|-------|-------|------|
| E1 | **Screen as Sensor** — Accelerometro per pitch | ⏳ Pending | 1 giorno | DeviceOrientation API |
| E2 | **Spectator Mode** — Replay IA piloti | ⏳ Pending | 3 giorni | Input recording |
| E3 | **Ghost Data** — Esporta/importa replay | ⏳ Pending | 2 giorni | File format |
| E4 | **AI Coach** — Suggerimenti basati su statistiche | ⏳ Pending | 4 giorni | Analytics locale |

---

## 📈 Progresso Globale

```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░  ~50%
FASE A ████████████████████████████░░░░░░░░  2/3 completato
FASE B ██████████████████████████░░░░░░░░░░░░  3/4 completato
FASE C ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0/6
FASE D ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0/4
FASE E ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0/4
```

---

## 🔮 Roadmap & Future Implementations

### 📊 3-Phase Vision

```
┌──────────────────────────────────────────────────────────────────┐
│  FASE 1: SCALABILITÀ (Q1–Q2)                                         │
│  → Solid architecture, performance, full PWA                      │
├──────────────────────────────────────────────────────────────────┤
│  FASE 2: INTELLIGENZA (Q3–Q4)                                        │
│  → Local AI, personalization, game intelligence                   │
├──────────────────────────────────────────────────────────────────┤
│  FASE 3: ECOSISTEMA (Q5–Q6)                                          │
│  → Multiplayer, social, gamification, ambient context             │
└──────────────────────────────────────────────────────────────────┘
```

> ⚡ **Visione Aggiornata**: Oltre alle 3 fasi originali, sono state aggiunte le FASI A–E per l'espansione del gameplay. Vedi tabella sopra per il piano dettagliato.

---

### 🏗️ FASE 1: Scalabilità (Q1–Q2)

| ID | Task | Tech | Impact |
|---|---|---|---|
| 1.1 | **Service Worker completo** — Cache-First per Three.js + Fonts, Stale-While-Revalidate per asset dinamici | Cache API | PWA installabile, offline completo |
| 1.2 | **Web App Manifest** — Icone, theme, fullscreen, share-target configurato | manifest.json | Installabile da home screen |
| 1.3 | **Lazy Loading** — Dynamic `import()` per Effects, Audio, Enemy subtypes | `import()` | TTI ridotto del 40% |
| 1.4 | **Refactoring ObjectPool** — Free-list stack O(1) ✅ *Completato* | — | Performance sotto carico |
| 1.5 | **Background Sync** — Salvataggio high-score in IndexedDB con sync quando online | Background Sync API | Integrità dati |
| 1.6 | **Error Boundary System** — Try/catch nel game loop + error reporting non-intrusivo | Custom | Zero crash silent |
| 1.7 | **Adaptive Quality** — FPS-based shadow map & particle quality | `requestAnimationFrame` timing | GPU-friendly |
| 1.8 | **Lighthouse Optimization** — Rimuovere render-blocking, ottimizzare CSS, meta tag | Varie | Score >90 in tutte le categorie |

---

### 🧠 FASE 2: Intelligenza (Q3–Q4)

| ID | Task | Tech | Impact |
|---|---|---|---|
| 2.1 | **Enemy AI Prediction** — Micro-modello Transformers.js che predice traiettoria del drone dai pattern di input | Transformers.js + WebNN | Ennemi adattivi, gameplay imprevedibile |
| 2.2 | **Procedural Mission Generation** — LLM locale (distilizzato) genera briefing e obiettivi dinamici | WebLLM / ONNX | Rigiocabilità infinita |
| 2.3 | **Voice Commands** — "Fire!", "Missile!", "Boost!", "Pause!" via Web Speech API | `Web Speech API` | Hands-free, accessibilità |
| 2.4 | **Biometric Player Profile** — Riconoscimento impronta/viso per profili giocatore multipli | `WebAuthn` API | Multi-user su stesso dispositivo |
| 2.5 | **Adaptive Soundtrack** — Generazione musica ambientale basata su stato di gioco (tensione, velocità) | Web Audio API + algoritmo | Immersione profonda |
| 2.6 | **Smart Replay System** — Registra input (non video), riproduci con simulazione deterministica | Input logging + timestamp | Share delle migliori battaglie |

---

### 🌍 FASE 3: Ecosistema (Q5–Q6)

| ID | Task | Tech | Impact |
|---|---|---|---|
| 3.1 | **Multiplayer PvP** — WebRTC DataChannel per duelli drone | `WebRTC` + WebSocket | Gioco competitivo in tempo reale |
| 3.2 | **Web Share Target Challenges** — Condividi "Survived 3 waves!" con amici | `Web Share Target API` | Viralità, engagement |
| 3.3 | **Geolocation Arena** — Arena virtuale basata su posizione GPS reale | `Geolocation API` + Mapbox | Contestuale, immersivo |
| 3.4 | **Push Notifications** — "Your high score was beaten! Play again." | `Push API` + Service Worker | Retention |
| 3.5 | **Gamification Engine** — XP, livelli, badge, daily quests, achievement system | Custom + IndexedDB | Engagement a lungo termine |
| 3.6 | **File System Access** — Esporta replay/battle report come PDF | `File System Access API` | Utilità professionale |
| 3.7 | **Ambient Context** — Cambia gameplay in base a ora del giorno (notte = visibilità ridotta), meteo via API, sensori accelerometro (inclinazione dispositivo = input aggiuntivo) | `Screen Wake Lock` + `Ambient Light` + `DeviceMotionEvent` | Esperienza unica e viva |
| 3.8 | **Home Screen Mini-App** — Mini-drone game direttamente dall'icona home | `Manifest shortcuts` | Instant engagement |

---

### 🗺️ Visual Roadmap

```
╔══════════════════════════════════════════════════════════╗
║                   🛸 DRONE STRIKE — ROADMAP 2026               ║
╠══════════════════════════════════════════════════════════╣
║                                                                ║
║  ┌──── Q1──Q2 ────┐  ┌──── Q3──Q4 ────┐  ┌──── Q5──Q6 ────┐  ║
║  │ SCALABILITÀ     │  │ INTELLIGENZA    │  │ ECOSISTEMA     │  ║
║  │                  │  │                  │  │                │  ║
║  │ ✅ PWA Complete  │  │ 🤖 AI Enemy     │  │ 🌐 Multiplayer │  ║
║  │ ✅ Lazy Loading  │  │ 🧠 LLM Missions │  │ 📤 Share Target│  ║
║  │ ✅ Error Handle  │  │ 🎙️ Voice Ctrl  │  │ 📍 Geo Arena   │  ║
║  │ ✅ Lighthouse    │  │ 👤 Biometric    │  │ 🔔 Push Notif  │  ║
║  │ ✅ BG Sync       │  │ 🎵 Smart Audio  │  │ 🏆 Gamification│  ║
║  │ ✅ Adaptive Q.   │  │ 🎬 Replay Sys   │  │ 🌍 Ambient     │  ║
║  └─────────────────┘  └─────────────────┘  └────────────────┘  ║
║                                                                ║
╚══════════════════════════════════════════════════════════╝
```

---

## 📄 Licenza

MIT — Sentiti libero di usare, modificare e distribuire.
