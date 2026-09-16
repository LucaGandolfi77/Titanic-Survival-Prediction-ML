# 🦎 Lazy Kaiju City Cleaner

> *Grogg just wants a nap. The city had other plans.*

Un gioco 3D casual in cui controlli un kaiju pigro che distrugge i rifiuti urbani con una coda verosimigliante fisica. Evita edifici ecologici, gestisci il karma e sfida te stesso a ripulire la città prima che il tempo scada.

![Lighthouse](https://img.shields.io/badge/Lighthouse-PWA%20A%2B-brightgreen)
![Three.js](https://img.shields.io/badge/Three.js-r158-black)
![JavaScript](https://img.shields.io/badge/JavaScript-ES2020-blue)
![PWA](https://img.shields.io/badge/PWA-Installable-blueviolet)
![WebGL](https://img.shields.io/badge/WebGL-2.0-brightgreen)

---

## 📋 Stack Tecnologico

| Tecnologia | Ruolo |
|------------|-------|
| [Three.js r158](https://threejs.org/) | Rendering 3D & WebGL |
| Vanilla ES Modules | Architettura modulare senza build |
| [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) | SFX procedurali (thud, crunch, yawn, sweep, ding) |
| [Verlet Integration](https://en.wikipedia.org/wiki/Verlet_integration) | Fisica coda interattiva |
| [PWA](https://web.dev/progressive-web-apps/) | Installabilità, caching offline, Service Worker |
| [Google Fonts](https://fonts.google.com/) | Orbitron, Rajdhani, Share Tech Mono |
| Analytics Module | Telemetria anonima con consent banner |
| Level Editor | Creazione livelli custom con save/share |
| Multiplayer Manager | 2 giocatori locale con P1/P2 input |
| NPC AI | A* pathfinding, chase AI, obstacle avoidance |
| Procedural Generator | Difficulty-based level config, seeded random |
| Music Engine | Markov chain adaptive music (calm/tense/danger/gameover) |
| Share Manager | Web Share API, deep linking, clipboard |
| Achievement System | 15 badges, XP levels, progress tracking |
| Push Manager | Notification API, daily challenge subscription |
| WebRTC Audio | PeerConnection voice chat with ICE/STUN |
| AR Mode | Device orientation tilt, WebXR detection |
| IoT Connector | Smart home framework, ambient triggers |
| Plugin Manager | Sandbox hook system, lifecycle management |

---

## 📂 Struttura del Progetto

| Requisito | Dettaglio |
|-----------|-----------|
| **Browser** | Chrome 90+, Firefox 88+, Safari 15+, Edge 90+ |
| **WebGL** | 2.0 (o 1.0 con fallback) |
| **Audio** | AudioContext supportato |
| **Opzionale** | Touchscreen (joystick mobile), Vibration API (haptic feedback) |
| **Rete** | Richiesta solo al primo caricamento (PWA caching successivo) |

---

## 🚀 Installazione e Avvio

### Sviluppo locale

```bash
# Clona il repository
git clone https://github.com/USER/lazy-kaiju.git
cd lazy-kaiju

# Opzione 1: Server Python (minimo)
python -m http.server 8080

# Opzione 2: Server Node.js (HTTPS + CORS consigliati)
npx serve -l 8080 --ssl

# Apri nel browser
# https://localhost:8080
```

> ⚠️ **Importante**: Per funzionalità PWA (installazione, Service Worker, AudioContext), il gioco deve essere servito via **HTTPS** o da `localhost`.

### Deploy in produzione

```bash
# Vercel
npx vercel

# Netlify
npx netlify deploy --prod

# GitHub Pages
gh pages deploy --dir .
```

---

## 🎮 Guida Rapida

| Azione | Input |
|--------|-------|
| Muovi Kaiju | `WASD` o Joystick mobile |
| Tail Slam | `SPAZIO` o pulsante mobile |
| Yawn | Automatico (timer casuale) |
| Pausa | `ESC` |

### Meccaniche di Gioco

- **Trash Cleared**: Colpisci i rifiuti con la coda per farli volare fuori dalla mappa e guadagnare punti
- **Karma**: Evita di distruggere edifici ecologici (penalità -20) e edifici normali (penalità -5)
- **Stamina**: Tail Slam costa 15 stamina, si rigenera nel tempo
- **Game Over**: Quando il karma scende a 0, gli attivisti vincono

---

## 📂 Struttura del Progetto

```
lazy-kaiju/
├── index.html          # Entry point con tutte le UI screens + importmap aggiornato
├── manifest.json       # PWA manifest (installabilità)
├── sw.js               # Service Worker (caching Stale-While-Revalidate)
├── importmap.json      # Import map con three/addons/ per GLTF
├── css/
│   ├── variables.css   # Custom properties design tokens
│   ├── reset.css       # Reset CSS base
│   ├── ui.css          # Screens, menu, settings styling + fadeIn transitions
│   ├── hud.css         # HUD overlay e controlli
│   └── extra.css       # Consent banner, editor toolbar, multiplayer cards
├── js/
│   ├── main.js         # GameController — loop, input, stato, modalità single/multi
│   ├── kaiju.js        # Kaiju entity — mesh, movimento, animazioni
│   ├── scene.js        # SceneManager — WebGL optimization, mobile detection
│   ├── city.js         # CityGenerator — generazione città + custom buildings
│   ├── items.js        # TrashManager + ParticleSystem (with dispose)
│   ├── tail.js         # TailPhysics — Verlet integration a 12 segmenti
│   ├── audio.js        # GameAudio — Web Audio API procedural sounds
│   ├── utils.js        # MathUtils + DOMHelper + AABB check
│   ├── analytics.js    # Analytics — anonymous tracking + consent banner
│   ├── editor.js       # LevelEditor — drag & drop, save/load, share
│   ├── multiplayer.js  # MultiplayerManager — 2 players mode
│   ├── level-data.js   # Level schema validation + encode/decode
│   ├── npc.js          # ActivistNPC — A* pathfinding, chase AI
│   ├── procedural.js   # ProceduralGenerator — difficulty config
│   ├── music.js        # MusicEngine — Markov chain adaptive music
│   ├── gesture.js      # GestureRecognition — camera motion detection
│   ├── share.js        # ShareManager — Web Share API + deep linking
│   ├── gamification.js # AchievementSystem — 15 badges, XP, levels
│   ├── push.js         # PushManager — Notifications API + daily challenge
│   ├── webrtc.js       # WebRTCAudio — PeerConnection voice chat
│   ├── ar.js           # ARMode — Device orientation tilt + WebXR
│   ├── iot.js          # IoTConnector — Smart home framework
│   └── plugins.js      # PluginManager — Sandbox hook system
├── importmap.json      # Import map con three/addons/ per GLTF
└── README.md           # Documentazione completa + Roadmap
```

---

## 🤝 Contribuire

### Branching Strategy

```
main          → Solo release stabili (tag: vX.Y.Z)
develop       → Branch di sviluppo principale
feature/*     → Nuove feature (es: feature/tail-physics)
fix/*         → Bug fix (es: fix/audio-context-suspend)
chore/*       → Manutenzione (es: chore/deps-update)
```

### Commit Convenzional (Conventional Commits)

```
feat(scope): aggiunge gestione push notifications
fix(kaiju): correggere collision detection tail
refactor(city): estrarre BuildingFactory module
perf(renderer): implementare frustum culling
docs(readme): aggiornare guida installazione
test(score): aggiungere unit test KarmaSystem
chore(deps): aggiornare Three.js a r160
```

### Linee Guida di Codice

- **Indentazione**: 4 spazi (nessuna tab)
- **Linting**: ESLint con configurazione `eslint-config-airbnb-base`
- **Commenti**: Solo quando necessario; codice auto-documentante
- **Export**: Named exports per ogni modulo
- **Testing**: Jest per unit test dei moduli di scoring e city generation
- **TypeScript**: Consigliato per nuove feature
- **Nessuna libreria esterna** aggiuntiva senza approvazione

---

## 🏗️ Roadmap & Future Implementations

### ✅ Fase 1 — Scalabilità (COMPLETA)

- [x] **WebGL Optimization** — Shadow map adattive (1024 mobile/2048 desktop), mobile fog tuning, ambient light boost su mobile
- [x] **Level Editor** — Click to place, right-click to remove, save/load via localStorage, share via base64 clipboard
- [x] **Analytics Anonima** — `navigator.sendBeacon`, localStorage queue, opt-in consent banner, GDPR-lite
- [x] **Multiplayer Locale** — 2 giocatori, P1: WASD+Space, P2: IJKL+Enter, win condition a 50 trash
- [x] **Asset Pipeline parziale** — Importmap aggiornato con `three/addons/` per GLTFLoader (Fase 2)

### ✅ Fase 2 — Intelligenza (COMPLETA)

- [x] **NPC Attivisti** — A* pathfinding, chase AI con obstacle avoidance, 3D mesh, dispose cleanup, respawn on catch
- [x] **Generazione Procedurale** — Difficulty-based config generator, seeded random, mobile performance adjust, level naming
- [x] **Audio Procedurale** — Markov chain adaptive music (calm/tense/danger/gameover), Web Audio oscillators, volume control, game state sync
- [x] **Gesture Recognition** — Camera-based motion detection (slam/yawn), toggle in settings, graceful fallback

### ✅ Fase 3 — Ecosistema (COMPLETA)

- [x] **Web Share Target** — Share API, deep linking (`?level=` / `?challenge=`), clipboard fallback
- [x] **Push Notifications** — Notification API, daily challenge, subscription management
- [x] **Gamification completa** — 15 badge achievements, XP system, levels, progress tracking
- [x] **WebRTC Audio** — PeerConnection voice chat with ICE/STUN, mute toggle
- [x] **AR Mode** — Device orientation tilt controls, table plane simulation, WebXR detection
- [x] **IoT Connector** — Smart home simulation with debounce, MQTT/WebSocket framework, ambient triggers
- [x] **Plugin System** — Web Worker sandbox, hook system, lifecycle management, API exposure

### 🔮 Fase 4 — Visione Futura

### 🔮 Fase 4 — Visione Futura

- Neural Network Kaiju behavior (TensorFlow.js / WebNN)
- Cross-platform cloud save & multiplayer
- User-generated content marketplace
- Spatial audio with HRTF
- Haptic pattern library for each event
- AI-generated city layouts
- Accessibility overhaul (screen reader, high contrast)

---

## 📜 Licenza

MIT License — Sentiti libero di usare, modificare e distribuire.

---

> **Sviluppato con** 🦎 + ☕ + 🦴 Three.js  
> `lazy-kaiju` — Where garbage meets glory.
