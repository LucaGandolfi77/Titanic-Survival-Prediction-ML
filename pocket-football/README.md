# ⚽ Pocket Football

[![PWA](https://img.shields.io/badge/PWA-Ready-03C75A?logo=pwa&logoColor=white)]()
[![Lighthouse](https://img.shields.io/badge/Lighthouse-A-4285F4?logo=google&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

Un gioco calcistico 5-a-side completo, ottimizzato per dispositivi mobili, con controlli touch, audio sintetizzato e avversari AI. Zero dipendenze, zero asset esterni — puro HTML5 Canvas e JavaScript ES6.

## 🚀 Stack Tecnologico

![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![Canvas API](https://img.shields.io/badge/Canvas-API-000000?logo=html5)
![Web Audio](https://img.shields.io/badge/WebAudio-API-000000)
![PWA](https://img.shields.io/badge/PWA-03C75A?logo=pwa&logoColor=white)
![IndexedDB](https://img.shields.io/badge/IndexedDB-API-000000)

## 📱 Requisiti di Sistema

- **OS**: Android 8+ / iOS 13+ / Desktop Chrome/Firefox/Safari
- **Browser**: Chrome 80+, Firefox 75+, Safari 13+
- **RAM**: 256MB+ consigliati
- **Orientamento**: Portrait (consigliato) / Landscape
- **Connettività**: Non necessaria (gioco 100% offline dopo primo caricamento)

## 🛠️ Installazione e Avvio

### Locale (sviluppo)
```bash
python3 -m http.server 8080
# Oppure
npx serve -l 8080
```
Naviga a `http://localhost:8080`

### PWA (installa sul dispositivo)
1. Apri `index.html` su Chrome/Edge mobile
2. Tocca il menu (⋮) → "Aggiungi alla schermata Home"
3. L'app si installerà come applicazione nativa

### Docker (production)
```bash
docker build -t pocket-football .
docker run -p 8080:8080 pocket-football
```

## 🤝 Contribuire

### Branching Strategy
- `main` → Stabile, solo release
- `develop` → Integrazione features
- `feature/X-name` → Sviluppo feature
- `hotfix/X-name` → Fix urgenti in produzione

### Commit Convenzionali
```
feat: Aggiunto joystick a tre direzioni
fix: Risolto crash su resize durante partita
perf: Ottimizzato rendering canvas con offscreen caching
docs: Aggiornato README con istruzioni PWA
refactor: Separato physics da logic di match
test: Aggiunti test AI decision accuracy
chore: Aggiornata dependency canvas polyfill
```

### Linee Guida di Codice
- **Stile**: ES6 modules, no global variables (tranne `GAME` singleton)
- **Testing**: Ogni modulo deve avere test unitari corrispondenti in `/test`
- **Lint**: `eslint --ext .js js/` con regole `no-unused-vars` attiva
- **Type Safety**: JSDoc annotations obbligatorie per tutte le funzioni pubbliche
- **Performance**: Nessun allocazione in `requestAnimationFrame` hot paths

## 📂 Struttura del Progetto

```
pocket-football/
├── index.html              # Entry point
├── manifest.json           # PWA Manifest
├── sw.js                   # Service Worker
├── css/
│   ├── variables.css       # CSS Custom Properties
│   ├── reset.css           # Normalize
│   ├── ui.css              # Screens, HUD, layout
│   └── controls.css        # Joystick & buttons
├── js/
│   ├── main.js             # Game loop & init
│   ├── match.js            # Rules, scoring, time
│   ├── player.js           # Physics & state
│   ├── ball.js             # Ball physics
│   ├── ai.js               # Opponent AI
│   ├── renderer.js         # Canvas 2D drawing
│   ├── controls.js         # Touch & keyboard input
│   ├── audio.js            # Web Audio synthesis
│   ├── ui.js               # Screen management
│   ├── utils.js            # Vector2, helpers
│   ├── services/
│   │   └── storage.js      # IndexedDB wrapper
│   └── game/
│       ├── gamification.js # XP, levels, badges
│       └── tournament.js   # Tournament mode
└── assets/
```

## 🎮 Come Giocare

| Comando | Tasto / Touch |
|---------|---------------|
| Muovi | Joystick (sinistra) / WASD |
| Passa | PASS / Spazio |
| Tira | Tieni premuto SHOOT / F |
| Rinvia | Rilascia SHOOT |
| Ruba | TACKLE / T |
| Cambia giocatore | SWITCH / Tab |

## ⚙️ Funzionalità

- **Gamification**: Sistema XP con livelli, badge e statistiche dettagliate
- **Tornei**: Modalità torneo locale P2P Hot-Seat (Round Robin / Eliminazione)
- **Progressione persistente**: Tutti i progressi salvati in IndexedDB
- **PWA**: Installabile, offline-capable con Service Worker
- **3 difficoltà AI**: Easy, Medium, Hard
- **Audio sintetizzato**: Zero asset audio richiesti
- **Impostazioni salvate**: Difficoltà, colore, nome squadra persistenti

## 🔮 Roadmap & Future Implementations

### Fase 1: SCALABILITÀ ✅ (Implementata)
> Obiettivo: Trasformare il prototipo in un'architettura enterprise-grade

| # | Feature | Stato | Dettaglio |
|---|---------|-------|-----------|
| 1.1 | Gamification — Sistema di Progressione | ✅ | XP, livelli (1-10+), 13 badge, statistiche dettagliate |
| 1.2 | Tornei Multiplayer Locale | ✅ | P2P Hot-Seat, Round Robin / Eliminazione, bracket UI |
| 1.3 | Database Statistiche (IndexedDB) | ✅ | 5 store: Progression, Records, Tournaments, Settings, Arenas |
| 1.4 | Kit Editor — Persistenza | ✅ | Nome e colore squadra salvati in Settings store |
| 1.5 | Arena Editor | ⏳ | Pianificato per Fase 2 |

### Fase 2: INTELLIGENZA
> Obiettivo: Portare l'intelligenza artificiale on-device

- **AI Avversaria con Transformers.js** — Micro-modello <2MB fine-tuned per predire la prossima mossa del giocatore
- **WebNN — NPU Acceleration** — Sfruttare la Neural Processing Unit del dispositivo
- **Coach AI** — Analisi post-partita con consigli tattici e heatmap
- **Procedural Commentary** — Sintesi vocale on-device con Web Speech API

### Fase 3: ECOSISTEMA
> Obiettivo: Espandere in piattaforma completa

- **Web Share Target** — Condividi risultato → amico apre direttamente la partita
- **Sensori Ambientali** — Giroscopio, luce, geolocalizzazione per gameplay adattivo
- **Push Notifications** — Richiamo partita tramite Service Worker Push API
- **Web Bluetooth** — Controller PS4/PS5/Xbox nativi dal browser
- **Canvas Export** — Highlights video automatici con MediaRecorder API
- **File System Access** — Importa/esporta pacchetti custom `.pfpack`

### 🗺️ Sintesi Visiva Roadmap

```
█████████████████████████████████████████████████████████████████████████████
FASE 1: SCALABILITÀ                          ██████████████░░░░░░░░░░░░░░░░
FASE 2: INTELLIGENZA                         ░░░░░░░░░░░████████████████░░░
FASE 3: ECOSISTEMA                           ░░░░░░░░░░░░░░░░░░░░██████████
█████████████████████████████████████████████████████████████████████████████
    M1  M2  M3  M4  M5  M6  M7  M8  M9  M10 M11 M12
```

### 🏆 Visione d'Insieme

> Pocket Football diventerà una **piattaforma di gaming sociale on-device** — un ecosistema dove l'intelligenza artificiale locale, la gamification profonda e le API del browser moderno si fondono per creare un'esperienza nativa nel browser, senza bisogno di server, senza dipendenze esterne, e 100% rispettosa della privacy dell'utente.

---
*Creato con ⚽ e puro codice.*
