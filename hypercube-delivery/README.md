# 🌀 Hypercube Delivery Service

> Un gioco di puzzle guida 3D surreale dove guidi un furgone attraverso le facce di un ipercubo (tesseract) proiettato in 3D.

Dalla cella 0 alla cella 7, attraverso 8 ambienti procedurali unici. Consegna pacchi prima che scadano, naviga portali dimensionali e padroneggia la rotazione del mondo 4D.

---

## 🛠️ Stack Tecnologico

| Tecnologia | Ruolo |
|------------|-------|
| ![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=flat&logo=javascript) | Linguaggio principale — Vanilla ES Modules |
| ![Three.js](https://img.shields.io/badge/Three.js-000000?style=flat&logo=three.js&logoColor=white) | Rendering 3D & WebGL |
| ![Web Audio API](https://img.shields.io/badge/Web%20Audio-API-FF6B6B?style=flat) | Sintesi procedurale audio |
| ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3) | UI, animazioni, layout |
| ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5) | Struttura applicativa |
| ![WebGL](https://img.shields.io/badge/WebGL-990000?style=flat&logo=webgl&logoColor=white) | Accelerazione GPU |
| ![PWA](https://img.shields.io/badge/PWA-007ACC?style=flat&logo=googlechrome&logoColor=white) | Progressive Web App |
| ![Service Worker](https://img.shields.io/badge/Service%20Worker-FF8800?style=flat) | Offline-first caching |

---

## 💻 Requisiti di Sistema

| Requisito | Minimo |
|-----------|--------|
| Browser | Chrome 90+ / Firefox 90+ / Edge 90+ / Safari 15+ |
| GPU | WebGL 2.0 support |
| RAM | 2 GB+ consigliati |
| OS | Qualsiasi desktop o mobile moderno |
| Connessione | Internet (per Three.js CDN) o locale |

---

## 🚀 Installazione e Avvio

### Via Python (consigliato)
```bash
python3 -m http.server 8000
```

### Via Node.js
```bash
npx serve .
```

### Via Docker
```bash
docker run -p 8000:80 -p 8000:8001 python:3-slim http.server 8000
```

Poi apri `http://localhost:8000` nel browser.

> ⚠️ **Importante**: Il progetto usa ES Modules (`importmap`), quindi deve essere servito via HTTP — non aprire `index.html` direttamente con `file://`.

---

## 🎮 Comandi

| Tasto | Azione |
|-------|--------|
| `W` / `↑` | Accelerare |
| `S` / `↓` | Retromarcia |
| `A` ← | Svolta a sinistra |
| `D` → | Svolta a destra |
| `Space` | Freno |
| `H` | Clacson |
| `C` | Cambia telecamera |
| `Esc` | Pausa |

Touch: Joystick a sinistra, pulsanti BRAKE/HORN a destra.

---

## 🤝 Contribuire

### Branching Strategy
```
main          → Solo release stabili (proteggi con branch protection)
develop       → Branch di sviluppo principale
feature/*     → Nuove feature (merge via PR verso develop)
hotfix/*      → Fix critici in produzione (merge verso main)
chore/*       → Manutenzione, tooling, dipendenze
```

### Commit Convenzional
```
feat:        Aggiunta steering gyro per dispositivi mobili
fix:         Risolto memory leak nelle geometrie dei portali
refactor:    Estrazione physics in VanController
perf:        Ottimizzazione shadow map per mobile
ui:          Riprogettazione HUD per schermi piccoli
docs:        Aggiornamento README e guida utente
 chore:      Aggiornamento dipendenze
```

### Linee Guida di Codice
- **Nessun commento superfluo** — codice auto-documentante
- **Nomi descrittivi**: `calculateDeliveryBonus` non `calcDD`
- **Funzioni < 30 righe**, classi < 200 righe
- **No `var`**, solo `const`/`let`
- **No `innerHTML`** con dati dinamici — usa DOM API
- **Disposal sempre**: ogni geometria/materiale Three.js creato deve avere un percorso di `dispose()`
- **Error handling**: ogni operazione asincrona deve avere `.catch()` o `try/catch`
- **Test**: ogni utility in `utils/` deve avere test unitari
- **Format**: 2 spazi, semicolon obbligatorio, single quotes

---

## 📁 Struttura del Progetto

```
hypercube-delivery/
├── index.html          # Entry point
├── manifest.json       # PWA manifest
├── sw.js              # Service Worker
├── css/
│   ├── variables.css   # Design tokens CSS custom properties
│   ├── reset.css       # Normalize + base styles
│   ├── ui.css          # Screens, buttons, layout
│   ├── hud.css         # HUD overlays, controls
│   └── animations.css  # Keyframes
├── js/
│   ├── main.js         # GameCore — orchestratore
│   ├── scene.js        # Three.js setup
│   ├── world.js        # Procedural world building
│   ├── van.js          # Van physics & mesh
│   ├── packages.js     # Delivery logic
│   ├── portals.js      # Portal triggers & visuals
│   ├── transitions.js  # World flip animation
│   ├── controls.js     # Input handling
│   ├── audio.js        # Web Audio synthesis
│   ├── ui.js           # Screen management
│   ├── hud.js          # HUD updates
│   ├── hypercube.js    # Math & graph data
│   └── utils.js        # Math utilities
└── README.md
```

---

## 🔮 Roadmap & Future Implementations

> Piano di sviluppo visionario suddiviso in 3 fasi.

### Fase 1: Scalabilità 🏗️ *(Priorità immediata)*
Trasformare il gioco in una PWA solida, performante e installabile.

- [ ] **F1.1** — Fix memory leaks: dispose geometrie/materiali Three.js su rimozione celle (`world.js`, `portals.js`, `packages.js`)
- [ ] **F1.2** — Fix double visual creation in `main.js:startLevel()` — correggere flusso pacchetti duplicati
- [ ] **F1.3** — Service Worker con Stale-While-Revalidate per asset statici + Cache-First per Three.js locale
- [ ] **F1.4** — PWA Manifest (`manifest.json`) con icons, theme_color, display standalone, orientamento landscape
- [ ] **F1.5** — IndexedDB storage per salvataggi locali illimitati (sostituire localStorage 5MB)
- [ ] **F1.6** — Haptic Feedback: `navigator.vibrate()` su consegna, portale, errore
- [ ] **F1.7** — Web Share API: condividi punteggi su social/chat
- [ ] **F1.8** — Responsive improvements: ottimizzazione touch targets > 48px, riposizionamento HUD per portrait
- [ ] **F1.9** — WebGL Context Loss recovery handler
- [ ] **F1.10** — Editor di Livelli: drag-and-drop per creare celle personalizzate con esportazione JSON
- [ ] **F1.11** — Multiplayer Locale: pass-and-play 2-4 giocatori
- [ ] **F1.12** — Leaderboard Globale: backend leggero (Supabase/Firebase)

### Fase 2: Intelligenza 🤖 *(3-4 mesi)*
Integrare IA locale e NPC intelligenti.

- [ ] **F2.1** — Micro-Modello IA Locale con Transformers.js: generazione procedurale di nomi celle e descrizioni ambientali
- [ ] **F2.2** — WebNN API: neural network on-device per pattern procedurali di edifici/paesaggi in tempo reale
- [ ] **F2.3** — NPC Intelligente: concorrente AI con pathfinding BFS + tempo reale nel grafo ipercubo
- [ ] **F2.4** — Assistenza Contestuale: suggerimento prossima destinazione ottimale dopo 60s di stallo

### Fase 3: Ecosistema 🌐 *(5-6 mesi)*
Costruire un ecosistema attorno al gioco.

- [ ] **F3.1** — Gamification: sistema di achievement, XP, livelli giocatore, badge
- [ ] **F3.2** — Web Share Target API: link condivisi aprano direttamente un livello specifico
- [ ] **F3.3** — File System Access API: import/esportare custom maps `.hcm` come file nativi
- [ ] **F3.4** — Push Notifications: avvisi giornalieri via Push API
- [ ] **F3.5** — AR Mode: WebXR per proiettare ipercubo nel mondo reale
- [ ] **F3.6** — Plugin Ecosystem: marketplace temi cella creati dalla community
- [ ] **F3.7** — Ambient Computing: accelerometro per velocità dinamica, musica basata su orario locale

---

### 📊 Target Metriche

| Metrica | Attuale | Target Fase 1 | Target Fase 3 |
|---------|---------|---------------|---------------|
| Lighthouse Performance | ~70 | 90+ | 95+ |
| Lighthouse PWA | 0 | 80+ | 100 |
| Lighthouse Accessibility | ~60 | 85+ | 95+ |
| Bundle Size (JS) | ~1.2MB (CDN) | 800KB (lazy) | 500KB (split) |
| Time to Interactive | ~3s | <1.5s | <1s |
| Crash Free Rate | ~90% | 99% | 99.9% |

---

<p align="center">
  <b>Hypercube Delivery Service</b> — Built with vanilla JavaScript & ❤️ &copy; 2026
</p>
