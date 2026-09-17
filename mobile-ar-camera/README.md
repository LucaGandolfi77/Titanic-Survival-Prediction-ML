# 📸 AR Snap Camera

> Snapchat-style augmented reality camera PWA — con filtri shader WebGL in tempo reale,
> tracking facciale e corporeo MediaPipe, e oltre **30 filtri AR** con particle systems.

[![PWA](https://img.shields.io/badge/PWA-Installable-brightgreen)]()
[![Lighthouse](https://img.shields.io/badge/Lighthouse-95%2B-blue)]()
[![Three.js](https://img.shields.io/badge/Three.js-0.169-red)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange)]()
[![WebGL](https://img.shields.io/badge/WebGL-2.0-purple)]()

---

## 🚀 Stack Tecnologico

| Tecnologia | Ruolo |
|------------|-------|
| **Three.js 0.169** | Rendering WebGL, shader post-processing |
| **MediaPipe Tasks Vision** | Face / Hand / Pose tracking |
| **WebGL 2.0 + GLSL** | 10 shader filters in tempo reale |
| **Canvas 2D** | AR overlay rendering |
| **Web App Manifest** | PWA installabile |
| **Service Worker** | Caching offline, model preload |
| **Web Share API** | Condivisione foto nativa |
| **Vite** | Build tool + HMR |

---

## 📋 Requisiti di Sistema

| Requisito | Minimo |
|-----------|--------|
| Browser | Chrome 90+, Safari 15+, Firefox 90+, Edge 90+ |
| HTTPS | Richiesto (eccetto localhost) |
| GPU | WebGL 2.0 supportato |
| RAM | 2 GB+ consigliati per MediaPipe |
| Camera | Fronte o posteriore con autofocus |

---

## ⚙️ Installazione e Avvio

### Prerequisiti

- [Node.js 18+](https://nodejs.org/) oppure [Python 3](https://www.python.org/)
- Un browser moderno con supporto WebGL 2.0

### Con Node.js (Vite — consigliato)

```bash
cd mobile-ar-camera
npm install
npm run dev
```

### Con Python (sviluppo rapido)

```bash
cd mobile-ar-camera
python3 -m http.server 8080
```

### Con Nginx (produzione)

```bash
npm run build
# Servi la cartella dist/ con Nginx
```

> ⚠️ **Importante:** L'app richiede un **contesto sicuro** (HTTPS o localhost). Aprire `http://localhost:8080` per lo sviluppo.

### Configurazione HTTPS per sviluppo locale

```bash
# Opzione 1: Vite con TLS
npm run dev -- --https

# Opzione 2: mkcert per certificati locali trusted
mkcert localhost
```

---

## 🤝 Contribuire

### Branching Strategy

```
main          → Stabilo, sempre deployabile
develop       → Integrazione feature
feature/*     → Nuove funzionalità
fix/*         → Bug fix
hotfix/*      → Fix critici in produzione
```

### Commit Convenzional

```
feat(shaders): add vintage film filter
fix(camera): handle permission denial gracefully
perf(render): throttle MediaPipe to 10fps
refactor(ui): extract carousel component
test(overlay): add particle system tests
docs(readme): update installation guide
ci: upgrade Node to v20
chore(deps): bump three from 0.169 to 0.170
```

### Linee Guida di Codice

- **Lint:** ESLint con `eslint:recommended` + `plugin:import/recommended`
- **Format:** Prettier (2 spazi, single quote, no semicolon)
- **Test:** Vitest per unit test, Playwright per e2e
- **Tipi:** JSDoc per tutte le funzioni pubbliche, TypeScript consigliato per nuove feature
- **CSS:** Custom properties per design tokens, BEM per componenti
- **GLSL:** GLSL ES 3.0, commenti `//` per uniform/variabili
- **File max:** 300 righe per file (split se superato)

### Pull Request Checklist

- [ ] Codice passa lint e typecheck
- [ ] Test coprono tutte le nuove funzionalità
- [ ] Nessun `console.log` / `debugger` rimasto
- [ ] Manifest `version` aggiornato se necessario
- [ ] Service Worker cache version aggiornata
- [ ] README aggiornato se necessario

---

## 📁 Struttura del Progetto

```
mobile-ar-camera/
├── public/                # Asset statici, manifest, service worker
│   ├── manifest.json
│   ├── sw.js
│   └── icons/
├── src/
│   ├── core/              # State management, lifecycle, constants
│   ├── camera/            # Camera control (MediaStream, constraints)
│   ├── rendering/         # Three.js, shaders, overlay, AR drawers
│   ├── ai/                # MediaPipe providers and trackers
│   ├── ui/                # HUD, carousels, sheets, toasts
│   └── media/             # Capture, share, save
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── icons/
├── index.html
├── style.css
├── app.js
├── shaders.js
├── ar-filters.js
├── ar-filters-extended.js
├── mediapipe-init.js
├── package.json
├── vite.config.js
└── vitest.config.js
```

---

## 📊 Metriche Lighthouse Target

| Categoria | Target |
|-----------|--------|
| Performance | ≥ 90 |
| PWA | ≥ 95 |
| Accessibility | ≥ 90 |
| Best Practices | ≥ 90 |
| SEO | ≥ 85 |

---

## 🔮 Roadmap & Future Implementations

### Fase 1: Scalabilità (Mese 1-2)

- [ ] **Modularizzazione completa** — Separare in moduli ESM con `import()` dinamici
- [ ] **Testing suite completa** — Unit test per shaders, integration test per MediaPipe, e2e per capture/share
- [ ] **Multi-language i18n** — Supporto per EN, IT, ES, FR, DE con `Intl` API
- [ ] **Analytics privacy-first** — Self-hosted Plausible/Umami per monitorare filtri più usati senza tracking personali
- [ ] **CMS filtri remoti** — Configurazione filtri AR via JSON remoto (feature-flag per rilasci graduali)
- [ ] **Plugin system** — Interfaccia `ARFilterPlugin` per permettere terze parti di aggiungere filtri senza fork

### Fase 2: Intelligenza (Mese 3-4)

- [ ] **🎮 Gamification** — Sistema di achievement:
  - "Selfie Master" — 100 foto scattate
  - "Filter Explorer" — Tutti i 30 filtri provati in una sessione
  - "Streak Keeper" — App aperta 7 giorni consecutivi
  - Leaderboard locale con `localStorage` + export/share
- [ ] **🤖 WebNN / Transformers.js** — On-device AI:
  - Stile transfer in tempo reale con modello MobileNet quantizzato
  - Segmentation semantica per separare soggetto/sfondo
  - Riconoscimento emozioni dal volto con generazione reattiva di filtri
  - Suggerimenti filtri intelligenti basati sulla scena
- [ ] **📍 Funzionalità Ambientali:**
  - Cambio automatico filtro in base all'ora del giorno
  - Weather API → filtri reattivi (pioggia → "Rainy Day" AR)
  - Geolocation-based AR landmarks
  - Sensori device: accelerometro per particelle reattive al movimento

### Fase 3: Ecosistema (Mese 5-6)

- [ ] **🛍️ Filter Marketplace** — Catalogo filtri con IndexedDB, installazione dinamica via Web App Manifest
- [ ] **📹 Video Recording** — WebRTC `MediaRecorder` per registrazione video con filtri in tempo reale, export MP4
- [ ] **👥 Social Integration** — WebRTC peer-to-peer per foto battle, collaborative AR sessions
- [ ] **🔌 Integrazione IoT** — Web Bluetooth per controllare drone/camera remota
- [ ] **🧠 On-device Training** — Fine-tuning micro-modello con WebNN per riconoscere oggetti personali
- [ ] **🌐 WebGPU Migration** — Migrazione da WebGL a WebGPU per compute shaders e rendering multi-threaded
- [ ] **📱 Native Wrappers** — Capacitor/TAO per distribuzione su App Store / Play Store

---

> 💡 Questo progetto è nato come sperimentazione su PWA mobile AR e si evolve in una piattaforma creativa completa. Ogni fase è progettata per essere **incrementale e deployabile indipendentemente**.
