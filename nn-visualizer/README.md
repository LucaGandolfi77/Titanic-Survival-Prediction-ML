# 🧠 NN Visualizer

> Visualizzazione interattiva di reti neurali — PWA moderna con supporto offline, pan/zoom fluidi e rendering SVG ottimizzato.

Un'applicazione web progressiva (PWA) per visualizzare, esplorare e analizzare l'architettura di reti neurali artificiali. Interagisci con i nodi, esamina pesi e bias, e ottieni un'esperienza nativa su qualsiasi dispositivo.

[![PWA](https://img.shields.io/badge/PWA-Ready-brightgreen?logo=googlechrome&logoColor=white)](https://web.dev/progressive-web-apps/)
[![Lighthouse](https://img.shields.io/badge/Lighthouse-95%2B-blue?logo=lighthouse)](https://developer.chrome.com/docs/lighthouse)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎨 Filtri e Minigiochi

L'app include un sistema modulare di **filtri visuali**, **strumenti educativi** e **minigiochi** che si sovrappongono alla visualizzazione della rete neurale. Tutti i filtri sono toggle indipendenti e attivabili via toolbar, voce o gesture.

### 🎪 Filtri Simaptici Divertenti

| Filtro | Icona | Descrizione |
|--------|-------|-------------|
| 🌈 **Rainbow Mode** | 🌈 | Tutto il network pulsa di colori arcobaleno ciclici — velocità modulata dai pesi |
| 🫧 **Bubble Wrap** | 🫧 | Ogni nodo diventa una bolla trasparente — clicca per POP con particle burst |
| 🐛 **Worm Connections** | 🐛 | Le connessioni diventano vermi ondulanti che si muovono lungo il percorso |
| 🌀 **Galaxy Drift** | 🌀 | I nodi orbitano lentamente attorno al centro del loro layer |
| 🎨 **Paint Splash** | 🎨 | I nodi sanguinano colori come macchie di vernice — ogni nodo ha colore unico |
| 🕸️ **Cobweb Mode** | 🕸️ | Connessioni tremolanti da ragnatele con gocce di rugiada luminose |
| 🎭 **Mask Mode** | 🎭 | Espressioni facciali sui nodi basate sull'attivazione: 😊😐😵 |
| 🎆 **Fireworks** | 🎆 | I nodi più attivi esplodono in fuochi d'artificio periodici |

### 🧠 Filtri Intelligenti per Apprendimento

| Filtro | Icona | Descrizione |
|--------|-------|-------------|
| 🔬 **Microscope** | 🔬 | Zoom 400% su un nodo con heatmap dei pesi, barre dei bias, derivate |
| 📖 **Story Mode** | 📖 | Narrazione testuale sopra ogni layer con evidenziazione sincronizzata |
| ⚡ **Gradient Flow** | ⚡ | Frecce del gradiente su ogni connessione — direzione e intensità |
| 🗺️ **Knowledge Map** | 🗺️ | Nodi trasformati in icone concettuali con raggruppamento visivo |
| 📊 **Analytics** | 📊 | Pannello trasparente con metriche live: accuracy, loss, histogram |
| 🧪 **Lab Mode** | 🧪 | Slider interattivi per modificare i pesi e vedere l'effetto in tempo reale |
| 📈 **Learning Curve** | 📈 | Curva loss-vs-epoch sovrapposta con gradient fill animato |
| 🎓 **Professor** | 🎓 | Narrazione passo-passo sincronizzata con highlight dei layer |
| 🧩 **Quiz Mode** | 🧩 | Predici l'output con layer nascosto — feedback e spiegazione |
| 🔗 **Connection Explorer** | 🔗 | Tocca una connessione per vedere peso, segno e contributo |

### 🎮 Minigiochi in Sovraimpressione

| Gioco | Icona | Descrizione |
|-------|-------|-------------|
| 🎯 **Hit the Neuron** | 🎯 | Clicca i nodi che si illuminano entro il tempo — 30 secondi |
| 🔗 **Connect the Dots** | 🔗 | Traccia le connessioni corrette tra i layer per punti |
| 🧠 **Memory Match** | 🧠 | Gira le carte dei nodi — trova coppie con stesso peso |
| ⚡ **Speed Classification** | ⚡ | Classifica l'input in 3 secondi — 10 round |
| 🎵 **Music Neurons** | 🎵 | Ogni nodo suona una nota — compone la tua melodia |
| 🏃 **Data Race** | 🏃 | Pacchetti di dati corrono attraverso la rete — scommetti |
| 🐍 **Neural Snake** | 🐍 | Serpente che mangia nodi attivi sulla topologia della rete |
| ⚖️ **Balance Game** | ⚖️ | Regola i pesi per bilanciare l'output — percentuale bilancia |
| 🃏 **Weight Cards** | 🃏 | Gira le carte delle connessioni — trova coppie stesso segno |
| 🔮 **Prediction Oracle** | 🔮 | Predici l'output con difficoltà crescente e statistiche |

---

## 🛠️ Stack Tecnologico

| Tecnologia | Ruolo |
|------------|-------|
| **HTML5 + CSS3** | Struttura e stili (CSS Grid, Flexbox, Custom Properties) |
| **Vanilla JavaScript (ES6+)** | Logica applicativa — zero framework, massima performance |
| **SVG** | Rendering vettoriale della rete neurale |
| **MediaPipe Hands** | Hand tracking e gesture recognition via webcam |
| **PWA** | Installabile, offline-capable, standalone mode |
| **Service Worker** | Caching Stale-While-Revalidate |
| **Speech Synthesis** | Feedback vocale (app parla) |
| **Web Speech API** | Voice control bilingual EN/IT, trascrizione real-time |
| **Web Audio API** | Music mode — note per ogni nodo |
| **Web Share API** | Condivisione architetture |
| **Vibration API** | Feedback aptico mobile |
| **WebNN (roadmap)** | Inferenza on-device con acceleratori hardware |

---

## 💻 Requisiti di Sistema

| Requisito | Minimo |
|-----------|--------|
| Browser | Chrome 88+, Firefox 85+, Safari 14+, Edge 88+ |
| OS | Android 8+, iOS 14+, Windows 10+, macOS 11+ |
| RAM | 2 GB (per modelli > 500 nodi, consigliati 4 GB) |
| Storage | 5 MB (app shell) + modelli |

---

## 🚀 Installazione e Avvio

### Avvio in locale (sviluppo)

> ⚠️ **Importante**: il Service Worker richiede un contesto HTTPS o `localhost`. Non aprire `index.html` direttamente via `file://`.

```bash
# Opzione 1: Python (raccomandato)
cd nn-visualizer
python -m http.server 8080

# Opzione 2: Node.js
npx serve .

# Opzione 3: VS Code
# Installa l'estensione "Live Server" → Click "Go Live"
```

Apri http://localhost:8080 nel browser.

### Installazione come PWA

1. Apri l'app su Chrome/Edge su desktop → Click ⋮ → **"Installa NN Visualizer"**
2. Su mobile → Safari/Chrome mostrerà il prompt **"Aggiungi alla Home Screen"**
3. L'app si aprirà in modalità **standalone** (senza barra browser)

---

## 📁 Struttura del Progetto

```
nn-visualizer/
├── index.html              # Shell applicativa
├── manifest.json           # PWA Manifest
├── sw.js                   # Service Worker
├── css/
│   └── styles.css          # Stili separati
├── js/
│   ├── app.js              # Entry point, bootstrap
│   ├── state.js            # State management + utils
│   ├── renderer.js         # Rendering SVG ottimizzato
│   ├── interaction.js      # Pan/zoom + touch gestures
│   ├── model-loader.js     # Caricamento/parsing modelli
│   ├── training.js         # Simulazione training
│   ├── share.js            # Web Share API
│   ├── audio.js            # Music mode (Web Audio)
│   ├── voice.js            # Voice control (Web Speech API)
│   ├── gestures.js         # Gesture control (MediaPipe Hands)
│   ├── haptics.js          # Vibrazioni patterns
│   ├── undo.js             # Undo/Redo system
│   ├── utils.js            # Utility helpers
│   ├── filters/            # Visual filters & minigames
│   │   ├── rainbow.js      # 🌈 Rainbow mode (CSS hue-rotate)
│   │   ├── story.js        # 📖 Story mode (layer labels overlay)
│   │   └── music.js        # 🎵 Music Neurons minigame
│   └── ui/
│       └── sidebar.js      # Sidebar DOM logic
├── models/                 # Modelli NN (JSON)
├── icons/                  # Icone PWA
└── tests/                  # Test suite
```

---

## 🤝 Contribuire

### Branching Strategy

```
main          → Produzione stabile (protetto)
develop       → Integrazione continua
feature/*     → Nuove funzionalità
fix/*         → Bug fixes
refactor/*    → Ristrutturazioni
hotfix/*      → Fix urgenti in produzione
```

### Stile del Codice

- **JavaScript**: ES6+ con `const`/`let`, arrow functions, destructuring. Nessun `var`.
- **CSS**: BEM naming, Custom Properties per i temi.
- **HTML**: Semantic tags, `aria-*` attributes per accessibilità.
- **Formatting**: Prettier (2 spazi, single quotes).
- **Nessun commento** nel codice a meno che non sia una JSDoc per API pubbliche.

### Commit Convenzional

```
feat:      Aggiunto supporto per modello CNN
fix:       Corretto bug nel calcolo zoom
refactor:  Separato renderer in modulo dedicato
docs:      Aggiornato README con istruzioni PWA
chore:     Aggiornato dependency
perf:      Ottimizzato rendering per modelli >500 nodi
test:      Aggiunti test per pan-zoom logic
```

### Pull Request

1. Crea un branch da `develop`: `git checkout -b feature/nome-feat`
2. Commit con messaggi convenzionali
3. Apri PR con:
   - Descrizione delle modifiche
   - Screenshot/gif dei cambiamenti UI
   - Checklist: [ ] Lighthouse score > 90, [ ] Test passanti, [ ] Accessibile con screen reader

---

## 🔮 Roadmap & Future Implementations

### 🎯 Fase 1: Scalabilità

- [ ] **Multi-model support** — Tabs per confrontare architetture affiancate
- [ ] **Import/Export JSON** — Salva/carica modelli con File System Access API
- [ ] **Batch rendering** — Canvas 2D fallback per modelli con >1000 nodi
- [ ] **Undo/Redo system** — Storico delle interazioni con Command Pattern
- [ ] **Test suite completa** — Unit test (Jest) + E2E test (Playwright)
- [ ] **Code splitting** — Dynamic import per renderer pesanti e UI modules
- [ ] **Lazy loading** — Modelli caricati on-demand con skeleton UI

### 🧠 Fase 2: Intelligenza

- [ ] **WebNN Backend** — Esecuzione inferenza on-device con acceleratori hardware
- [ ] **TensorFlow.js Integration** — Addestramento visuale in tempo reale
- [ ] **Smart Suggestions** — Modello IA locale (Transformers.js) che suggerisce architetture ottimali basate sui dati
- [ ] **Anomaly Detection** — Rilevamento automatico di pesi anomali con evidenziazione
- [ ] **A/B Comparison Mode** — Confronto interattivo di due modelli con metriche
- [ ] **Push Notifications** — Alert quando training completato (tramite backend)
- [ ] **Web Share Target** — Ricezione di architetture condivise da altre app

### 🌍 Fase 3: Ecosistema

- [ ] **Plugin System** — Architettura a plugin per renderer custom, data source, export formats
- [ ] **Gamification** — Badge per esplorazione completa della rete, "Speed Run" per trovare tutti i nodi, leaderboard locale
- [ ] **Multiplayer Mode** — Collaborazione in real-time via WebRTC per team ML
- [ ] **Ambient Context Engine** — Suggerimenti basati su ora del giorno, posizione (GPS), e sensori del dispositivo
- [ ] **VS Code Extension** — Visualizza reti neurali direttamente dall'editor
- [ ] **Marketplace** — Repository community di modelli pre-configurati
- [ ] **File System Access API** — Salva/carica architetture direttamente dal file system
- [ ] **Haptic + Visual Synergy** — Feedback aptico avanzato con pattern personalizzati per eventi diversi (errore, successo, scoperta)

---

> 💬 **Hai un'idea geniale?** Apri una Issue con il tag `enhancement`!

---

## 🚧 In Corso di Implementazione

### ✅ Completate

- [x] 🎵 **Music Mode** — Ogni nodo emette una nota musicale in base dell'attivazione (Web Audio API). L'inferenza diventa una sinfonia.
- [x] 📤 **Web Share** — Condividi l'architettura del modello come testo strutturato tramite Web Share API.
- [x] 🎯 **Training Simulation** — Simulazione di training con animazione live dei pesi, display loss/accuracy (requestAnimationFrame).
- [x] 🌗 **Theme Toggle** — Toggle Dark/Light mode con persistenza in localStorage.
- [x] ⏪ **Undo/Redo System** — Storico delle interazioni con Command Pattern (max 50 stati).
- [x] 🔔 **Haptic Patterns** — Pattern di vibrazione personalizzati per diversi eventi (click, success, error, discovery).
- [x] 📊 **Activation in Detail** — Mostra il valore di attivazione sigmoidale nel dettaglio nodo.
- [x] 🎮 **Toolbar Icons** — Interfaccia toolbar con icone emoji per accesso rapido.
- [x] 🎨 **Cosmic Mode** — Rendering alternativo con effetti glow SVG, filtri gaussiani, gradienti nebula e nodi stellari.
- [x] 🗣️ **Voice Control (Enhanced)** — Speech Synthesis feedback, bilingual EN/IT, real-time transcription, confidence threshold, natural language ("more", "again"), context-aware commands
- [x] ✋ **Gesture Control** — Hand tracking via MediaPipe Hands (Webcam): pinch=zoom, swipe=switch/undo, fist=fit, open palm=reset, thumbs up=play, thumbs down=stop, peace=cosmic
- [x] 🌈 **Rainbow Mode** — Tutto il network pulsa di colori arcobaleno ciclici (filter-rainbow)
- [x] 📖 **Story Mode** — Narrazione testuale sopra ogni layer con evidenziazione sincronizzata (filter-story)
- [x] 🎵 **Music Neurons** — Minigioco: clicca i nodi per comporre la tua melodia (minigame-music)
- [ ] 🫳 **Gesture-Only Mode (enhanced)** — Full hands-free navigation without mouse/keyboard

### ⏳ In Attesa

- [ ] 🫳 **Gesture-Only Mode** — Controllo a gesti tramite MediaPipe Hands (opzionale)
- [ ] 🧬 **DNA Mode** — Carica sequenze DNA e visualizzale come rete neurale colorata
- [ ] 🔊 **Haptic Advanced** — Pattern di vibrazione avanzati con sincronizzazione musicale
- [ ] 📊 **Multi-Model Comparison** — Confronto affiancato di 2-3 modelli addestrati con seed diversi
- [ ] 🕰️ **Time Machine** — Timeline degli stati durante il training, scrollabile

### 🎪 Filtri Divertenti — In Sviluppo

- [ ] 🫧 **Bubble Wrap** — Nodi a bolle trasparenti con POP
- [ ] 🐛 **Worm Connections** — Connessioni vermi ondulanti
- [ ] 🌀 **Galaxy Drift** — Nodi in orbita per layer
- [ ] 🎨 **Paint Splash** — Macchie di vernice dai nodi
- [ ] 🕸️ **Cobweb Mode** — Ragnatele tremolanti
- [ ] 🎭 **Mask Mode** — Espressioni facciali sui nodi
- [ ] 🎆 **Fireworks** — Esplosioni periodiche

### 🧠 Filtri Educativi — In Sviluppo

- [ ] 🔬 **Microscope** — Zoom 400% con heatmap pesi
- [ ] ⚡ **Gradient Flow** — Frecce gradiente su connessioni
- [ ] 🗺️ **Knowledge Map** — Icone concettuali sui nodi
- [ ] 📊 **Analytics** — Metriche live in overlay
- [ ] 🧪 **Lab Mode** — Slider interattivi per pesi
- [ ] 📈 **Learning Curve** — Curva loss sovrapposta
- [ ] 🎓 **Professor Mode** — Narrazione sincronizzata
- [ ] 🧩 **Quiz Mode** — Predici con layer nascosto
- [ ] 🔗 **Connection Explorer** — Tocca connessione per dettagli

### 🎮 Minigiochi — In Sviluppo

- [ ] 🎯 **Hit the Neuron** — Clicca i nodi che si illuminano entro il tempo
- [ ] 🏎️ **Speed Classification** — Classifica in 3 secondi, 10 round
- [ ] 🏃 **Data Race** — Pacchetti di dati in gara attraverso la rete
- [ ] 🐍 **Neural Snake** — Serpente che mangia nodi attivi
- [ ] ⚖️ **Balance Game** — Regola i pesi per bilanciare l'output
- [ ] 🃏 **Weight Cards** — Gira le carte, trova coppie stesso segno
- [ ] 🔮 **Prediction Oracle** — Predici l'output con difficoltà crescente
- [ ] 🔗 **Connect the Dots** — Traccia le connessioni corrette
- [ ] 🧠 **Memory Match** — Gira i nodi come carte, trova coppie
