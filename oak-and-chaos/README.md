# 🌳 OAK & CHAOS — Idle Casino Simulator

> A surreal idle simulation where you grow **Zarghun**, a sentient oak tree that breeds with plants, animals, and Taliban slot machine operators while managing an underground casino.

[![Platform](https://img.shields.io/badge/Platform-Web%2FPWA-blue)](.)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Vanilla%20JS--ESModules-orange)](.)
[![Build](https://img.shields.io/badge/Build-No%20Dependencies-lightgrey)](.)

---

## 📖 Descrizione

**OAK & CHAOS** è un simulatore idle browser-based in cui coltivi **Zarghun**, una quercia senziente che accoppia con piante, animali e operatori di slot machine Taliban, gestendo contemporaneamente un casinò sotterraneo. Usa Energia, Acorn, Punti DNA e Monete per far crescere l'albero, generare prole, acquistare upgrade e sbloccare macchine da gioco. Raggiungi 100m di altezza per ascendere e vincere.

### Funzionalità Principali
- 🌱 **Sistema di crescita idle** con ciclo giorno/notte
- 🧬 **DNA Upgrades** permanenti che alterano le statistiche dell'albero
- 🎰 **6 macchine slot** con NPC unici e meccaniche speciali
- 🌰 **Sistema di breeding** con 25 partner e generazione procedurale di prole
- 📜 **18 eventi random** (positivi, negativi, weird)
- 💾 **Auto-save** ogni 30s in `localStorage` con supporto offline (max 2h)
- ⚡ **Velocità giocabile**: 1×, 2×, 5× e pausa
- 📱 **PWA** — installabile, offline-capable, service worker
- 🎮 **Scorciatoie da tastiera**: Space/P (pausa), 1/2/5 (velocità)
- 📳 **Feedback aptico** su dispositivi mobile supportati

---

## 🛠️ Stack Tecnologico

| Tecnologia | Scopo |
|------------|-------|
| ![JS](https://img.shields.io/badge/JavaScript-ES2022-yellow?logo=javascript) | Linguaggio principale (ES Modules) |
| ![CSS](https://img.shields.io/badge/CSS3-Custom%20Properties-purple?logo=css3) | Layout, animazioni, design tokens |
| ![HTML5](https://img.shields.io/badge/HTML5-Semantic-orange?logo=html5) | Struttura DOM |
| ![PWA](https://img.shields.io/badge/PWA-Service%20Worker-blue) | Installabilità, caching offline |
| ![LocalStorage](https://img.shields.io/badge/LocalStorage-Persistence-blue) | Salvataggio dati client-side |
| Google Fonts | Press Start 2P, VT323, Inter |
| No build step | Nessun dipendenza, nessun bundler |
| **SeededRNG** | Utility per test riproducibili (opzionale) |

---

## 💻 Requisiti di Sistema

| Requisito | Minimo |
|-----------|--------|
| Browser | Chrome 90+, Firefox 90+, Safari 15+, Edge 90+ |
| JavaScript | ES Modules + `requestAnimationFrame` |
| Storage | `localStorage` abilitato (min 5MB) |
| Display | 800×600px minimo (responsive da 360px) |
| OS | Qualsiasi (Web-based) |

---

## 🚀 Installazione e Avvio

### Metodo 1 — Apertura diretta
```bash
git clone https://github.com/oak-and-chaos/oak-and-chaos.git
cd oak-and-chaos
open index.html        # macOS
xdg-open index.html    # Linux
start index.html       # Windows
```

### Metodo 2 — Server locale
```bash
python -m http.server 8080
# Visita http://localhost:8080
```

### Metodo 3 — Installazione PWA
1. Apri `index.html` in Chrome/Edge
2. Clicca l'icona installabile nella barra degli indirizzi
3. L'app appare come applicazione standalone

---

## 🎮 Guida Rapida

1. **Grow** 🌱 — spendi 50⚡ per aumentare altezza e foglie
2. **Acorn** 🌰 — spendi 30⚡ per produrre acorn (valuta breeding)
3. **Meditate** 🧘 — attiva per guadagnare DNA points (1/30s)
4. **Breed** — spendi 40⚡ + 1🌰 per generare prole con partner
5. **Casino** — investi in macchine slot per guadagno passivo
6. **Spend Coins** — acquista buff temporanei e sblocca macchine
7. **Win** — raggiungi **100m** di altezza!

### Scorciatoie da Tastiera

| Tasto | Azione |
|-------|--------|
| `Space` / `P` | Pausa / Riprendi |
| `1` | Velocità 1× |
| `2` | Velocità 2× |
| `5` | Velocità 5× |

---

## 🤝 Contribuzione

### Branching Strategy
```
main          ← Stabile, solo release taggate
develop       ← Integrazione feature
feature/*     ← Nuove funzionalità
fix/*         ← Bug fix
hotfix/*      ← Fix urgenti
```

### Commit Convenzional
```
feat:       Nuova funzionalità
fix:        Bug fix
refactor:   Rifattorizzazione
perf:       Ottimizzazione performance
docs:       Documentazione
chore:      Manutenzione
test:       Test
```

### Linee Guida per il Codice
- **Nessun `console.log`** in codice di produzione
- **Nessuna dipendenza esterna** — tutto vanilla JS
- **Nomina variabili** in `camelCase`, costanti in `UPPER_SNAKE_CASE`
- **Funzioni** max 30 righe, max 3 parametri
- **CSS**: usa variabili da `:root`, mai colori hardcoded
- **HTML**: tag semantici, no inline styles

### PR Checklist
- [ ] Nessun `console.log` rimasto
- [ ] Testato su Chrome, Firefox, Safari
- [ ] Nessuna nuova dipendenza (o giustificata)
- [ ] Commit message segue Conventional Commits
- [ ] README aggiornato se necessario

---

## 📁 Struttura del Progetto

```
oak-and-chaos/
├── index.html              # Main HTML shell + modali + templates
├── manifest.json           # PWA manifest + share target
├── sw.js                   # Service Worker (Cache-First + SWR)
├── README.md               # Questo file
├── css/
│   ├── variables.css       # Design tokens, colori, font
│   ├── reset.css           # Reset + mobile touch + user-select
│   ├── layout.css          # Grid, tabs, responsive, content-visibility
│   ├── oak.css             # CSS art albero (6 stadi)
│   ├── casino.css          # Slot machine, NPC, monete
│   ├── breeding.css        # Partner, breeding UI, famiglia
│   ├── animations.css      # @keyframes (GPU-optimized)
│   └── ui.css              # Toast, modali, win screen, achievements
└── js/
    ├── main.js             # Game loop, save/load, rAF control, stop()
    ├── oak.js              # OakTree: crescita, DNA, energia, upgrades
    ├── casino.js           # Slot machines, NPCs, spin timeout tracking
    ├── breeding.js         # BreedingLab (console.log rimossi)
    ├── population.js       # Partners & offspring management
    ├── events.js           # Random event system (18 events)
    ├── renderer.js         # DOM rendering (race-condition fixed)
    ├── ui.js               # Tabs, modali, buttons, keyboard, haptics
    └── utils.js            # RNG, formatting, name generator
```

---

## 🔧 Fix e Miglioramenti Implementati

### Bug Critici Risolti
| Bug | File | Fix |
|-----|------|-----|
| Memory leak — rAF infinito | `js/main.js:69` | `cancelAnimationFrame` in `game.stop()` |
| Debug panel in produzione | `js/ui.js:294` | `_initDebugPanel()` rimosso completamente |
| Race condition meditating | `js/renderer.js:13` | Tracking via `data-stage` attribute |
| Keyboard shortcuts mancanti | (documentati ma non implementati) | `_bindKeyboard()` in `ui.js` |
| Salto silenzioso del save | `js/main.js:197` | Messaggi di errore espliciti |
| Offline progress doppio toast | `js/main.js:242` | Toast solo in `load()`, non in `_applyOfflineProgress()` |

### Code Quality
- **12 `console.log`** rimossi da `breeding.js`, `ui.js` e `renderer.js`
- Debug logging sostituito con `console.warn`/`console.error` appropriati
- Game reset ora ferma il loop rAF prima del reload

### PWA & Performance
- **`manifest.json`** aggiunto con share target e icone SVG
- **`sw.js`** con strategia Cache-First (statici) + Stale-While-Revalidate (dinamici)
- **Meta tag PWA** aggiunti a `index.html` (theme-color, apple-mobile-web-app, ecc.)
- **CSS GPU-optimized**: `winOakGrow` usa `transform: scale()` invece di `font-size`
- **Mobile optimizations**: `touch-action: manipulation`, `-webkit-tap-highlight-color: transparent`, `user-select: none` su buttoni
- **`content-visibility: auto`** su tab-content per rendering lazy
- **Haptic feedback**: `navigator.vibrate` su toast (30ms) e achievement ([50,30,50])

---

## 🌱 Growth Stages

| Stage | Height | Unlock |
|-------|--------|--------|
| Sapling | 0m | Start |
| Young Oak | 2m | — |
| Mature Oak | 8m | — |
| Ancient Oak | 20m | Machine 5, Taliban breeding |
| Cosmic Oak | 50m | Machine 6, cosmic breeding |
| ASCENDED | 100m | **YOU WIN** |

## 🧬 DNA Upgrades

- **Super Roots** (5 DNA) — +50% energy generation
- **Aphrodisiac Bark** (10 DNA) — +30% breeding success
- **Telepathic Leaves** (15 DNA) — Taliban communication
- **Quantum Acorns** (20 DNA) — Cross-dimensional breeding
- **Beard of Moss** (8 DNA) — +20 Charisma
- **Carnivore Mode** (25 DNA) — Eat failed offspring for energy

## 🎰 Slot Machines

6 unique machines, each with themed symbols and payout rates:
1. Holy Wheel of Fortune
2. Desert Storm
3. Mountain Glory
4. Opium Dreams
5. Zarghun's Revenge *(unlocks at 20m)*
6. The Ascension *(unlocks at 50m)*

---

## 📜 Licenza

MIT License — vedi [LICENSE](LICENSE)

---

## 🔮 Roadmap & Future Implementations

### FASE 1: SCALABILITÀ — Fondamenta
> *Costruire la piattaforma per milioni di partite*

| # | Feature | Descrizione |
|---|---------|-------------|
| 1.1 | **Web Worker per breeding** | Spostare `generateOffspring()` in un Web Worker |
| 1.2 | **IndexedDB migrator** | Sostituire localStorage con IndexedDB + schema migration |
| 1.3 | **State Machine formalizzata** | GAME → PLAYING → PAUSED → WIN → RESTART |
| 1.4 | **Web Share Target API** | Condivisione stato partita via SMS/WhatsApp |
| 1.5 | **Gesture system** | Swipe tra tab, pinch-to-zoom albero, pull-to-refresh |
| 1.6 | **Fullscreen PWA API** | Button fullscreen nella top bar |

### FASE 2: INTELLIGENZA — Algoritmi Viventi
> *Rendere l'albero e il casinò intelligenti*

| # | Feature | Descrizione |
|---|---------|-------------|
| 2.1 | **Micro-Modello IA Locale (Transformers.js)** | distilGPT-2 <5MB per lore procedurale on-device |
| 2.2 | **Prognosi Eventi** | Pattern analysis per anticipare eventi |
| 2.3 | **Procedural Generation Avanzata** | Partner, NPC con personalità evolutiva |
| 2.4 | **Text-to-Speech Narratore** | Web Speech API per eventi e dialoghi |
| 2.5 | **Daily Challenges & Ladder** | Obiettivi giornalieri, classifiche locali |

### FASE 3: ECOSISTEMA — Mondo Vivente
> *Trasformare OAK & CHAOS in un universo*

| # | Feature | Descrizione |
|---|---------|-------------|
| 3.1 | **Sensori Ambientali** | Meteo reale, GPS, ora → gameplay adattivo |
| 3.2 | **Gamification Completa** | Achievement 3-tier, badge, level-up, streak |
| 3.3 | **Multigiocatore P2P** | WebRTC — 2-4 alberi, casinò condiviso |
| 3.4 | **Desktop Widget** | Stats in tempo reale, notifiche push |
| 3.5 | **Audio Dinamico** | Musica adattiva (calmo/epico/sinistro) |
| 3.6 | **AR Mode** | WebXR — Zarghun nel mondo reale |

### Timeline

| Fase | Durata |
|------|--------|
| Fase 1: Scalabilità | 2-3 mesi |
| Fase 2: Intelligenza | 3-4 mesi |
| Fase 3: Ecosistema | 4-6 mesi |
| **Totale** | **9-13 mesi** |

> 💡 **Nota dell'Architetto**: Il codice attuale è solido come MVP. La priorità immediata: (1) completare Fase 1, (2) validare retention con analytics, (3) iterare sui feedback utente.

---

## 🧪 Segreti & Easter Eggs — Segreti nel Codice

> Questi segreti sono nascosti nel codice come reliquie. Trovalili, sbloccali, goditeli.

| # | Segreto | Trigger Suggerito | Ricompensa |
|---|---------|-------------------|------------|
| 🥚 **1** | **The Forbidden Acorn** — Cerca un acorn con nome buff in localStorage | Modifica manualmente `acorns` a 666 in console | Albero oscuro, stato segreto |
| 🥚 **2** | **Konami Code** — ↑↑↓↓←→←→BA | Sequenza tastiera rapida | +9999 di tutte le risorse, Chaos Mode |
| 🥚 **3** | **The 4th Wall** — L'albero ti guarda | Lascia la tab aperta 24h | Messaggio inaspettato nel log |
| 🥚 **4** | **Chaos Mode** — Clicka 50 volte velocemente | Click rapidi sul logo | Stats raddoppiate, eventi caotici |
| 🥚 **5** | **The Whisper** — Acquisisci tutti i tratti | Combina tutte le razze | Foglia d'oro nell'albero |

---

## 🎭 Gameplay Experiments — TODO Sperimentali

> Idee folli da testare in feature branch. Non garantiti al 100%.

- [ ] **🤖 Opponent AI** — Un albero AI rivale che cresce in parallelo. Chi arriva a 100m per primo vince davvero?
- [ ] **🎰 Casino Mode** — Gioca il casinò con monete reali (virtuali). Modalità high-stakes con moltiplicatore.
- [ ] **🌦️ Weather Engine** — Ciclone, siccità, grandine influenzano le statistiche in tempo reale.
- [ ] **🧬 Cloning** — Clona un offspring esistente al costo del 200% — ma con un glitch casuale.
- [ ] **🖼️ Oak Gallery** — Modalità "esposizione" dove l'albero è esposto come opera d'arte CSS. Screenshot ottimizzato.
- [ ] **📈 Stock Market** — Le monete diventano azioni. Investi, specula, bancarotta.
- [ ] **🃏 Tarot Mode** — Ogni giorno un tarot draw cambia le probabilità per 24h.
- [ ] **🌑 Shadow Oak** — Versione oscura dell'albero sbloccata dopo la vittoria. New Game+.
- [ ] **🎼 Soundscape** — Il motore audio genera musica ambientale basata sullo stato dell'albero.
- [ ] **🗺️ World Map** — L'albero "viaggia" in diverse biome (deserto, oceano, spazio) con regole uniche.

---

## 🏗️ Tech Debt TODO — Da Risolvere Prima della v2

- [ ] **[CRIT]** Migrare a TypeScript per type safety sugli oggetti complessi
- [ ] **[CRIT]** Aggiungere `<noscript>` fallback in index.html
- [ ] **[HIGH]** Implementare error boundary wrapper nel render loop
- [ ] **[HIGH]** Sostituire `Math.random()` con seeded RNG (opzionale + riproducibile)
- [ ] **[MED]** Aggiungere `lang` alternativo per i18n framework-ready
- [ ] **[MED]** Separare CSS con `@import` per lazy loading effettivo
- [ ] **[LOW]** Documentare API pubblica di ogni classe (JSDoc)
- [ ] **[LOW]** Aggiungere `aria-label` a tutti i bottoni per screen readers

---

## 🌌 Visione a Lungo Termine — TODO Cinematografici

> Questi TODO richiedono un salto creativo. Sono le fasi finali del gioco.

- [ ] **Filmare l'ascensione** — Una sequenza animazione cinematografica quando Zarghun ascende (Web Animations API + WebGL)
- [ ] **Libro dei ricordi** — Un "giornale" generato automaticamente che racconta la tua partita come un romanzo
- [ ] **Multiplayer Async** — Lascia il tuo albero in "standby" e visita quello di un amico per un giorno
- [ ] **Soundtrack generativa** — Audio procedurale che evolve con lo stato del gioco (Web Audio API oscillators)
- [ ] **The Sequel** — Dopo la vittoria, il figlio di Zarghun eredita il casinò. New Game+ con twist narrativo
- [ ] **Crossover** — Un evento speciale dove altri alberi famosi del portfolio visitano il tuo (cross-promotion)
- [ ] **The End?** — Dopo la vittoria, scopri che l'ascensione era solo l'inizio. L'albero diventa un universo
