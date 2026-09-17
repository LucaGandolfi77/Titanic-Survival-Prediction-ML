# SRT Generator Suite

> **Tutto il processing avviene nel browser. Nessun dato lascia mai il tuo dispositivo.**

Suite di 5 strumenti professionali per la generazione e modifica di sottotitoli SRT, trascrizione audio, editing video e burn-in di sottotitoli — alimentati da intelligenza artificiale locale (Whisper via Transformers.js) e FFmpeg.wasm. Zero server, zero API key, zero costi.

![PWA](https://img.shields.io/badge/PWA-Ready-03A9F4?logo=pwa&logoColor=white)
![Offline](https://img.shields.io/badge/Offline-First-4CAF50?logo=offline&logoColor=white)
![Privacy](https://img.shields.io/badge/Privacy-100%%-E91E63?logo=shield&logoColor=white)
![Transformers.js](https://img.shields.io/badge/Transformers.js-v2.17-FF6F00?logo=transformers&logoColor=white)
![FFmpeg.wasm](https://img.shields.io/badge/FFmpeg.wasm-v0.11-000000?logo=ffmpeg&logoColor=white)

---

## 📦 Stack Tecnologico

| Tecnologia | Utilizzo |
|------------|----------|
| **Vanilla JavaScript (ES Modules)** | 100% del codice — zero framework |
| **Transformers.js v2.17** | Trascrizione speech-to-text con Whisper |
| **FFmpeg.wasm v0.11** | Editing video, estrazione audio, burn-in sottotitoli |
| **Web Audio API** | Decoding audio e risampling a 16kHz |
| **Web Share Target API** | Condivisione file da altre app |
| **Service Worker** | Caching offline, Stale-While-Revalidate |
| **FFmpeg Web Worker** | FFmpeg off main thread per editing video |
| **MediaRecorder Streaming** | Registrazione mic → trascrizione Whisper |
| **Keyboard Shortcuts** | Vim-style shortcuts + tool switching |
| **Waveform Visualization** | Audio waveform real-time con Web Audio |
| **CSS Custom Properties** | Design system con theming dark-first |
| **Cache Storage API** | Gestione modelli ML e asset statici |

---

## 🧩 Strumenti Disponibili

| File | Funzione | Motore |
|------|----------|--------|
| `srt_editor.html` | Editor grafico SRT con timeline, drag/resize, undo/redo, zoom, esportazione multi-formato | Puro JS + Shared Utils |
| `audio_to_srt_free.html` | Trascrizione audio → SRT + **registrazione mic live con waveform** | Transformers.js + Whisper + MediaRecorder |
| `video_editor.html` | Taglia, split, copia/incolla segmenti su timeline, **FFmpeg in Web Worker** | FFmpeg.wasm (worker) |
| `video_to_audio.html` | Estrae audio da video in 6 formati, **FFmpeg in Web Worker** | FFmpeg.wasm (worker) |
| `video_with_subs.html` | Burn-in sottotitoli con stile personalizzabile, **FFmpeg in Web Worker** | FFmpeg.wasm (worker) |
| `js/ffmpeg-worker.js` | Web Worker per FFmpeg (single-thread, no COOP/COEP richiesti) | ffmpeg.wasm |

---

## 🎮 Scorciatoie da Tastiera

| Scorciatoia | Azione |
|-------------|--------|
| `1-5` | Cambia strumento (nella shell) |
| `?` | Mostra tutte le shortcut del tool corrente |
| `J/K` | Naviga tra blocchi/segmenti |
| `D` | Elimina blocco/segmento selezionato |
| `C/V` | Copia/Incolla segmento (Video Editor) |
| `S` | Split al playhead |
| `Space` | Play/Pause |
| `Ctrl+Enter` | Avvia azione principale (trascrizione/export) |
| `Ctrl+R` | Registra microfono (Audio → SRT) |
| `Esc` | Chiudi / Deseleziona |

## 🎙️ Registrazione Live & Streaming

Il tool **Audio → SRT** ora supporta la **registrazione dal microfono** con visualizzazione waveform in tempo reale e trascrizione Whisper per chunk. Clicca **🎙️ Registra & Trascrivi** per iniziare.

## ⚡ FFmpeg in Web Worker

Tutte le operazioni FFmpeg (Video Editor, Video → Audio, Video + Sottotitoli) ora girano in un **Web Worker** dedicato (`js/ffmpeg-worker.js`), mantenendo l'interfaccia responsiva durante l'elaborazione.

> ⚠️ **Nota**: FFmpeg.wasm in modalità single-thread non richiede header COOP/COEP speciali. Per la modalità multi-thread (più veloce), il server deve inviare `Cross-Origin-Opener-Policy: same-origin` e `Cross-Origin-Embedder-Policy: require-corp`.

---

## ✨ Nuove Funzionalità

### 🎹 Scorciatoie da Tastiera
| Scorciatoia | Azione |
|-------------|--------|
| `1-5` | Cambia strumento (nella shell) |
| `?` | Mostra tutte le shortcut del tool corrente |
| `J/K` | Naviga tra blocchi/segmenti |
| `D` | Elimina blocco/segmento selezionato |
| `Ctrl+Z` | Undo (SRT Editor) |
| `Ctrl+Shift+Z` / `Ctrl+Y` | Redo (SRT Editor) |
| `Ctrl+R` | Registra microfono (Audio → SRT) |
| `Ctrl+Enter` | Avvia azione principale |
| `Esc` | Chiudi / Deseleziona |

### 💾 Auto-Save
Tutti i tool salvano automaticamente il contenuto in `localStorage`. Quando riapri la pagina, il contenuto precedente viene ripristinato automaticamente con un toast di notifica.

### 🌓 Tema Chiaro/Scuro
Clicca il pulsante 🌓 in alto a destra per passare tra tema dark e light. La scelta viene salvata in `localStorage`.

### 📄 Esportazione Multi-formato
Oltre al formato `.srt`, ora puoi esportare in:
- **`.vtt`** — WebVTT (standard web)
- **`.txt`** — Testo semplice (solo testo dei sottotitoli)

### ✓ Validazione SRT
Il pulsante "Valida" verifica:
- Formato dei timestamp
- Indici sequenziali
- Sovrapposizioni temporali
- Blocchi vuoti o malformati

### 🔍 Zoom Timeline
Nel SRT Editor, usa il **mouse wheel** sulla timeline per zoomare (+/-).

### ⏪ Undo/Redo (SRT Editor)
- `Ctrl+Z` — Annulla ultima azione
- `Ctrl+Shift+Z` o `Ctrl+Y` — Ripristina azione annullata

---

## ⚙️ Requisiti di Sistema

| Requisito | Minimo |
|-----------|--------|
| **Browser** | Chrome 111+, Edge 111+, Firefox 111+ |
| **RAM** | 4 GB (8 GB consigliati per video > 10 min) |
| **Storage** | 500 MB liberi (modelli Whisper: 40–244 MB) |
| **GPU** | Opzionale ma consigliata per FFmpeg.wasm |
| **HTTPS** | Richiesto per Service Worker e Web Audio API |
| **Web Server** | Qualsiasi server HTTP statico (non `file://`) |

---

## 🚀 Installazione e Avvio

### Metodo 1 — Python (consigliato)

```bash
cd srt-generator
python3 -m http.server 8080
# Apri http://localhost:8080
```

### Metodo 2 — Node.js

```bash
cd srt-generator
npx serve .
```

### Metodo 3 — Docker

```bash
cd srt-generator
docker run --rm -p 8080:80 -v "$PWD":/usr/share/nginx/html:ro nginx:alpine
```

### Metodo 4 — VS Code Live Server

Installa l'estensione **Live Server**, poi click destro su `index.html` → *Open with Live Server*.

> ⚠️ **Importante**: non aprire i file con `file:///` — Service Worker, Web Audio API e Fetch API richiedono HTTP/HTTPS.

---

## 📱 Installazione come PWA

1. Apri la suite su Chrome/Edge da un dispositivo mobile o da desktop
2. Clicca il menu ⋮ → **"Installa app"** (o **"Aggiungi alla schermata Home"**)
3. L'app apparirà come icona nativa con splash screen e fullscreen

Il Service Worker precachera automaticamente la shell dell'app per l'uso offline. I modelli Whisper vengono scaricati una sola volta e memorizzati nella **Cache Storage** del browser.

---

## 📱 Web Share Target

Dopo l'installazione come PWA, puoi condividere file audio/video da qualsiasi app (WhatsApp, File Manager, ecc.) direttamente in SRT Generator. Il file verrà importato automaticamente nello strumento appropriato.

---

## 🤝 Contribuire

### Branching Strategy — Conforme a Git-Flow

```
main          → Produzione stabile (solo tag e hotfix)
develop       → Integrazione feature (branch di sviluppo)
feature/X     → Nuova funzionalità (da develop, merge pull request)
hotfix/X      → Fix produzione (da main, merge pull request)
```

### Stile del Codice

- **Nessun framework** — codice in JavaScript vanilla ES2022+
- **Moduli ES** — ogni feature in file separato con `export`/`import`
- **CSS** — Custom Properties per design tokens, BEM per classi, zero framework
- **HTML** — Semantico, accessibile (WCAG 2.1 AA), no inline styles
- **Nessun `alert()`** — usare `showToast()` dalla shared utils
- **Nessun `console.log`** in produzione
- **Errori** — `try/catch` obbligatorio su tutte le operazioni asincrone

### Commit Convenzional

```
feat(audio-to-srt): add Whisper medium model support
fix(srt-editor): resolve drag offset when zoom > 2x
perf(video-editor): lazy load ffmpeg wasm on first use
docs(readme): update installation instructions
refactor(css): extract shared design tokens
chore(deps): bump transformers.js to v2.18
test(srt-parser): add edge cases for malformed SRT
```

### Pull Request

1. Crea branch da `develop`: `git checkout -b feature/nome-feature develop`
2. Commit con messaggi convenzionali
3. Verifica Lighthouse score ≥ 90 su tutte le metriche
4. Testa su mobile (iOS Safari + Chrome Android)

---

## 🔧 Sviluppo Locale

Per FFmpeg.wasm con SharedArrayBuffer, il server deve inviare header speciali:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Per uso senza header speciali, FFmpeg.wasm funziona in modalità single-thread rimuovendo `{ enableSharedArrayBuffer: true }` dalla configurazione.

---

## 🔮 Roadmap & Future Implementations

> Un piano di sviluppo visionario per trasformare questa suite da strumento utilissimo a piattaforma AI-native per contenuti multimediali.

---

### 📊 Fase 1 — Scalabilità (M1–M3)

**Obiettivo**: Trasformare da 5 file HTML isolati a piattaforma modulare e installabile.

| ID | Todo | Impatto |
|----|------|---------|
| 1.1 | ✅ Shell PWA unificata con navigazione a tab | Fonda di tutto |
| 1.2 | ✅ Service Worker con Stale-While-Revalidate | Offline 100% |
| 1.3 | 🔲 Refactoring in moduli ES con pattern Observer | Manutenibilità |
| 1.4 | ✅ Web App Manifest con icons, screenshots | PWA Score 100% |
| 1.5 | 🔲 Code splitting — carica ogni feature solo quando aperta | TTI −40% |
| 1.6 | ✅ Web Share Target API | UX nativa |
| 1.7 | ✅ Libreria componenti condivisa (drop-zone, timeline, toast) | Coerenza UI |
| 1.8 | 🔲 Test unitari con Vitest per parser SRT, utils | Qualità |
| 1.9 | 🔲 Lighthouse CI con soglia PWA ≥ 95 | Automazione |
| 1.10 | 🔲 Supporto multi-lingua UI con Intl | Portabilità |

### 🧠 Fase 2 — Intelligenza (M4–M6)

**Obiettivo**: Aggiungere intelligenza artificiale locale e automatizzazione.

| ID | Todo | Tecnologia | Impatto |
|----|------|------------|---------|
| 2.1 | Modelli Whisper multi-formato con auto-detect qualità | Transformers.js | Flessibilità |
| 2.2 | **Traduzione SRT offline** con modello T5/LaMTA via WebNN | WebNN + Transformers.js | AI locale |
| 2.3 | Sottotitoli intelligenti con speaker diarization | Whisper + vad-events | Pro |
| 2.4 | Generazione SRT da testo → sottotitoli temporizzati | Transformers.js TTS reverse | Nuovo |
| 2.5 | Upscale e restaurazione audio con DCUModel | Diffusion model | Pro |
| 2.6 | Chatbot contestuale ("Ripeti questo blocco", "Rendi più breve") | LLM locale quantizzato | UX |
| 2.7 | Background Sync per coda trascrizioni offline | Background Sync API | Affidabilità |
| 2.8 | Push Notifications per completamento trascrizione | Push API | Mobile |
| 2.9 | Timeline magnetica — snap intelligente a beat audio | Web Audio API | Precisione |
| 2.10 | Memory pool — gestione lifecycle Web Workers | Worker Pool | Performance |

### 🌍 Fase 3 — Ecosistema (M7–M12)

**Obiettivo**: Diventare un hub di creazione contenuti connesso e collaborativo.

| ID | Todo | Tecnologia | Impatto |
|----|------|------------|---------|
| 3.1 | **Gamification**: badge, XP, streak per uso | localStorage + Canvas | Engagement |
| 3.2 | Plugin Marketplace — sistema di plugin per nuovi formati | ES Modules dynamic import | Estensibilità |
| 3.3 | Edizione collaborativa via WebRTC | WebRTC + Yjs CRDT | Collaborazione |
| 3.4 | Integrazione API esterne (SRT.svc, Subtitle Horse) | Fetch + OAuth | Distribuzione |
| 3.5 | Ambient Computing — UI adatta a ora, luce, movimento | Sensors API | Contestuale |
| 3.6 | Voice commands — controllo hands-free | Web Speech API | Accessibilità |
| 3.7 | File System Access API — gestione cartelle progetti | File System Access | Produttività |
| 3.8 | Biometric unlock — Face ID / fingerprint | WebAuthn | Sicurezza |
| 3.9 | Scene detection — auto-segmentazione video | ONNX Runtime Web | AI Pro |
| 3.10 | Export universale: VTT, SCC, PDF, DOCX | Formatter multipli | Compatibilità |

---

### 🏆 Metriche di Successo

| Fase | KPI Target |
|------|-----------|
| Fase 1 | Lighthouse PWA ≥ 95, TTI < 2s, installazioni PWA > 100 |
| Fase 2 | Trascrizione < 30s per 10 min audio, traduzione offline funzionante, push notifications attive |
| Fase 3 | Plugin marketplace con > 10 plugin, editing collaborativo stabile, gamification con > 5 badge |

---

## 📜 Licenza

MIT — libera di usare, modificare e distribuire.

---

> *Built with ❤️ — 100% in the browser, 0% on servers, 100% yours.*
