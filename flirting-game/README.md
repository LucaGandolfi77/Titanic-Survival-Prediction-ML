<div align="center">

# 💘 Speed Crush

### *A flirty interactive story PWA — fast choices unlock secret scenes.*

**Speed Crush** is a lightning-fast, offline-first Progressive Web App where every second counts. Pick your vibe, dive into an interactive flirtation, and race against a reaction timer to unlock secret bonus scenes before the moment slips away. No install friction, no network required — just pure, playful chemistry in your pocket.

[![PWA Ready](https://img.shields.io/badge/PWA-Ready-ff4fa3?style=for-the-badge&logo=pwa&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
[![Vanilla JS](https://img.shields.io/badge/JavaScript-Vanilla-f7df1e?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Offline First](https://img.shields.io/badge/Offline-First-4df3ff?style=for-the-badge&logo=offline&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Offline_Service_workers)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-8cff77?style=for-the-badge&logo=dotenv&logoColor=black)](#)
[![License](https://img.shields.io/badge/License-MIT-ffd44d?style=for-the-badge&logo=open-source-initiative&logoColor=black)](#-license)

</div>

---

## 📖 About the Project

**Speed Crush** is a mobile-first interactive fiction game built as a fully installable Progressive Web App. Instead of reading static paragraphs, you react in real time: a **reaction timer** pulses while you choose between flirty comebacks, and **fast answers** (under the threshold) unlock hidden **secret scenes** and stack a **fast streak** bonus.

The experience is designed around a "night out" narrative arc spread across three chapters — from the first glance at the drinks table, to the balcony, to the final question under the city lights. Endings adapt to your charm score, and every playthrough is randomized between multiple character profiles.

### ✨ Key Features

- 🎭 **Multiple character profiles** — randomized between female and male love interests, each with unique palettes and personalities.
- 🌳 **Graph-based dialog engine** — scenes branch through a dialog graph (`dialogues.json`) with secret detours and multiple endings; content and logic are fully separated.
- ⏱️ **Reaction timer engine** — an 8-second decision window with visual feedback (green → amber → red gradient), driven by `requestAnimationFrame` and pause-safe when the tab is hidden.
- ⚡ **Fast streak & secret scenes** — answer while at least 65% of the timer remains to unlock hidden bonus scenes and stack combo bonuses.
- 🏁 **Adaptive endings** — three ending tiers (*Electric*, *Sweet*, *Missed Signal*) based on your final answer and charm score.
- 👆 **Swipe gestures & tap-to-skip** — swipe left for the choice history bottom sheet, tap to skip the reply beat; Pointer Events with Touch Events fallback.
- 🎬 **View Transitions API** — smooth slide/fade transitions between screens, with a CSS entrance-animation fallback.
- 🧠 **AI mode** *(opt-in)* — infinite procedural scenes, custom flirty lines with sentiment scoring, and mood-biased detours — all running locally.
- 🌡️ **Mood Match ambient engine** — the narrative adapts to the player's context (hour of day, IANA timezone region, ambient light, battery) with zero permissions required.
- 📈 **Adaptive difficulty** — a local heuristic tracks reaction times and recalibrates the timer window per chapter, keeping the game in the flow zone.
- 🎙️ **Voice flirt mode** — the character speaks with a per-character pitch (SpeechSynthesis) and you can answer by voice (SpeechRecognition with fuzzy choice matching).
- 🔒 **Stats vault with Passkey** — personal stats protected by Face ID / Touch ID via WebAuthn, fully local.
- 🔄 **Offline Background Sync** — relationship milestones are queued in IndexedDB and drained by the Service Worker when connectivity returns.
- 🌍 **Web Share Target** — install the app and share any text from any app → Speed Crush weaves it into a custom opening scene (*"You shared this — convince me it's worth a story"*).
- 📤 **Story export/import** — File System Access API (with fallbacks): export the complete playthrough as JSON, replay a friend's story, or load a community story pack (schema-validated before it becomes playable).
- 🎬 **Story trailer** — Canvas + MediaRecorder: an animated recap of your playthrough (choice highlights + ending) recorded as a shareable WebM video.
- 👯 **Duet mode** — PWA-to-PWA co-op flirtation: one player plays the suitor, the other plays the crush and reacts to every choice — their reaction is what the suitor's device shows. BroadcastChannel on the same device, WebRTC with manual invite-code signaling across devices. No server.
- 🏆 **Gamification** — 10 achievements, daily streaks (🔥 badge on the setup screen), all visible in the stats vault.
- ☙ **Ambient "night out" mode** — a glanceable low-distraction UI (timer + dialogue only); pick up the phone (DeviceMotion spike) to wake it back up.
- 🌐 **i18n** — full UI translation layer (English, Italiano, Español, Français) with auto-detection from the browser language and a picker in the setup screen.
- 📳 **Haptic feedback** — structured vibration patterns per tone (where supported).
- 📲 **Web Share API** — share your final score with a native share sheet, with clipboard + toast fallback.
- 💾 **Personal records** — best score, best streak, games played and unlocked endings persist in `localStorage`.
- 🔌 **Fully offline** — Service Worker with layered caching (Network-first navigations, Stale-While-Revalidate assets, Cache-First fonts) plus a dedicated offline fallback page.
- 🔄 **Silent updates** — "New version available" toast powered by the SW message channel + `SKIP_WAITING`.
- 🌈 **Dynamic UI** — glassmorphism panels, animated gradient orbs, and a neon-pink theme.
- ♿ **Accessible** — `prefers-reduced-motion` support, semantic HTML, keyboard-friendly navigation and safe-area insets.

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) | Semantic structure, app shell |
| | ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white) | Glassmorphism, animations, responsive grid |
| | ![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-F7DF1E?style=flat-square&logo=javascript&logoColor=black) | Game engine, timer, state management |
| **PWA** | ![Service Worker](https://img.shields.io/badge/Service%20Worker-API-4df3ff?style=flat-square) | Offline caching, background updates |
| | ![Web App Manifest](https://img.shields.io/badge/Manifest-JSON-8cff77?style=flat-square) | Installability, icons, shortcuts |
| | ![Background Sync](https://img.shields.io/badge/Background%20Sync-API-4df3ff?style=flat-square) | Offline milestone queue |
| | ![Web Share API](https://img.shields.io/badge/Web%20Share-API-ff4fa3?style=flat-square) | Native share sheet integration |
| | ![Vibration API](https://img.shields.io/badge/Vibration-API-ffd44d?style=flat-square) | Haptic feedback on mobile |
| **AI / Device** | ![Transformers.js](https://img.shields.io/badge/Transformers.js-opt--in-ffd44d?style=flat-square&logo=huggingface&logoColor=black) | Local neural sentiment (zero server, zero keys) |
| | ![WebNN](https://img.shields.io/badge/WebNN-API-4df3ff?style=flat-square) | Hardware-accelerated inference (experimental) |
| | ![WebGPU](https://img.shields.io/badge/WebGPU-API-8cff77?style=flat-square) | GPU inference fallback |
| | ![Web Speech API](https://img.shields.io/badge/Web%20Speech-API-ff4fa3?style=flat-square) | Voice flirt mode (TTS + recognition) |
| | ![WebAuthn](https://img.shields.io/badge/WebAuthn-Passkeys-ffd44d?style=flat-square) | Biometric stats vault (Face ID / Touch ID) |
| | ![IndexedDB](https://img.shields.io/badge/IndexedDB-Queue-4df3ff?style=flat-square) | Offline milestone storage |
| **Assets** | ![SVG](https://img.shields.io/badge/SVG-Icons-ffb199?style=flat-square&logo=svg&logoColor=white) | Scalable vector icons (192px, 512px) |
| **Fonts** | ![Google Fonts](https://img.shields.io/badge/Google%20Fonts-Inter-4285F4?style=flat-square&logo=googlefonts&logoColor=white) | Typography (Inter, weights 400–800) |

---

## 📋 Requirements

- **Browser:** Chrome 90+, Edge 90+, Firefox 88+, Safari 15.4+ (iOS 15.4+ for full PWA features)
- **Node.js** *(optional, only for serving locally)*: v18+
- **HTTPS or localhost** — Service Workers require a secure context.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/flirting-game.git
cd flirting-game
```

### 2. Serve locally

Because Service Workers require a secure context, you **cannot** just open `index.html` with `file://`. Use one of these:

**Option A — Python (zero dependencies):**
```bash
python3 -m http.server 8080
```

**Option B — Node.js:**
```bash
npx serve .
```

**Option C — VS Code Live Server:**
Install the *Live Server* extension → right-click `index.html` → *Open with Live Server*.

### 3. Open in browser

Navigate to:
```
http://localhost:8080
```

### 4. Install as PWA (optional)

- **Desktop Chrome/Edge:** click the install icon (⊕) in the address bar, or *Menu → Install Speed Crush*.
- **iOS Safari:** tap *Share → Add to Home Screen*.
- **Android Chrome:** tap *Menu → Install app*.

---

## 🎮 How to Play

1. **Pick your vibe** — choose your character gender and who you want to date.
2. **Read the scene** — a reaction timer starts pulsing (8 seconds, adaptive after chapter 1).
3. **Choose fast** — answer while at least **65% of the timer remains** (~5.2s) to unlock the **secret scene** and stack a **fast streak** bonus.
4. **Watch the colors** — green timer = plenty of time, amber = getting risky, red = almost too late.
5. **Swipe left** — review your choice history in the bottom sheet; tap to skip a reply beat.
6. **Enable AI mode** — unlock infinite procedural scenes, mood-biased detours, custom flirty lines with smart scoring, and the optional neural model.
7. **Talk back** — turn on voice mode 🔊 to hear your crush speak, or answer by voice 🎙️ with fuzzy choice matching.
8. **Reach chapter 3** — your final answer and charm score decide between three endings: *Electric*, *Sweet*, or *Missed Signal*.

---

## 📁 Project Structure

Feature-driven architecture — zero build tooling, pure ES modules:

```
flirting-game/
├── index.html               # App shell + 3 screens + 4 overlays (history/stats/duet)
├── offline.html             # Offline fallback for navigation requests
├── style.css                # Glassmorphism UI, animations, responsive grid
├── manifest.json            # PWA manifest (maskable icons, shortcuts, share target)
├── sw.js                    # Service Worker (v4: caching, sync, notifications)
├── dialogues.json           # Graph-based dialog data (scenes, secrets, endings)
├── src/
│   ├── main.js              # Entry point: bootstrapping, SW, share target, deep links
│   ├── core/
│   │   ├── state.js         # Central state + pub/sub emitter
│   │   ├── timer.js         # rAF timer engine (pause-safe, no race conditions)
│   │   ├── storage.js       # localStorage wrapper (records, prefs, streaks)
│   │   ├── haptics.js       # Vibration API wrapper with feature detection
│   │   ├── ambient.js       # Mood Match: hour, timezone, light, battery
│   │   ├── difficulty.js    # Adaptive difficulty (flow-zone heuristic)
│   │   ├── webauthn.js      # Passkey registration/verification
│   │   ├── queue.js         # IndexedDB milestone queue (SW-safe)
│   │   ├── i18n.js          # Translation layer (en/it/es/fr) + applyTranslations
│   │   └── motion.js        # DeviceMotion spike detection (ambient wake)
│   ├── ai/
│   │   ├── capabilities.js  # WebNN / WebGPU / Speech / Passkey detection
│   │   ├── sentiment.js     # Heuristic + Transformers.js neural scoring
│   │   └── generator.js     # Procedural infinite scene generator
│   ├── data/
│   │   ├── dialogues.js     # Dialog engine: loader + graph traversal
│   │   └── validate.js      # Story pack schema validation (marketplace)
│   ├── features/
│   │   ├── setup/setup.js   # Setup screen (prefs, AI, language, notifications, duet)
│   │   ├── game/game.js     # Game engine + gestures + voice + ambient mode
│   │   ├── ending/ending.js # Endings + achievements + trailer + share (lazy)
│   │   ├── voice/voice.js   # Voice flirt mode (TTS + recognition)
│   │   ├── stats/stats.js   # Passkey-protected stats vault
│   │   ├── duet/duet.js     # Co-op flirtation (BroadcastChannel + WebRTC)
│   │   ├── story/story.js   # Story export/import + replay (File System Access)
│   │   ├── story/trailer.js # Canvas + MediaRecorder story trailer
│   │   ├── achievements/achievements.js  # Gamification + daily streaks
│   │   └── notifications/notifications.js # Push + Notification Triggers
│   └── ui/
│       ├── dom.js           # Cached DOM references
│       ├── router.js        # Screen manager + View Transitions API
│       └── toast.js         # Non-blocking toast notifications
└── icons/
    ├── icon-192.svg         # App icon 192×192 (any)
    ├── icon-512.svg         # App icon 512×512 (any)
    ├── maskable-192.svg     # Maskable icon 192×192
    └── maskable-512.svg     # Maskable icon 512×512
```

---

## 🤝 Contributing

Contributions are welcome and encouraged! To keep the codebase clean and the history readable, please follow these guidelines **strictly**.

### 🌿 Branching strategy

We use a simplified **Git Flow**:

```
main          ← production-ready, protected
 ├── develop  ← integration branch
 │    ├── feature/<name>     ← new features
 │    ├── fix/<name>         ← bug fixes
 │    └── chore/<name>       ← maintenance, docs, deps
```

- `main` is protected: no direct pushes, only PRs from `develop` or hotfixes.
- `feature/`, `fix/`, and `chore/` branches are created **from `develop`** and merged back via PR.
- Branch names use `kebab-case` (e.g., `feature/swipe-gestures`, `fix/timer-pause-bug`).

### 🎨 Code style

- **JavaScript:** ES6+, no transpilation, no external frameworks. Prefer **const/let** over `var`. Use **arrow functions** for callbacks, **template literals** for string interpolation.
- **Naming:** `camelCase` for variables/functions, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- **CSS:** `kebab-case` class names, CSS custom properties (`--var`) for theming, mobile-first media queries.
- **Comments:** only when necessary; prefer self-documenting code.
- **Max line length:** 100 characters.

### 📝 Commit conventions (Conventional Commits)

We follow [**Conventional Commits 1.0.0**](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary in imperative mood>

[optional body]

[optional footer(s)]
```

**Types:**

| Type | Purpose | Example |
|------|---------|---------|
| `feat` | New feature | `feat(game): add swipe gestures for scene history` |
| `fix` | Bug fix | `fix(timer): resolve pause race condition on visibilitychange` |
| `chore` | Maintenance | `chore(deps): update Google Fonts weights` |
| `docs` | Documentation | `docs(readme): add contribution guidelines` |
| `style` | Code formatting | `style(css): format choice-btn variants` |
| `refactor` | Refactoring | `refactor(state): extract event emitter pattern` |
| `perf` | Performance | `perf(sw): switch fonts to cache-first strategy` |
| `test` | Tests | `test(timer): add unit tests for TimerEngine` |

**Examples of good commits:**
```
feat(haptics): add structured vibration patterns per tone
fix(manifest): add maskable icons for Lighthouse PWA score
perf(assets): lazy-load ending module with dynamic import
```

### 🔄 Pull request process

1. Fork the repo and create your branch from `develop`.
2. Ensure your code follows the style guide above.
3. Test manually on **desktop Chrome** and **mobile Chrome (or iOS Safari)**.
4. Verify the PWA works **offline** (DevTools → Network → Offline → reload).
5. Open a PR with a clear title following Conventional Commits.
6. Link any related issues.

---

## 🔮 Roadmap & Future Implementations

> ✅ **All three phases are complete.** What follows was the vision; the checkboxes tell the story. New ideas keep flowing below the completed phases.

This roadmap is not a boring TODO list — it's a visionary plan to transform **Speed Crush** from a polished PWA into an **intelligent, ambient, ecosystem-level experience** that feels like it's alive in your pocket.

### 🚀 Phase 1 — Scalability *(Foundation & Polish)* ✅ *(completed)*

> *Goal: turn the prototype into a maintainable, high-performance, feature-driven codebase.*

- [x] **Refactor to Feature-Driven architecture** — split `script.js` into `core/`, `features/`, `ui/` modules with ES modules + dynamic `import()` for the ending screen.
- [x] **Fix the CSS filename mismatch** (`styles.css` → `style.css`) and add a proper `offline.html` fallback for navigation requests.
- [x] **Load `dialogues.json`** and replace the hardcoded `getScenes()` with the graph-based dialog engine (with `next`, `secret.condition`, `ending` branching).
- [x] **Rewrite the timer engine** using `requestAnimationFrame` instead of `setInterval` — eliminate the 100ms race condition and the pause-on-visibilitychange bug.
- [x] **Implement View Transitions API** between the three screens with CSS fallback — slide/scale transitions for a native-app feel.
- [x] **Add swipe gestures** (`Touch Events` + `Pointer Events`) to review the choice log or skip the response.
- [x] **Optimize Lighthouse** — maskable icons, font subsetting (only weights 400 + 800), `content-visibility: auto` on inactive panels, `prefers-reduced-motion` support.
- [x] **Add a "New version available" toast** using SW `message` + `skipWaiting` for silent background updates.
- [x] **Introduce localStorage persistence** — best score, preferred character, unlocked endings history.

### 🧠 Phase 2 — Intelligence *(Local AI & Contextual Awareness)* ✅ *(completed)*

> *Goal: make the game feel smart, personal, and aware — fully offline, zero cloud dependency.*

- [x] **Micro-model IA locale con Transformers.js** — a distilled sentiment model (`Xenova/distilbert-base-uncased-finetuned-sst-2-english`, ~60MB quantized) runs **directly in the browser** on explicit opt-in, scoring the player's custom flirty lines. Zero server, zero API keys; the model + library are cached by the SW (Cache-First cross-origin) so neural sentiment works **fully offline after the first download**. Infinite custom dialogues come from the procedural generator (`src/ai/generator.js`) — template pools biased by ambient mood and character, injected as graph detours that always return to the intended next scene.
- [x] **WebNN API integration** — `src/ai/capabilities.js` detects `navigator.ml` (WebNN) and `navigator.gpu` (WebGPU); the sentiment model loads with a device cascade (**WebNN → WebGPU → WASM**), using hardware acceleration where available and falling back gracefully everywhere else.
- [x] **"Mood Match" ambient engine** — `src/core/ambient.js` combines the local hour (5 narrative phases: late-night/morning/afternoon/evening/night), the **IANA timezone region** (zero-permission region detection), the **Ambient Light Sensor** and **Battery API** (both feature-detected, silent-fail). The mood biases the procedural scene locations (intimate/bright/fresh/warm pools), shows a context chip on the setup screen, and opens each game with a mood line toast.
- [x] **Adaptive difficulty curve** — `src/core/difficulty.js` tracks reaction times with an exponential moving average and recalibrates the timer window **per chapter**: consistently fast players get a tighter window (down to 4s), struggling players get a wider one (up to 12s); the fast threshold follows (bounded 50–80%). The profile persists in `localStorage` across sessions.
- [x] **Sentiment-aware choices** — a **"Write your own line"** input lets the player improvise; the line is scored by the heuristic keyword engine (instant, offline) or by the neural micro-model (when downloaded), and the character's reaction + score adapt to the sentiment. Custom lines can trigger secret detours and even decide the ending tier on the final scene.
- [x] **Voice flirt mode** — Web Speech API: the character speaks scene lines and reactions with a **deterministic per-character pitch** (derived from the character id), and the player can answer by voice via SpeechRecognition with fuzzy keyword matching against the choice buttons. Both APIs feature-detected; the mode degrades to text-only.
- [x] **Biometric Passkey unlock** — WebAuthn platform authenticator (Face ID / Touch ID) protects the **stats vault** overlay: registration with local challenges, verification required on every open, fully local with no server.
- [x] **Offline Background Sync** — relationship milestones (game completions, tier, region, phase) are queued in **IndexedDB** and registered via `SyncManager`; the Service Worker (module worker) drains the queue on the `sync` event and notifies the clients. When the API is missing but the page is online, the queue drains immediately as a fallback.

### 🌍 Phase 3 — Ecosystem *(Social, Ambient & Cross-App)* ✅ *(completed)*

> *Goal: transform the PWA into a social, ambient, cross-platform experience that lives beyond the app itself.*

- [x] **Web Share Target integration** — `share_target` in the manifest (GET with title/text/url params): install the app, share any text from any app → Speed Crush opens and **weaves the shared content into a custom opening scene** built around the character, then continues on the normal story graph. The shared prompt is consumed once per launch.
- [x] **File System Access API** — `src/features/story/story.js`: **export** the complete playthrough (scenes, choices, ending, stats) as a `.json` file via `showSaveFilePicker` (anchor-download fallback); **import** a friend's story (replayed read-only in the history sheet) or a community story pack (validated then playable immediately).
- [x] **Push Notifications with rich actions** — contextual reminders scheduled via the experimental **Notification Triggers API** ("It's 11pm — the right hour for trouble") when available; the Service Worker handles `notificationclick` with **quick-reply actions** ("Play now" focuses the app and starts a story, via postMessage + deep link).
- [x] **PWA-to-PWA multiplayer flirt mode** — **Duet mode** (`src/features/duet/duet.js`): one player plays the suitor (the normal game), the other plays the crush and **reacts to every choice** — their reaction is what the suitor's device shows instead of the local one, with a 6s fallback. Same device via **BroadcastChannel**; two devices via **WebRTC data channels with manual copy/paste invite codes** (zero signaling server).
- [x] **Gamification layer** — `src/features/achievements/achievements.js`: **10 achievements** (First Move, Speed Demon, Secret Keeper, Electric, Streak Master, Night Owl, Custom Charmer, Voice Flirt, Duet Partner, Collector) evaluated over the playthrough context with staggered unlock toasts; **daily streaks** (consecutive days, 🔥 badge on the setup screen); everything visible in the stats vault.
- [x] **Ambient "night out" mode** — a ☙ toggle switches the UI to a **glanceable low-distraction mode** (timer + dialogue only, decorative clutter hidden); **DeviceMotion spike detection** (with iOS permission gate + debounce) wakes the app back up when the phone is picked up.
- [x] **Cross-app shortcuts & widgets** — PWA `shortcuts` ("Quick Play" deep link) shipped in the manifest since Phase 1; native home-screen widgets would require a platform shell (kept as a future idea below).
- [x] **Progressive story sharing** — `src/features/story/trailer.js`: an **animated trailer** of the playthrough (title card → choice highlights with tone colors → ending card) rendered on Canvas and recorded via **MediaRecorder** → shareable WebM video (Web Share with files, download fallback).
- [x] **Internationalization (i18n)** — `src/core/i18n.js`: full UI translation layer in **English, Italiano, Español, Français** (50+ keys: chrome, stats vault, mood phases, toasts) with `data-i18n` annotations, `{param}` interpolation, auto-detection from `navigator.language`, a picker in the setup screen, and persistence in prefs. Story content stays in the story-pack language.
- [x] **Open story marketplace** — `src/data/validate.js`: the **story pack schema** (identical to `dialogues.json`) with full graph-integrity validation (scenes, choices, `next` references, secret detours, ending tiers); packs are imported via File System Access and playable immediately after validation — Speed Crush is now a platform for interactive fiction creators.

---

## 📄 License

Distributed under the **MIT License**. See the `LICENSE` file for more information.

---

<div align="center">

**Made with 💘 and zero dependencies.**

*Speed Crush — because fast choices unlock the best stories.*

</div>
