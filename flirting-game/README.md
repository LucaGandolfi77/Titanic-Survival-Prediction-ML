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
| | ![Web Share API](https://img.shields.io/badge/Web%20Share-API-ff4fa3?style=flat-square) | Native share sheet integration |
| | ![Vibration API](https://img.shields.io/badge/Vibration-API-ffd44d?style=flat-square) | Haptic feedback on mobile |
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
2. **Read the scene** — a reaction timer starts pulsing (8 seconds).
3. **Choose fast** — answer while at least **65% of the timer remains** (~5.2s) to unlock the **secret scene** and stack a **fast streak** bonus.
4. **Watch the colors** — green timer = plenty of time, amber = getting risky, red = almost too late.
5. **Swipe left** — review your choice history in the bottom sheet; tap to skip a reply beat.
6. **Reach chapter 3** — your final answer and charm score decide between three endings: *Electric*, *Sweet*, or *Missed Signal*.

---

## 📁 Project Structure

Feature-driven architecture — zero build tooling, pure ES modules:

```
flirting-game/
├── index.html               # App shell + 3 screens (setup, game, end)
├── offline.html             # Offline fallback for navigation requests
├── style.css                # Glassmorphism UI, animations, responsive grid
├── manifest.json            # PWA manifest (maskable icons, shortcuts)
├── sw.js                    # Service Worker (versioned, layered caching)
├── dialogues.json           # Graph-based dialog data (scenes, secrets, endings)
├── src/
│   ├── main.js              # Entry point: bootstrapping + SW update flow
│   ├── core/
│   │   ├── state.js         # Central state + pub/sub emitter
│   │   ├── timer.js         # rAF timer engine (pause-safe, no race conditions)
│   │   ├── storage.js       # localStorage wrapper (records + preferences)
│   │   └── haptics.js       # Vibration API wrapper with feature detection
│   ├── data/
│   │   └── dialogues.js     # Dialog engine: loader + graph traversal
│   ├── features/
│   │   ├── setup/setup.js   # Setup screen (prefs, personal best)
│   │   ├── game/game.js     # Game engine + rendering + swipe gestures
│   │   └── ending/ending.js # Endings + Web Share (lazy-loaded via import())
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

### 🧠 Phase 2 — Intelligence *(Local AI & Contextual Awareness)*

> *Goal: make the game feel smart, personal, and aware — fully offline, zero cloud dependency.*

- [ ] **Micro-model IA locale con Transformers.js** — run a small text-generation model (e.g., a distilled GPT-2 or a fine-tuned dialogue model) **directly in the browser** via WebAssembly/WebGPU to generate **infinite custom dialogues** based on the user's vibe, time of day, and past choices. Zero server, zero API keys, fully offline.
- [ ] **WebNN API integration** — use the native Neural Network API (where available) for hardware-accelerated inference of the dialogue model on mobile devices.
- [ ] **"Mood Match" ambient engine** — combine `Geolocation API` (reverse-geocoded to city/countryside), `Ambient Light Sensor`, and `Intl.DateTimeFormat` to adapt the narrative tone: moonlit rooftop dialogues at night, neon arcade energy during the day, cozy café vibes on rainy afternoons (via `weather` heuristics or optional API).
- [ ] **Adaptive difficulty curve** — a local heuristic model that tracks the player's reaction times and adjusts the `TIME_LIMIT` and `FAST_THRESHOLD` dynamically per chapter, keeping the game in the "flow zone".
- [ ] **Sentiment-aware choices** — use a local sentiment analysis micro-model to tag player-written custom responses and adapt the character's reaction accordingly.
- [ ] **Voice flirt mode** — `Web Speech API` (SpeechRecognition + SpeechSynthesis) to speak choices aloud and hear the character respond with a synthesized voice, pitch-shifted per character.
- [ ] **Biometric Passkey unlock** — `WebAuthn` Passkeys to protect a local leaderboard and personal stats with Face ID / Touch ID.
- [ ] **Offline Background Sync** — `Background Sync API` to queue "relationship milestones" and sync them (with a future cloud backend) when connectivity returns.

### 🌍 Phase 3 — Ecosystem *(Social, Ambient & Cross-App)*

> *Goal: transform the PWA into a social, ambient, cross-platform experience that lives beyond the app itself.*

- [ ] **Web Share Target integration** — register Speed Crush as a **Share Target**: share any text, image, or URL from any app → Speed Crush uses it as a prompt to generate a custom flirtation scene. *"Flirt about this photo"* mode.
- [ ] **File System Access API** — export the complete "relationship story" (all choices, secrets, endings) as a `.json` or `.txt` file; import a friend's story to replay it from their perspective.
- [ ] **Push Notifications with rich actions** — "Your crush is waiting" notifications with quick-reply action buttons, scheduled via `Notification Triggers API` (experimental) for context-aware reminders (e.g., *"It's 11pm — the right hour for trouble"*).
- [ ] **PWA-to-PWA multiplayer flirt mode** — `WebRTC` + `BroadcastChannel` to let two players on the same device (or across devices via WebRTC data channels) play a **co-op flirtation**: one plays the character, the other the suitor, in real time.
- [ ] **Gamification layer** — achievements, daily streaks, seasonal "vibe" badges, a local leaderboard with Passkey-protected personal records, and unlockable cosmetic palettes per character.
- [ ] **Ambient "night out" mode** — `DeviceOrientation` + `DeviceMotion` sensors to detect when the phone is in a pocket or on a table, and switch the UI to an **ambient glanceable mode** (minimal UI, only the timer and the current dialogue line).
- [ ] **Cross-app shortcuts & widgets** — PWA `shortcuts` for "Quick Play", plus (on Android/iOS) home-screen widgets showing daily streaks or the current character's mood.
- [ ] **Progressive story sharing** — generate a shareable "trailer" of your playthrough (choice highlights + ending) as an animated GIF or short video via `Canvas API` + `MediaRecorder API`.
- [ ] **Internationalization (i18n)** — full translation layer (Italian, Spanish, French) with `Intl` APIs for locale-aware date/time/number formatting and culturally-adapted flirtation lines.
- [ ] **Open story marketplace** — a JSON schema for community-authored "story packs" (characters, scenes, secrets) that can be imported via File System Access or Web Share Target, making Speed Crush a platform for interactive fiction creators.

---

## 📄 License

Distributed under the **MIT License**. See the `LICENSE` file for more information.

---

<div align="center">

**Made with 💘 and zero dependencies.**

*Speed Crush — because fast choices unlock the best stories.*

</div>
