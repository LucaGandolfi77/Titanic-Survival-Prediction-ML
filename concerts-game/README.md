# 🎵 Concerts Chase

> Hunt concerts · Book flights · Rock out — A world-tour management game!

Explore a world map, discover hidden concerts, book flights & hotels, and play unique minigames — including a quick Pocket Football match — to attend shows by Taylor Swift, Harry Styles, Måneskin, Billie Eilish, Beyoncé, BLACKPULSE, El Fuego & The Thunder. 🎸🎤🕺

---

## ✨ What's New

Two sessions of development — 16 items completed from the 42-item roadmap:

### Session 1

| ID  | Feature                                             | File(s) Changed                                                                          |
| --- | --------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| G-4 | Screen transition animations                        | `css/screens.css`                                                                        |
| G-5 | Reactive HUD (low-budget pulse, flash on change)    | `css/hud.css`, `js/state.js`                                                             |
| G-8 | HiDPI canvas rendering                              | `js/utils.js`, `js/cg-swift.js`, `js/cg-maneskin.js`, `js/cg-eilish.js`, `js/concert.js` |
| F-1 | localStorage high scores + leaderboard on game over | `js/utils.js`, `js/results.js`                                                           |
| T-1 | ESLint + Prettier configuration                     | `.prettierrc.json`, `eslint.config.js`, `package.json`                                   |
| T-2 | Vite dev server + build pipeline                    | `vite.config.js`, `package.json`                                                         |

### Session 2 (Phase A)

| ID   | Feature                                       | File(s) Changed |
| ---- | --------------------------------------------- | --------------- |
| G-7  | Lucide Icons (settings, mute, calendar icons) | `index.html`    |
| F-2  | Settings menu (volume, animations, reset)     | `index.html`    |
| F-18 | Keyboard shortcuts overlay (?)                | `index.html`    |
| F-20 | Minigame timer pause (Esc)                    | `js/utils.js`   |
| F-19 | Calendar sort/filter (status + artist)        | `index.html`    |
| G-9  | Mini-map route visualization in booking panel | `js/booking.js` |

### Session 3 (Phase B)

| ID   | Feature                                              | File(s) Changed                               |
| ---- | ---------------------------------------------------- | --------------------------------------------- |
| F-3  | Achievement system (15 badges, toasts, localStorage) | `js/utils.js`, `js/state.js`, `js/concert.js` |
| F-12 | Screen shake on concert climax                       | `js/utils.js`, `js/concert.js`                |
| F-13 | Procedural crowd noise (Web Audio API)               | `js/audio.js`, `js/concert.js`                |
| G-2  | Animated flight paths on map (SVG animate)           | `js/map.js`                                   |
| G-6  | Dynamic gradient mesh backgrounds per screen         | `css/screens.css`                             |

### Session 4 (Phase C)

| ID   | Feature                                           | File(s) Changed                                      |
| ---- | ------------------------------------------------- | ---------------------------------------------------- |
| F-6  | 3 new artists + 12 concerts (K-pop, Latin, Rock)  | `js/data.js`, `js/concert.js` + 3 new minigame files |
| F-7  | Street Team discovery minigame (drag posters)     | `js/mg-streetteam.js`, `js/data.js`                  |
| F-8  | Merch shop (4 items with permanent stat boosts)   | `js/merch.js`, `js/data.js`                          |
| F-9  | Travel upgrades (flight & hotel tiers)            | `js/travel.js`, `js/booking.js`                      |
| F-10 | Photo mode (canvas filters: B&W, sepia, vignette) | `js/photo.js`                                        |

---

## 🚧 TODO / Roadmap

All planned improvements are tracked below. Check off items as they're completed!

### 🎨 Graphics & Visuals

| ID   | Task                                                                                                       | Status |
| ---- | ---------------------------------------------------------------------------------------------------------- | ------ |
| G-1  | Particle system for concert screens — sparks, confetti, laser beams synced to gameplay                     | ⬜     |
| G-2  | Animated flight paths + day/night cycle overlay on world map                                               | ✅     |
| G-3  | Audio visualizer during concert minigames — waveforms & frequency bars reacting to SFX                     | ⬜     |
| G-4  | Screen transition animations — slide/fade/scale between screens                                            | ✅     |
| G-5  | Reactive HUD — glow on value changes, low-budget red pulse, points flash                                   | ✅     |
| G-6  | Dynamic gradient mesh backgrounds — unique per screen (ocean blues for map, neon stage lights for concert) | ✅     |
| G-7  | Migrate emoji → Lucide Icons for consistent, scalable, themeable UI icons                                  | ✅     |
| G-8  | HiDPI canvas rendering — devicePixelRatio scaling for crisp Retina/4K display                              | ✅     |
| G-9  | Mini-map route visualization in booking panel — flight path from current city                              | ✅     |
| G-10 | Cinematic result card — spotlight effect, stadium crowd silhouette, dynamic score counter                  | ⬜     |
| G-11 | Pocket Football — discovery minigame (pitch SVG, ball physics, AI keeper, timer)                           | ✅     |

### 🏗️ Structure & Architecture

| ID   | Task                                                                                             | Status |
| ---- | ------------------------------------------------------------------------------------------------ | ------ |
| S-1  | ES Modules migration — convert all `<script>` tags to `import`/`export`                          | ⬜     |
| S-2  | Observable state store — subscribe/notify pattern decoupling UI from state                       | ⬜     |
| S-3  | Render layer separation — all `innerHTML` into pure render functions                             | ⬜     |
| S-4  | Event bus system (`G.on` / `G.emit`) for inter-module communication                              | ⬜     |
| S-5  | Data layer extraction — concerts/artists to JSON or mock API                                     | ⬜     |
| S-6  | Audio service abstraction — clean `AudioService` interface (`playSfx`, `playMusic`, `setVolume`) | ⬜     |
| S-7  | Canvas minigame framework — base class with init/update/render/destroy lifecycle                 | ⬜     |
| S-8  | Error handling layer — try/catch around minigames, graceful audio fallback                       | ⬜     |
| S-9  | Directory restructure — `js/core/`, `js/screens/`, `js/games/`, `js/data/`                       | ⬜     |
| S-10 | JSDoc type annotations on all major functions and state shape                                    | ⬜     |

### 🛠️ Tools & Developer Experience

| ID   | Task                                                                           | Status |
| ---- | ------------------------------------------------------------------------------ | ------ |
| T-1  | ESLint + Prettier configuration — `.eslintrc`, `.prettierrc`, consistent style | ✅     |
| T-2  | Vite dev server — HMR, bundling, asset optimization, production builds         | ✅     |
| T-3  | Vitest unit test suite — state logic, minigame scoring, booking calculations   | ⬜     |
| T-4  | GitHub Actions CI — auto-run lint + tests on PR/push                           | ⬜     |
| T-5  | Stylelint for CSS — `stylelint-config-standard`                                | ⬜     |
| T-6  | Git hooks — husky + lint-staged (format on commit, lint on commit)             | ⬜     |
| T-7  | Bundle size analysis — rollup-plugin-visualizer                                | ⬜     |
| T-8  | UI component dev/preview page for isolated screen design                       | ⬜     |
| T-9  | Conventional commits + changelog generator                                     | ⬜     |
| T-10 | Lighthouse CI — enforce performance budgets                                    | ⬜     |

### 🚀 Features — Persistence & Progression

| ID  | Feature                                                                                                           | Status |
| --- | ----------------------------------------------------------------------------------------------------------------- | ------ |
| F-1 | 🏆 localStorage high scores & stats persistence — leaderboard on game over                                        | ✅     |
| F-2 | ⚙️ Settings menu — volume, music toggle, animation toggle, reset game                                             | ✅     |
| F-3 | 🎖️ Achievement system — 15+ badges ("First Discovery", "Budget Master", "Festival Month King", "All On Stage"...) | ⬜     |
| F-4 | 🎚️ Difficulty levels — Casual (more budget, easier), Tour Manager (balanced), Rockstar (hard)                     | ⬜     |
| F-5 | 📅 Weekly challenge mode — pre-built scenarios with leaderboard                                                   | ⬜     |

### 🚀 Features — Gameplay Modes

| ID   | Feature                                                                    | Status |
| ---- | -------------------------------------------------------------------------- | ------ |
| F-21 | ⚽ Pocket Football — Quick football minigame discovery mode (25s, 2 goals) | ✅     |

### 🚀 Features — Content Expansion

| ID   | Feature                                                                                                | Status |
| ---- | ------------------------------------------------------------------------------------------------------ | ------ |
| F-6  | 🎤 3 new artists + 12 new concerts — K-pop group, Latin star, Classic rock (each with unique minigame) | ✅     |
| F-7  | 📰 New discovery minigame: "Street Team" — drag-and-drop poster placement                              | ✅     |
| F-8  | 🛍️ Merch shop — spend points on merch with permanent stat boosts                                       | ✅     |
| F-9  | ✈️ Travel upgrades — unlock faster/cheaper flights & better hotels                                     | ✅     |
| F-10 | 📸 Photo mode — in-concert screenshots with artistic filters                                           | ✅     |

### 🚀 Features — Audio & Immersion

| ID   | Feature                                                                             | Status |
| ---- | ----------------------------------------------------------------------------------- | ------ |
| F-11 | 🎵 Background music — dynamic per-screen playlists (ambient map, energetic concert) | ⬜     |
| F-12 | 📳 Screen shake on concert climax moments, synced to bass hits                      | ⬜     |
| F-13 | 👥 Procedural crowd noise — cheering intensifies as score increases                 | ⬜     |

### 🚀 Features — Social & Sharing

| ID   | Feature                                                                 | Status |
| ---- | ----------------------------------------------------------------------- | ------ |
| F-14 | 📋 Shareable tour results card — Canvas-generated PNG with stats        | ⬜     |
| F-15 | 👻 Friend ghost system — fictional friends' dots on map, compare scores | ⬜     |
| F-16 | 🐦 Social sharing — Twitter/X + copy link for tour results              | ⬜     |

### 🚀 Features — Quality of Life

| ID   | Feature                                                                        | Status |
| ---- | ------------------------------------------------------------------------------ | ------ |
| F-17 | 📡 Service Worker + offline mode — cache all assets, playable without internet | ⬜     |
| F-18 | ⌨️ Keyboard shortcut overlay — press `?` to show all shortcuts                 | ✅     |
| F-19 | 📊 Calendar sort/filter — by artist, by status (upcoming/attended/expired)     | ✅     |
| F-20 | ⏸️ Minigame timer pause option — accessibility pause for time-based games      | ✅     |

---

## 📈 Priority Matrix

| Phase         | Key Tasks                                         | Timeline |
| ------------- | ------------------------------------------------- | -------- |
| 🔴 **Now**    | T-1, T-2, G-4, G-7, S-1, S-2, F-1, F-2            | Week 1–2 |
| 🟡 **Soon**   | S-3, S-4, T-3, T-4, G-5, G-6, F-3, F-11, F-17     | Week 3–4 |
| 🔵 **Active** | F-6, F-7, F-8, F-9, F-10 ✅                       | Week 4–5 |
| 🟢 **Later**  | S-5 to S-10, G-1 to G-3, F-4 to F-16, T-5 to T-10 | Month 2+ |

---

## 🎮 How to Play

1. **🔍 Discover** — Click ❓ dots on the map to play minigames and reveal concert dates
2. **✈️ Book** — Once discovered, book a flight and hotel (earlier = cheaper!)
3. **🎶 Attend** — When the concert date arrives, play the artist's unique minigame
4. **🎤 Organise** — Create your own concerts with fictional friends for extra budget
5. **💰 Budget** — Start with €5,000 — manage wisely!
6. **📅 Time** — Each action advances the calendar. Don't miss concert dates!
7. **🏆 Scoring** — <40% = No entry · 40-69% = Seated · 70-89% = Parterre · 90%+ = On Stage!
8. **🎉 Festival Month** — Attend 3+ concerts in the same month for a ×1.5 multiplier!
9. **🛍️ Merch** — Press `M` from the map to open the shop and spend points on permanent boosts
10. **📷 Photo** — Press `P` during a concert to capture the moment with artistic filters
11. **✈️ Travel** — Unlock cheaper flights & hotels from the map screen → Travel button
12. **⚽ Pocket Football** — Play a quick 25-second football match to score 2 goals and discover a concert

**New minigames:**

- **🎤 K-pop (Rhythm Sync)** — Press `1-5` for 5-lane rhythm notes
- **💃 Latin (Dance Battle)** — Repeat arrow sequences with ↑↓←→
- **🥁 Rock (Drum Solo)** — Press `D F J K` for 4-lane drum notes

**Discovery minigames:**

- **🔤 Cipher** — Unscramble song lyrics
- **📱 Gossip Feed** — Tap posts with real concert hints
- **🔨 Ticket Auction** — Bid within the right price range
- **🧩 Fan Photo Puzzle** — Slide tiles to order 1-8
- **📰 Street Team** — Drag posters to correct cities
- **⚽ Pocket Football** — Score 2 goals in 25 seconds

---

## 🛠️ Tech Stack

| Layer    | Technology                                                        |
| -------- | ----------------------------------------------------------------- |
| Frontend | Vanilla HTML5, CSS3, JavaScript (ES6+)                            |
| Graphics | SVG (world map), Canvas 2D (minigames), CSS animations            |
| Audio    | Web Audio API (synthesized sound effects)                         |
| Font     | [Outfit](https://fonts.google.com/specimen/Outfit) (Google Fonts) |
| Platform | Browser (desktop + mobile)                                        |

## 🛠️ Development

```bash
# Install dependencies
npm install

# Run dev server (with hot reload)
npm run dev

# Lint
npm run lint

# Format
npm run format

# Check formatting
npm run format:check

# Production build
npx vite build
```
