# ⛸️ SKATE MANAGER

A fully playable figure skating team management game built with vanilla HTML5, CSS3, and JavaScript (ES6 modules).

## 🎮 How to Play

Open `index.html` in any modern browser. No build step, no dependencies.

### Core Loop
1. **Create a team** — choose name, color, and difficulty
2. **Manage your squad** — 16 active skaters + reserves, buy/sell on the transfer market
3. **Enter competitions** — pay entry fees, compete in tiered events (Regional → World Championship)
4. **Perform routines** — real-time 60-second Canvas mini-game directing formation choreography
5. **Advance weeks** — collect sponsor income, pay wages, handle random events
6. **Win the season** — earn the most points across 12 weeks to become champion

### Mini-Game Controls
| Action | Effect |
|--------|--------|
| **Tempo buttons** (Slow/Med/Fast/Max) | Change music BPM & score multiplier — higher = riskier |
| **Formation buttons** | Command skaters into synchronized formations (difficulty × points/sec) |
| **Click/Tap wobbling skaters** | Save them before they fall (+10 pts saved, -50 pts fallen) |
| **Keyboard: 1–4** | Tempo selection |
| **Keyboard: Q–P** | Formations (left-to-right) |
| **Keyboard: ESC / ⏸ button** | Pause & resume the routine |

### Key Mechanics
- **Formations** unlock with fame — 10 total from STAR (×1.0) to CROWN ROYAL (×3.0)
- **Tempo** multiplies formation score (higher = riskier wobbles) and feeds the high-tempo bonus
- **Wobble probability** scales with tempo and inversely with skater stamina
- **Sponsors** provide weekly income with breach conditions; CoolBreeze boosts high-tempo routines, QuantumIce boosts sync
- **Contracts** expire after their duration — skaters leave the team (a reserve is auto-promoted to fill the squad)
- **Injured skaters** sit out routines (marked ✚ on the rink)
- **Competitions** can be entered or withdrawn each week (withdrawing forfeits the entry fee)
- **Market** refreshes every 2 weeks; scout for hidden talent (25% chance of star)
- **Random events** each week — injuries, morale boosts, fan donations, rival scandals

## 📁 Project Structure

```
skate-manager/
├── index.html              # Complete DOM with all screens (i18n-ready)
├── README.md
├── ARCHITECTURE.md         # Game loop, module map, data flow, tick order
├── package.json            # type:module + test/lint/format scripts
├── eslint.config.js        # ESLint flat config
├── .prettierrc
├── manifest.webmanifest    # PWA manifest
├── icon.svg                # App icon
├── sw.js                   # Service worker (offline play)
├── tests/
│   ├── smoke.test.mjs      # Economy, placements, sponsors, save, formations
│   └── scoring.test.mjs    # Mini-game scoring unit tests
├── css/
│   ├── variables.css       # CSS custom properties (colors, fonts)
│   ├── reset.css           # Box-sizing, scrollbar, defaults
│   ├── layout.css          # Screens, header, tabs, panels
│   ├── roster.css          # Skater cards, stats bars, market
│   ├── rink.css            # Mini-game layout, tempo/formation UI
│   ├── ui.css              # Modals, toasts, save slots
│   └── animations.css      # @keyframes for all animations
└── js/
    ├── main.js             # Entry point, event wiring, delegated actions, weekly tick
    ├── config.js           # ALL tunable constants (balance, economy, tempo, wobble, staff)
    ├── i18n.js             # UI strings (en/it), t() + applyI18n()
    ├── types.js            # JSDoc typedefs for the data model
    ├── state.js            # GameState singleton, versioned save/load (3 slots) + migration
    ├── utils.js            # Helpers, name generation, stat math
    ├── skaters.js          # Skater model, generation, weekly updates
    ├── formations.js       # 10 formation definitions (16 positions each)
    ├── squad.js            # Roster, market, contracts, staff, findSkater
    ├── competitions.js     # Calendar, entry/withdraw, quick-sim, scoring, rivals
    ├── sponsors.js         # 8 sponsors with deal lifecycle + perks
    ├── music.js            # Web Audio API music synthesis + SFX
    ├── rink-renderer.js    # Canvas 2D rink view (HiDPI-aware)
    ├── minigame.js         # Mini-game controller (DOM, HUD, sound, keyboard, pause)
    ├── minigame/
    │   ├── engine.js       # RoutineEngine — rules & state, no DOM, emits events
    │   └── scoring.js      # Pure final-score computation
    └── ui.js               # UI facade — re-exports ui/*
        └── ui/
            ├── screens.js      # showScreen, switchTab
            ├── feedback.js     # Toasts, modals, confirms
            ├── modals.js       # Skater detail, sell, buy, comp entry
            ├── skater-card.js  # skaterCardHTML
            ├── refresh.js      # refreshAllPanels
            └── panels/         # One renderer per tab
                ├── overview.js
                ├── squad.js
                ├── market.js
                ├── calendar.js
                ├── sponsors.js
                ├── standings.js
                ├── results.js
                └── stats.js
```

## 🛠️ Tech Stack

- **Vanilla JS** (ES6 modules) — zero runtime frameworks, zero runtime dependencies
- **Canvas 2D API** — rink rendering with gradient ice, skater dots, sparkles (HiDPI-aware)
- **Web Audio API** — synthesized music at 4 tempo levels + 8 SFX
- **Google Fonts** — Orbitron (display), Inter (body), Share Tech Mono (mono)
- **localStorage** — versioned saves (3 slots) with migration + file export/import
- **PWA** — web manifest + service worker for offline play and installability
- **Dev tooling** — ESLint + Prettier + Node test runner (`npm run lint` / `npm test`); CI via GitHub Actions
- See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the module map, game loop and tick order

## 🎯 Difficulty Levels

| Level | Starting Squad | Money |
|-------|---------------|-------|
| Amateur | Mostly Tier 1 youth | €150,000 |
| Semi-Pro | Mix of Tier 1-2 | €100,000 |
| Elite | Tier 2-3 performers | €80,000 |

## ⭐ Tips

- Start with **SLOW** tempo until you have high-stamina skaters
- **Scout** the market early — 25% chance of finding a Tier 4 star
- Unlock **harder formations** via fame for higher point multipliers
- Keep **3 sponsors** active at all times for maximum weekly income
- Watch for **wobbling skaters** — missing one costs 50 points + team morale

---

## 🗺️ Roadmap / TODO

### 🐛 Bugs & Errors (code audit findings)
- [x] **Fix competition balance** — rival scores (~3,500–15,500 pts from `strength*100 + tier*1500`) always beat the player's realistic max (~1,900 pts), so the player can never podium or win; rival scores now scale with difficulty (551–1,687) and formation points scale with tempo
- [x] **Fix impossible competition minimums** — `minOverall` was `tier*35` (tier 3 needed 105, tier 4 needed 140 — unreachable since max overall is 99); now `15 + tier*15` (30/45/60/75)
- [x] **Implement the promised tempo score multiplier** — formation points now scale with `tempoMultiplier`; higher tempo = higher reward AND higher wobble risk
- [x] **Fix double-counted wobble penalty** — falls subtract 50 pts live during the routine only; `finishRoutine()` no longer subtracts them again, and the breakdown row is labeled "applied during routine"
- [x] **Resolve starting-money inconsistency** — code gave €5,000,000, README said €50,000; now €150k/€100k/€80k by difficulty (via `STARTING_MONEY`), wages rebalanced (`overall*8+150`), economy verified by smoke test (6–14 week runway)
- [x] **Fix MiniGame event-listener leaks** — `touchstart` handler stored & removed, `stop()` now called on time-up, tempo buttons bound once per instance (no more duplicate listeners)
- [x] **Hide "COMPETE NOW" after competing** — button only shows while the competition can still be played; overview shows ✔ COMPETED
- [x] **Fix market refresh info** — refresh cadence now driven by `marketRefreshWeek`; shows real countdown or "Refreshes when you advance"
- [x] **Persist mini-game morale** — rink skaters hold a live `ref` to the GameState skater; morale changes now persist (stats already did)
- [x] **Use QuantumIce sync bonus in final score** — final sync bonus is now `cohesion × (0.5 + syncBonus)`
- [x] **Implement CoolBreeze `tempoBonus` perk** — +25% on the high-tempo music bonus, shown as a "CoolBreeze Boost" breakdown row
- [x] **Add withdraw/forfeit for entered competitions** — withdraw via the Overview card (entry fee lost); advancing shows a clearer message
- [x] **Handle contract expiry** — contracts now expire: skaters leave with a 2-week warning, best reserve auto-promoted
- [x] **Penalize fielding injured skaters** — injured skaters sit out routines (dimmed ✚ on the rink) instead of performing at full stats
- [x] **Harden save/load** — save version (`SAVE_VERSION`) + v1→v2 migration, validation of merged state, transient minigame fields no longer persisted; squad section titles now show live counts (e.g. "Active Squad (14/16)")

### 🏗️ Architecture & Code Clarity
- [x] Extract all magic numbers (squad sizes 16/8/24, costs, tempo maps, economy values, wobble formulas) into a central `config.js` / balance module
- [x] Remove dead code: `swapActivePositions`, `deepClone`, `weightedRandom`, `shuffle`, `delay`, `GameState.soundEnabled`, `MusicEngine.setEnabled`, unused CSS (`.sale-slider-wrapper`, `.sale-price-display`)
- [x] Deduplicate helpers: `findSkater` now lives once in `squad.js` (shared by main.js and the UI); wage/cohesion accessed via `squad.js` facades
- [x] Cache DOM element references in the mini-game loop instead of `getElementById` every frame; timer/score/tempo update only on change; cooldown buttons update only when the visible value changes
- [x] Split `minigame.js` into `minigame/engine.js` (rules/state, no DOM, event-driven), `minigame/scoring.js` (pure scoring) and `minigame.js` (controller: DOM, HUD, sound)
- [x] Split `ui.js` (716 lines) into `ui/screens|feedback|modals|skater-card|refresh` + per-tab `ui/panels/*`; one delegated listener in `main.js` replaces re-wiring all listeners on every render
- [x] Bind static mini-game controls (tempo buttons, canvas) once instead of re-binding on every `init()`
- [x] Add ESLint + Prettier config and a minimal CI workflow (lint + test) — `.eslintrc` replaced by flat `eslint.config.js`; 17 lint errors found and fixed
- [x] Add unit tests for pure logic (economy math, `calculatePlacements`, formation positions, sponsor lifecycle, save/load round-trip, `computeFinalScore`) — `npm test`
- [x] Add JSDoc types to the data model (skater, competition, sponsor, formation) — `js/types.js`
- [x] Document the game loop (screens, transitions, weekly tick order) in `ARCHITECTURE.md`

### ✨ Features & Ideas
- [x] **Quick-sim competitions** — ⚡ QUICK SIM button next to COMPETE NOW; estimates the routine score from squad stats (overall + cohesion + form, injured sit out); deliberately weaker than a well-played routine so playing still pays off
- [x] **Contract renewals** — click a squad skater for detail → "📝 Renew Contract (+12 wks)" at half wage (loyalty discount); contract weeks shown on every card; expiring contracts leave a 2-week warning and auto-promote a reserve
- [x] **Smarter rival AI** — rival signings now raise their `strength` for future competitions (was: points only), so the AI genuinely improves over the season
- [x] **Training focus + coaching staff** — pick the stat to train (Balanced/Technique/Stamina/Rhythm/Sync/Charisma); hire 4 permanent staff members with passive weekly bonuses (Fitness Coach, Technique Coach, Team Psychologist, Head Scout)
- [x] **More random events** — 5 new: blizzard training (+stamina), sellout crowd (gate receipts), mentorship (veteran boosts a youth), rivalry tension (morale drop) — 13 events total
- [x] **Save slots + export/import** — 3 save slots (LOAD / SAVE HERE via the CONTINUE modal), plus file export/import in Settings; slot 1 stays backward-compatible
- [x] **Keyboard controls for the mini-game** — 1–4 for tempo, Q–P for formations, ESC to pause; larger tap targets on mobile; hint shown in the controls panel
- [x] **Responsive / HiDPI canvas** — devicePixelRatio-aware backing store (crisp on retina), canvas re-fits on window resize (positions rescale proportionally), pause button (⏸/▶) in the mini-game header
- [x] **PWA support** — web manifest + SVG icon + service worker (cache-first, offline play); installable when served over http(s)
- [x] **Stats & history screen** — new 📊 Stats tab: club records (best score, best placement, wins, podiums, prize), score-progression bar chart, most-fielded skaters, Hall of Fame
- [x] **Post-season evolution** — youth intake draft at season start, legends (75+ OVR) enter the Hall of Fame on retirement, season history persists
- [x] **i18n** — `js/i18n.js` with `t(key, params)` + `applyI18n()`; English + Italian dictionaries for the full UI chrome; language setting in Settings (persisted); game-event messages can migrate incrementally

**All phases complete** — 15 bug fixes, 11 architecture improvements, 12 new features.

---

*Built with ❄️ and vanilla JavaScript*
