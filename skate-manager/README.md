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
├── index.html              # Complete DOM with all screens
├── README.md
├── css/
│   ├── variables.css       # CSS custom properties (colors, fonts)
│   ├── reset.css           # Box-sizing, scrollbar, defaults
│   ├── layout.css          # Screens, header, tabs, panels
│   ├── roster.css          # Skater cards, stats bars, market
│   ├── rink.css            # Mini-game layout, tempo/formation UI
│   ├── ui.css              # Modals, toasts, sliders
│   └── animations.css      # @keyframes for all animations
└── js/
    ├── main.js             # Entry point, event wiring, game loop
    ├── ui.js               # Toasts, modals, panel rendering
    ├── state.js            # GameState singleton, save/load
    ├── utils.js            # Helpers, name generation, stat math
    ├── skaters.js          # Skater model, generation, weekly updates
    ├── formations.js       # 10 formation definitions (16 positions each)
    ├── squad.js            # Roster management, market buy/sell
    ├── competitions.js     # Calendar generation, scoring, rivals
    ├── sponsors.js         # 8 sponsors with deal lifecycle
    ├── music.js            # Web Audio API music synthesis + SFX
    ├── rink-renderer.js    # Canvas 2D ice rink + skater rendering
    └── minigame.js         # 60-second real-time routine engine
```

## 🛠️ Tech Stack

- **Vanilla JS** (ES6 modules) — zero frameworks, zero dependencies
- **Canvas 2D API** — rink rendering with gradient ice, skater dots, sparkles
- **Web Audio API** — synthesized music at 4 tempo levels + 8 SFX
- **Google Fonts** — Orbitron (display), Inter (body), Share Tech Mono (mono)
- **localStorage** — automatic save/load

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
- [ ] Extract all magic numbers (squad sizes 16/8/24, costs, tempo maps, economy values, wobble formulas) into a central `config.js` / balance module
- [ ] Remove dead code: `swapActivePositions`, `deepClone`, `weightedRandom`, `shuffle`, `delay`, `GameState.soundEnabled`, `MusicEngine.setEnabled`, unused CSS (`.sale-slider-wrapper`, `.sale-price-display`)
- [ ] Deduplicate helpers: `findSkater` (main.js) vs `findSkaterById` (ui.js), wage/cohesion wrapper functions spread across skaters.js/squad.js
- [ ] Cache DOM element references in the mini-game loop instead of `getElementById` every frame; update timer/score/morale UI only on change; reduce per-frame DOM churn in `updateFormationCooldownUI()`
- [ ] Split `minigame.js` (553 lines) into engine (state/rules), view (canvas), and scoring modules
- [ ] Split `ui.js` (716 lines) into per-tab renderer modules; use event delegation instead of re-wiring all listeners on every render
- [ ] Bind static mini-game controls (tempo buttons, canvas) once instead of re-binding on every `init()`
- [ ] Add ESLint + Prettier config and a minimal CI workflow (syntax check, lint)
- [ ] Add unit tests for pure logic (economy math, `calculatePlacements`, formation positions, sponsor lifecycle, save/load round-trip)
- [ ] Add JSDoc types to the data model (skater, competition, sponsor) for editor-time checking
- [ ] Document the game loop (screens, transitions, weekly tick order) in an `ARCHITECTURE.md`

### ✨ Features & Ideas
- [ ] **Quick-sim competitions** — skip the mini-game and simulate the score from squad stats (accessibility + mobile-friendly)
- [ ] **Transfer windows & contract renewals** — negotiate contracts, free agents at season end, transfer deadlines
- [ ] **Smarter rival AI** — rival signings currently grant points only; make them raise rival `strength` for future competitions
- [ ] **Training focus** — choose which stat to train per session; hire coaching staff for passive weekly bonuses
- [ ] **More random events & event chains** — weather chaos, media days, mentorship arcs; add season objectives and achievements
- [ ] **Save slots + export/import** — multiple saves and a downloadable save file
- [ ] **Keyboard controls for the mini-game** — 1–4 for tempo, Q–P for formations, plus larger tap targets on mobile
- [ ] **Responsive / HiDPI canvas** — devicePixelRatio scaling, pause button, canvas resize handling
- [ ] **PWA support** — web manifest + service worker for offline play and installability
- [ ] **Stats & history screen** — per-skater season history, records, score progression graphs
- [ ] **Post-season evolution** — aging curve mid-season, guaranteed retirements, youth intake draft, hall of fame
- [ ] **i18n** — externalize UI strings for translations

**Suggested order:** Bugs first (balance + leaks), then architecture refactors (config, tests), then features (quick-sim and contracts add the most depth for the least effort).

---

*Built with ❄️ and vanilla JavaScript*
