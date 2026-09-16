# ⛸️ Skate Manager — Architecture

Vanilla ES6 modules, no build step, no framework. This document describes how the
game is structured, how data flows, and the exact order of the weekly tick.

## Module map

```
js/
├── main.js                 Entry point: event wiring, screens routing, weekly tick
├── config.js               ALL tunable constants (balance, economy, tempo, wobble…)
├── i18n.js                 UI strings (en/it): t(key, params), applyI18n()
├── types.js                JSDoc typedefs (Skater, Competition, Sponsor, …) — no runtime code
├── state.js                GameState singleton + save/load (versioned, validated, migrated)
├── utils.js                Pure helpers + name generation + stat/value/wage math
├── skaters.js              Skater model, generation, weekly stat updates
├── squad.js                Roster management, market buy/sell, findSkater (shared lookup)
├── competitions.js         Calendar generation, entry/withdraw, rival scoring, placements
├── sponsors.js             Sponsor catalog, deal lifecycle, perk getters
├── music.js                Web Audio music synthesis + SFX (MusicEngine)
├── rink-renderer.js        Canvas 2D view of the rink (RinkRenderer)
├── minigame.js             Mini-game CONTROLLER: DOM wiring, HUD (cached), buttons
├── minigame/
│   ├── engine.js           RoutineEngine: rules & state, no DOM, emits events
│   └── scoring.js          Pure scoring: computeFinalScore()
└── ui.js                   UI facade — re-exports everything below
    └── ui/
        ├── screens.js      showScreen, switchTab
        ├── feedback.js     showToast, showModal, hideModal, confirmModal
        ├── modals.js       skater detail, sell, buy, competition entry
        ├── skater-card.js  skaterCardHTML
        ├── refresh.js      refreshAllPanels()
        └── panels/         one module per tab:
            ├── overview.js   updateHeader, renderOverview, renderEventLog
            ├── squad.js      renderSquad
            ├── market.js     renderMarket
            ├── calendar.js   renderCalendar
            ├── sponsors.js   renderSponsors
            ├── standings.js  renderStandings
            └── results.js    renderResults, renderSeasonEnd
```

## Layering rules

1. **config.js** is imported by everything, imports nothing.
2. **Domain modules** (`skaters`, `squad`, `competitions`, `sponsors`) mutate `GameState`
   through their action functions — the UI never mutates game state directly.
3. **UI modules** (`ui/*`) only read state and render HTML; card actions are handled
   by a single delegated listener in `main.js` (no per-render listener re-wiring).
4. **Mini-game** is split into engine (rules, no DOM) / scoring (pure) / controller
   (DOM, sound, HUD). The engine emits events; the controller presents them.

## Screens & transitions

```
screen-menu ──NEW GAME──▶ screen-setup ──START──▶ screen-game
     │                        │
     └──CONTINUE──▶ screen-game (loaded from localStorage)
                        │
                        ├── ADVANCE WEEK ──▶ (weekly tick, see below)
                        ├── COMPETE NOW ──▶ screen-minigame ──▶ screen-results ──▶ screen-game
                        └── week > maxWeeks ──▶ screen-season-end ──▶ screen-menu or new season
```

## Weekly tick order (`main.js → advanceWeek`)

1. **Guard**: if an entered competition hasn't been played or withdrawn → block.
2. **Wages** deducted.
3. **Sponsors**: income paid, breach warnings (2 → deal cancelled), deals expire.
4. **Stat fluctuation** for every skater (form ±10, morale ±5, injury heal, contract tick).
5. **Contract warnings** at 2 weeks remaining (event log).
6. **Contract expiries**: skaters with 0 weeks leave; best reserve auto-promoted.
7. **AI market activity**: rivals sign market skaters / buy your listings.
8. **Market refresh** when `week >= marketRefreshWeek` (then `marketRefreshWeek = week + 2`).
9. **Random events** (injury, morale, form, donations, prospects, scandals, equipment, fame).
10. **Bankruptcy warning** below −€20,000.
11. `week++`; season end check → `endSeason()` if past `maxWeeks`.

## Mini-game architecture

```
MiniGame (controller, js/minigame.js)
 ├── RoutineEngine (js/minigame/engine.js)   ← rules & state, no DOM
 │     skaters[16] (ref → GameState skater: stats & morale persist)
 │     score, tempo, formation state, timers, wobbles, popups
 │     events: onWobble / onSave / onFall / onFormationStart /
 │             onFormationComplete / onMilestone
 ├── RinkRenderer (js/rink-renderer.js)      ← canvas view
 ├── MusicEngine (js/music.js)               ← Web Audio
 └── HUD (cached DOM refs; writes only on change: timer, score,
           tempo, morale bar, wobble alerts, judges, cooldown buttons)
```

- **Entrance** (first 3s): skaters lerp from the left onto their oval paths.
- **Scoring**: formation points = `difficulty × 10 × tempoMultiplier × (1 + syncBonus)` per second.
- **Risk**: wobble chance = `TEMPO_RISK[tempo] × (1 − stamina/100) × WOBBLE_BASE_CHANCE` per frame-equivalent; low morale ×1.5.
- **Falls**: −50 points live; perfect routine (no falls) +300 at the end.
- **Final score** (pure function in `scoring.js`): `base + musicBonus + tempoBoost + syncBonus + perfectBonus` — components are pre-rounded so the results breakdown always adds up exactly.

## Save format

- Key: `skate-manager-save` (localStorage), version: `SAVE_VERSION` (currently 2).
- 3 slots: slot 1 = legacy key (backward compatible), slots 2–3 = extra slots;
  file export/import available in Settings.
- `serializeState()` excludes transient fields (`minigameActive`, `currentCompetition`).
- `loadGame()` validates the parsed object, then `migrateState()` upgrades older
  saves (fills missing keys with defaults; v1→v2 fixes `marketRefreshWeek` semantics).

## i18n

- `t(key, params)` resolves the active language (fallback: English, then the key).
- Static HTML is localized via `data-i18n` attributes + `applyI18n()`;
  JS-rendered panels call `t()` directly.
- Language persists in `GameState.language`; switchable in Settings.

## Tooling

- `npm test` — smoke + scoring tests (pure logic, run under Node)
- `npm run lint` — ESLint (flat config, `eslint.config.js`)
- `npm run format` — Prettier
- CI: `.github/workflows/skate-manager.yml` (lint + test on push/PR)
