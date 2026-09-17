# UNRAVEL: Chaos Knots

A mobile PWA gacha puzzle-tactics game. Match threads to untangle chaos, deploy Unravelers with unique abilities, and pull for rare characters.

## Play

Open `index.html` in a browser, or serve via any HTTP server:

```bash
cd chaos-knots && python3 -m http.server 8080
# or
cd chaos-knots && npx serve .
```

Then open `http://localhost:8080`. On iPhone, "Add to Home Screen" for full PWA mode.

## How to Play

- **Match 3+ threads** of the same color by swapping adjacent threads
- **Deploy Unravelers** from the bottom hotbar to clear large areas (tap a Unraveler, then tap a target cell)
- **Defeat Boss knots** (👹) at levels 10, 20, 30
- **Void Pull** to collect new Unravelers with rarity-based odds
- Threads earned from matches can be spent on gacha pulls

## Features

- 6×8 match-3 board with 5 thread types (Red, Blue, Gold, Shadow, Chaos)
- 9+ collectible Unravelers (Common → Legendary) with unique abilities
- Clash Royale-style cooldown deployment (4 Unravelers on hotbar)
- Gacha system with pity mechanics (10 pulls = guaranteed Rare+)
- 30 levels across 4 regions with progressive difficulty
- Boss battles every 10 levels
- Web Audio API synthesized sound effects
- Haptic feedback on iOS
- Service worker caching for offline play
- Soft/pastel theme optimized for iPhone portrait mode

## Tech Stack

- Vanilla JavaScript (ES6+)
- CSS Custom Properties, Grid, Flexbox
- Web Audio API (synthesized sounds)
- Service Worker (cache-first)
- localStorage (persistent state)
- No build tools or dependencies

## File Structure

```
chaos-knots/
├── index.html        ← PWA shell, all screens, manifest inline
├── css/
│   └── style.css     ← Pastel theme, iPhone-optimized styles
├── js/
│   ├── board.js      ← Match-3 engine (match, gravity, cascade)
│   ├── combat.js     ← Unraveler abilities, deployment, boss damage
│   ├── gacha.js      ← Pull system, pity, roster management
│   ├── audio.js      ← Web Audio API sound synthesis
│   ├── storage.js    ← localStorage persistence layer
│   └── app.js        ← Main game controller (all screens, logic)
└── sw.js             ← Service worker
```

## Game Systems

| System | Details |
|--------|---------|
| Match-3 | Swap adjacent threads → 3+ match → cut → gravity → cascade |
| Unravelers | 4 on hotbar, each with unique AoE/single-target ability |
| Cooldowns | Common 6s, Rare 4s, Epic 3s, Legendary 2s |
| Gacha | Common 55%, Rare 30%, Epic 12%, Legendary 3% |
| Pity | Rare at 10 pulls, Epic at 10 consecutive, Legendary at 50 |
| Boss | Appears at levels 10, 20, 30 with HP bar |
| Idle | Earn threads while away (after 120s) |
| Thread Gain | Floating "+X 🧶" notifications on matches |
| Continue | Resume from last played level on menu |

## Screens

1. **Menu** — Play, Continue, Void Pull, Roster + thread balance + idle earnings
2. **Level Select** — 30 levels across 4 regions, star ratings
3. **Game Board** — Match-3 board + hotbar + score + boss HP
4. **Result Overlay** — Victory/defeat with rewards
5. **Void Pull** — Gacha with spinning animation then reveal
6. **Roster** — All collected Unravelers with levels and shards
