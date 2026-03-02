# ⛸️ SKATE MANAGER

A fully playable figure skating team management game built with vanilla HTML5, CSS3, and JavaScript (ES6 modules).

## 🎮 How to Play

Open `index.html` in any modern browser. No build step, no dependencies.

### Core Loop
1. **Create a team** — choose name, color, and difficulty
2. **Manage your squad** — 16 active skaters + 8 reserves, buy/sell on the transfer market
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
- **Wobble probability** scales with tempo and inversely with skater stamina
- **Sponsors** provide weekly income with breach conditions
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
| Amateur | Mostly Tier 1 youth | €50,000 |
| Semi-Pro | Mix of Tier 1-2 | €50,000 |
| Elite | Tier 2-3 performers | €50,000 |

## ⭐ Tips

- Start with **SLOW** tempo until you have high-stamina skaters
- **Scout** the market early — 25% chance of finding a Tier 4 star
- Unlock **harder formations** via fame for higher point multipliers
- Keep **3 sponsors** active at all times for maximum weekly income
- Watch for **wobbling skaters** — missing one costs 50 points + team morale

---

*Built with ❄️ and vanilla JavaScript*
