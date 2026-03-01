# 🌳 OAK & CHAOS — Idle Casino Simulator

A surreal idle simulation where you grow **Zarghun**, a sentient oak tree that breeds with plants, animals, and Taliban slot machine operators while managing an underground casino.

## 🎮 How to Play

1. Open `index.html` in any modern browser (no server needed)
2. **Grow** your oak by spending energy (50⚡ per growth)
3. **Produce acorns** (30⚡) — needed for breeding
4. **Meditate** to earn DNA points (1 DNA/30s)
5. **Breed** with partners to create offspring (costs 40⚡ + 1🌰)
6. **Assign roles** to offspring to boost the casino
7. **Spend coins** on nutrients, fertilizer, and bribes
8. **Reach 100m** height to ascend and win!

## 📁 Project Structure

```
oak-and-chaos/
├── index.html              # Main HTML shell
├── README.md               # This file
├── css/
│   ├── variables.css       # Design tokens, colors, fonts
│   ├── reset.css           # CSS reset + scrollbars
│   ├── layout.css          # Grid layout, tabs, responsive
│   ├── oak.css             # CSS-only oak tree art (6 stages)
│   ├── casino.css          # Slot machines, NPCs, coins
│   ├── breeding.css        # Partner cards, breeding UI
│   ├── animations.css      # All @keyframes animations
│   └── ui.css              # Toasts, modals, win screen
└── js/
    ├── main.js             # Entry: game loop, state, save/load
    ├── oak.js              # OakTree class: growth, DNA, energy
    ├── casino.js           # Casino: 6 slot machines, 6 NPC operators
    ├── breeding.js         # BreedingLab: trait generation, roles
    ├── population.js       # Partners + offspring management
    ├── events.js           # Random event system (18 events)
    ├── renderer.js         # DOM rendering functions
    ├── ui.js               # UiManager: tabs, buttons, modals
    └── utils.js            # RNG, formatting, name generation
```

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

## 💾 Features

- **Auto-save** every 30 seconds to localStorage
- **Offline progress** (up to 2 hours)
- **Speed controls**: 1×, 2×, 5× and pause
- **Day/night cycle** affecting energy generation
- **18 random events** (positive, negative, weird)
- **CSS-only art** — no images or canvas

## 🛠️ Tech Stack

- Vanilla JavaScript (ES Modules)
- CSS Custom Properties
- Google Fonts: Press Start 2P, VT323, Inter
- No dependencies, no build step

## License

MIT
