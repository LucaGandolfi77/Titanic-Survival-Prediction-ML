# CardForge

A fully playable, vanilla-JavaScript card game with a **Deck Builder** and a **Game Table**, all running in the browser — no server, no frameworks, no build step.

## Features

| Area | Highlights |
|------|-----------|
| **Deck Builder** | 50-card fantasy database, search & filter (type / element / rarity / cost / power), add up to 4 copies of each card, max 60 cards, save / load from localStorage |
| **Game Table** | Draw pile, discard pile, hand fan, play zone with free positioning, drag-and-drop (hand→table, reposition, table→discard), flip cards face-down, right-click context menus |
| **UI / UX** | Hash-based SPA router with fade transitions, responsive layout, toast notifications, modal card details, keyboard shortcuts, CSS-only card backs, fallback gradient faces when images are missing |

## Getting Started

```bash
# No install needed — just open index.html
open card-game/index.html        # macOS
xdg-open card-game/index.html    # Linux
start card-game/index.html       # Windows
```

Or serve it locally (required if your browser blocks ES module imports from `file://`):

```bash
npx serve card-game
# or
python3 -m http.server 8000 -d card-game
```

## Project Structure

```
card-game/
├── index.html
├── README.md
├── css/
│   ├── variables.css      Design tokens
│   ├── reset.css           CSS reset + utilities
│   ├── animations.css      @keyframes & animation classes
│   ├── layout.css          Header, nav, sections, buttons, toasts
│   ├── card.css            Card component (sizes, rarity, flip, etc.)
│   ├── deck-builder.css    Filter panel + card grid + deck panel
│   ├── game-table.css      Piles, play zone, hand area, controls
│   ├── modal.css           Modals & zoom overlay
│   └── responsive.css      Breakpoints, touch, reduced motion
└── js/
    ├── main.js             Entry point
    ├── utils.js            Shared helpers (toast, modal, etc.)
    ├── database.js         50-card collection
    ├── state.js            Reactive state + localStorage
    ├── router.js           Hash-based SPA router
    ├── card-renderer.js    Card DOM generation
    ├── filters.js          Search & filter logic
    ├── deck-builder.js     Deck builder controller
    ├── drag-drop.js        HTML5 drag-and-drop
    └── game.js             Game controller
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `D` | Draw a card (game view) |
| `N` | New game (game view) |
| `S` | Save deck (deck builder) |
| `Z` | Zoom hovered card |
| `1` | Switch to Deck Builder |
| `2` | Switch to Game Table |
| `Esc` | Close modal / zoom |

## Tech Stack

- **Vanilla JS** (ES2022+, ES modules)
- **CSS** custom properties, no preprocessors
- **HTML5** Drag-and-Drop API
- **Google Fonts**: Inter, Cinzel, JetBrains Mono

No React. No Vue. No jQuery. No Bootstrap. No Tailwind. No bundler.

## License

MIT
