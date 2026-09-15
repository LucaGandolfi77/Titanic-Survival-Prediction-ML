# Chaos Deckbuilder

A minimal real-time 1v1 deckbuilder built with Flask, Flask-SocketIO, and vanilla HTML/CSS/JS.

## Core idea

- Each round, both players:
  - buy **one** card from the market,
  - select up to **three** cards from hand,
  - lock in their choices,
  - resolve the round simultaneously.
- Cards grant power, shield, healing, poison, draw, and bonus coins.
- Every day the whole game world gets a **global modifier**.
- Each player has a **secret mission**.
- The cosmetic shop is **useless but beautiful**.
- Achievements are intentionally a bit ridiculous.

## Features

### Core Gameplay
- Real-time private rooms via WebSocket
- Daily modifier system (global rule changes every day)
- Secret missions with a Chaos Crown reward card
- Minimal market-based deckbuilding
- Cosmetic loadout panel (tables, card backs, particles)
- Silly achievements
- Responsive UI
- AI bot fallback after 12 seconds if no second player joins
- Weighted rarity system: common, uncommon, rare, epic, legendary
- Tribal synergies: beast, ocean, machine, cult, warrior, cursed
- Cursed combo cards with extra payoff
- 20+ additional cards beyond the starter deck

### Commercial Features (v2)
- **SoundFX** - All sounds synthesized via Web Audio API (zero external audio files)
- **Card Reveal Animation** - 3D perspective flip on card display, legendary cards pulse with golden glow
- **Emote System** - Quick reactions during matches (GG!, Wow!, Sick!, RNG!, Nope!)
- **Spectator Mode** - Watch any room in read-only mode for content creation or learning
- **Post-Game Stats** - Round-by-round damage, shield, and healing bar charts on game over screen
- **Deck Archetype Badge** - Auto-detected deck type displayed at game end (e.g., WARRIOR ENGINE, CURSED CHAOS, TURTLE DEFENSE)
- **Tutorial** - First-time interactive guide explaining shop → hand → lock-in → resolve flow
- **Daily Challenge** - Same deck, same market, same modifier. Compete for the best time on today's daily.
- **Boss Rush Mode** - Solo mode against "The Auditor" (40 HP, boss-tier deck)
- **Trading Post** - Trade a random card with your opponent between rounds
- **Incremental Mission Progress** - Visual progress bar with percentage tracking for all missions

## Tech Stack

| Technology | Version | Usage |
|------------|---------|-------|
| Python | 3.x | Backend language |
| Flask | 3.1.0 | HTTP framework, routing, templates |
| Flask-SocketIO | 5.5.1 | Real-time WebSocket communication |
| python-socketio | 5.12.1 | SocketIO client/server protocol |
| simple-websocket | 1.1.0 | WebSocket transport layer |
| Vanilla JavaScript | ES6+ | Frontend logic, game state, UI rendering |
| HTML5 | Latest | Page structure, semantic markup |
| CSS3 | Latest | Styling, animations, responsive design |
| Web Audio API | Browser-native | Sound synthesis (buy, lock, hit, win, rare) |
| CSS Keyframe Animations | Browser-native | Card reveal, emote bubbles, legendary pulse |
| localStorage | Browser-native | Tutorial seen flag persistence |

## How to play

### Basic Loop
1. **Shop Phase** - Buy one card from the market (5 cards available). Rare cards appear less often.
2. **Select Cards** - Click up to 3 cards from your hand to select them.
3. **Lock In** - Click "Lock In Cards" to submit your selections.
4. **Resolve** - Both players reveal simultaneously. Check the Battle Log for results.

### Tribal Synergies
Building around a tribe creates bonus effects during resolution:

| Tribe | Bonus |
|-------|-------|
| Beast (2+) | +2 power |
| Ocean (2+) | +2 shield |
| Machine (2+) | +1 gold |
| Cult (2+) | +2 heal |
| Warrior (2+) | +1 power, +1 shield |
| Cursed (2+) | +3 power, +1 poison, -1 heal |

### Combo Cards
- **Doom Seed + Saint of Debt**: +4 extra power
- **Hex Mirror + Venom**: +2 extra poison
- **Deep Oracle + Kraken Receipt**: +1 extra draw

### Daily Modifiers
Every day the entire game world gets a global modifier that changes strategy:

| Modifier | Effect |
|----------|--------|
| Everything Is Underwater | Blue cards cost 1 less. All attacks lose 1 power. |
| Ridiculous Heatwave | Red cards gain +1 power. Healing is reduced by 1. |
| Hall of Echoes | Cards with draw gain +1 extra draw. |
| Heavy Gravity Day | Shield cards gain +1 shield. Draw effects lose 1. |

### Secret Missions
Each player gets a random secret mission. Completing it grants a **Chaos Crown** card:

| Mission | Goal |
|---------|------|
| Underwater Collector | Buy 3 blue cards |
| Dramatic Entrance | Deal 6+ damage in one round |
| Suspiciously Kind | Heal a total of 6 HP |
| Turtle Mode | Reach 6 shield in one round |

### Achievements
- **Lost 7 Times With Dignity** - Lose 7 games in a row
- **Fashion Victim** - Change cosmetics 10 times in one game
- **One HP Main Character** - Win the game with exactly 1 HP left
- **Certified Mermaid** - Buy 4 blue cards on the underwater daily

## Game Mechanics Explained

### Resolution
When both players lock in, all cards are discarded and bundles are computed:
1. Sum all stats from selected cards (power, shield, heal, etc.)
2. Apply daily modifier effects
3. Apply tribal synergy bonuses
4. Apply cursed combo bonuses
5. Calculate damage: `max(0, attacker_power - defender_shield) + attacker_poison`
6. Apply damage, healing, draw, and gold bonuses
7. Check for game over conditions

### Mission Progress
Missions track progress incrementally:
- **Buy missions**: Count blue cards purchased
- **Damage missions**: Trigger when dealing 6+ damage in a round
- **Heal missions**: Track total healing across all rounds (capped at goal)
- **Shield missions**: Trigger when reaching 6 shield in a round

### Spectator Mode
Spectators can join any room and watch in real-time:
- Buy, select, and cosmetic buttons are disabled
- UI displays "SPECTATOR MODE" badge
- All game events are visible in real-time
- Ideal for content creators or learning the game

## Architecture

```
chaos-deckbuilder/
├── app.py                  # Backend: Flask routes, SocketIO handlers, game logic
├── templates/
│   ├── base.html           # Base HTML template
│   ├── index.html          # Landing page with create/join/daily forms
│   └── room.html           # Game room page with all UI panels
├── static/
│   ├── app.js              # Frontend: game state, rendering, event handlers, sound synthesis
│   ├── style.css           # Styling, animations, themes
│   └── sounds/             # Sound files (legacy - now synthesized via Web Audio API)
└── README.md
```

### Backend (app.py)
- **Flask routes**: `/` (home), `/create` (room creation), `/join` (join room), `/daily` (daily challenge), `/room/<code>` (game room view)
- **SocketIO events**: `join_game`, `buy_card`, `submit_cards`, `set_cosmetic`, `send_emote`, `boss_rush`, `trade_request`
- **Game logic**: `resolve_round`, `compute_bundle`, `effective_cost`, `maybe_play_bot_turn`
- **State management**: In-memory `ROOMS` dictionary with full serialization per client

### Frontend (static/app.js)
- **Web Audio API**: Procedural sound synthesis (no external files needed)
- **State rendering**: Modular render functions for each panel (stats, market, hand, log, cosmetics, mission, achievements)
- **Event-driven**: SocketIO state updates trigger full re-renders with smooth animations

## Feature Implementation Details

| Feature | Files Changed | Lines Added | Approach |
|---------|--------------|-------------|----------|
| SoundFX | `static/app.js` | ~200 | Web Audio API oscillators, noise buffer for hit, chord sequences for lock/win |
| Card Reveal | `static/app.js`, `static/style.css` | ~100 | CSS keyframe animation `cardReveal` with perspective rotateY, legendary pulse |
| Emotes | `app.py`, `static/app.js`, `templates/room.html` | ~80 | SocketIO event + floating CSS animation |
| Spectator | `app.py`, `static/app.js` | ~60 | `is_spectator` flag in player state, UI disable logic |
| Post-Game Stats | `app.py`, `static/app.js`, `static/style.css` | ~150 | `round_history` array, bar chart with CSS flexbox |
| Archetype Badge | `static/app.js` | ~40 | Tribe counting algorithm on hand, gradient badge |
| Tutorial | `templates/room.html`, `static/app.js` | ~60 | Modal with localStorage persistence |
| Daily Challenge | `app.py`, `templates/index.html` | ~50 | Date-seeded route with dedicated form |
| Boss Rush | `app.py` | ~20 | SocketIO event replacing bot with boss deck |
| Trading Post | `app.py`, `static/app.js` | ~50 | SocketIO event exchanging random card between players |
| Mission Progress | `app.py`, `static/app.js`, `static/style.css` | ~100 | Capped progress values, progress bar CSS, percentage display |

## API Reference

### HTTP Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Home page with create/join/daily forms |
| POST | `/create` | Create a new room |
| POST | `/join` | Join an existing room |
| POST | `/daily` | Start a daily challenge |
| GET | `/room/<room_code>` | Game room view |

### SocketIO Events (Client → Server)

| Event | Data | Description |
|-------|------|-------------|
| `join_game` | `{room, username, spectator}` | Join a room (spectator optional) |
| `buy_card` | `{room, index}` | Buy market card at index |
| `submit_cards` | `{room, cards}` | Lock in up to 3 selected cards |
| `set_cosmetic` | `{room, category, value}` | Change table/back/particles |
| `send_emote` | `{room, emote}` | Send an emote reaction |
| `boss_rush` | `{room}` | Start boss rush mode |
| `trade_request` | `{room}` | Request a trade with opponent |

### SocketIO Events (Server → Client)

| Event | Data | Description |
|-------|------|-------------|
| `state` | Game state object | Full game state update |
| `emote_received` | `{username, emote}` | Emote from another player |
| `trade_response` | `{success, card, from}` | Trade result |
| `error_message` | `{message}` | Error notification |

### Game State Object

```python
{
    "room": str,
    "phase": str,           # "lobby", "shop", "battle", "game_over"
    "round": int,
    "daily": dict,          # {id, name, desc}
    "market": list,         # 5 serialized cards
    "you": {
        "username": str,
        "is_spectator": bool,
        "hp": int,
        "coins": int,
        "deck_count": int,
        "discard_count": int,
        "hand": list,
        "mission": {name, desc, goal, progress, done},
        "cosmetics": {table, back, particles},
        "achievements": list,
        "submitted": list,
        "bought": bool,
    },
    "opponents": list,      # Mini player objects
    "players_in_room": int,
    "round_history": list,  # Round-by-round combat data
    "log": list,            # Last 8 log lines
    "cosmetics_catalog": dict,
}
```

## Card Library

### Starter Deck
`Strike (x2), Guard (x2), Spark, Medic, Greed Coin, Venom Pin`

### Market Cards (weighted by rarity)
20+ cards across 7 tribes and 5 rarities. See `app.py` `CARD_POOL` for full card data.

## Daily Challenge

Same deck, same market, same daily modifier for everyone. The room code is always `DAILY` + last 4 digits of today's date (e.g., `DAILY0915`).

## Boss Rush

After joining a room, emit `boss_rush` to replace the bot opponent with "The Auditor" - a 40 HP boss with a hand-picked deck of powerful cards. Escalate difficulty by modifying The Auditor's deck programmatically.

## Trading Post

During any round after submitting cards, emit `trade_request` to receive a random card from your opponent's hand. You can only trade once per round if you haven't bought a card yet.

## Local Run

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python app.py
```

### 4. Open in browser

- **Home page**: http://localhost:5000
- **Create room**: Fill form on home page
- **Join room**: Enter 5-character room code
- **Daily challenge**: Click "Daily Challenge" form on home page

## Development

### Code Style
- Python: Standard indentation, descriptive variable names
- JavaScript: ES6+, const/let, template literals, arrow functions where appropriate
- CSS: CSS custom properties (variables), flexbox/grid layouts, keyframe animations
- No build step required - all files are served as-is

### File Conventions
- Backend logic: `app.py` (single file, all routes and handlers)
- Frontend logic: `static/app.js` (modular render functions, event listeners)
- Styling: `static/style.css` (CSS variables for theming, animation keyframes at bottom)
- Templates: `templates/` (Jinja2, extends `base.html`)

## License

Chaos Deckbuilder - Minimal Flask Deckbuilder
