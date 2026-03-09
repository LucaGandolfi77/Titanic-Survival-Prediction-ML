# Briscola — Single-player Web Demo

A lightweight single-player implementation of the classic Italian card game "Briscola".
This demo runs a Flask server as the backend and a simple client UI in the `client` folder. You play against a basic AI.

## Features
- Full Briscola trick logic and scoring
- Simple AI opponent with heuristic play
- In-memory single-game state served by Flask

## Quick start

1. Open a terminal in the `briscola` folder:

```bash
cd briscola
```

2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

3. Install dependencies:

```bash
pip install flask flask-cors
```

4. Run the server:

```bash
python server.py
```

5. Open your browser at `http://localhost:5000` to use the client UI.

## API Endpoints
- `POST /api/new_game` — start a new game; returns full client state JSON.
- `GET  /api/game_state` — returns current game state for the client.
- `POST /api/play_card` — play a card as the player. Body JSON: `{ "card": "R_S" }` where `R_S` is `RANK_SUIT` (e.g. `A_C`). Returns updated state.

The client (`client/index.html` and `client/game.js`) uses these endpoints to run the game flow.

## Important files
- Server: [briscola/server.py](briscola/server.py)
- Client UI: [briscola/client/index.html](briscola/client/index.html)

## Notes for developers
- Game state is stored in the global `game` variable in `server.py`. Restarting the server resets any active game.
- Card encoding uses `RANK_SUIT` strings (e.g. `A_C`, `3_D`). See `server.py` helpers for utilities like `card_value()` and `trick_winner()`.
- The AI logic lives in `ai_play()` inside `server.py`. It's deterministic but simple — a good spot to add improvements.

## Contributing ideas
- Add persistent multiplayer support (WebSockets / Redis)
- Improve AI strategy and add difficulty levels
- Add visual improvements and animations in the client

Enjoy playing Briscola!
