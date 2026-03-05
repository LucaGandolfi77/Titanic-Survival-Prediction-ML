# Cozy Farm RPG

>A small browser-based cozy farming RPG built with plain JavaScript (ES modules), Canvas 2D and Web Audio.

This README explains how to run the game locally, play it, and troubleshoot common issues.

---

## Quick start (macOS / Linux)

1. Open a terminal and change into the game folder:

```bash
cd python/rpg
```

2. Run a simple static server (recommended — browsers require modules to be served over HTTP):

Option A — Python 3 built-in server:

```bash
python3 -m http.server 8000
# then open http://localhost:8000 in your browser
```

Option B — Node (if you have Node.js):

```bash
# using npx
npx serve . -l 8000
# or if you have http-server installed
npx http-server -p 8000
```

3. Open the game in your browser:

http://localhost:8000

Open DevTools (F12 / Cmd+Opt+I) if you encounter issues.

---

## Files of interest

- `index.html` — single module entry (`js/game.js`).
- `js/game.js` — application entry point, constructs systems and starts the loop.
- `js/` — game modules (player, tilemap, renderer, camera, farming, particles, ui, dialogue, shop, etc.).
- `css/` — `style.css` and `ui.css` for visual layout and overlays.

Edit these files to tweak behavior and visuals.

---

## Controls

- Move: `W A S D` or Arrow keys
- Use tool / interact: `Space`
- Cycle tools: `E` (next) / `Q` (previous)
- Open shop (when nearby): `B`
- Open/close journal: `J`
- Change selected inventory slot: `Tab`
- Pause / Close overlays: `Esc`
- Mouse: interact with shop overlays (click items)
- Mobile: touchscreen controls appear automatically on touch devices

Tools include: `hand` (harvest/pickup), `hoe` (till), `watering` (water crops), `axe` (cut trees), `pickaxe` (break rocks), `rod` (fish).

---

## Gameplay basics

- Planting: Till soil with the `hoe`, plant seeds (inventory items) into tilled soil, water with the `watering` tool. Crops grow over days and can be harvested with `hand`.
- Energy: Performing actions consumes `energy`. Energy resets when a new day begins.
- Currency: `leaves` are a currency used in the shop.
- Inventory: 8 quickslots; items automatically stack where possible.
- NPCs: Talk to NPCs to trigger dialogue and shops. Press `B` near the vendor (Lily) to open the shop.
- Journal & Quests: Press `J` to open the journal which tracks logs, quests and the bestiary.
- Save: The game saves to `localStorage` under the key `cozyfarm_save`. Saves occur on day advance and when the game triggers a save.

---

## Development notes

- Modules are ES modules — editing `js/game.js` and other files requires reloading the page.
- The canvas internal resolution is `640×480` (CSS can scale it up for crisp pixel art). Tile size and default globals are set in `js/game.js` if you need to change them (look for `window.TILE_SIZE`, `window.MAP_W`, etc.).
- Utilities are in `js/utils.js` (helpers used across modules).
- Particle presets live in `js/particles.js` and are triggered by farming/player actions.

---

## Troubleshooting / FAQ

- Q: The game is a blank screen or `Uncaught SyntaxError: Cannot use import statement outside a module` appears.
  - A: Make sure you run a local HTTP server (see Quick start). Loading `index.html` directly via `file://` may cause module and CORS errors.

- Q: `TypeError: Failed to fetch` or module 404s in the console.
  - A: Verify you started the server from the `python/rpg` folder and open the correct URL (http://localhost:8000).

- Q: Audio doesn't play / I hear nothing.
  - A: Browsers require a user gesture to unlock Web Audio. Interact with the page (click or press a key) and try again. Also check the tab's audio mute state.

- Q: My save is corrupted or I want to reset progress.
  - A: Open DevTools Console and run:

```js
localStorage.removeItem('cozyfarm_save');
location.reload();
```

- Q: I see `Module not found` or missing asset errors.
  - A: Check console for path and filename. Ensure files exist under `js/`, `css/` and the paths in `index.html` match. After edits, reload the page (no build step required).

- Q: Pygame / ravenhollow.py errors when running `python ravenhollow.py`.
  - A: That is a separate Python/Pygame program in this repo. It requires Python and Pygame installed and is unrelated to the browser game. To run it, use a Python virtualenv and `pip install -r requirements.txt` (if present) or `pip install pygame`.

---

## Contributing

Pull requests and fixes welcome. If you plan to modify major systems, open an issue first to discuss architecture and avoid overlapping changes.

---

## Credits

- Author / Maintainer: Your project repository owner
- Built with: Vanilla JavaScript, Canvas 2D, Web Audio

If you want, I can add a short `CONTRIBUTING.md`, a license file, or inline developer notes — tell me which.
