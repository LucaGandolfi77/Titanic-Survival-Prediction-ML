# Tiny Lives

Tiny Lives is a cozy, browser-based 3D life-simulation PWA built with vanilla JavaScript and Three.js. Small autonomous characters wander a shared plaza, form relationships, hold conversations, and express moods. The app is designed to be lightweight, educational, and fun to tinker with.

This README documents how to run and develop the project, explains important systems and constants, and provides a detailed section describing how characters interact with one another.

---

## Quick start

- Open the project folder: `pwa/tiny-lives`
- Serve via a static HTTP server (service workers require HTTP/HTTPS). Example using Python 3 from the `pwa/tiny-lives` folder:

```bash
python3 -m http.server 8080
# then open http://localhost:8080 in your browser
```

- The app loads Three.js from the CDN and registers a service worker located at `service-worker.js`.

## Files

- `index.html` — app shell, toolbar, canvas container and links to `styles.css` and `app.js`.
- `styles.css` — UI styling including toolbar, profile panel, speech bubbles and responsive rules.
- `app.js` — the complete simulation engine (scene setup, characters, behavior, persistence, UI). This is the primary file to edit.
- `manifest.webmanifest` — PWA manifest for installability.
- `service-worker.js` — basic offline cache for the app shell (note: caching cross-origin CDN files may need adjustment).

## Important constants (tweak to change simulation behavior)

- `WORLD_SIZE` — world radius/extent.
- `TICK_MS` — simulation tick interval (how often needs decay and behavior decisions run).
- `TALK_RANGE` — maximum distance for conversations to start.
- `TALK_DURATION` — length of a conversation in milliseconds.
- `WANDER_SPEED` — base movement speed when characters walk.
- `NEED_DECAY` — how fast character needs decrease each tick.

These constants are defined at the top of `app.js` and are intentionally easy to modify for experimentation.

## How to use the UI

- Toolbar: Pause / Resume the simulation, change `Speed` (1×, 2×, 4×), and `Reset` to clear saved progress.
- Click a character in the 3D scene to open the profile panel showing mood, needs, traits and relationships.
- Speech bubbles appear briefly over characters during conversations.

## Persistence

The simulation saves periodically to `localStorage` under the key `tinyLivesSave`. Saving includes character needs, positions and relationships. Talks in-progress are saved as idle so conversations do not resume mid-talk on reload.

## Service Worker and offline

`service-worker.js` caches the app shell for offline use. Note: caching external CDN resources (like Three.js) may cause SW install warnings in some browsers due to cross-origin policies — if you encounter SW install errors, host the Three.js file locally in the project and update `index.html` and `service-worker.js` accordingly.

---

## Character systems and interactions (detailed)

This section explains the internal rules that govern how characters behave and interact. If you want to tune the simulation, this is where to start.

### Character state

Each character object maintains these core properties (see `createCharacters` in `app.js`):

- `id`, `name`, `age`, `traits`, `bio`, `avatar`, `color` — static metadata.
- `mood` — derived value representing overall well-being (0–100).
- `energy`, `hunger`, `social`, `fun`, `hygiene` — primary needs (0–100) that decay over time.
- `money` — currency (present but not heavily used by default).
- `currentAction` — `'idle' | 'walking' | 'resting' | 'talking'`.
- `relationships` — map of other character ids → score (0–100+), increased by social interactions.

### Needs, mood and trait effects

- Needs decay each simulation tick via `decayNeeds()` using `NEED_DECAY` and per-trait modifiers (defined in `TRAIT_EFFECTS`).
- `updateMood()` computes mood as a smoothed average of the needs; some traits alter mood recovery speed.
- Traits (e.g., `Outgoing`, `Shy`, `Lazy`) change behavior probabilities and decay multipliers via the `traitFactor()` helper. For example:
  - `Outgoing` increases `talkChance` and social decay (more social activity).
  - `Shy` decreases `talkChance` and reduces social decay (less likely to start conversations).

### Movement and idle behavior

- Characters alternate between `idle` and `walking`. When idling long enough they pick a new random target and switch to `walking` (chance modulated by `moveChance` trait factor).
- When walking they move toward `targetX/targetZ` each render frame using `updateMovement(delta)` and play a small bob animation.

### When conversations start

Conversations are initiated in `checkProximityInteractions()`:

1. For each pair of characters A, B that are not already talking or resting, compute the planar distance.
2. If the distance is less than `TALK_RANGE`, compute a base `chance` to start talking (default ~0.25).
3. Multiply the chance by both characters' `talkChance` trait factors.
4. If either character has very low `social` need, the chance is increased (they are more likely to accept social contact).
5. If Math.random() < chance, `startConversation(A, B)` is called.

### Conversation effects

- `startConversation` sets both participants to `talking` and sets their `talkTimer` (duration controlled by `TALK_DURATION`).
- Social and relationship boosts are applied immediately: `social` is increased by `NEED_TALK_BOOST`, `fun` receives a small bump, and the relationship score between A and B increases by `REL_TALK_BOOST + random(0..3)`.
- Each participant displays a random line from the `SPEECH_LINES` pool as a speech bubble. The speech pool contains silly, snarky, and crazy lines to give flavor to interactions.

### Conversation termination

- When the talk timer expires, the participants revert to `idle`, and a short idle timeout is applied to avoid immediately restarting conversation loops.

### Relationship stages

Relationships are measured as a numeric score per pair. The UI groups them into stages via `RELATIONSHIP_STAGES`:

- `Stranger`: 0–14
- `Acquaintance`: 15–39
- `Friend`: 40–69
- `Close Friend`: 70+

Interaction boosts from conversations gradually move relationships upward, and the profile panel shows both numeric score and stage.

### Edge cases and special behavior

- If energy is very low a character goes into `resting` to recover energy; resting prevents talking and walking until energy is sufficient.
- Characters flagged `talking` are excluded from new conversation checks until their timer ends.

---

## Modding and development notes

- To modify the speech pool, edit the `SPEECH_LINES` array at the top of `app.js` — lines are randomly sampled during conversations.
- To add new traits, extend `TRAIT_EFFECTS` and reference the new keys in the simulation (e.g., `energyDecay`, `moveChance`, `talkChance`).
- To add behaviors (e.g., shops, events), add world objects in `buildWorld()` and expand `behaviorUpdate()` / `checkProximityInteractions()` to consider event triggers.

## Troubleshooting

- If the scene does not render, open the browser console and ensure the Three.js script loaded and no SW errors occurred. If service worker prevents updates, unregister it from DevTools > Application > Service Workers.
- If speeches do not appear, confirm DOM element `scene-container` exists and that `bubbleContainer` is appended in `app.js` boot sequence.

## Credits

Built as part of a learning/demo collection. Feel free to fork, tweak traits, and contribute improvements.

---

If you want, I can now search the workspace and update other projects that use speech lines (for example to expand dialogue pools there as well). Alternatively I can add translations or a developer guide describing tuning parameters.
