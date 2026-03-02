# BUREAU OF MISPLACED PORTALS

A first-person 3D puzzle game set in a brutalist government office where gravity changes with every portal, logic is broken, and the only escape is paperwork.

## GAMEPLAY

You are an unnamed employee trapped in the Bureau of Misplaced Portals, a 12-room government office where every doorway is a portal that doesn't work correctly. To escape, you must:

1. **Collect the required documents:** Form 27-Γ, Rubber Stamp, Cabinet Key, Director's Coffee, Meeting Minutes
2. **Navigate 12 interconnected rooms** with incorrect portal installations
3. **Survive the Auditor** — a terrifying bureaucrat that pursues you when your sanity falls below 50
4. **Manage your sanity** as the environment becomes increasingly non-Euclidean
5. **File the forms correctly** at the EXIT DESK to be formally terminated and escape

## THREE ENDINGS

### NORMAL ENDING
Collect all required items (Form, Stamp, Key, Coffee, Minutes) and file the form at the EXIT DESK. You will be formally terminated and released into the mundane afternoon.

### SECRET ENDING ⭐
Collect all 8 **sticky notes** hidden throughout the office. This unlocks access to **Greg's Supply Closet**, where you find the Portal Remote. With this device, you can recalibrate all portals to work correctly and escape with Greg into freedom.

### BAD ENDING
Allow your sanity to drop below -100. You become a permanent NPC bureaucrat, shuffling papers endlessly through the corridors.

## CONTROLS

| Action | Key |
|--------|-----|
| Move Forward/Backward | W / S |
| Strafe Left/Right | A / D |
| Look Around | Mouse (click to lock cursor) |
| Jump | Spacebar |
| Crouch | C |
| Interact (pick up items) | E |
| Toggle Flashlight | F |
| Show Inventory | I |
| Pause Game | P or ESC |
| Run | Hold Shift |

## THE 12 ROOMS

**Room 0 - LOBBY**
The entry point with a reception desk and two guard NPCs. Your first portal (to Records) is here.

**Room 1 - RECORDS**
Filing cabinets everywhere. **Form 27-Γ is hidden in Cabinet #3 on the back wall.**

**Room 2 - INVERTED OFFICE**
Upside-down desks and chairs. Grappling with inverted gravity is required to navigate.

**Room 3 - INFINITE CORRIDOR**
A hallway that seems to stretch forever due to clever portal loops. **The loop lever is here** — pull it to break the infinite connection and move forward.

**Room 4 - BREAK ROOM**
Coffee machine and vending machine. **Director's Coffee is on the table** — it restores sanity and helps calm NPCs.

**Room 5 - VOID OFFICE**
Minimal, dark furniture. **The Rubber Stamp and Small Key are hidden here.** Approach carefully — the void vignette effect indicates spatial distortion.

**Room 6 - DIRECTOR'S OFFICE**
A large, imposing office with a **portrait frame whose eyes track your movement.** The **Master Key is on the desk** — it opens all locked areas. Director's portrait is quite unsettling.

**Room 7 - COPY ROOM**
Copy machines and paper everywhere. Mostly a transition space toward the conference area.

**Room 8 - CONFERENCE ROOM**
A long table with 8 chairs (8 bureaucrat NPCs sit here). **Meeting Minutes are on the table.** The Auditor may be lurking here if sanity gets too low.

**Room 9 - SERVER ROOM**
Server racks with blinking LEDs. **Portal Calibration Device is hidden here** — use it once to fix a broken portal connection.

**Room 10 - MAINTENANCE SHAFT**
A tight crawlspace (height 1.5 units — must crouch). The only way to reach the Exit Hall.

**Room 11 - EXIT HALL**
The final destination with an **EXIT DESK and escalator.** File your form here to win.

## THE PORTAL TYPES

Greg, the Bureau's portal maintenance technician, has installed each portal incorrectly. Each type exhibits unique properties:

### Normal Portal (Blue)
Standard connection between rooms. Gravity unchanged. **Use these to navigate safely.**

### Upside-Down Portal (Red)
Entering inverts all gravity. Your "down" becomes "up." Movement controls invert. Very disorienting.

### Sideways Portals (Orange)
Rotate gravity 90 degrees. Left-sideways rotates right, right-sideways rotates left. Requires rotating your perspective.

### Forward-Down Portal
Gravity points toward you (into the screen). Highly unstable.

### Loop Portal (Green)
Incorrectly connects a room to itself. **Entering costs -5 sanity.** Lever in Room 3 breaks this loop.

### Mirror Portal (Blue)
Destination room is horizontally mirrored. Left becomes right, confusing spatial memory.

### Scaled Portal (Purple)
Destination appears 0.5× larger or smaller. Alters your perception of room scale.

### Time-Lag Portal (Purple)
Destination view is delayed by 60 frames (~1 second). Your inputs lag behind visual feedback. Extremely dangerous.

### VOID Portal (Black)
**DO NOT ENTER. Instant game over.** The void pulls at your consciousness. There is no escape from the void.

### Correct Portal (Blue, Rare)
Works as intended. Only accessible if you unlock the secret ending.

## SANITY SYSTEM

Your sanity (0–100, can go negative) decreases from:

- **Loop Portals:** -5 per entry
- **Void Proximity:** -0.5 per frame when within 5 units
- **NPC Interaction:** -3 when an NPC speaks to you
- **Time-Lag Portal:** -2 per frame in destination
- **Breathing Walls / Distortion:** -1 per frame at low sanity

Sanity recovers from:

- **Collecting Coffee:** +20
- **Completing Objectives:** +10 per objective
- **Using Correct Portals:** +5
- **Time:** Passive recovery +0.1 per frame while safe

### Sanity Thresholds

- **70–100:** Normal vision. Green health bar.
- **50–70:** Mild hallucinations. Yellow bar.
- **30–50:** Screen warp and chromatic aberration. Red bar.
- **<0:** Desaturation and noise overlay. Purple bar.
- **<−50:** Auditor spawns and hunts you.
- **≤−100:** Game Over — you become a bureaucrat.

## THE AUDITOR (Boss)

When sanity falls below -50, a tall, thin figure in a red suit materializes in the EXIT HALL. **The Auditor glides through walls, ignores collision, and moves at 2 units per second toward you.**

Contact with The Auditor costs -20 sanity and ends the game immediately if sanity was already low.

**How to stop the Auditor:** If you have the Rubber Stamp, you can stamp the form in front of the Auditor. This triggers a 10-second animation as they "process" your paperwork, briefly paralyzing them.

The Auditor will say: **"YOUR PAPERWORK IS INCOMPLETE."**

## ITEMS & INVENTORY

Your inventory holds a maximum of **6 items** at once. Items can be collected by walking near them (within 2 units) and pressing **E**.

### Required Items for Normal Exit

1. **Form 27-Γ** (yellow document) — Records Room, Cabinet #3
2. **Rubber Stamp** (red pad) — Void Office
3. **Cabinet Key** (small brass key) — Void Office
4. **Director's Coffee** (brown cup) — Break Room
5. **Meeting Minutes** (bound papers) — Conference Room

### Optional Items

- **Master Key** — Director's Office. Opens all locked areas (not required for exit but helpful).
- **Portal Calibration Device** — Server Room. Can repair one broken portal connection.
- **Batteries** (×3) — Scattered across rooms. Recharge flashlight to 100%.
- **Sticky Notes** (×8) — Hidden in various rooms. Collect all 8 to unlock the secret ending.

## FLASHLIGHT SYSTEM

Your flashlight activates with **F** and drains battery at 0.5% per second of use. 

- **Battery at 100%:** Full brightness, white light
- **Battery at 50%:** Dimmed yellow light
- **Battery depleted:** Flashlight turns off (can still collect batteries to recharge)

Batteries fully recharge your flashlight to 100%.

## NPCs & DIALOG

Throughout the office, you'll encounter **bureaucrat NPCs** who patrol, sit at desks, or guard doorways. When you approach within 2 units, they stop and stare. If you stay near them for 3 seconds, they speak:

### Sample NPC Quotes
- "Have you submitted Form 27-B/6?"
- "Portal scheduled for Q4 2047."
- "Per regulation 44-Ω, all visitors must carry their badge."
- "I've been filing this cabinet for 11 years."
- "The badge says VISITOR but the scanner ate the log."

Each NPC interaction costs -3 sanity. They're not hostile, just... bureaucratic.

## TECHNICAL DETAILS

### Engine & Dependencies
- **Three.js r158** (via CDN, ES6 modules)
- **Pure Vanilla JavaScript** (no external frameworks beyond Three.js)
- **HTML5 Canvas + WebGL** (stencil buffer enabled)

### Rendering

**Portal Rendering (Stencil Buffer Technique):**
1. **Stencil Write Pass:** Render portal frame to stencil buffer with unique ID (no color output)
2. **Scene Pass:** Render destination room from virtual camera position, only where stencil ID matches
3. **Depth Repair:** Re-render portal frame to fix depth values for subsequent portals

Maximum recursive depth: 2 levels (portals visible through portals, but not infinitely).

### Physics

- **Custom AABB Collision** — Player collision against room bounds via axis-aligned bounding boxes
- **Raycasting** — Ground detection using ray-plane intersection
- **Variable Gravity** — Gravity vector lerped over 2 seconds when transiting portals
- **No External Physics Engine** — All physics implemented from first principles

### Audio

- **Web Audio API** — 100% synthesis, zero external audio files
- **Synthesized Sounds:**
  - Footsteps (noise burst at 150Hz)
  - Portal swoosh (sine sweep)
  - Item pickup (C5/E5/G5 arpeggio)
  - NPC dialog (sawtooth pitch wave)
  - Ambient hum (60Hz office tone)
  - Void rumble (30Hz sub-bass)
  - Lever pull (mechanical noise + ring)

Master gain at 0.3 (adjustable in settings). Context resume required on first user interaction.

### Non-Euclidean Effects

- **Breathing Walls** — Vertex shader deformation using Perlin noise on wall geometry
- **Screen Distortion** — Canvas overlay applying wave, chromatic aberration, desaturation, noise based on sanity
- **Infinite Corridor Effect** — CubeCamera reflection for infinite-seeming hallway
- **Time Lag Effect** — 60-frame buffer storing camera positions for delayed-view portals

### Build & Deployment

Simply open `index.html` in a modern browser (Chrome, Firefox, Safari, Edge). All code is transpiled-free (native ES6 modules).

**No build step required.**
**No dependencies beyond Three.js r158 (loaded via CDN).**

## DEBUGGING

Press the **DEBUG** toggle in Settings to enable:
- Collision box visualization
- Portal frustum visualization
- NPC pathfinding visualization
- Sanity meter debug output to console

## LORE

The Bureau of Misplaced Portals was once a normal government office. Then Greg, the portal technician, was tasked with upgrading the hallway doors to dimensional portals for "efficiency." Greg misread the specifications. Every portal was installed incorrectly. Gravity rotates. Rooms loop. The void lurks beyond certain doorways.

The Auditor is an entity that emerged from the corruption. It is neither alive nor dead, but *processed*. Its only purpose is to audit the paperwork of those trapped here.

Your only escape is to become complicit in the bureaucracy — file the form, get approved for termination, and leave.

Or, collect the hidden notes that Greg left behind. He realized his mistake and hid a master remote in his supply closet. With it, you can restore all portals to their correct configurations and help both yourself and Greg escape.

But first, you have to navigate 12 rooms of warped space and convince a bureaucrat to sign off on your existence.

## WINNING STRATEGIES

### Path to Normal Ending (Safest)
1. Collect Form 27-Γ from Records (Cabinet #3)
2. Collect Rubber Stamp from Void Office
3. Collect Cabinet Key from Void Office
4. Collect Coffee from Break Room
5. Collect Minutes from Conference Room
6. Navigate to Exit Hall (via Maintenance Shaft)
7. Interact with Exit Desk and file form

**Estimated time:** 15–20 minutes (first playthrough)

### Path to Secret Ending (Challenging)
1. Find all 8 sticky notes before reaching Exit Hall
   - Room 0 (Lobby): Near reception desk
   - Room 1 (Records): Behind cabinets
   - Room 3 (Corridor): On floor
   - Room 5 (Void Office): On desk
   - Room 6 (Director): In trash
   - Room 8 (Conference): On table
   - Room 9 (Server): On a rack
   - Room 11 (Exit Hall): Under escalator
2. Return to Lobby after collecting all notes
3. Portal #0 recalibrates to Greg's Supply Closet
4. Navigate the closet (one final surreal room)
5. Retrieve Portal Remote from Greg's office
6. Escape with purpose, not termination

**Estimated time:** 25–35 minutes (expert playthrough)

### Maintaining Sanity
- Avoid loop portals if possible
- Stay away from void portals entirely
- Minimize NPC interactions
- Collect coffee early and use it near bureaucrats to offset dialog penalties
- Use correct portals (rare) whenever available

## CREDITS

Designed and built as an exercise in non-Euclidean game design, variable gravity mechanics, particle systems, and Web Audio synthesis.

Inspired by Portal (Valve), The Office (Greg Daniels & Michael Schur), Braid (Jonathan Blow), and every government building that feels like an Escher painting.

---

**Welcome to the Bureau. Your forms are incomplete.**