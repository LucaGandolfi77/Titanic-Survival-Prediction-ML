
***

### `01_style_world.md` — Visual palette + world map
```markdown
# STEP 1 — Renderer + TileMap + Camera

## Output these classes: Renderer, TileMap, Camera

---

## TileMap
40x30 grid of tiles, each 16x16px.
Store map as a 2D array of tile IDs.

### Tile types and their canvas drawing:
| ID | Name       | Drawing description |
|----|------------|---------------------|
| 0  | Grass      | Fill #5a7a50, add 2px #6b8f5e highlight top-left |
| 1  | Soil       | Fill #5c3d1e, rough texture with #7a4e2d dots |
| 2  | Tilled     | Fill #3d2510, horizontal #5c3d1e lines every 4px |
| 3  | Water      | Animated: cycle #4a7fa8 → #5a9ac0 shimmer |
| 4  | Stone path | Fill #7a7060, #8a8070 highlight, #606050 border |
| 5  | Sand       | Fill #c4a060, dotted #d4b070 noise |
| 6  | Flower     | Grass base + small 4px colored dot (rotate palette) |
| 7  | Tree       | 8x8 brown trunk (#5c3d1e) + 14px round #4e8045 canopy |
| 8  | Building   | #7a5a3a walls, #5c3d1e roof, 4x6 #ffd890 window glow |
| 9  | Mountain   | #5a5060 fill, #6a6070 highlight, triangle peak |
| 10 | Bridge     | #8b5e3c planks with #5c3d1e gaps horizontal |
| 11 | Mine entry | Dark #2d1b0e arch on #5a5060 mountain face |

### World zone layout (use this exact map):
- Rows 0-8:   FOREST ZONE (tiles 7, 0, 6)
- Rows 9-18:  FARM ZONE (tiles 1, 2, 8 for barn, 0)
- Rows 19-22: TOWN ZONE (tiles 4, 8, 0)
- Rows 23-26: RIVER/LAKE (tiles 3, 10, 5)
- Rows 27-29: MINE ENTRANCE (tiles 9, 11)
- Column 38-39: Stone path border between zones

Place these landmark tiles manually:
- Fairy ring: (5, 5) — flower tiles in 3x3 circle pattern
- Farm cottage: (18, 11) — 3x3 building tiles
- Town well: (22, 20) — single building tile with special draw
- Mine entrance: (20, 28) — mine entry tile

### Tile animations:
- Water (ID 3): shimmer offset += 0.02 per frame, use sin wave
- Flowers (ID 6): sway ±1px on x using sin(time + tileX * 0.5)
- Building windows: flicker alpha 0.8-1.0 at night only

---

## Camera
- Follows player with lerp factor 0.08
- Clamps to world bounds (0 to MAP_W*TILE_SIZE - CANVAS_W)
- Exposes worldToScreen(wx, wy) → {x, y} method
- Exposes screenToWorld(sx, sy) → {x, y} method

---

## Renderer
Renders in strict layer order each frame:
1. Sky gradient (interpolated by time of day)
2. Ground layer (all tiles)
3. Object layer (trees, buildings, decorations)
4. Entity layer (NPCs, player, particles)
5. Weather layer (rain, falling leaves if active)
6. UI layer (HUD — drawn last, never scrolls)

Sky gradient interpolation:
- 6am-8am  dawn:   #d4805a → #e8a87c → #a8daf0
- 8am-6pm  day:    #7ab8d4 → #c5eaf8
- 6pm-8pm  dusk:   #e8a87c → #d4805a → #3d2510
- 8pm-2am  night:  #1a1428 → #0e0c18
Stars: draw 40 white dots at fixed positions, twinkle alpha = 0.4 + 0.6*sin(time*2+i)
