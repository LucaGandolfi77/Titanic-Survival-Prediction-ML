# STEP 2 — Player + FarmSystem

## Output these classes: Player, FarmSystem

---

## Player
16x16 sprite drawn each frame with canvas primitives.

### Sprite drawing (warm palette):
- Boots:    6x4px rectangle, #b84c2a, at bottom of sprite
- Body:     10x8px rectangle, #d4a35a (shirt)
- Arms:     2x5px each side, #a06b3e
- Head:     8x7px, #e8b870 (skin tone)
- Hat:      10x4px, #5c3d1e (dark brim) + 8x5px #7a4e2d (top)
- Eyes:     2x2px #2d1b0e dots
- Direction offset: shift arm/leg position by 1px based on facing

### Movement:
- WASD or Arrow keys
- Speed: 1.5 tiles/second (24px/s)
- Diagonal movement normalized
- Footstep sound every 0.4s of movement

### Tools (cycle Q / E):
| Tool         | Icon color | Action on SPACE |
|--------------|------------|-----------------|
| Hoe          | #8b5e3c    | Till soil tile  |
| Watering Can | #5a9ac0    | Water crop      |
| Axe          | #a06b3e    | Chop tree       |
| Fishing Rod  | #d4a35a    | Start fishing   |
| Pickaxe      | #7a7060    | Mine rock       |
| Hand (default)| #e8b870   | Pick up / talk  |

Energy cost per action: Hoe=5, Water=3, Axe=8, Fish=2, Mine=6, Pick=1

### Stats:
- energy: 100 max, shown as amber bar top-left
- leaves: starts at 100 (currency)
- inventory: array of 8 slots {item, qty}

### Inventory HUD (bottom bar):
- 8 slots, each 36x36px, #2d1b0e background, #a06b3e border
- Selected slot: #a06b3e bright border, slight scale-up
- Item icon drawn as 16x16 colored shape inside slot
- Current Leaves: ♣ symbol + amount, #ffd890, bottom-right

---

## FarmSystem
Manages all soil/crop state.

### Data structure per soil tile:
```js
{ tilled: bool, watered: bool, crop: null | {
    type: 'turnip'|'carrot'|'pumpkin'|'tomato',
    day_planted: int,
    days_watered: int,
    stage: 0|1|2|3   // 3 = harvestable
}}
