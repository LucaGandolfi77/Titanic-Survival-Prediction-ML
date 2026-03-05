# PROJECT: Cozy Farm RPG (Stardew Valley-inspired)

You will build this game across 6 steps.
Each step I will give you a focused spec file.
After each step, output ONLY the JavaScript class(es) or
HTML section requested — no extra commentary.

## OUTPUT FORMAT (follow strictly)
- Each file you produce is a standalone JS module (ES6 class)
- At the end I will ask you to assemble everything into one index.html
- Use // FILE: ClassName.js as header comment in each output
- Maintain the warm cozy palette defined in Step 1 across all files

## GLOBAL CONSTANTS (available to all classes)
```js
const PALETTE = {
  bg:       ['#2d1b0e','#3d2510','#1a2e1a','#2a3d1a'],
  soil:     ['#5c3d1e','#7a4e2d','#8b5e3c','#a06b3e'],
  grass:    ['#4a6741','#5a7a50','#6b8f5e','#7da668'],
  accent:   ['#d4a35a','#e8b870','#f2c97a','#ffd890'],
  red:      ['#b84c2a','#d45f35','#e87048'],
  green:    ['#3d6b3a','#4e8045','#62975a'],
  sky:      ['#7ab8d4','#8fcae0','#a8daf0','#c5eaf8'],
  dawn:     ['#e8a87c','#d4805a'],
  ui:       { bg:'#2d1b0e', border:'#a06b3e', text:'#ffd890' },
  water:    ['#4a7fa8','#5a9ac0','#72b8d4'],
};
const TILE_SIZE = 16;
const MAP_W = 40, MAP_H = 30;
const CANVAS_W = 640, CANVAS_H = 480;
